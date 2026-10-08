"""单卡完整训练闭环：RTX 5090 / BF16，兼容旧 v1 checkpoint。

部署到 /root/decoder-only-pretrain/scripts/，数据放在同项目的 data/，
checkpoint 保存到 checkpoints/。路径以脚本位置为准，不依赖启动目录。
"""
from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields, replace
import hashlib
from itertools import accumulate, islice
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

FINAL_CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FINAL_CODE_DIR.parent
if str(FINAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_CODE_DIR))
from model import ModelConfig, MinimalLlamaStyleCausalLM
from generation import load_tokenizer, run_beam_search_evaluation

MODEL_CONFIG = ModelConfig(
    vocab_size=48000, block_size=2048, hidden_size=1024, n_layers=24,
    n_heads=16, n_kv_heads=4, hidden_dim=3072, dropout=0.0,
)


@dataclass
class TrainingConfig:
    manifest_path: str = str(PROJECT_ROOT / "data/tokenized_manifest.json")
    tokenizer_path: str = str(PROJECT_ROOT.parent / "tokenizer/spiece.model")
    output_dir: str = str(PROJECT_ROOT / "checkpoints")
    epochs: int = 4
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    min_learning_rate_ratio: float = 0.1
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0
    num_workers: int = 0
    pin_memory: bool = True
    seed: int = 20260712
    log_every_steps: int = 10
    save_every_steps: int = 500
    epoch_checkpoint_every: int = 1
    resume: bool = True
    beam_size: int = 4
    max_new_tokens: int = 128
    length_penalty: float = 0.8
    early_stopping: bool = True
    eos_token_id: int = 3
    pad_token_id: int = 0
    bos_token_id: int = 2
    smoke_test: bool = False
    smoke_train_batches: int = 1
    smoke_valid_batches: int = 1
    smoke_max_new_tokens: int = 2
    compile_model: bool = True
    compile_mode: str = "default"
    fused_optimizer: bool = True
    loss_weighting: str = "microbatch"  # 旧语义；大 batch 调优可显式选择 sample。


TRAINING_CONFIG = TrainingConfig()
SMOKE_TEST = False
SMOKE_MODEL_CONFIG = ModelConfig(
    vocab_size=48000, block_size=2048, hidden_size=8, n_layers=1,
    n_heads=1, n_kv_heads=1, hidden_dim=16, dropout=0.0,
)


@dataclass
class Progress:
    epoch: int = 0
    shard_position: int = 0
    batch_position: int = 0
    global_step: int = 0
    tokens_seen: int = 0
    best_valid_loss: float = float("inf")
    epoch_loss_sum: float = 0.0
    epoch_loss_batches: int = 0


def format_duration(seconds: float) -> str:
    hours, remainder = divmod(math.ceil(max(seconds, 0)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TrainingTimer:
    """只计本次运行；近期速度取最近 10 个日志区间，跳过首次编译区间。"""
    def __init__(self, started_at: float):
        self.started_at = started_at
        self.intervals = deque(maxlen=10)
        self.epoch_overheads = []
        self.warming_up = True

    def start_epoch(self, progress: Progress) -> None:
        # 验证、生成和轮末保存单独估算，避免摊进下一轮的每步耗时。
        self.last_sample = (perf_counter(), progress.global_step, progress.tokens_seen)

    def format(self, progress: Progress, epoch_end_step: int, total_steps: int,
               remaining_epochs: int) -> str:
        now = perf_counter()
        previous_time, previous_step, previous_tokens = self.last_sample
        self.last_sample = (now, progress.global_step, progress.tokens_seen)
        elapsed = f"elapsed={format_duration(now - self.started_at)}"
        if self.warming_up:
            self.warming_up = False
            return elapsed + " | s/step=-- | tokens/s=-- | epoch_eta=-- | total_eta=-- (warming_up)"
        self.intervals.append((now - previous_time, progress.global_step - previous_step,
                               progress.tokens_seen - previous_tokens))
        seconds, steps, tokens = map(sum, zip(*self.intervals))
        step_seconds = seconds / steps
        overhead = sum(self.epoch_overheads) / len(self.epoch_overheads) if self.epoch_overheads else 0.0
        epoch_eta = max(epoch_end_step - progress.global_step, 0) * step_seconds + overhead
        total_eta = max(total_steps - progress.global_step, 0) * step_seconds + overhead * remaining_epochs
        return (f"{elapsed} | s/step={step_seconds:.3f} | tokens/s={tokens / seconds:,.0f} | "
                f"epoch_eta={format_duration(epoch_eta)} | total_eta={format_duration(total_eta)}")


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


class PackedShardDataset(Dataset):
    """保持原有 int32 [N, 2048] 数据契约，一次仅加载一个 shard。"""
    def __init__(self, shard_path: Path, expected_split: str) -> None:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        ids = shard.get("input_ids") if isinstance(shard, dict) else None
        if not isinstance(ids, torch.Tensor) or ids.dtype != torch.int32:
            raise ValueError(f"shard input_ids 必须为 int32 Tensor：{shard_path}")
        if ids.ndim != 2 or ids.shape[1] != 2048:
            raise ValueError(f"input_ids shape 必须是 [N, 2048]：{ids.shape}")
        if shard.get("split") != expected_split:
            raise ValueError(f"shard split 不匹配，期望 {expected_split}：{shard_path}")
        if ids.numel() and (ids.min() < 0 or ids.max() >= 48000):
            raise ValueError(f"token id 超出 [0, 48000)：{shard_path}")
        self.input_ids = ids

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.input_ids[index]


def load_data_manifest(training_config: TrainingConfig):
    manifest_path = _resolve_path(training_config.manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("vocab_size") != 48000 or manifest.get("seq_len") != 2048:
        raise ValueError("manifest 必须满足 vocab_size=48000、seq_len=2048")

    def entries(split):
        result = []
        for entry in manifest.get("splits", {}).get(split, {}).get("shards", []):
            original_path = _resolve_path(entry["path"])
            # 上传后优先使用 manifest 旁的 train/valid；保留原路径兼容旧布局。
            relocated_path = manifest_path.parent / split / original_path.name
            path = relocated_path if relocated_path.is_file() else original_path
            if not path.is_file():
                raise FileNotFoundError(f"缺少 {split} shard：{relocated_path}（原路径：{original_path}）")
            if not isinstance(entry.get("num_sequences"), int) or entry["num_sequences"] < 0:
                raise ValueError(f"shard num_sequences 必须是非负整数：{path}")
            result.append({**entry, "resolved_path": path})
        if not result:
            raise ValueError(f"manifest 中没有 {split} shard")
        return result
    return manifest, entries("train"), entries("valid")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device() -> torch.device:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("正式训练需要支持 BF16 的 CUDA GPU")
    return torch.device("cuda")


def build_optimizer(model: torch.nn.Module, training_config: TrainingConfig):
    decay, no_decay = [], []
    for parameter in model.parameters():  # parameters() 对共享 embedding/head 去重。
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    fused = training_config.fused_optimizer and next(model.parameters()).device.type == "cuda"
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": training_config.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=training_config.learning_rate, betas=(training_config.beta1, training_config.beta2),
        **({"fused": True} if fused else {}),
    )


def shuffled_train_entries(train_entries, seed, epoch):
    entries = list(train_entries)
    random.Random(seed + epoch).shuffle(entries)
    return entries


def optimizer_steps_per_epoch(train_entries, training_config: TrainingConfig, epoch=0) -> int:
    remaining = training_config.smoke_train_batches if training_config.smoke_test else math.inf
    steps = 0
    for entry in shuffled_train_entries(train_entries, training_config.seed, epoch):
        batches = min(math.ceil(entry["num_sequences"] / training_config.batch_size), remaining)
        steps += math.ceil(batches / training_config.gradient_accumulation_steps)
        remaining -= batches
    return steps


def build_scheduler(optimizer, total_steps, training_config):
    warmup = min(training_config.warmup_steps, max(total_steps - 1, 0))

    def multiplier(step):
        if step < warmup:
            return (step + 1) / warmup
        position = min((step - warmup) / (total_steps - warmup), 1.0)
        minimum = training_config.min_learning_rate_ratio
        return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * position))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Trainer:
    """训练状态只有一个所有者；对外入口仍为 run_training(config, model_config)。"""
    def __init__(
        self, training_config: TrainingConfig, model_config: ModelConfig, *, device=None,
    ):
        self.timer = TrainingTimer(perf_counter())
        self.config, self.model_config = training_config, model_config
        self._validate_config()
        set_random_seed(self.config.seed)
        self.device = choose_device() if device is None else torch.device(device)  # 显式 CPU 注入仅供测试。
        self.amp_dtype = torch.bfloat16 if self.device.type == "cuda" else None
        _, self.train_entries, self.valid_entries = load_data_manifest(self.config)
        self.raw_model = MinimalLlamaStyleCausalLM(model_config).to(self.device)
        self.optimizer = build_optimizer(self.raw_model, self.config)
        epoch_steps = [optimizer_steps_per_epoch(self.train_entries, self.config, e)
                       for e in range(self.config.epochs)]
        self.epoch_end_steps = list(accumulate(epoch_steps))
        self.steps_per_epoch, self.total_steps = epoch_steps[0], max(sum(epoch_steps), 1)
        self.scheduler = build_scheduler(self.optimizer, self.total_steps, self.config)
        self.tokenizer = load_tokenizer(self.config)
        self.run_metadata = self._metadata()
        self.output_dir = _resolve_path(self.config.output_dir)
        self.latest_path = self.output_dir / "checkpoint_latest.pt"
        self.best_path = self.output_dir / "checkpoint_best.pt"
        self.progress = Progress()
        if self.config.resume and self.latest_path.is_file():
            self.load_checkpoint(self.latest_path)
        elif self.config.resume:
            print(f"未找到 latest checkpoint，从头训练：{self.latest_path}", flush=True)
        # 保存和生成始终使用原模型；compile 不改变 v1 的权重名称。
        self.train_loss = self.loss
        if self.config.compile_model and self.device.type == "cuda":
            self.train_loss = torch.compile(self.loss, mode=self.config.compile_mode)
        self.raw_model.train()
        print(f"manifest={_resolve_path(self.config.manifest_path)} | "
              f"tokenizer={_resolve_path(self.config.tokenizer_path)} | checkpoints={self.output_dir}", flush=True)
        print(f"device={self.device} | amp={self.amp_dtype} | "
              f"batch={self.config.batch_size} × accumulation={self.config.gradient_accumulation_steps} | "
              f"parameters={sum(p.numel() for p in self.raw_model.parameters()):,} | "
              f"steps_per_epoch={self.steps_per_epoch} | total_steps={self.total_steps} | "
              f"compile={self.train_loss != self.loss} | "
              f"fused={self.optimizer.param_groups[0].get('fused')}", flush=True)
        print("计时：elapsed 为本次启动耗时；s/step、tokens/s 为最近 10 个日志区间的实际速度。"
              "ETA 首个日志区间预热后开始估算；验证、生成和轮末保存耗时在首轮完成后纳入。", flush=True)

    def _validate_config(self):
        c = self.config
        if self.model_config.vocab_size != 48000 or self.model_config.block_size != 2048:
            raise ValueError("ModelConfig 必须满足 vocab_size=48000、block_size=2048")
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("本脚本仅支持单卡，请勿使用多进程 torchrun 启动")
        positive = ("epochs", "batch_size", "gradient_accumulation_steps", "log_every_steps",
                    "save_every_steps", "epoch_checkpoint_every")
        if c.smoke_test:
            positive += ("smoke_train_batches", "smoke_valid_batches")
        for name in positive:
            if getattr(c, name) <= 0:
                raise ValueError(f"{name} 必须大于 0")
        if c.loss_weighting not in ("microbatch", "sample"):
            raise ValueError("loss_weighting 必须为 microbatch 或 sample")
        if not (0 <= c.min_learning_rate_ratio <= 1 and 0 <= c.beta1 < 1 and 0 <= c.beta2 < 1):
            raise ValueError("学习率比例或 Adam betas 超出范围")
        if c.learning_rate <= 0 or c.max_grad_norm <= 0 or min(c.num_workers, c.warmup_steps, c.weight_decay) < 0:
            raise ValueError("学习率/裁剪阈值必须为正；worker/warmup/weight_decay 不能为负")

    def _metadata(self):
        c = self.config
        return {
            **{key: getattr(c, key) for key in ("learning_rate", "warmup_steps", "min_learning_rate_ratio", "epochs")},
            "steps_per_epoch": self.steps_per_epoch, "total_steps": self.total_steps,
            "tokenized_manifest_sha256": sha256_file(_resolve_path(c.manifest_path)),
            "sentencepiece_model_sha256": sha256_file(_resolve_path(c.tokenizer_path)),
            "train_shard_count": len(self.train_entries), "valid_shard_count": len(self.valid_entries),
            "train_sequence_count": sum(e["num_sequences"] for e in self.train_entries),
            "valid_sequence_count": sum(e["num_sequences"] for e in self.valid_entries),
        }

    def autocast(self):
        return nullcontext() if self.amp_dtype is None else torch.autocast(self.device.type, dtype=self.amp_dtype)

    def loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 模型只读 labels；shift 仍在模型内部完成。不保留返回的整份 logits。
        return self.raw_model(input_ids, labels=input_ids)[1]

    def _loader(self, entry, split, seed):
        c = self.config
        dataset = PackedShardDataset(entry["resolved_path"], split)
        if len(dataset) != entry["num_sequences"]:
            raise ValueError(f"manifest num_sequences 与 shard 不符：{entry['resolved_path']}")
        return DataLoader(
            dataset, batch_size=c.batch_size, shuffle=split == "train", num_workers=c.num_workers,
            pin_memory=c.pin_memory and self.device.type == "cuda",
            generator=torch.Generator().manual_seed(seed), drop_last=False,
        )

    def checkpoint(self) -> dict[str, Any]:
        return {
            "checkpoint_version": 1, "model_state_dict": self.raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(), "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": None,  # 保留 v1 字段；BF16 不使用 GradScaler。
            "model_config": asdict(self.model_config), "training_config": asdict(self.config),
            **asdict(self.progress), **self.run_metadata,
            "python_random_state": random.getstate(), "numpy_random_state": np.random.get_state(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": [torch.cuda.get_rng_state(self.device)] if self.device.type == "cuda" else None,
            "position_semantics": "epoch/shard_position/batch_position point to the next batch",
            "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                        "device": str(self.device), "amp_dtype": str(self.amp_dtype),
                        "compile": self.config.compile_model and self.device.type == "cuda",
                        "fused": self.optimizer.param_groups[0].get("fused")},
        }

    def save_checkpoint(self, *paths: Path) -> None:
        if not paths:
            raise ValueError("至少需要一个 checkpoint 路径")
        source, *copies = dict.fromkeys(Path(p) for p in paths)
        atomic_write(source, lambda temporary: torch.save(self.checkpoint(), temporary))
        for target in copies:
            atomic_write(target, lambda temporary: shutil.copyfile(source, temporary))

    def load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        expected = {**self.run_metadata, "model_config": asdict(self.model_config),
                    "checkpoint_version": 1, "scaler_state_dict": None}
        keys = ("batch_size", "gradient_accumulation_steps", "seed", "weight_decay",
                "beta1", "beta2", "max_grad_norm", "smoke_test", "loss_weighting")
        if self.config.smoke_test:
            keys += ("smoke_train_batches", "smoke_valid_batches")
        old_config = {"loss_weighting": "microbatch", **checkpoint["training_config"]}
        changed = [key for key, value in expected.items() if checkpoint[key] != value]
        changed += [f"TrainingConfig.{key}" for key in keys
                    if old_config[key] != getattr(self.config, key)]
        if changed:
            raise ValueError(f"checkpoint 不兼容：{', '.join(changed)}")
        progress = Progress(**{field.name: checkpoint[field.name] for field in fields(Progress)})
        counts = (progress.epoch, progress.shard_position, progress.batch_position, progress.global_step,
                  progress.tokens_seen, progress.epoch_loss_batches)
        if (min(counts) < 0 or progress.epoch > self.config.epochs or progress.global_step > self.total_steps
                or progress.shard_position > len(self.train_entries)
                or (progress.shard_position == len(self.train_entries) and progress.batch_position)
                or (progress.epoch == self.config.epochs and (progress.shard_position or progress.batch_position))):
            raise ValueError("checkpoint progress 超出合法范围")
        if checkpoint["scheduler_state_dict"]["last_epoch"] != progress.global_step:
            raise ValueError("checkpoint scheduler 与 global_step 不一致")
        cuda_rng = checkpoint["torch_cuda_rng_state"]
        if (self.device.type == "cuda") != (cuda_rng is not None):
            raise ValueError("checkpoint CUDA RNG 与当前设备不兼容")
        state = checkpoint["optimizer_state_dict"]
        # 旧普通 AdamW 可恢复到 fused AdamW；加载前决定 step 状态所在设备。
        for saved, current in zip(state["param_groups"], self.optimizer.param_groups):
            saved.update({key: current.get(key) for key in ("fused", "foreach", "capturable", "differentiable")})
        self.raw_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(state)
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_cpu_rng_state"])
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng[0], self.device)
        self.progress = progress
        print(f"已恢复 checkpoint：{path} | epoch={progress.epoch} | shard={progress.shard_position} | "
              f"batch={progress.batch_position} | global_step={progress.global_step}", flush=True)

    def train_group(self, batches: list[torch.Tensor]) -> tuple[list[float], int, torch.Tensor]:
        """接收已校验的非空 batch 组；组末检查 loss/梯度后更新。"""
        sizes = [batch.size(0) for batch in batches]
        sample_count = sum(sizes)
        try:
            self.optimizer.zero_grad(set_to_none=True)
            detached_losses = []
            for batch, size in zip(batches, sizes):
                ids = batch.to(device=self.device, dtype=torch.long, non_blocking=True)
                with self.autocast():
                    loss = self.train_loss(ids)
                    scaled_loss = (loss * (size / sample_count)
                                   if self.config.loss_weighting == "sample" else loss / len(batches))
                scaled_loss.backward()
                detached_losses.append(loss.detach())
            loss_values = torch.stack(detached_losses).cpu().tolist()
            if not all(math.isfinite(value) for value in loss_values):
                raise FloatingPointError("train loss 出现 NaN/Inf，当前组未更新权重")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.raw_model.parameters(), self.config.max_grad_norm, error_if_nonfinite=True,
            )
            self.optimizer.step()
            self.scheduler.step()
            return loss_values, sum(batch.numel() for batch in batches), grad_norm.detach()
        finally:
            self.optimizer.zero_grad(set_to_none=True)

    def train_epoch(self, epoch: int) -> float:
        p, c = self.progress, self.config
        if p.epoch != epoch:
            raise ValueError("train_epoch 必须从 progress.epoch 开始")
        self.timer.start_epoch(p)
        self.raw_model.train()
        entries = shuffled_train_entries(self.train_entries, c.seed, epoch)
        first_shard = p.shard_position
        for shard in range(first_shard, len(entries)):
            if c.smoke_test and p.epoch_loss_batches >= c.smoke_train_batches:
                break
            loader = self._loader(entries[shard], "train", c.seed + epoch * 100000 + shard)
            start = p.batch_position if shard == first_shard else 0
            if start > len(loader):
                raise ValueError(f"checkpoint batch_position={start} 超过 shard batch 数 {len(loader)}")
            end = min(len(loader), start + c.smoke_train_batches - p.epoch_loss_batches) if c.smoke_test else len(loader)
            pending = []
            for position, batch in islice(enumerate(loader), start, end):
                pending.append(batch)
                if len(pending) < c.gradient_accumulation_steps and position + 1 < end:
                    continue
                losses, tokens, norm = self.train_group(pending)
                p.tokens_seen += tokens
                p.epoch_loss_sum += sum(losses)
                p.epoch_loss_batches += len(pending)
                p.global_step += 1
                p.shard_position, p.batch_position = shard, position + 1
                if p.batch_position == len(loader):
                    p.shard_position, p.batch_position = shard + 1, 0
                pending.clear()
                if p.global_step % c.log_every_steps == 0:
                    # 复用原有 grad_norm 读数的同步点，不额外增加 CUDA 同步。
                    grad_norm = float(norm)
                    timing = self.timer.format(p, self.epoch_end_steps[epoch], self.total_steps, c.epochs - epoch)
                    print(f"epoch={epoch + 1} | shard={shard + 1}/{len(entries)} | batch={position + 1}/{len(loader)} | "
                          f"global_step={p.global_step} | loss={losses[-1]:.6f} | "
                          f"lr={self.optimizer.param_groups[0]['lr']:.8e} | grad_norm={grad_norm:.6f} | "
                          f"tokens_seen={p.tokens_seen} | {timing}", flush=True)
                if p.global_step % c.save_every_steps == 0:
                    self.save_checkpoint(self.latest_path)
            if end == len(loader):
                p.shard_position, p.batch_position = shard + 1, 0
            del loader
        if not p.epoch_loss_batches:
            raise RuntimeError("当前 epoch 没有训练 batch")
        return p.epoch_loss_sum / p.epoch_loss_batches

    @torch.inference_mode()
    def validate(self) -> float:
        was_training = self.raw_model.training
        self.raw_model.eval()
        try:
            loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
            samples, processed = 0, 0
            for entry in self.valid_entries:
                loader = self._loader(entry, "valid", self.config.seed)
                limit = self.config.smoke_valid_batches - processed if self.config.smoke_test else len(loader)
                for batch in islice(loader, max(limit, 0)):
                    ids = batch.to(device=self.device, dtype=torch.long, non_blocking=True)
                    with self.autocast():
                        loss = self.loss(ids)
                    loss_sum.add_(loss, alpha=ids.size(0))
                    samples += ids.size(0)
                    processed += 1
                del loader
                if self.config.smoke_test and processed >= self.config.smoke_valid_batches:
                    break
            loss_sum = loss_sum.item()
            if not samples or not math.isfinite(loss_sum):
                raise FloatingPointError("valid 没有样本或 loss 出现 NaN/Inf")
            return loss_sum / samples
        finally:
            self.raw_model.train(was_training)

    def finish_epoch(self, epoch: int, train_loss: float, valid_loss: float) -> None:
        p = self.progress
        improved = valid_loss < p.best_valid_loss
        p.best_valid_loss = min(p.best_valid_loss, valid_loss)
        p.epoch, p.shard_position, p.batch_position = epoch + 1, 0, 0
        p.epoch_loss_sum, p.epoch_loss_batches = 0.0, 0
        paths = [self.latest_path]
        if improved:
            paths.append(self.best_path)
        if p.epoch % self.config.epoch_checkpoint_every == 0:
            paths.append(self.output_dir / f"checkpoint_epoch_{p.epoch:04d}.pt")
        self.save_checkpoint(*paths)
        run_beam_search_evaluation(
            self.raw_model, self.tokenizer, self.config, self.device, p.epoch, p.global_step,
            p.tokens_seen, train_loss, valid_loss, self.latest_path, self.amp_dtype,
        )
        if self.config.smoke_test:
            print("smoke test 已完成一个受限 epoch。", flush=True)

    def fit(self) -> dict[str, Any]:
        for epoch in range(self.progress.epoch, self.config.epochs):
            train_loss = self.train_epoch(epoch)
            evaluation_started = perf_counter()
            self.finish_epoch(epoch, train_loss, self.validate())
            self.timer.epoch_overheads.append(perf_counter() - evaluation_started)
        print(f"训练完成 | global_step={self.progress.global_step} | "
              f"elapsed={format_duration(perf_counter() - self.timer.started_at)} | total_eta=00:00:00", flush=True)
        return {
            "model": self.raw_model, "optimizer": self.optimizer, "scheduler": self.scheduler,
            "scaler": None, "progress": asdict(self.progress), "latest_checkpoint": self.latest_path,
        }


def run_training(training_config: TrainingConfig, model_config: ModelConfig) -> dict[str, Any]:
    return Trainer(training_config, model_config).fit()


def main():
    config, model = TRAINING_CONFIG, MODEL_CONFIG
    if SMOKE_TEST:
        config = replace(config, output_dir=str(FINAL_CODE_DIR / "smoke_checkpoints_kv_cache"),
                         epochs=1, batch_size=1, gradient_accumulation_steps=1, warmup_steps=0,
                         log_every_steps=1, save_every_steps=1, resume=False, smoke_test=True)
        model = SMOKE_MODEL_CONFIG
    run_training(config, model)


if __name__ == "__main__":
    main()
