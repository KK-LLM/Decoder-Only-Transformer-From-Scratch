"""单机 1/2/4 卡 BF16 训练；复用同目录 train.py 的配置、数据与通用工具。

torchrun --standalone --nproc_per_node=2 scripts/train_ddp.py
默认每卡 batch=8、全局 batch=32；原 model.py / generation.py / train.py 均不修改。
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields, replace
from datetime import timedelta
from itertools import accumulate, islice
import math
import os
from pathlib import Path
import random
import shutil
from time import perf_counter

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader

import train as base


@dataclass
class TrainingConfig(base.TrainingConfig):
    output_dir: str = str(base.PROJECT_ROOT / "checkpoints_ddp")
    save_every_steps: int = 3000
    max_steps: int = 0  # 仅限制本次运行，不改变完整训练的学习率计划；0 表示不限制。
    extend_training: bool = False  # 显式延长已完成的计划；epochs 始终是目标总轮数。
    mix_shards: int = 8  # 将这些分片的完整序列混合打乱；1 保留原单分片顺序。
    rewarm_lr: bool = False  # 显式为新增阶段建立一次升温 + 余弦衰减。
    peak_learning_rate: float | None = None  # 新阶段默认 1.2e-4；恢复时 None 沿用已保存值。
    rewarm_steps: int | None = None  # 新阶段默认 1000 个 optimizer 更新步。
    end_learning_rate: float | None = None  # 新阶段默认 1e-5。


class ShardBatches:
    """先保持原 microbatch 分组，再按 rank 分配；无真实样本的尾部槽位仅供零梯度同步。"""
    def __init__(self, size, batch_size, rank, world_size, seed, start, end):
        self.size, self.batch_size = size, batch_size
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.start, self.end = start, end

    def __len__(self):
        return math.ceil((self.end - self.start) / self.world_size)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        # 原 DataLoader 在 RandomSampler 之前消耗一次 base_seed，保持原样本顺序。
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        order = torch.randperm(self.size, generator=generator).tolist()
        for first in range(self.start, self.end, self.world_size):
            position = first + self.rank
            yield (order[position * self.batch_size:(position + 1) * self.batch_size]
                   if position < self.end else order[:1])


class TrainingLoss(torch.nn.Module):
    """仅取原模型返回的 loss；不替换其 forward 或内部模块。"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids):
        return self.model(ids, labels=ids)[1]


class Trainer:
    def __init__(self, config, model_config=base.MODEL_CONFIG, *, device=None):
        self.config, self.model_config = config, model_config
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device = torch.device(device) if device is not None else base.choose_device()
        self.amp_dtype = torch.bfloat16 if self.device.type == "cuda" else None
        self.group_batches = config.gradient_accumulation_steps * self.world_size
        self._validate_config()
        self.timer, self.progress = base.TrainingTimer(perf_counter()), base.Progress()
        _, self.train_entries, self.valid_entries = base.load_data_manifest(config)
        self.output_dir = base._resolve_path(config.output_dir)
        self.latest_path = self.output_dir / "checkpoint_latest.pt"
        self.best_path = self.output_dir / "checkpoint_best.pt"
        if self.latest_path.is_file() and not config.resume:
            raise FileExistsError(f"输出目录已有 checkpoint，请续训或使用新目录：{self.output_dir}")
        phase_options = (config.peak_learning_rate, config.rewarm_steps, config.end_learning_rate)
        if (config.extend_training or config.rewarm_lr or any(value is not None for value in phase_options)) and not self.latest_path.is_file():
            raise FileNotFoundError(f"追加训练必须从已有 latest checkpoint 恢复：{self.latest_path}")
        checkpoint = torch.load(self.latest_path, map_location="cpu", weights_only=False) if self.latest_path.is_file() else None
        self.sampling = dict(mix_shards=config.mix_shards, mix_start_epoch=0)
        if checkpoint is not None:
            if checkpoint.get("checkpoint_version") not in (2, 3, 4, 5):
                raise ValueError("本入口只恢复 DDP v2/v3/v4/v5 checkpoint；旧单卡 checkpoint 不作隐式迁移")
            if checkpoint.get("checkpoint_version") >= 4:
                self.sampling = checkpoint["contract"].get("sampling")
                self.count_steps(config.epochs, self.sampling)
                if self.sampling["mix_shards"] != config.mix_shards:
                    raise ValueError("续训条件不一致：mix_shards；请保持 checkpoint 的混合池大小")
            else:
                # 旧 checkpoint 当前轮维持旧顺序；轮末处理也完成后才能切换采样。
                pending = any(checkpoint[key] for key in ("shard_position", "batch_position", "epoch_loss_sum", "epoch_loss_batches"))
                self.sampling["mix_start_epoch"] = checkpoint["epoch"] + int(pending) if config.mix_shards > 1 else 0
        self.epoch_steps = self.count_steps(config.epochs, self.sampling)
        self.epoch_end_steps = list(accumulate(self.epoch_steps))
        self.total_steps = sum(self.epoch_steps)
        self.schedule_total_steps = self.total_steps
        self.schedule_plan = dict(epochs=config.epochs, **self.sampling)
        self.lr_phase = None
        if not self.total_steps:
            raise ValueError("训练数据没有有效 batch")
        base.set_random_seed(config.seed)
        self.raw_model = base.MinimalLlamaStyleCausalLM(model_config).to(self.device)
        self.optimizer = base.build_optimizer(self.raw_model, config)
        self.scheduler = base.build_scheduler(self.optimizer, self.total_steps, config)
        loss_module = TrainingLoss(self.raw_model)
        self.ddp = (DDP(loss_module, device_ids=[self.device.index] if self.device.type == "cuda" else None,
                        broadcast_buffers=False, gradient_as_bucket_view=True)
                    if self.world_size > 1 else loss_module)
        # 先 DDP 后 compile，让编译器保留按梯度 bucket 重叠通信的机会。
        self.train_loss = (torch.compile(self.ddp, mode=config.compile_mode)
                           if config.compile_model else self.ddp)
        self.contract = self._contract()
        self.tokenizer = base.load_tokenizer(config) if self.rank == 0 else None
        base.set_random_seed(config.seed + self.rank)
        if checkpoint is not None:
            self.load_checkpoint(self.latest_path, checkpoint)
        del checkpoint
        self.log(f"DDP | world_size={self.world_size} | per_gpu_batch={config.batch_size} | "
                 f"accumulation={config.gradient_accumulation_steps} | "
                 f"global_batch={config.batch_size * self.group_batches} | amp={self.amp_dtype} | "
                 f"compile={config.compile_model} | fused={self.optimizer.param_groups[0].get('fused')} | "
                 f"steps_per_epoch={self.epoch_steps} | checkpoints={self.output_dir}")
        self.log("tokens/s 为所有卡合计；计时跳过首个日志区间，轮末验证/生成/保存耗时随后纳入 ETA。")
        self.log("推理展示：每轮约 50% 和 100% 各一次；中途先更新 latest，完整验证仍在轮末。")
        self.log(f"数据混合 | mix_shards={config.mix_shards} | 从第 {self.sampling['mix_start_epoch'] + 1} 轮生效 | "
                 "pool 为混合池序号；只打乱完整序列，序列内部 token 顺序不变")

    def _validate_config(self):
        c = self.config
        if (c.extend_training or c.rewarm_lr) and not c.resume:
            raise ValueError("追加训练必须启用 resume")
        if not c.resume and any(value is not None for value in (c.peak_learning_rate, c.rewarm_steps, c.end_learning_rate)):
            raise ValueError("升温参数只用于 checkpoint 续训")
        positive = (c.epochs, c.batch_size, c.gradient_accumulation_steps, c.log_every_steps,
                    c.save_every_steps, c.epoch_checkpoint_every, c.mix_shards)
        if c.smoke_test:
            positive += (c.smoke_train_batches, c.smoke_valid_batches)
        if min(positive) <= 0 or min(c.num_workers, c.warmup_steps, c.weight_decay, c.max_steps) < 0:
            raise ValueError("batch/epoch/保存间隔须为正，worker/warmup/weight_decay/max_steps 不能为负")
        if (c.loss_weighting not in ("microbatch", "sample") or c.learning_rate <= 0 or c.max_grad_norm <= 0
                or not (0 <= c.min_learning_rate_ratio <= 1 and 0 <= c.beta1 < 1 and 0 <= c.beta2 < 1)):
            raise ValueError("loss 权重、学习率或 Adam 参数无效")
        if self.device.type == "cuda" and (self.model_config.vocab_size, self.model_config.block_size) != (48000, 2048):
            raise ValueError("正式训练的模型必须匹配 vocab_size=48000、seq_len=2048")

    def count_steps(self, epochs, sampling):
        if (type(epochs) is not int or epochs <= 0 or not isinstance(sampling, dict)
                or set(sampling) != {"mix_shards", "mix_start_epoch"}
                or any(type(value) is not int for value in sampling.values())
                or sampling["mix_shards"] <= 0 or not 0 <= sampling["mix_start_epoch"] <= epochs):
            raise ValueError("采样或学习率计划无效")
        return [sum(math.ceil(n / self.group_batches) for _, n in self.plan(epoch, sampling)) for epoch in range(epochs)]

    def plan(self, epoch, sampling=None):
        sampling = self.sampling if sampling is None else sampling
        width = sampling["mix_shards"] if epoch >= sampling["mix_start_epoch"] else 1
        entries = base.shuffled_train_entries(self.train_entries, self.config.seed, epoch)
        remaining = self.config.smoke_train_batches if self.config.smoke_test else math.inf
        for first in range(0, len(entries), width):
            group = entries[first:first + width]
            count = min(math.ceil(sum(entry["num_sequences"] for entry in group) / self.config.batch_size), remaining)
            yield group, count
            remaining -= count

    def log(self, message):
        if self.rank == 0:
            print(message, flush=True)

    def primary_call(self, function):
        """仅 rank 0 执行保存/验证/生成，并将结果或错误同步给其他进程。"""
        result = [None, None]
        if self.rank == 0:
            try:
                result[0] = function()
            except Exception as error:
                result[1] = f"{type(error).__name__}: {error}"
        if self.world_size > 1:
            dist.broadcast_object_list(result, src=0)
        if result[1]:
            raise RuntimeError(result[1])
        return result[0]

    def autocast(self):
        return torch.autocast("cuda", dtype=self.amp_dtype) if self.amp_dtype else nullcontext()

    def _contract(self):
        keys = ("epochs", "batch_size", "gradient_accumulation_steps", "seed", "learning_rate", "warmup_steps",
                "min_learning_rate_ratio", "weight_decay", "beta1", "beta2", "max_grad_norm", "loss_weighting",
                "compile_model", "compile_mode", "fused_optimizer", "smoke_test", "smoke_train_batches", "smoke_valid_batches")
        return dict(world_size=self.world_size, model_config=asdict(self.model_config), total_steps=self.total_steps,
                    sampling=self.sampling,
                    settings={key: getattr(self.config, key) for key in keys}, torch=str(torch.__version__),
                    device_type=self.device.type, manifest=base.sha256_file(base._resolve_path(self.config.manifest_path)),
                    tokenizer=base.sha256_file(base._resolve_path(self.config.tokenizer_path)),
                    model_source=base.sha256_file(base.FINAL_CODE_DIR / "model.py"))

    def rng_state(self):
        return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state(self.device) if self.amp_dtype else None)

    def check_lr_phase(self, phase, step, total_steps):
        keys = {"start_step", "end_step", "start_lrs", "peak_lr", "end_lr", "warmup_steps"}
        if not isinstance(phase, dict) or set(phase) != keys:
            raise ValueError("升温阶段元数据无效")
        start, end, warmup = (phase[key] for key in ("start_step", "end_step", "warmup_steps"))
        rates = phase["start_lrs"]
        if (any(type(value) is not int for value in (start, end, warmup))
                or not 0 <= start <= step or not start < end <= total_steps or not 0 < warmup < end - start
                or start not in [0, *self.epoch_end_steps] or end not in self.epoch_end_steps
                or not isinstance(rates, list) or len(rates) != len(self.optimizer.param_groups)):
            raise ValueError("升温阶段起止步、升温长度或参数组无效")
        values = [*rates, phase["peak_lr"], phase["end_lr"]]
        if (any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
                or min(rates) < 0 or not 0 < phase["end_lr"] < phase["peak_lr"] or max(rates) > phase["peak_lr"]):
            raise ValueError("升温阶段学习率必须有限，且峰值不低于起点并高于正的末端值")

    def build_scheduler(self, original_steps, phase):
        if phase is None:
            return base.build_scheduler(self.optimizer, original_steps, self.config)
        functions = []
        for group, start_lr in zip(self.optimizer.param_groups, phase["start_lrs"]):
            def multiplier(step, start_lr=start_lr, base_lr=group["initial_lr"]):
                offset = max(0, step - phase["start_step"])
                warmup = phase["warmup_steps"]
                if offset <= warmup:
                    rate = start_lr + (phase["peak_lr"] - start_lr) * offset / warmup
                else:
                    position = min((offset - warmup) / (phase["end_step"] - phase["start_step"] - warmup), 1.0)
                    rate = phase["end_lr"] + (phase["peak_lr"] - phase["end_lr"]) * 0.5 * (1 + math.cos(math.pi * position))
                return rate / base_lr
            functions.append(multiplier)
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, functions)

    def save_checkpoint(self, *paths):
        states = [None] * self.world_size if self.rank == 0 else None
        if self.world_size > 1:
            dist.gather_object(self.rng_state(), states, dst=0)
        else:
            states[0] = self.rng_state()

        def write():
            checkpoint = dict(checkpoint_version=5, model_state_dict=self.raw_model.state_dict(),
                              optimizer_state_dict=self.optimizer.state_dict(), scheduler_state_dict=self.scheduler.state_dict(),
                              model_config=asdict(self.model_config), training_config=asdict(self.config),
                              contract=self.contract, schedule_total_steps=self.schedule_total_steps, schedule_plan=self.schedule_plan,
                              lr_phase=self.lr_phase, rank_rng_states=states, **asdict(self.progress))
            source, *copies = dict.fromkeys(paths)
            base.atomic_write(source, lambda temporary: torch.save(checkpoint, temporary))
            for target in copies:
                base.atomic_write(target, lambda temporary: shutil.copyfile(source, temporary))
        self.primary_call(write)

    def load_checkpoint(self, path, checkpoint=None):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False) if checkpoint is None else checkpoint
        version = checkpoint.get("checkpoint_version")
        if version not in (2, 3, 4, 5):
            raise ValueError("本入口只恢复 DDP v2/v3/v4/v5 checkpoint；旧单卡 checkpoint 不作隐式迁移")
        saved_contract = checkpoint["contract"]
        old_epochs = saved_contract["settings"]["epochs"]
        if version < 4:
            pending = any(checkpoint[key] for key in ("shard_position", "batch_position", "epoch_loss_sum", "epoch_loss_batches"))
            start = checkpoint["epoch"] + int(pending) if self.config.mix_shards > 1 else 0
            if self.sampling != dict(mix_shards=self.config.mix_shards, mix_start_epoch=start):
                raise ValueError("旧 checkpoint 的采样计划尚未准备，请通过 resume 构造 Trainer 恢复")
        extending = self.config.extend_training and self.config.epochs > old_epochs
        expected = self.contract
        old_steps = self.count_steps(old_epochs, dict(mix_shards=1, mix_start_epoch=0)) if version < 4 else self.epoch_steps[:old_epochs]
        if extending or version < 4:
            expected = {**self.contract, "total_steps": sum(old_steps),
                        "settings": {**self.contract["settings"], "epochs": old_epochs}}
        if version < 4:
            expected.pop("sampling")
        if self.config.epochs != old_epochs and not extending:
            raise ValueError("续训条件不一致：epochs；增加目标轮数须使用 --extend-training")
        if saved_contract != expected:
            changed = [key for key, value in expected.items() if saved_contract.get(key) != value]
            raise ValueError(f"续训条件不一致：{', '.join(changed)}；请保持卡数、全局分组和训练计划")
        p = base.Progress(**{field.name: checkpoint[field.name] for field in fields(base.Progress)})
        if extending and (p.epoch != old_epochs or p.global_step != saved_contract["total_steps"]
                          or any((p.shard_position, p.batch_position, p.epoch_loss_sum, p.epoch_loss_batches))):
            raise ValueError("只能延长已完整完成的训练计划；请先按原 epochs 完成训练、验证和保存")
        if min(p.epoch, p.shard_position, p.batch_position, p.global_step, p.tokens_seen, p.epoch_loss_batches) < 0:
            raise ValueError("checkpoint 含负数进度")
        plan = list(self.plan(p.epoch)) if p.epoch < self.config.epochs else []
        if p.epoch > self.config.epochs or p.shard_position > len(plan):
            raise ValueError("checkpoint epoch/shard 越界")
        limit = plan[p.shard_position][1] if p.shard_position < len(plan) else 0
        if p.batch_position > limit or p.batch_position % self.group_batches:
            raise ValueError("checkpoint 不在有效的全局更新边界")
        expected_step = (sum(self.epoch_steps[:p.epoch])
                         + sum(math.ceil(n / self.group_batches) for _, n in plan[:p.shard_position])
                         + p.batch_position // self.group_batches)
        if p.global_step != expected_step or checkpoint["scheduler_state_dict"]["last_epoch"] != p.global_step:
            raise ValueError("checkpoint global_step/scheduler 与数据位置不一致")
        states = checkpoint["rank_rng_states"]
        if len(states) != self.world_size:
            raise ValueError("checkpoint 缺少对应 rank 的随机状态")
        schedule_steps = (saved_contract["total_steps"] if version == 2
                          else checkpoint["schedule_total_steps"])
        old_ends = list(accumulate(old_steps))
        schedule_plan = checkpoint.get("schedule_plan") if version >= 4 else None
        if version < 4 and schedule_steps in old_ends:
            schedule_plan = dict(epochs=old_ends.index(schedule_steps) + 1, mix_shards=1, mix_start_epoch=0)
        if not isinstance(schedule_plan, dict) or set(schedule_plan) != {"epochs", "mix_shards", "mix_start_epoch"}:
            raise ValueError("checkpoint 原学习率计划无效")
        origin_epochs = schedule_plan["epochs"]
        origin_sampling = {key: value for key, value in schedule_plan.items() if key != "epochs"}
        origin_steps = self.count_steps(origin_epochs, origin_sampling)
        if type(schedule_steps) is not int or not 0 < origin_epochs <= old_epochs or schedule_steps != sum(origin_steps):
            raise ValueError("checkpoint 原学习率计划长度无效")
        if p.epoch < origin_epochs and old_epochs > origin_epochs:
            raise ValueError("checkpoint 追加阶段的进度早于原计划结束位置")
        if version == 5 and "lr_phase" not in checkpoint:
            raise ValueError("checkpoint 缺少 lr_phase 元数据")
        phase = checkpoint["lr_phase"] if version == 5 else None
        if phase is not None:
            self.check_lr_phase(phase, p.global_step, saved_contract["total_steps"])
        # LambdaLR 的 state_dict 不保存闭包；必须按原计划重建，再恢复优化器中的当前 LR。
        self.scheduler = self.build_scheduler(schedule_steps, phase)
        expected_lrs = [lr * function(p.global_step) for lr, function in
                        zip(self.scheduler.base_lrs, self.scheduler.lr_lambdas)]
        saved_lrs = [group["lr"] for group in checkpoint["optimizer_state_dict"]["param_groups"]]
        if (len(saved_lrs) != len(expected_lrs) or
                any(not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15) for a, b in zip(saved_lrs, expected_lrs))):
            raise ValueError("checkpoint 学习率与原计划不一致")
        if extending and not self.config.rewarm_lr and min(saved_lrs) <= 0:
            raise ValueError("原计划结束学习率为 0，不能用固定末端学习率追加训练")
        c = self.config
        requested = {key: value for key, value in dict(peak_lr=c.peak_learning_rate,
                     warmup_steps=c.rewarm_steps, end_lr=c.end_learning_rate).items() if value is not None}
        if c.rewarm_lr and extending:
            phase = dict(start_step=p.global_step, end_step=self.total_steps, start_lrs=saved_lrs,
                         peak_lr=1.2e-4, end_lr=1e-5, warmup_steps=1000)
            phase.update(requested)
            self.check_lr_phase(phase, p.global_step, self.total_steps)
            self.scheduler = self.build_scheduler(schedule_steps, phase)
        else:
            if c.rewarm_lr and (phase is None or phase["end_step"] != self.total_steps):
                raise ValueError("首次升温需要 --extend-training 增加已完成计划的总轮数")
            if requested and (phase is None or any(phase[key] != value for key, value in requested.items())):
                raise ValueError("续训条件不一致：升温参数；已有阶段须沿用 checkpoint 中的值")
        self.lr_phase = phase
        self.schedule_total_steps = schedule_steps
        self.schedule_plan = schedule_plan
        self.raw_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        state = states[self.rank]
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["cpu"])
        if self.amp_dtype:
            torch.cuda.set_rng_state(state["cuda"], self.device)
        self.progress = p
        self.log(f"恢复 DDP checkpoint | epoch={p.epoch + 1} | shard={p.shard_position} | "
                 f"global_microbatch={p.batch_position} | global_step={p.global_step}")
        if self.lr_phase is not None and p.global_step < self.lr_phase["end_step"]:
            phase = self.lr_phase
            self.log(f"学习率阶段 | start_step={phase['start_step']} | start_lr={phase['start_lrs']} | "
                     f"rewarm_steps={phase['warmup_steps']} | peak_lr={phase['peak_lr']:.8g} | "
                     f"end_step={phase['end_step']} | end_lr={phase['end_lr']:.8g} | 当前 lr={saved_lrs}")
        elif self.total_steps > self.schedule_total_steps:
            completed_step = self.lr_phase["end_step"] if self.lr_phase is not None else self.schedule_total_steps
            self.log(f"追加训练计划 | target_epochs={self.config.epochs} | total_steps={self.total_steps} | "
                     f"已完成学习率阶段截止 step={completed_step} | 固定学习率，不再升温 | lr={saved_lrs}")

    def train_group(self, batches, first, end, shard_size):
        c = self.config
        count = end - first
        samples = min(end * c.batch_size, shard_size) - first * c.batch_size
        loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        self.optimizer.zero_grad(set_to_none=True)
        for offset, batch in enumerate(batches):
            position = first + self.rank + offset * self.world_size
            valid = position < end
            weight = ((len(batch) / samples if c.loss_weighting == "sample" else 1 / count)
                      * self.world_size if valid else 0.0)
            context = self.ddp.no_sync() if self.world_size > 1 and offset + 1 < len(batches) else nullcontext()
            # no_sync 必须同时包住 forward 和 backward，只有末次 backward 同步梯度。
            with context:
                ids = batch.to(self.device, dtype=torch.long, non_blocking=True)
                with self.autocast():
                    loss = self.train_loss(ids)
                (loss * weight).backward()
            loss_sum.add_(loss.detach().double() * int(valid))
        if self.world_size > 1:
            dist.all_reduce(loss_sum)
        loss_sum = loss_sum.item()
        if not math.isfinite(loss_sum):
            raise FloatingPointError("全局 loss 非有限值，本组未更新")
        norm = torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), c.max_grad_norm, error_if_nonfinite=True)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return loss_sum, samples * batches[0].shape[1], norm.detach()

    def train_epoch(self, epoch):
        p, c = self.progress, self.config
        midpoint = self.epoch_end_steps[epoch] - self.epoch_steps[epoch] // 2
        self.raw_model.train()
        self.timer.start_epoch(p)
        plan = list(self.plan(epoch))
        for shard, (entries, end) in enumerate(plan):
            if shard < p.shard_position:
                continue
            start = p.batch_position if shard == p.shard_position else 0
            if start == end:
                p.shard_position, p.batch_position = shard + 1, 0
                continue
            datasets = [base.PackedShardDataset(entry["resolved_path"], "train") for entry in entries]
            for entry, dataset in zip(entries, datasets):
                if len(dataset) != entry["num_sequences"]:
                    raise ValueError(f"manifest 与分片行数不一致：{entry['resolved_path']}")
            dataset = ConcatDataset(datasets)
            seed = c.seed + epoch * 100000 + shard
            sampler = ShardBatches(len(dataset), c.batch_size, self.rank, self.world_size, seed, start, end)
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=c.num_workers,
                                pin_memory=c.pin_memory and bool(self.amp_dtype),
                                generator=torch.Generator().manual_seed(seed))
            iterator = iter(loader)
            for first in range(start, end, self.group_batches):
                stop = min(first + self.group_batches, end)
                batches = [next(iterator) for _ in range(math.ceil((stop - first) / self.world_size))]
                loss_sum, tokens, norm = self.train_group(batches, first, stop, len(dataset))
                p.tokens_seen += tokens
                p.epoch_loss_sum += loss_sum
                p.epoch_loss_batches += stop - first
                p.global_step += 1
                p.shard_position, p.batch_position = (shard + 1, 0) if stop == end else (shard, stop)
                if self.rank == 0 and p.global_step % c.log_every_steps == 0:
                    grad_norm = float(norm)
                    timing = self.timer.format(p, self.epoch_end_steps[epoch], self.total_steps, c.epochs - epoch)
                    self.log(f"epoch={epoch + 1} | pool={shard + 1}/{len(plan)} | shards={len(entries)} | "
                             f"global_step={p.global_step} | loss={loss_sum / (stop - first):.6f} | "
                             f"lr={self.optimizer.param_groups[0]['lr']:.8e} | grad_norm={grad_norm:.6f} | "
                             f"tokens_seen={p.tokens_seen} | {timing}")
                stopping = c.max_steps and p.global_step >= c.max_steps
                preview_due = self.epoch_steps[epoch] > 1 and p.global_step == midpoint
                if stopping or preview_due or p.global_step % c.save_every_steps == 0:
                    self.save_checkpoint(self.latest_path)
                if preview_due:
                    self.log(f"epoch={epoch + 1} | 中途推理（约 50%）；本次不运行验证，valid_loss=nan 表示未计算。")
                    self.preview(epoch + 1, p.epoch_loss_sum / p.epoch_loss_batches)
                if stopping:
                    return None
            del iterator, loader, dataset, datasets
        return p.epoch_loss_sum / p.epoch_loss_batches

    @torch.inference_mode()
    def validate(self):
        c = self.config
        self.raw_model.eval()
        loss_sum, samples, processed = 0.0, 0, 0
        for entry in self.valid_entries:
            dataset = base.PackedShardDataset(entry["resolved_path"], "valid")
            if len(dataset) != entry["num_sequences"]:
                raise ValueError(f"验证分片行数不符：{entry['resolved_path']}")
            loader = DataLoader(dataset, batch_size=c.batch_size, num_workers=c.num_workers,
                                pin_memory=c.pin_memory and bool(self.amp_dtype),
                                generator=torch.Generator().manual_seed(c.seed))
            limit = c.smoke_valid_batches - processed if c.smoke_test else len(loader)
            for batch in islice(loader, max(0, limit)):
                ids = batch.to(self.device, dtype=torch.long, non_blocking=True)
                with self.autocast():
                    loss = self.raw_model(ids, labels=ids)[1]
                loss_sum += float(loss) * len(batch)
                samples += len(batch)
                processed += 1
            if c.smoke_test and processed >= c.smoke_valid_batches:
                break
        self.raw_model.train()
        if not samples or not math.isfinite(loss_sum):
            raise FloatingPointError("验证集为空或 loss 非有限值")
        return loss_sum / samples

    def preview(self, epoch, train_loss, valid_loss=float("nan")):
        def generate():
            # 推理仅在 rank 0 执行，并隔离其 Torch RNG，不改变后续训练的随机序列。
            with torch.random.fork_rng(devices=[self.device] if self.amp_dtype else []):
                base.run_beam_search_evaluation(
                    self.raw_model, self.tokenizer, self.config, self.device, epoch,
                    self.progress.global_step, self.progress.tokens_seen, train_loss,
                    valid_loss, self.latest_path, self.amp_dtype)
        self.primary_call(generate)

    def fit(self):
        p, c = self.progress, self.config
        for epoch in range(p.epoch, c.epochs):
            if c.max_steps and p.global_step >= c.max_steps:
                break
            train_loss = self.train_epoch(epoch)
            if train_loss is None:
                break
            started = perf_counter()
            valid_loss = self.primary_call(self.validate)
            improved = valid_loss < p.best_valid_loss
            p.best_valid_loss = min(p.best_valid_loss, valid_loss)
            p.epoch, p.shard_position, p.batch_position = epoch + 1, 0, 0
            p.epoch_loss_sum, p.epoch_loss_batches = 0.0, 0
            paths = [self.latest_path] + ([self.best_path] if improved else [])
            if p.epoch % c.epoch_checkpoint_every == 0:
                paths.append(self.output_dir / f"checkpoint_epoch_{p.epoch:04d}.pt")
            self.save_checkpoint(*paths)
            self.preview(p.epoch, train_loss, valid_loss)
            self.timer.epoch_overheads.append(perf_counter() - started)
        self.log(f"{'训练完成' if p.epoch == c.epochs else '达到本次步数上限'} | global_step={p.global_step} | "
                 f"elapsed={base.format_duration(perf_counter() - self.timer.started_at)} | latest={self.latest_path}")
        return dict(model=self.raw_model, optimizer=self.optimizer, scheduler=self.scheduler,
                    progress=asdict(p), latest_checkpoint=self.latest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainingConfig()
    parser.add_argument("--batch-size", type=int, default=8, help="每张卡的 microbatch")
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--mix-shards", type=int, default=defaults.mix_shards, help="每个混合池的分片数；1 保持旧单分片顺序")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--extend-training", action="store_true", help="原计划完成后延长至目标总轮数；默认沿用旧调度，可配合 --rewarm-lr 升温")
    parser.add_argument("--rewarm-lr", action="store_true", help="新增阶段升温一次再余弦衰减；同一阶段恢复不重复升温")
    parser.add_argument("--peak-lr", type=float, help="新增阶段峰值，默认 1.2e-4；恢复时默认沿用 checkpoint")
    parser.add_argument("--rewarm-steps", type=int, help="新增阶段升温更新步数，默认 1000")
    parser.add_argument("--end-lr", type=float, help="新增阶段末端学习率，默认 1e-5")
    parser.add_argument("--manifest", default=defaults.manifest_path)
    parser.add_argument("--tokenizer", default=defaults.tokenizer_path)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--max-steps", type=int, default=0, help="仅提前停止并保存，不缩短学习率计划")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-mode", default="default", choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"))
    parser.add_argument("--smoke", action="store_true", help="真实数据与小模型的训练/验证/保存/生成检查")
    args = parser.parse_args()
    world_size, local_rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("LOCAL_RANK", "0"))
    if int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size))) != world_size:
        parser.error("本入口仅支持单机多卡，请在同一台服务器启动全部进程")
    if min(args.batch_size, args.global_batch_size) <= 0 or args.global_batch_size % (world_size * args.batch_size):
        parser.error("global-batch-size 必须是 卡数 × batch-size 的正整数倍")
    if not torch.cuda.is_available():
        parser.error("正式入口需要 CUDA GPU；CPU 仅由测试显式注入")
    torch.cuda.set_device(local_rank)
    base.choose_device()
    config = replace(defaults, batch_size=args.batch_size,
                     gradient_accumulation_steps=args.global_batch_size // (world_size * args.batch_size),
                     epochs=args.epochs, manifest_path=args.manifest, tokenizer_path=args.tokenizer,
                     output_dir=args.output_dir, max_steps=args.max_steps, resume=args.resume,
                     compile_model=args.compile, compile_mode=args.compile_mode, extend_training=args.extend_training,
                     mix_shards=args.mix_shards, rewarm_lr=args.rewarm_lr, peak_learning_rate=args.peak_lr,
                     rewarm_steps=args.rewarm_steps, end_learning_rate=args.end_lr)
    model_config = base.MODEL_CONFIG
    if args.smoke:
        config = replace(config, epochs=1, smoke_test=True, smoke_train_batches=world_size * config.gradient_accumulation_steps,
                         log_every_steps=1, save_every_steps=1, warmup_steps=0)
        model_config = base.SMOKE_MODEL_CONFIG
    try:
        if world_size > 1:
            dist.init_process_group("nccl", timeout=timedelta(minutes=30))
        Trainer(config, model_config, device=f"cuda:{local_rank}").fit()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
