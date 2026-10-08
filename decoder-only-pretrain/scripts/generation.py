"""基于 model.py 的 KV Cache Beam Search。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Optional, Protocol

import sentencepiece as spm
import torch
import torch.nn.functional as F


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from model import (
    KVCache,
    MinimalLlamaStyleCausalLM,
)


@dataclass
class ScoredSequence:
    sequence: torch.Tensor
    score: float

    def normalized_score(self, prompt_len: int, length_penalty: float) -> float:
        generated_len = max(self.sequence.size(1) - prompt_len, 1)
        if length_penalty == 0:
            return self.score
        return self.score / (generated_len ** length_penalty)


@dataclass
class ActiveBeam(ScoredSequence):
    kv_cache: KVCache
    last_logits: torch.Tensor


class GenerationEvaluationConfig(Protocol):
    """调用方可额外提供 repetition_penalty；缺省 1.0 表示关闭。"""

    tokenizer_path: str | Path
    smoke_test: bool
    smoke_max_new_tokens: int
    max_new_tokens: int
    beam_size: int
    length_penalty: float
    early_stopping: bool
    eos_token_id: int
    pad_token_id: int
    bos_token_id: int


def sort_scored_sequences(
    beams: list[ScoredSequence],
    prompt_len: int,
    length_penalty: float,
) -> None:
    beams.sort(
        key=lambda beam: beam.normalized_score(prompt_len, length_penalty),
        reverse=True,
    )


def filter_generation_logits(
    logits: torch.Tensor,
    config: GenerationEvaluationConfig,
    terminal_ids: set[int],
    generated_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    filtered_logits = logits.to(dtype=torch.float32, copy=True)
    blocked_token_ids = {config.pad_token_id, config.bos_token_id} - {None} - terminal_ids
    for token_id in blocked_token_ids:
        filtered_logits[:, token_id] = -1e9
    penalty = getattr(config, "repetition_penalty", 1.0)
    if penalty != 1.0 and generated_ids is not None and generated_ids.numel():
        # batch_size=1：仅统计当前 beam 的回答，同一 token 每步只惩罚一次。
        seen_ids = generated_ids.unique()
        excluded_ids = torch.tensor(
            sorted(terminal_ids | blocked_token_ids), device=seen_ids.device, dtype=torch.long)
        seen_ids = seen_ids[~torch.isin(seen_ids, excluded_ids)]
        scores = filtered_logits[:, seen_ids]
        filtered_logits[:, seen_ids] = torch.where(scores < 0, scores * penalty, scores / penalty)
    return filtered_logits


def clone_kv_cache(
    model: MinimalLlamaStyleCausalLM,
    kv_cache: KVCache,
) -> KVCache:
    # 每个候选 beam 都需要独立 Cache，避免兄弟分支相互覆盖。
    cloned_cache = model.create_kv_cache(
        batch_size=kv_cache.batch_size,
        device=kv_cache.k_cache.device,
        dtype=kv_cache.k_cache.dtype,
    )
    cloned_cache.k_cache.copy_(kv_cache.k_cache)
    cloned_cache.v_cache.copy_(kv_cache.v_cache)
    return cloned_cache


def validate_kv_generation_args(
    model: MinimalLlamaStyleCausalLM,
    input_ids: torch.Tensor,
    config: GenerationEvaluationConfig,
    max_new_tokens: int,
    end_token_id: Optional[int],
) -> None:
    prompt_len = input_ids.size(1)
    if prompt_len == 0:
        raise ValueError("input_ids must contain at least one prompt token")
    for name, value in (("max_new_tokens", max_new_tokens), ("beam_size", config.beam_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    if config.beam_size == 0:
        raise ValueError("beam_size must be greater than zero")
    penalty = config.length_penalty
    if (
        isinstance(penalty, bool) or not isinstance(penalty, (int, float))
        or not math.isfinite(penalty) or penalty < 0
    ):
        raise ValueError(f"length_penalty must be finite and >= 0, got {penalty!r}")
    repetition_penalty = getattr(config, "repetition_penalty", 1.0)
    if (
        isinstance(repetition_penalty, bool) or not isinstance(repetition_penalty, (int, float))
        or not math.isfinite(repetition_penalty) or repetition_penalty < 1.0
    ):
        raise ValueError(f"repetition_penalty must be finite and >= 1, got {repetition_penalty!r}")
    vocab_size = model.config.vocab_size
    for name, token_id in (
        ("eos_token_id", config.eos_token_id), ("end_token_id", end_token_id),
        ("pad_token_id", config.pad_token_id), ("bos_token_id", config.bos_token_id),
    ):
        if token_id is None:
            continue
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError(f"{name} must be an integer or None, got {token_id!r}")
        if not 0 <= token_id < vocab_size:
            raise ValueError(f"{name}={token_id} is outside vocab_size={vocab_size}")
    if prompt_len + max_new_tokens > model.config.block_size:
        raise ValueError(
            f"T_prompt + max_new_tokens = {prompt_len + max_new_tokens} exceeds "
            f"block_size={model.config.block_size}. Sliding window is not supported."
        )


def prefill_kv_cache(
    model: MinimalLlamaStyleCausalLM,
    input_ids: torch.Tensor,
) -> tuple[KVCache, torch.Tensor]:
    B, T_prompt = input_ids.shape
    device = input_ids.device
    cache_dtype = next(model.parameters()).dtype
    if torch.is_autocast_enabled(device.type):
        cache_dtype = torch.get_autocast_dtype(device.type)

    kv_cache = model.create_kv_cache(
        batch_size=B,
        device=device,
        dtype=cache_dtype,
    )
    input_pos = torch.arange(T_prompt, device=device)
    logits = model(
        input_ids,
        kv_cache=kv_cache,
        input_pos=input_pos,
        return_last_logits_only=True,
    )
    return kv_cache, logits


@torch.no_grad()
def generate_beam_search_with_kv_cache(
    model: MinimalLlamaStyleCausalLM,
    input_ids: torch.Tensor,
    config: GenerationEvaluationConfig,
    max_new_tokens: int,
    end_token_id: Optional[int] = None,
) -> torch.Tensor:
    """返回含输入的 token 序列；重复惩罚只作用于新生成的回答，不统计输入。"""
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be 2D [B, T], got {input_ids.shape}")

    if input_ids.size(0) != 1:
        raise ValueError(
            "This teaching KV-cache beam search only supports batch_size=1. "
            f"Got input_ids.shape={input_ids.shape}"
        )

    validate_kv_generation_args(
        model=model,
        input_ids=input_ids,
        config=config,
        max_new_tokens=max_new_tokens,
        end_token_id=end_token_id,
    )

    if max_new_tokens == 0:
        return input_ids

    was_training = model.training
    model.eval()
    try:
        device = input_ids.device
        prompt_len = input_ids.size(1)

        initial_cache, logits = prefill_kv_cache(model=model, input_ids=input_ids)
        initial_last_logits = logits[:, -1, :]
        terminal_ids = {config.eos_token_id, end_token_id} - {None}

        # 每个 active beam 保存序列、累计分数、独立 Cache 和下一步 logits。
        active_beams = [ActiveBeam(input_ids, 0.0, initial_cache, initial_last_logits)]
        completed_beams: list[ScoredSequence] = []

        for generated_len in range(max_new_tokens):
            candidate_beams: list[tuple[torch.Tensor, float, KVCache]] = []
            is_final_generation_step = generated_len + 1 == max_new_tokens
            for beam in active_beams:
                next_token_logits = filter_generation_logits(
                    logits=beam.last_logits,
                    config=config,
                    terminal_ids=terminal_ids,
                    generated_ids=beam.sequence[:, prompt_len:],
                )
                next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)
                top_k = min(config.beam_size, next_token_log_probs.size(-1))
                top_log_probs, top_token_ids = torch.topk(
                    next_token_log_probs,
                    k=top_k,
                    dim=-1,
                )
                for beam_index in range(top_k):
                    next_token = top_token_ids[:, beam_index : beam_index + 1]
                    next_token_id = int(next_token.item())
                    next_token_score = float(top_log_probs[0, beam_index].item())
                    sequence = torch.cat([beam.sequence, next_token], dim=1)
                    score = beam.score + next_token_score
                    if next_token_id in terminal_ids:
                        completed_beams.append(ScoredSequence(sequence, score))
                    else:
                        candidate_beams.append((sequence, score, beam.kv_cache))

            if candidate_beams and completed_beams and not is_final_generation_step:
                sort_scored_sequences(completed_beams, prompt_len, config.length_penalty)
                completed_beams = completed_beams[:config.beam_size]
            if config.early_stopping and len(completed_beams) >= config.beam_size:
                break

            # 同一轮候选的生成长度相同，按累计分数选出下一轮 Beam。
            candidate_beams.sort(key=lambda item: item[1], reverse=True)
            if is_final_generation_step or not candidate_beams:
                completed_beams.extend(
                    ScoredSequence(sequence, score)
                    for sequence, score, _ in candidate_beams[:config.beam_size]
                )
                break

            input_pos = torch.tensor(
                [prompt_len + generated_len], dtype=torch.long, device=device
            )
            active_beams = []
            for sequence, score, parent_cache in candidate_beams[:config.beam_size]:
                candidate_cache = clone_kv_cache(model=model, kv_cache=parent_cache)
                logits = model(sequence[:, -1:], kv_cache=candidate_cache, input_pos=input_pos)
                active_beams.append(
                    ActiveBeam(sequence, score, candidate_cache, logits[:, -1, :])
                )

        sort_scored_sequences(completed_beams, prompt_len, config.length_penalty)
        return completed_beams[0].sequence
    finally:
        model.train(was_training)


GENERATION_TEST_CASES = [
    ("中文基础问答", "请用简单的语言解释什么是机器学习。", True),
    ("Transformer 技术问答", "请解释 Transformer 中 self-attention 的作用。", True),
    ("KV Cache 技术问答", "请解释 Decoder-only 模型推理时 KV Cache 的作用。", True),
    (
        "英文指令",
        "Explain the difference between training and inference in a language model.",
        True,
    ),
    (
        "英译中",
        "Translate into Chinese: Artificial intelligence is changing the way people work and learn.",
        True,
    ),
    ("中译英", "将下面句子翻译成英文：深度学习模型需要大量高质量数据。", True),
    (
        "数学题",
        "一辆汽车每小时行驶60千米，3.5小时一共行驶多少千米？请给出计算过程和答案。",
        True,
    ),
    ("Python 代码", "请使用 Python 编写一个函数，返回列表中的最大值。", True),
    (
        "总结任务",
        "请将下面内容总结为一句话：机器学习通过从数据中学习规律，使计算机能够完成分类、预测和生成等任务。",
        True,
    ),
    ("纯文本续写", "人工智能的发展经历了多个重要阶段，其中", False),
]


def load_tokenizer(training_config: GenerationEvaluationConfig) -> spm.SentencePieceProcessor:
    tokenizer_path = PROJECT_ROOT / training_config.tokenizer_path
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"缺少 SentencePiece tokenizer：{tokenizer_path}")

    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    expected_tokens = {
        "<|pad|>": 0,
        "<|bos|>": 2,
        "<|eos|>": 3,
        "<|system|>": 4,
        "<|user|>": 5,
        "<|assistant|>": 6,
        "<|end|>": 7,
    }
    for piece, expected_id in expected_tokens.items():
        actual_id = tokenizer.piece_to_id(piece)
        if actual_id != expected_id:
            raise ValueError(
                f"tokenizer special token 不匹配：{piece}={actual_id}，期望 {expected_id}"
            )
    if tokenizer.vocab_size() != 48000:
        raise ValueError(f"tokenizer vocab_size 必须为 48000，实际为 {tokenizer.vocab_size()}")
    return tokenizer


def format_generation_prompt(text: str, is_chat: bool) -> str:
    if not is_chat:
        return text
    return (
        "<|system|>你是一个有帮助的中英文助手。\n"
        f"<|user|>{text}\n"
        "<|assistant|>"
    )


def run_beam_search_evaluation(
    model: MinimalLlamaStyleCausalLM,
    tokenizer: spm.SentencePieceProcessor,
    training_config: GenerationEvaluationConfig,
    device: torch.device,
    completed_epoch: int,
    global_step: int,
    tokens_seen: int,
    train_loss: float,
    valid_loss: float,
    checkpoint_path: Path,
    amp_dtype: Optional[torch.dtype],
) -> None:
    cases = GENERATION_TEST_CASES[:1] if training_config.smoke_test else GENERATION_TEST_CASES
    max_new_tokens = (
        training_config.smoke_max_new_tokens
        if training_config.smoke_test
        else training_config.max_new_tokens
    )
    chat_end_token_id = tokenizer.piece_to_id("<|end|>")

    print("=" * 60)
    print(f"Epoch {completed_epoch} Beam Search Evaluation")
    print(f"global_step: {global_step}")
    print(f"tokens_seen: {tokens_seen}")
    print(f"train_loss: {train_loss:.6f}")
    print(f"valid_loss: {valid_loss:.6f}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"beam_size: {training_config.beam_size}")
    print(f"max_new_tokens: {max_new_tokens}")
    print(f"length_penalty: {training_config.length_penalty}")
    print("=" * 60, flush=True)

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for index, (name, text, is_chat) in enumerate(cases, start=1):
                prompt = format_generation_prompt(text, is_chat)
                prompt_ids = tokenizer.encode(prompt, add_bos=False, add_eos=False)
                if len(prompt_ids) + max_new_tokens > model.config.block_size:
                    raise ValueError(f"测试用例 {name} 超过 block_size")

                input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                amp_context = (
                    nullcontext() if amp_dtype is None
                    else torch.autocast(device_type=device.type, dtype=amp_dtype)
                )
                with amp_context:
                    generated_ids = generate_beam_search_with_kv_cache(
                        model=model,
                        input_ids=input_ids,
                        max_new_tokens=max_new_tokens,
                        config=training_config,
                        end_token_id=chat_end_token_id if is_chat else None,
                    )
                continuation_ids = generated_ids[0, len(prompt_ids) :].tolist()
                if is_chat and continuation_ids[-1:] == [chat_end_token_id]:
                    continuation_ids = continuation_ids[:-1]
                generated_text = tokenizer.decode(continuation_ids)

                print(f"\n[测试用例 {index}：{name}]")
                print("Prompt:")
                print(prompt)
                print("\nGenerated:")
                print(generated_text)
                print("\n" + "-" * 60, flush=True)
    except Exception as exc:
        print(f"Beam Search 测试失败：{type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        model.train(was_training)
