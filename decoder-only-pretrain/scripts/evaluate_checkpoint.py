"""一次评测一个 checkpoint；只生成并保存回答，评分在生成完成后进行。"""
import argparse
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
import generation
from model import MinimalLlamaStyleCausalLM, ModelConfig

# 所有可修改参数放在这里；换权重时只需修改 CHECKPOINT_PATH。
ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT.parent / "weights/pretrain/checkpoint_epoch_0001.pt"
QUESTIONS_PATH = ROOT / "evaluation/training_aligned_bank_v3_review/questions.jsonl"
# 固定题库的完整性约束；参考答案与评分要点已合并到 questions.jsonl。
QUESTIONS_SHA256 = "09d98af905d18884338de272abd4c2c42b68fc2534aa8516a5e34d48ac680414"
EXPECTED_CASES = 200
EXPECTED_GENERATION_TURNS = 200
TOKENIZER_PATH = ROOT.parent / "tokenizer/spiece.model"
OUTPUT_DIR = ROOT.parent / "outputs/pretrain_evaluation"
DEVICE = "cpu"  # 本机短测：4线程CPU比MPS更快；6个checkpoint统一使用此配置。
NUM_THREADS = 4
AMP_DTYPE = torch.bfloat16 if DEVICE.startswith("cuda") else None
BEAM_SIZE = 1
MAX_NEW_TOKENS = 256
CODE_MAX_NEW_TOKENS = 512
SEED = 0
GENERATION_CONFIG = SimpleNamespace(
    tokenizer_path=TOKENIZER_PATH, beam_size=BEAM_SIZE, length_penalty=0.0,
    early_stopping=True, eos_token_id=3, pad_token_id=0, bos_token_id=2,
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_questions(path):
    """校验合并题库，只返回原有题目字段，参考答案和评分要点不进入生成流程。"""
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != QUESTIONS_SHA256:
        raise ValueError("题库内容与脚本记录的 SHA-256 不一致")
    questions = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    if (len(questions) != EXPECTED_CASES or len({q["id"] for q in questions}) != len(questions)
            or sum(len(q["user_turns"]) for q in questions) != EXPECTED_GENERATION_TURNS):
        raise ValueError("题目数量、题号或对话轮次不完整")
    fields = ("id", "category", "difficulty", "language", "user_turns")
    return [{field: question[field] for field in fields} for question in questions]


def load_model():
    # 本项目自己的训练文件含 RNG 等状态；mmap 避免把优化器张量全部读入内存。
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False, mmap=True)
    tokenizer_hash = sha256(TOKENIZER_PATH)
    expected_hash = checkpoint.get("contract", {}).get("tokenizer")
    if expected_hash and expected_hash != tokenizer_hash:
        raise ValueError("tokenizer 与 checkpoint 记录不一致")
    model = MinimalLlamaStyleCausalLM(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    info = {key: checkpoint.get(key) for key in ("checkpoint_version", "epoch", "global_step", "model_config")}
    return model.to(DEVICE).eval(), dict(info, tokenizer_sha256=tokenizer_hash)


@torch.inference_mode()
def answer_turns(question, model, tokenizer):
    history = []
    limit = CODE_MAX_NEW_TOKENS if question["category"] == "code" else MAX_NEW_TOKENS
    for index, user in enumerate(question["user_turns"], 1):
        prompt = (generation.format_generation_prompt(user, True) if index == 1
                  else f"\n<|user|>{user}\n<|assistant|>")
        prompt_ids = history + tokenizer.encode(prompt, add_bos=False, add_eos=False)
        if len(prompt_ids) + limit > model.config.block_size:
            raise ValueError(f"{question['id']} 第{index}轮超过上下文上限，未截断历史")
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
        started = perf_counter()
        amp = nullcontext() if AMP_DTYPE is None else torch.autocast(torch.device(DEVICE).type, dtype=AMP_DTYPE)
        with amp:
            output = generation.generate_beam_search_with_kv_cache(
                model, inputs, GENERATION_CONFIG, limit, end_token_id=7)
        generated = output[0, len(prompt_ids):].tolist()
        reason = {3: "eos", 7: "end"}.get(generated[-1] if generated else None, "max_new_tokens")
        content = generated[:-1] if reason in ("eos", "end") else generated
        history = prompt_ids + content + [7]  # 保留原始回答 token；只统一上一轮结束边界。
        yield dict(turn=index, user=user, answer=tokenizer.decode(content), prompt_ids=prompt_ids,
                   generated_ids=generated, generated_tokens=len(generated), finish_reason=reason,
                   seconds=round(perf_counter() - started, 3))


def main():
    global CHECKPOINT_PATH, OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    CHECKPOINT_PATH = args.checkpoint.expanduser().resolve()
    OUTPUT_DIR = args.output_dir.expanduser().resolve()
    torch.set_num_threads(NUM_THREADS)
    torch.manual_seed(SEED)
    questions = load_questions(QUESTIONS_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{CHECKPOINT_PATH.stem}_answers.jsonl"
    if output_path.exists():
        raise FileExistsError(f"保留已有回答；重新测试请修改 OUTPUT_DIR：{output_path}")
    tokenizer = generation.load_tokenizer(GENERATION_CONFIG)
    model, info = load_model()
    info.update(checkpoint=str(CHECKPOINT_PATH.resolve()), checkpoint_bytes=CHECKPOINT_PATH.stat().st_size,
                questions_sha256=sha256(QUESTIONS_PATH), device=DEVICE, amp_dtype=str(AMP_DTYPE),
                torch_version=str(torch.__version__), num_threads=NUM_THREADS, seed=SEED, generation=vars(GENERATION_CONFIG),
                max_new_tokens=MAX_NEW_TOKENS, code_max_new_tokens=CODE_MAX_NEW_TOKENS,
                sources={name: sha256(Path(__file__).with_name(name))
                         for name in ("model.py", "generation.py", "evaluate_checkpoint.py")})
    output_path.with_suffix(".config.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    started = perf_counter()
    print(f"checkpoint={CHECKPOINT_PATH.name} | device={DEVICE} | cases={len(questions)} | output={output_path}", flush=True)
    with output_path.open("x", encoding="utf-8") as stream:
        for index, question in enumerate(questions, 1):
            record = dict(id=question["id"], category=question["category"], difficulty=question["difficulty"],
                          language=question["language"], user_turns=question["user_turns"], turns=[], status="incomplete")
            try:
                for answer in answer_turns(question, model, tokenizer):
                    record["turns"].append(answer)
                record["status"] = "ok"
            except Exception as error:
                record.update(status="error", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
            print(f"[{index}/{len(questions)}] {question['id']} | elapsed={perf_counter() - started:.1f}s", flush=True)
    print("回答生成完成；status=ok 仅表示运行成功，不代表答案正确。", flush=True)


if __name__ == "__main__":
    main()
