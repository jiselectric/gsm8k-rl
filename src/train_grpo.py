"""GRPO (RL with verifiable rewards) on GSM8K train, using TRL.

    python src/train_grpo.py configs/grpo_smoke.yaml          # 2-step CPU smoke test (Mac)
    python src/train_grpo.py configs/grpo_qwen3_17b.yaml      # real run (cloud GPU + vLLM)
    python src/train_grpo.py configs/grpo_qwen3_17b.yaml max_steps=20 run_name=pilot  # overrides

Prompts come from prompts.py and rewards from rewards.py -> grading.py, so the
model is trained on exactly the prompt + grader that eval.py measures.
Train on data/train.jsonl only; dev/test are never seen here.
"""
import json
import subprocess
import sys
from pathlib import Path

import yaml
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from prompts import ROOT, build_messages
from rewards import lenient_correct, strict_correct


def load_rows(path: str, n: int | None = None) -> Dataset:
    """jsonl of {question, answer, ...} -> TRL dataset with the zero-shot eval prompt.
    Paths are relative to the project root (e.g. data/train_filtered.jsonl)."""
    rows = [json.loads(l) for l in open(ROOT / path)]
    rows = rows[:n] if n else rows
    return Dataset.from_list([
        {"prompt": build_messages(r["question"], examples=[]),  # same zero-shot prompt as eval
         "answer": r["answer"]}
        for r in rows
    ])


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """`key=value` args override the YAML, e.g. `max_steps=20 run_name=pilot`.
    Keys that live under `grpo:` are overridden there; others at the top level."""
    for o in overrides:
        key, value = o.split("=", 1)
        target = cfg["grpo"] if key in cfg["grpo"] or key not in cfg else cfg
        target[key] = yaml.safe_load(value)  # "20" -> 20, "true" -> True
    return cfg


def main():
    cfg = apply_overrides(yaml.safe_load(open(sys.argv[1])), sys.argv[2:])
    if not cfg.get("eval_n"):  # no dev slice -> no periodic eval (e.g. the pilot)
        cfg["grpo"]["eval_strategy"] = "no"
    out_dir = ROOT / "checkpoints" / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Record exactly what produced this checkpoint.
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip() or None
    (out_dir / "run_config.json").write_text(json.dumps({**cfg, "git_commit": commit}, indent=2))

    args = GRPOConfig(
        output_dir=str(out_dir),
        run_name=cfg["run_name"],
        model_init_kwargs={"revision": cfg["revision"], "dtype": cfg.get("dtype", "bfloat16")},
        # Same chat template switch as eval.py's `thinking` flag.
        chat_template_kwargs={"enable_thinking": cfg.get("thinking", False)},
        reward_weights=[1.0, 0.0],  # train on strict only; lenient is just logged
        **cfg["grpo"],
    )
    # Qwen3.5 is a vision-language model; we only train on text, so pass the plain
    # tokenizer (otherwise TRL loads the image processor too).
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    trainer = GRPOTrainer(
        model=cfg["model"],
        processing_class=tokenizer,
        reward_funcs=[strict_correct, lenient_correct],
        args=args,
        train_dataset=load_rows(cfg.get("train_file", "data/train.jsonl"), cfg.get("n_train")),
        # Small held-out dev slice, sampled like training: tracks reward on unseen
        # questions so we can spot overfitting / reward hacking during the run.
        eval_dataset=load_rows("data/dev.jsonl", cfg["eval_n"]) if cfg.get("eval_n") else None,
    )
    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    print(f"saved -> {out_dir / 'final'}")


if __name__ == "__main__":
    main()
