"""Evaluate a model on GSM8K and save every output for error analysis.

Usage:
    python src/eval.py --model Qwen/Qwen3-0.6B --split dev --n 50
    python src/eval.py --model microsoft/Phi-3.5-mini-instruct --split dev

Output: results/<split>/<model_name>__<tag>.jsonl  (one line per problem)
        + a printed accuracy summary.

The plumbing (loading, looping, saving) is done for you.
Prompting lives in prompts.py, extraction + grading in grading.py (shared with training).
"""
import argparse
import json
import time
from pathlib import Path

from mlx_lm import batch_generate, generate, load
from mlx_lm.sample_utils import make_sampler
from tqdm import tqdm

# Shared with training. Re-exported here so `from eval import ...` keeps working.
from grading import (extract_answer, extract_answer_with_rule,  # noqa: F401
                     extract_raw_with_rule, is_correct, normalize_number)
from prompts import (SYSTEM_PROMPT, build_messages, build_prompt,  # noqa: F401
                     load_fewshot_examples)

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Plumbing (done)
# ---------------------------------------------------------------------------
def load_split(split: str, n: int | None):
    rows = [json.loads(l) for l in open(ROOT / "data" / f"{split}.jsonl")]
    return rows[:n] if n else rows


def load_model(repo: str, revision: str | None = None):
    """Load a model at an exact commit. If no revision is given, resolve the
    current one, so every result records exactly which weights produced it."""
    if revision is None and not Path(repo).expanduser().exists():  # HF repo, not a local folder
        try:
            from huggingface_hub import model_info
            revision = model_info(repo).sha
        except Exception:  # offline: fall back to whatever is cached
            revision = None
    model, tokenizer = load(repo, revision=revision)
    return model, tokenizer, revision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None, help="exact commit sha (default: latest)")
    ap.add_argument("--shots", type=int, default=0, help="0 = zero-shot")
    ap.add_argument("--seed", type=int, default=0, help="which few-shot examples")
    ap.add_argument("--thinking", action="store_true", help="Qwen3/3.5 thinking mode")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--n", type=int, default=None, help="only first n problems")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temp", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="questions generated together (~3.7x faster at 8)")
    ap.add_argument("--tag", default="default", help="label for this config")
    args = ap.parse_args()

    cfg = {k: v for k, v in vars(args).items() if k != "tag"}
    out_path = result_path(cfg, args.tag)
    model, tokenizer, cfg["revision"] = load_model(args.model, args.revision)
    n_correct, n = run_eval(model, tokenizer, cfg, out_path)
    print(f"\n{args.model} [{args.split}, {args.tag}]: "
          f"{n_correct}/{n} = {100 * n_correct / n:.1f}%")
    print(f"saved -> {out_path.relative_to(ROOT)}")


def result_path(cfg: dict, tag: str) -> Path:
    out_dir = ROOT / "results" / cfg["split"]
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{cfg['model'].split('/')[-1]}__{tag}.jsonl"


def encode_prompt(tokenizer, prompt: str) -> list[int]:
    """Prompt string -> token ids, exactly as mlx_lm.generate() does it
    (add BOS only if the chat template didn't already put it there)."""
    add_special = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
    return tokenizer.encode(prompt, add_special_tokens=add_special)


def generate_batch(model, tokenizer, prompts: list[str], max_tokens: int, sampler) -> list[str]:
    """Generate completions for several prompts. One prompt -> plain generate();
    several -> batch_generate(), which shares each pass over the weights."""
    if len(prompts) == 1:
        return [generate(model, tokenizer, prompt=prompts[0],
                         max_tokens=max_tokens, sampler=sampler)]
    ids = [encode_prompt(tokenizer, p) for p in prompts]
    return batch_generate(model, tokenizer, ids, max_tokens=max_tokens, sampler=sampler).texts


def run_eval(model, tokenizer, cfg: dict, out_path: Path) -> tuple[int, int]:
    """Evaluate an already-loaded model under one config. Reused by sweep.py.

    cfg keys: model, split, n, shots, seed, thinking, max_tokens, temp, batch_size.
    batch_size changes outputs slightly (floating-point order), so only compare
    runs that used the same batch_size.
    """
    if cfg["split"] == "test":
        print("⚠️  Running on TEST. Only do this for final numbers.")

    bs = cfg.get("batch_size", 1)
    sampler = make_sampler(temp=cfg["temp"])
    rows = load_split(cfg["split"], cfg["n"])
    examples = load_fewshot_examples(cfg["shots"], cfg["seed"])
    config = {**cfg, "batch_size": bs, "system_prompt": SYSTEM_PROMPT}

    n_correct = 0
    tmp_path = out_path.with_suffix(".partial")  # renamed only when complete
    bar = tqdm(total=len(rows), desc=f"{cfg['model'].split('/')[-1]} {out_path.stem.split('__')[-1]}")
    with open(tmp_path, "w") as f:
        for start in range(0, len(rows), bs):
            chunk = rows[start:start + bs]
            prompts = [build_prompt(r["question"], tokenizer, examples, cfg["thinking"])
                       for r in chunk]
            t0 = time.time()
            completions = generate_batch(model, tokenizer, prompts, cfg["max_tokens"], sampler)
            secs = (time.time() - t0) / len(chunk)  # amortized per question
            for row, completion in zip(chunk, completions):
                n_tokens = len(tokenizer.encode(completion, add_special_tokens=False))
                pred_raw, rule = extract_raw_with_rule(completion)
                pred = normalize_number(pred_raw) if pred_raw is not None else None
                ok = is_correct(pred, row["answer"])
                n_correct += ok
                f.write(json.dumps({
                    "id": row["id"], "question": row["question"],
                    "gold": row["answer"], "pred": pred, "pred_raw": pred_raw,
                    "rule": rule, "correct": ok,
                    "completion": completion, "n_tokens": n_tokens,
                    "hit_max_tokens": n_tokens >= cfg["max_tokens"],
                    "seconds": round(secs, 2),
                    "config": config,
                }) + "\n")
            bar.update(len(chunk))
    bar.close()
    tmp_path.rename(out_path)
    return n_correct, len(rows)


if __name__ == "__main__":
    main()
