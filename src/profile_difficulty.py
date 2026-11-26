"""Difficulty profiling: how often does the base model solve each TRAIN question?

GRPO only learns from questions where the model's k answers DISAGREE (some right,
some wrong). Always-right and always-wrong questions give zero advantage = wasted
compute. So before training we sample k answers per question, exactly as training
will (same prompt, thinking mode, temperature, length cap, strict reward), and keep
only the "learnable" ones.

    # on the GPU pod (needs vLLM):
    python src/profile_difficulty.py configs/grpo_qwen3_17b.yaml            # all train
    python src/profile_difficulty.py configs/grpo_qwen3_17b.yaml --n 500    # quick check
    # anywhere (no GPU): rebuild the filtered set from saved results
    python src/profile_difficulty.py configs/grpo_qwen3_17b.yaml --filter-only

Outputs (resumable: finished questions are skipped on re-run):
    data/difficulty_<run>.jsonl  one row per question: strict/lenient pass counts,
                                 truncations, avg tokens, one sample completion
    data/train_filtered.jsonl    train rows with 1 <= strict passes <= k-1
"""
import argparse
import collections
import json

import yaml

from grading import extract_raw_with_rule, is_correct, normalize_number
from prompts import ROOT, build_prompt
from rewards import strict_correct


def grade(text: str, gold: str) -> tuple[float, bool]:
    """(strict reward exactly as in training, lenient correctness)."""
    strict = strict_correct([text], [gold])[0]
    raw, _ = extract_raw_with_rule(text)
    lenient = is_correct(normalize_number(raw) if raw is not None else None, gold)
    return strict, lenient


def select_learnable(rows: list[dict], k: int) -> list[dict]:
    """Keep questions with mixed outcomes. A '#### 72.0' (0.5) counts as a pass here:
    it still differs from a 0, so the group has signal."""
    return [r for r in rows if 1 <= r["strict_pass"] <= k - 1]


def summarize(rows: list[dict], k: int) -> None:
    hist = collections.Counter(r["strict_pass"] for r in rows)
    n = len(rows)
    print(f"\n{n} questions profiled, k={k}")
    print("strict passes  questions")
    for p in range(k + 1):
        bar = "#" * round(40 * hist.get(p, 0) / max(n, 1))
        print(f"  {p}/{k}         {hist.get(p, 0):5d}  {bar}")
    learn = len(select_learnable(rows, k))
    strict_acc = sum(r["strict_pass"] for r in rows) / (n * k)
    len_acc = sum(r["lenient_pass"] for r in rows) / (n * k)
    trunc = sum(r["n_truncated"] for r in rows) / (n * k)
    print(f"sample accuracy: strict {strict_acc:.1%}, lenient {len_acc:.1%}, truncated {trunc:.1%}")
    print(f"always right: {hist.get(k, 0)}, always wrong: {hist.get(0, 0)}, "
          f"LEARNABLE: {learn} ({learn / max(n, 1):.0%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="training YAML: model, revision, thinking, grpo.* are reused")
    ap.add_argument("--n", type=int, default=None, help="only the first n train questions")
    ap.add_argument("--k", type=int, default=None, help="samples per question (default: num_generations)")
    ap.add_argument("--chunk", type=int, default=512, help="questions per vLLM call (resume granularity)")
    ap.add_argument("--filter-only", action="store_true", help="skip generation, rebuild filtered set")
    ap.add_argument("--input", default="data/train.jsonl", help="questions to profile")
    ap.add_argument("--output", default="data/train_filtered.jsonl", help="learnable questions go here")
    ap.add_argument("--tag", default=None, help="name of the profile file (default: run_name)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    g = cfg["grpo"]
    k = args.k or g["num_generations"]
    out_path = ROOT / "data" / f"difficulty_{args.tag or cfg['run_name']}.jsonl"
    train = [json.loads(l) for l in open(ROOT / args.input)]
    train = train[:args.n] if args.n else train

    done = {}
    if out_path.exists():
        done = {r["id"]: r for r in map(json.loads, open(out_path))}
    todo = [r for r in train if r["id"] not in done]

    if todo and not args.filter_only:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
        llm = LLM(model=cfg["model"], revision=cfg["revision"], dtype=cfg.get("dtype", "bfloat16"),
                  max_model_len=g.get("vllm_max_model_length", 4608),
                  gpu_memory_utilization=0.85, seed=0)
        params = SamplingParams(n=k, temperature=g.get("temperature", 1.0),
                                top_p=g.get("top_p", 1.0),
                                max_tokens=g["max_completion_length"], seed=0)
        print(f"profiling {len(todo)} questions x {k} samples ({len(done)} already done)")
        with open(out_path, "a") as f:
            for start in range(0, len(todo), args.chunk):
                chunk = todo[start:start + args.chunk]
                prompts = [build_prompt(r["question"], tok, [], cfg.get("thinking", False))
                           for r in chunk]
                for row, out in zip(chunk, llm.generate(prompts, params)):
                    grades = [grade(o.text, row["answer"]) for o in out.outputs]
                    rec = {
                        "id": row["id"], "answer": row["answer"], "k": k,
                        "strict_pass": sum(s > 0 for s, _ in grades),
                        "strict_reward_sum": sum(s for s, _ in grades),
                        "lenient_pass": sum(l for _, l in grades),
                        "n_truncated": sum(o.finish_reason == "length" for o in out.outputs),
                        "avg_tokens": sum(len(o.token_ids) for o in out.outputs) / k,
                        "sample": out.outputs[0].text,
                    }
                    done[row["id"]] = rec
                    f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"  {min(start + args.chunk, len(todo))}/{len(todo)} done")

    rows = [done[r["id"]] for r in train if r["id"] in done]
    summarize(rows, k)
    passes = {r["id"]: r["strict_pass"] for r in select_learnable(rows, k)}
    filtered = [{**r, "strict_pass": passes[r["id"]]} for r in train if r["id"] in passes]
    filt_path = ROOT / args.output
    with open(filt_path, "w") as f:
        for r in filtered:
            f.write(json.dumps(r) + "\n")
    print(f"saved -> {out_path.relative_to(ROOT)}\nsaved -> {filt_path.relative_to(ROOT)} "
          f"({len(filtered)} learnable questions)")


if __name__ == "__main__":
    main()
