"""Fast GPU evaluation with vLLM: same prompt, decoding and grader as eval.py (MLX).

Use it on the cloud pod to score checkpoints in minutes instead of hours.
Output rows have the same fields as eval.py, so aggregate.py works on them.

    python src/eval_vllm.py --model Qwen/Qwen3-1.7B --revision 70d244c... --thinking --tag think_vllm
    python src/eval_vllm.py --model checkpoints/qwen3-17b-grpo-v1/checkpoint-100 --thinking --tag A_s100
    python src/eval_vllm.py ... --split test        # final numbers only

Greedy decoding on vLLM and on MLX can differ slightly (different kernels), so
compare base vs trained models with the SAME engine.
"""
import argparse
import json
import time
from pathlib import Path

from grading import extract_raw_with_rule, is_correct, normalize_number
from prompts import ROOT, SYSTEM_PROMPT, build_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local checkpoint folder")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--split", default="dev", help="dev | test | any data/<split>.jsonl, e.g. math_test")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--name", default=None, help="model name in the output file (default: last path part)")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in open(ROOT / "data" / f"{args.split}.jsonl")]
    rows = rows[:args.n] if args.n else rows
    name = args.name or Path(args.model.rstrip("/")).name
    if "checkpoint-" in name:  # checkpoints/<run>/checkpoint-100 -> <run>-checkpoint-100
        name = f"{Path(args.model.rstrip('/')).parent.name}-{name}"
    out_dir = ROOT / "results" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}__{args.tag}.jsonl"

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    llm = LLM(model=args.model, revision=args.revision, dtype="bfloat16",
              max_model_len=args.max_tokens + 512, gpu_memory_utilization=args.gpu_mem, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    prompts = [build_prompt(r["question"], tok, [], args.thinking) for r in rows]

    t0 = time.time()
    outs = llm.generate(prompts, params)
    secs = (time.time() - t0) / len(rows)
    config = {"model": args.model, "revision": args.revision, "split": args.split, "n": args.n,
              "shots": 0, "seed": 0, "thinking": args.thinking, "max_tokens": args.max_tokens,
              "temp": 0.0, "batch_size": "vllm", "engine": "vllm", "system_prompt": SYSTEM_PROMPT}

    n_correct = 0
    with open(out_path, "w") as f:
        for row, out in zip(rows, outs):
            o = out.outputs[0]
            raw, rule = extract_raw_with_rule(o.text)
            pred = normalize_number(raw) if raw is not None else None
            ok = is_correct(pred, row["answer"])
            n_correct += ok
            f.write(json.dumps({
                "id": row["id"], "question": row["question"], "gold": row["answer"],
                "pred": pred, "pred_raw": raw, "rule": rule, "correct": ok,
                "completion": o.text, "n_tokens": len(o.token_ids),
                "hit_max_tokens": o.finish_reason == "length",
                "seconds": round(secs, 3), "config": config,
            }) + "\n")
    strict = sum(1 for l in open(out_path) if (r := json.loads(l))["correct"] and r["rule"] == "hash")
    print(f"\n{name} [{args.split}, {args.tag}]: acc {100 * n_correct / len(rows):.1f}%  "
          f"strict {100 * strict / len(rows):.1f}%  ({len(rows)} questions)")
    print(f"saved -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
