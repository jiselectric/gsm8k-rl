"""Summarize every result file into one comparison table.

    python src/aggregate.py              # dev results
    python src/aggregate.py --split test

Prints the table and writes results/<split>/summary.csv.
All metrics come from the saved rows, so nothing is re-run.
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CONFIG_COLS = ["shots", "seed", "thinking", "max_tokens", "temp", "batch_size", "revision"]


def summarize(path: Path) -> dict:
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    cfg = rows[0]["config"]
    return {
        "model": cfg["model"].split("/")[-1],
        "tag": path.stem.split("__", 1)[1],
        **{k: cfg.get(k) for k in CONFIG_COLS},
        "n": n,
        "acc": 100 * acc,
        # 95% margin of error: differences smaller than this may be noise
        "±95%": 100 * 1.96 * math.sqrt(acc * (1 - acc) / n),
        # strict = only answers in the requested '#### N' format count
        "strict_acc": 100 * sum(r["correct"] and r["rule"] == "hash" for r in rows) / n,
        # correct by value but not written identically to gold ("72.00" vs "72",
        # "1,250" vs "1250"): an exact-string-match grader would mark these wrong
        "norm_only%": (100 * sum(r["correct"] and r["pred_raw"] != r["gold"] for r in rows) / n
                       if "pred_raw" in rows[0] else float("nan")),
        "format_ok%": 100 * sum(r["rule"] == "hash" for r in rows) / n,
        "no_answer%": 100 * sum(r["pred"] is None for r in rows) / n,
        "truncated%": 100 * sum(r["hit_max_tokens"] for r in rows) / n,
        "avg_tokens": sum(r["n_tokens"] for r in rows) / n,
        "sec/q": sum(r["seconds"] for r in rows) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()

    files = sorted((ROOT / "results" / args.split).glob("*__*.jsonl"))
    if not files:
        print(f"No results in results/{args.split}/ yet.")
        return
    df = pd.DataFrame([summarize(p) for p in files]).sort_values(["model", "tag"])
    out = ROOT / "results" / args.split / "summary.csv"
    df.to_csv(out, index=False)

    show = df.drop(columns=CONFIG_COLS)  # tag already names the config
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(show.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print(f"\nsaved -> {out.relative_to(ROOT)}  (full config columns included)")


if __name__ == "__main__":
    main()
