"""Build data/math_train.jsonl: MATH (Hendrycks) TRAIN problems whose answer is a whole number.

Harder than GSM8K, so far more questions land in the "sometimes right" range that GRPO
learns from. Whole-number answers only, so the same `#### N` grader works unchanged.
Only the MATH train split is used (never its test split, never any GSM8K eval data).

    python src/make_math.py              # -> data/math_train.jsonl
"""
import json
import re

from datasets import load_dataset

from prompts import ROOT

CONFIGS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
           "number_theory", "prealgebra", "precalculus"]


def last_boxed(s: str) -> str | None:
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    depth, j = 1, i + len("\\boxed{")
    while j < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        j += 1
    return s[i + len("\\boxed{"): j - 1] if depth == 0 else None


def integer_answer(solution: str) -> str | None:
    a = last_boxed(solution)
    if a is None:
        return None
    a = a.replace(",", "").replace("\\$", "").replace("$", "").replace("\\!", "").strip()
    return str(int(a)) if re.fullmatch(r"-?\d+", a) else None


def main():
    rows = []
    for c in CONFIGS:
        for i, r in enumerate(load_dataset("EleutherAI/hendrycks_math", c, split="train")):
            ans = integer_answer(r["solution"])
            if ans is not None:
                rows.append({"id": f"math-{c}-{i}", "question": r["problem"], "answer": ans,
                             "level": r["level"], "subject": c, "source": "math"})
    out = ROOT / "data" / "math_train.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} problems with whole-number answers -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
