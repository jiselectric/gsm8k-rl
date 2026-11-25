"""Download GSM8K and create a fixed train/dev split.

- dev  = 500 problems held out from the official TRAIN split (use this while developing)
- test = official 1,319-problem test split (only touch for final numbers!)

Run once:  python src/make_splits.py
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

SEED = 42
DEV_SIZE = 500
OUT = Path(__file__).resolve().parent.parent / "data"


def gold_answer(solution: str) -> str:
    # GSM8K solutions end with "#### <number>"
    return solution.split("####")[-1].strip().replace(",", "")


def dump(rows, path):
    with open(path, "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"id": i, "question": r["question"],
                                "solution": r["answer"],
                                "answer": gold_answer(r["answer"])}) + "\n")
    print(f"wrote {len(rows):5d} -> {path.relative_to(OUT.parent)}")


def main():
    OUT.mkdir(exist_ok=True)
    ds = load_dataset("openai/gsm8k", "main")
    train = list(ds["train"])
    random.Random(SEED).shuffle(train)
    dump(train[DEV_SIZE:], OUT / "train.jsonl")
    dump(train[:DEV_SIZE], OUT / "dev.jsonl")
    dump(list(ds["test"]), OUT / "test.jsonl")


if __name__ == "__main__":
    main()
