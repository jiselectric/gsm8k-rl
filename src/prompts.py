"""Prompt construction, shared by eval.py (MLX, Mac) and train_grpo.py (PyTorch, cloud).

No MLX/torch imports here, so training and evaluation build IDENTICAL prompts.
"""
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
# One fixed instruction for every model and every question. Changing this text
# changes the protocol, so log any change in notes/phase1_investigation.md.
SYSTEM_PROMPT = (
    "You are a helpful assistant solving Grade School Math word problems. "
    "Think step by step to solve the given question. "
    "End your response with the final numeric answer in the form '#### <number>'."
)

CALC_ANNOTATION = re.compile(r"<<[^>]*>>")  # GSM8K calculator markers: <<48/2=24>>


def load_fewshot_examples(k: int, seed: int = 0) -> list[dict]:
    """Pick k solved problems from TRAIN (never dev/test). Call once, before the loop.

    Same k + seed -> same examples, for every question and every model.
    """
    if k == 0:
        return []
    rows = [json.loads(l) for l in open(ROOT / "data" / "train.jsonl")]
    picked = random.Random(seed).sample(rows, k)
    return [{"question": r["question"],
             # strip <<...>> but keep the trailing "#### 72", which teaches the format
             "solution": CALC_ANNOTATION.sub("", r["solution"])}
            for r in picked]


def build_messages(question: str, examples: list[dict],
                   system: str | None = SYSTEM_PROMPT) -> list[dict]:
    """[system?] + [user/assistant pair per example] + [user: question]. No tokenizer needed."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    for ex in examples:
        messages.append({"role": "user", "content": ex["question"]})
        messages.append({"role": "assistant", "content": ex["solution"]})
    messages.append({"role": "user", "content": question})
    return messages


def build_prompt(question: str, tokenizer, examples: list[dict],
                 thinking: bool = False) -> str:
    """messages -> model-specific chat string (markers come from the tokenizer)."""
    messages = build_messages(question, examples)
    if tokenizer.chat_template is None:  # base model without a chat format
        return "\n\n".join(m["content"] for m in messages) + "\n\n"
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,                # return a string, not token ids
        add_generation_prompt=True,    # end with "assistant's turn" marker
        enable_thinking=thinking,      # Qwen3/3.5 only; other templates ignore it
    )
