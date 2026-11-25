"""Run: python -m pytest tests/ -q   (works in either env; no MLX/torch needed)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rewards import lenient_correct, strict_correct  # noqa: E402


def conv(text):  # TRL's conversational completion format
    return [{"role": "assistant", "content": text}]


CASES = [  # (completion, gold, strict, lenient)
    ("48/2 = 24\n#### 72", "72", 1.0, 1.0),
    ("#### 1,250", "1250", 1.0, 1.0),
    ("#### 72.00", "72", 0.5, 1.0),                    # right, but unclean decimal
    ("#### 71", "72", 0.0, 0.0),                       # wrong answer
    ("\\boxed{72}", "72", 0.0, 1.0),                    # right, wrong format
    ("so the answer is 72", "72", 0.0, 1.0),            # right, wrong format
    ("I think 72", "72", 0.0, 1.0),                     # last-number only
    ("3 5 9 72 100\n#### 100", "72", 0.0, 0.0),         # number spraying doesn't pay
    ("<think>so 72", "72", 0.0, 0.0),                   # unfinished thinking
    ("<think>hmm</think>\n#### 72", "72", 1.0, 1.0),
    ("", "72", 0.0, 0.0),
]


def test_strict_and_lenient():
    comps = [conv(c) for c, *_ in CASES]
    golds = [g for _, g, *_ in CASES]
    assert strict_correct(comps, golds) == [s for *_, s, _ in CASES]
    assert lenient_correct(comps, golds) == [l for *_, l in CASES]


def test_plain_string_completions():
    assert strict_correct(["#### 5"], ["5"]) == [1.0]


def test_strict_never_exceeds_lenient():
    comps = [conv(c) for c, *_ in CASES]
    golds = [g for _, g, *_ in CASES]
    assert all(s <= l for s, l in zip(strict_correct(comps, golds), lenient_correct(comps, golds)))
