"""RL reward functions for GRPO, built on the SAME grader as eval.py (grading.py).

TRL calls each function with the batch of completions plus every extra dataset
column as a keyword argument (here: `answer`, the gold number). Each returns
one float per completion.
"""
from grading import extract_raw_with_rule, is_correct, normalize_number


def _text(completion) -> str:
    """TRL passes conversational completions as [{"role": "assistant", "content": ...}]."""
    return completion[0]["content"] if isinstance(completion, list) else completion


def _grade(completion, gold: str) -> tuple[bool, str, str | None]:
    raw, rule = extract_raw_with_rule(_text(completion))
    pred = normalize_number(raw) if raw is not None else None
    return is_correct(pred, gold), rule, raw


UNCLEAN_PENALTY = 0.5  # '#### 72.0' when gold is '72': right math, but lm-eval's text match fails it


def strict_correct(completions, answer, **kwargs) -> list[float]:
    """THE training reward, for a correct answer in '#### N' form:
    1.0 if written cleanly ('#### 72'), 0.5 if written as a decimal ('#### 72.0').

    Strict on purpose: a lenient 'last number in the text' rule could be gamed
    by scattering numbers. Commas ('1,250') stay full reward: lm-eval strips them.
    All GSM8K golds are integers, so any '.' in a correct answer is a trailing '.0'.
    """
    out = []
    for c, gold in zip(completions, answer):
        ok, rule, raw = _grade(c, gold)
        if not (ok and rule == "hash"):
            out.append(0.0)
        else:
            out.append(UNCLEAN_PENALTY if "." in raw else 1.0)
    return out


def lenient_correct(completions, answer, **kwargs) -> list[float]:
    """Monitoring only (weight 0): correct under ANY extraction rule = `acc` in eval.

    If lenient rises but strict doesn't, the model is solving problems but
    drifting from the format. If strict > lenient ever, something is broken.
    """
    return [1.0 if _grade(c, gold)[0] else 0.0 for c, gold in zip(completions, answer)]
