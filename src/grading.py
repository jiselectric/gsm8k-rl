"""Answer extraction + correctness, shared by eval.py and the RL reward (rewards.py).

Pure Python (no MLX/torch), so the training reward and the eval score use the
exact same grader.
"""
import re
from decimal import Decimal, InvalidOperation


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
# A number: "1,250" (comma-grouped) or "1250" / "12.5", optionally negative.
# (?<![\d.]) stops "48-24" from yielding "-24" and "3.14" from yielding ".14"-style splits.
NUMBER = r"(?<![\d.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
NUMBER_RE = re.compile(NUMBER)

# Rules, from most explicit to least explicit. Each is a STYLE of answer, not a model.
HASH_RE = re.compile(r"####\s*\$?\s*(" + NUMBER + ")")
ANSWER_IS_RE = re.compile(
    # separator tolerates markdown: "Answer: 18", "answer is 18", "**Final Answer:** **18**"
    r"(?:final answer|the answer|answer)\s*(?:is)?[\s*:=]*\$?\s*(" + NUMBER + ")",
    re.IGNORECASE,
)

RULES = ("hash", "boxed", "answer_is", "last_number")  # strict -> lenient


def normalize_number(s: str) -> str:
    """'1,250.00' -> '1250', '-0.50' -> '-0.5', '72.' -> '72'."""
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return s


def boxed_contents(text: str) -> list[str]:
    """Contents of every \\boxed{...}, handling nested braces like \\boxed{\\text{72}}."""
    out = []
    start = 0
    while (i := text.find("\\boxed{", start)) != -1:
        depth, j = 1, i + len("\\boxed{")
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        if depth == 0:
            out.append(text[i + len("\\boxed{"): j - 1])
        start = j
    return out


def answer_region(completion: str) -> str | None:
    """Text where the final answer should be. Skips any <think>...</think> scratchpad.

    Returns None if thinking never finished (cut off by max_tokens): there is no
    final answer, and grabbing a number from half-finished thoughts would be a guess.
    """
    if "</think>" in completion:
        return completion.rsplit("</think>", 1)[1]
    if "<think>" in completion:
        return None
    return completion


def extract_raw_with_rule(completion: str) -> tuple[str | None, str]:
    """Return (number exactly as written, e.g. '1,250.00', or None; rule name).

    Always takes the LAST match of a rule: models often restate the question's
    numbers early and give the answer at the end.
    Rule names: see RULES, plus "unfinished_think" and "none".
    """
    text = answer_region(completion)
    if text is None:
        return None, "unfinished_think"

    if m := HASH_RE.findall(text):
        return m[-1], "hash"

    for content in reversed(boxed_contents(text)):
        if m := NUMBER_RE.findall(content):
            return m[-1], "boxed"

    if m := ANSWER_IS_RE.findall(text):
        return m[-1], "answer_is"

    if m := NUMBER_RE.findall(text):
        return m[-1], "last_number"

    return None, "none"


def extract_answer_with_rule(completion: str) -> tuple[str | None, str]:
    """Return (normalized number or None, name of the rule that matched)."""
    raw, rule = extract_raw_with_rule(completion)
    return (normalize_number(raw) if raw is not None else None), rule


def extract_answer(completion: str, strict: bool = False) -> str | None:
    """Final numeric answer as a normalized string, or None.

    strict=True accepts only the requested '#### <number>' format.
    """
    pred, rule = extract_answer_with_rule(completion)
    if strict and rule != "hash":
        return None
    return pred


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------
def to_decimal(s: str | None) -> Decimal | None:
    """'1,250.00' -> Decimal('1250.00'); None or unparseable -> None."""
    if s is None:
        return None
    try:
        return Decimal(s.replace(",", "").replace("$", "").strip())
    except InvalidOperation:
        return None


def is_correct(pred: str | None, gold: str) -> bool:
    """Exact numeric equality: '72.0' == '72' is True, '72.5' == '72' is False.

    Decimal (not float) so there's no rounding fuzz, and not int() so a decimal
    prediction is simply wrong instead of crashing. All GSM8K golds are integers.
    """
    p, g = to_decimal(pred), to_decimal(gold)
    return p is not None and g is not None and p == g
