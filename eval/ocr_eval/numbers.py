"""Monetary-amount accuracy: matched / misread / dropped counts against a reference text."""
import re
from collections import Counter
from dataclasses import dataclass, field

from app.domain.quality import _TAG

_AMOUNT = r"\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?|\d{7,}"
_NUM = re.compile(_AMOUNT)
_NEG_PAREN = re.compile(rf"\(\s*({_AMOUNT})\s*\)")
_NEG_MINUS = re.compile(rf"(?<![\w.,])[-−]\s*({_AMOUNT})")
_MIN_DIGITS = 7
_MAX_MISREAD_DIFF = 2  # digits that may differ for two amounts to count as one misread
_MAX_EXAMPLES = 4


@dataclass
class NumberScore:
    reference: int
    ocr: int
    matched: int
    misread: int
    dropped: int
    extra: int
    neg_reference: int
    neg_matched: int
    misread_examples: list[str] = field(default_factory=list)
    dropped_examples: list[str] = field(default_factory=list)


def _normalize(raw: str) -> str:
    """Digits only; a trailing 1-2 digit decimal is dropped when its separator is not the thousands one."""
    seps = re.findall(r"[.,]", raw)
    tail = re.search(r"([.,])(\d{1,2})$", raw)
    if tail and len(seps) > 1 and tail.group(1) != seps[0]:
        raw = raw[: tail.start()]
    elif tail and len(seps) == 1 and len(tail.group(2)) != 3:
        raw = raw[: tail.start()]
    return re.sub(r"\D", "", raw)


def _count(matches: list[str]) -> Counter[str]:
    keys = (_normalize(m) for m in matches)
    return Counter(k for k in keys if len(k) >= _MIN_DIGITS)


def extract_amounts(text: str) -> Counter[str]:
    return _count(_NUM.findall(_TAG.sub(" ", text)))


def extract_negative_amounts(text: str) -> Counter[str]:
    text = _TAG.sub(" ", text)
    return _count(_NEG_PAREN.findall(text) + _NEG_MINUS.findall(text))


def _near(a: str, b: str) -> bool:
    return len(a) == len(b) and sum(x != y for x, y in zip(a, b)) <= _MAX_MISREAD_DIFF


def _score(reference: Counter[str], reference_neg: Counter[str], ocr_text: str) -> NumberScore:
    ocr = extract_amounts(ocr_text)
    matched = (reference & ocr).total()
    dropped = list((reference - ocr).elements())
    extra = list((ocr - reference).elements())
    pairs: list[tuple[str, str]] = []
    unpaired: list[str] = []
    for e in extra:
        for d in dropped:
            if _near(e, d):
                pairs.append((d, e))
                dropped.remove(d)
                break
        else:
            unpaired.append(e)
    return NumberScore(
        reference=reference.total(),
        ocr=ocr.total(),
        matched=matched,
        misread=len(pairs),
        dropped=len(dropped),
        extra=len(unpaired),
        neg_reference=reference_neg.total(),
        neg_matched=(reference_neg & extract_negative_amounts(ocr_text)).total(),
        misread_examples=[f"{a} -> {b}" for a, b in pairs[:_MAX_EXAMPLES]],
        dropped_examples=dropped[:_MAX_EXAMPLES],
    )


def score_numbers(reference_text: str, ocr_text: str) -> NumberScore | None:
    """Score OCR amounts against the reference text; None when the reference has no amounts."""
    reference = extract_amounts(reference_text)
    if not reference:
        return None
    return _score(reference, extract_negative_amounts(reference_text), ocr_text)


def score_numbers_against(reference: Counter[str], ocr_text: str) -> NumberScore:
    """Like score_numbers for a prebuilt amount Counter; negatives are unknown, so neg_reference is 0."""
    return _score(reference, Counter(), ocr_text)
