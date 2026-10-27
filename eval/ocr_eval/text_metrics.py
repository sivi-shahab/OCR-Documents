"""Text-level metrics: word-overlap Dice and English function-word coverage."""
import re
from collections import Counter

from app.domain.quality import _TAG, _words

_MIN_REFERENCE_WORDS = 50
_MIN_EN_WORDS = 30
_EN = {"the", "of", "and", "for", "as", "which", "with", "are", "was", "were", "has", "have", "on", "by",
       "from", "this", "that"}
_LATIN = re.compile(r"[a-z]+")


def word_dice(ocr_text: str, reference_text: str) -> float | None:
    """Dice overlap of word multisets; None when the reference has fewer than 50 words."""
    ref = _words(reference_text)
    if ref.total() < _MIN_REFERENCE_WORDS:
        return None
    ocr = _words(_TAG.sub(" ", ocr_text))
    return 2 * (ref & ocr).total() / (ref.total() + ocr.total())


def _english(text: str) -> Counter[str]:
    return Counter(w for w in _LATIN.findall(text.lower()) if w in _EN)


def english_coverage(ocr_text: str, reference_text: str) -> float | None:
    """Share of the reference's English function words kept by the OCR; None below 30 such words."""
    ref = _english(reference_text)
    if ref.total() < _MIN_EN_WORDS:
        return None
    return (ref & _english(_TAG.sub(" ", ocr_text))).total() / ref.total()
