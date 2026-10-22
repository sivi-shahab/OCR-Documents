"""Cheap per-page quality checks on model output.

A generation loop repeats the same few tokens until max_tokens, so its text compresses far
better than real document text (prose and tables stay above roughly 0.2).

A born-digital page also carries the PDF's own text layer. The OCR output should contain
roughly the same words, so low word overlap points at misread or hallucinated output (or
at a text layer that is itself junk, e.g. a bad earlier OCR); either way a person should look.
The threshold is a starting point, not a measured value.
"""

import re
import zlib
from collections import Counter

_MIN_CHARS = 1000  # short outputs are too small for the ratio to mean anything
_MAX_COMPRESSION_RATIO = 0.1


def is_repetitive(text: str) -> bool:
    data = text.encode()
    if len(data) < _MIN_CHARS:
        return False
    return len(zlib.compress(data)) / len(data) < _MAX_COMPRESSION_RATIO


_TAG = re.compile(r"<[^>]+>")
_WORD = re.compile(r"\w+")
_MIN_TEXT_LAYER_WORDS = 50  # too few words to compare, e.g. a heading-only page
_MIN_WORD_OVERLAP = 0.5


def disagrees_with_text_layer(ocr_text: str, text_layer: str) -> bool:
    """True when the OCR output and the PDF text layer share too few words (Dice coefficient)."""
    layer_words = _words(text_layer)
    if layer_words.total() < _MIN_TEXT_LAYER_WORDS:
        return False
    ocr_words = _words(_TAG.sub(" ", ocr_text))
    overlap = 2 * (layer_words & ocr_words).total() / (layer_words.total() + ocr_words.total())
    return overlap < _MIN_WORD_OVERLAP


def _words(text: str) -> Counter[str]:
    return Counter(_WORD.findall(text.lower()))
