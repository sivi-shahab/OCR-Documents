"""Page classification: digital text, pure scan, or scan carrying an invisible OCR text layer."""
from enum import StrEnum
from pathlib import Path

import pymupdf

_MIN_CHARS = 50
_FULL_IMAGE_RATIO = 0.8
_INVISIBLE_TRACE = 3  # get_texttrace() span type for invisible text (render mode 3)


class PageKind(StrEnum):
    DIGITAL = "digital"
    SCAN = "scan"
    SCAN_WITH_HIDDEN_TEXT = "scan_with_hidden_text"


def classify_page(page: pymupdf.Page) -> PageKind:
    visible = hidden = 0
    for span in page.get_texttrace():
        if span["type"] == _INVISIBLE_TRACE:
            hidden += len(span["chars"])
        else:
            visible += len(span["chars"])
    if visible >= _MIN_CHARS:
        return PageKind.DIGITAL
    area = page.rect.width * page.rect.height
    full_image = any(
        (pymupdf.Rect(img["bbox"]) & page.rect).get_area() >= _FULL_IMAGE_RATIO * area
        for img in page.get_image_info()
    )
    if hidden >= _MIN_CHARS and full_image:
        return PageKind.SCAN_WITH_HIDDEN_TEXT
    return PageKind.SCAN


def classify_pdf(path: Path) -> dict[int, PageKind]:
    with pymupdf.open(path) as doc:
        return {i + 1: classify_page(page) for i, page in enumerate(doc)}
