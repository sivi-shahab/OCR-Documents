from dataclasses import dataclass, field
from enum import StrEnum


class PdfStrategy(StrEnum):
    """How a multi-page document is sent to the model."""

    PER_PAGE = "per_page"  # one request per page, run concurrently
    WHOLE_DOCUMENT = "whole_document"  # all pages in one long-horizon request


@dataclass(frozen=True, slots=True)
class PageImage:
    content: bytes
    mime_type: str


@dataclass(frozen=True, slots=True)
class EngineOutput:
    text: str
    truncated: bool = False  # generation stopped at max_tokens instead of finishing


@dataclass(frozen=True, slots=True)
class Block:
    """A layout region detected by the model."""

    category: str
    bbox: tuple[int, ...] | None
    text: str


@dataclass(frozen=True, slots=True)
class OcrResult:
    page: int | None  # None when the result spans the whole document
    markdown: str
    blocks: list[Block] = field(default_factory=list)
    raw: str = ""
    truncated: bool = False
