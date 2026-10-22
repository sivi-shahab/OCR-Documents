"""Turns raw Unlimited-OCR output into layout blocks and clean markdown.

The model emits one header per layout region, followed by its content:

    <|det|>title [117, 74, 581, 101]<|/det|>Annual Report
    <|det|>text [117, 130, 592, 148]<|/det|>First line
    continuation line of the same block

Bounding boxes are normalised to a 0-999 grid relative to the page.
"""

import re

from app.domain.models import Block

_DET_HEADER = re.compile(r"<\|det\|>([^<\s]+)(?:\s*\[([^\]]*)\])?\s*<\|/det\|>(.*)", re.DOTALL)
_SPECIAL_TOKEN = re.compile(r"<\|[^|>]+\|>")

# Regions that carry no text worth returning in the markdown output.
_NON_TEXT_CATEGORIES = frozenset({"image"})


def parse_blocks(raw: str) -> list[Block]:
    blocks: list[Block] = []
    category, bbox, lines = "text", None, []

    def flush() -> None:
        if lines or category in _NON_TEXT_CATEGORIES:
            blocks.append(Block(category=category, bbox=bbox, text="\n".join(lines)))

    for line in raw.splitlines():
        line = line.rstrip()
        header = _DET_HEADER.match(line)
        if header:
            flush()
            category, bbox = header.group(1), _parse_bbox(header.group(2))
            content = header.group(3).strip()
            lines = [content] if content else []
        elif line:
            lines.append(_strip_special_tokens(line))
    flush()
    return blocks


def to_markdown(blocks: list[Block]) -> str:
    return "\n\n".join(
        block.text for block in blocks if block.text and block.category not in _NON_TEXT_CATEGORIES
    )


def _parse_bbox(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    try:
        return tuple(int(float(part)) for part in value.split(","))
    except ValueError:
        return None


def _strip_special_tokens(text: str) -> str:
    return _SPECIAL_TOKEN.sub("", text)
