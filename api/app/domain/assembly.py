"""Joins per-page OCR results into one document and repairs what page breaks split apart.

Each page is OCR'd on its own, so the joined document still carries page furniture and
broken seams. The cleanup:

- drops running headers, footers and page numbers: blocks the model labels as such, plus
  short first/last lines that repeat on many pages (digits ignored, so "Page 7" matches);
- rejoins a paragraph split across pages, including a word hyphenated at the break;
- merges a table continued on the next page when the column counts match, dropping the
  repeated header row.

Moved text stays with the page it started on, so every page keeps its `<!-- page N -->` marker
and its span in the assembled document.
"""

import re
from collections import Counter
from dataclasses import dataclass, replace
from itertools import pairwise

from app.domain.jobs import PageOutcome, PageStatus
from app.domain.models import Block

_FURNITURE_CATEGORIES = frozenset({"header", "footer", "page_number"})
_NON_TEXT_CATEGORIES = frozenset({"image"})
_PROSE_CATEGORIES = frozenset({"text", "ref_text"})

# A line repeated at the top or bottom of this share of pages is treated as a running header/footer.
_RUNNING_LINE_MIN_PAGES = 3
_RUNNING_LINE_MIN_SHARE = 0.3
_RUNNING_LINE_MAX_CHARS = 200

_TERMINAL_PUNCTUATION = tuple('.!?:;"”’)]')
_HYPHENATED_END = re.compile(r"\w-$")
_TABLE_ROW = re.compile(r"<tr>.*?</tr>", re.DOTALL)
_TABLE_CELL = re.compile(r"<t[dh][\s>]")


@dataclass(slots=True)
class _Page:
    number: int
    status: PageStatus
    blocks: list[Block]


@dataclass(frozen=True, slots=True)
class PageSpan:
    """Where a page's text sits in the assembled markdown: markdown[start:end]."""

    page: int
    status: PageStatus
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class AssembledDocument:
    markdown: str
    spans: list[PageSpan]


def assemble_document(outcomes: list[PageOutcome], clean: bool = True) -> AssembledDocument:
    """Returns the document in page order, each page preceded by a `<!-- page N -->` marker,
    with the character span every page occupies."""
    pages = [_to_page(outcome) for outcome in outcomes]
    if clean:
        _drop_furniture(pages)
        _repair_seams(pages)
    chunks, spans, offset = [], [], 0
    for page in pages:
        chunk = _render(page)
        spans.append(PageSpan(page.number, page.status, offset, offset + len(chunk)))
        chunks.append(chunk)
        offset += len(chunk) + 2  # the "\n\n" separator
    return AssembledDocument("\n\n".join(chunks) + "\n", spans)


def assemble_markdown(outcomes: list[PageOutcome], clean: bool = True) -> str:
    return assemble_document(outcomes, clean).markdown


def _to_page(outcome: PageOutcome) -> _Page:
    result = outcome.result
    if result is None:
        blocks = []
    elif result.blocks:
        blocks = [block for block in result.blocks if block.category not in _NON_TEXT_CATEGORIES]
    else:  # text-layer fallback pages carry markdown only
        blocks = [Block(category="text", bbox=None, text=result.markdown)] if result.markdown else []
    return _Page(outcome.page, outcome.status, [block for block in blocks if block.text.strip()])


def _render(page: _Page) -> str:
    if page.status is PageStatus.FAILED:
        return f"<!-- page {page.number}: OCR failed -->"
    label = "" if page.status is PageStatus.SUCCEEDED else f" ({page.status})"
    body = "\n\n".join(block.text for block in page.blocks)
    return f"<!-- page {page.number}{label} -->\n\n{body}".rstrip()


def _drop_furniture(pages: list[_Page]) -> None:
    running = _running_lines(pages)
    for page in pages:
        page.blocks = [block for block in page.blocks if block.category not in _FURNITURE_CATEGORIES]
        if page.blocks and _signature(page.blocks[0].text) in running:
            page.blocks.pop(0)
        if page.blocks and _signature(page.blocks[-1].text) in running:
            page.blocks.pop()


def _running_lines(pages: list[_Page]) -> set[str]:
    counts: Counter[str] = Counter()
    for page in pages:
        blocks = [block for block in page.blocks if block.category not in _FURNITURE_CATEGORIES]
        edges = {_signature(block.text) for block in (blocks[:1] + blocks[-1:])}
        counts.update(edge for edge in edges if edge)
    threshold = max(_RUNNING_LINE_MIN_PAGES, _RUNNING_LINE_MIN_SHARE * len(pages))
    return {line for line, count in counts.items() if count >= threshold}


def _signature(text: str) -> str:
    if len(text) > _RUNNING_LINE_MAX_CHARS:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", text)).strip().lower()


def _repair_seams(pages: list[_Page]) -> None:
    for previous, current in pairwise(pages):
        if not previous.blocks or not current.blocks:
            continue
        last, first = previous.blocks[-1], current.blocks[0]
        joined = _join_prose(last, first) or _join_tables(last, first)
        if joined is not None:
            previous.blocks[-1] = joined
            current.blocks.pop(0)


def _join_prose(last: Block, first: Block) -> Block | None:
    if last.category not in _PROSE_CATEGORIES or first.category not in _PROSE_CATEGORIES:
        return None
    head, tail = last.text.rstrip(), first.text.lstrip()
    if not tail[:1].islower():
        return None
    if _HYPHENATED_END.search(head):
        return replace(last, text=head[:-1] + tail)
    if head.endswith(_TERMINAL_PUNCTUATION):
        return None
    return replace(last, text=f"{head} {tail}")


def _join_tables(last: Block, first: Block) -> Block | None:
    if last.category != "table" or first.category != "table":
        return None
    head_rows, tail_rows = _TABLE_ROW.findall(last.text), _TABLE_ROW.findall(first.text)
    if not head_rows or not tail_rows or _column_count(head_rows[0]) != _column_count(tail_rows[0]):
        return None
    if _normalise_row(tail_rows[0]) == _normalise_row(head_rows[0]):
        tail_rows = tail_rows[1:]  # header row repeated on the continuation page
    return replace(last, text="<table>" + "".join(head_rows + tail_rows) + "</table>")


def _column_count(row: str) -> int:
    return len(_TABLE_CELL.findall(row))


def _normalise_row(row: str) -> str:
    return re.sub(r"\s+", " ", row).strip().lower()
