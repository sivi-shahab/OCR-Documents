"""Table parsing (HTML and pipe markdown) and amount column-placement scoring."""
import re
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser

from ocr_eval.numbers import extract_amounts

Grid = list[list[str]]

_HTML_TABLE = re.compile(r"<table\b.*?(?:</table\s*>|\Z)", re.IGNORECASE | re.DOTALL)
_SEPARATOR_CELL = re.compile(r":?-+:?")


def _span(attrs: dict[str, str | None], name: str) -> int:
    try:
        return max(1, min(int(attrs.get(name) or 1), 1000))
    except ValueError:
        return 1


class _TableState:
    def __init__(self) -> None:
        self.rows: Grid = []
        self.cells: list[tuple[str, int, int]] | None = None
        self.text: list[str] | None = None
        self.span: tuple[int, int] = (1, 1)
        self.pending: dict[int, list] = {}  # col -> [remaining rows, text]

    def end_cell(self) -> None:
        if self.text is not None and self.cells is not None:
            self.cells.append((" ".join("".join(self.text).split()), *self.span))
        self.text = None

    def end_row(self) -> None:
        self.end_cell()
        if self.cells is None:
            return
        row: list[str] = []

        def fill_pending() -> None:
            while len(row) in self.pending:
                entry = self.pending[len(row)]
                row.append(entry[1])
                entry[0] -= 1
                if entry[0] <= 0:
                    del self.pending[len(row) - 1]

        for text, colspan, rowspan in self.cells:
            fill_pending()
            for _ in range(colspan):
                if rowspan > 1:
                    self.pending[len(row)] = [rowspan - 1, text]
                row.append(text)
        while self.pending and max(self.pending) >= len(row):
            if len(row) in self.pending:
                fill_pending()
            else:
                row.append("")
        self.rows.append(row)
        self.cells = None


class _HtmlTables(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.grids: list[Grid] = []
        self._stack: list[_TableState] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._stack.append(_TableState())
            return
        if not self._stack:
            return
        st = self._stack[-1]
        if tag == "tr":
            st.end_row()
            st.cells = []
        elif tag in ("td", "th"):
            st.end_cell()
            if st.cells is None:
                st.cells = []
            a = dict(attrs)
            st.span = (_span(a, "colspan"), _span(a, "rowspan"))
            st.text = []
        elif tag == "br" and st.text is not None:
            st.text.append(" ")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "br" and self._stack and self._stack[-1].text is not None:
            self._stack[-1].text.append(" ")
        elif tag != "br":
            self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if not self._stack:
            return
        if tag == "table":
            self._pop()
        elif tag == "tr":
            self._stack[-1].end_row()
        elif tag in ("td", "th"):
            self._stack[-1].end_cell()

    def handle_data(self, data: str) -> None:
        if self._stack and self._stack[-1].text is not None:
            self._stack[-1].text.append(data)

    def _pop(self) -> None:
        st = self._stack.pop()
        st.end_row()
        if st.rows:
            self.grids.append(st.rows)

    def finish(self) -> list[Grid]:
        self.close()
        while self._stack:
            self._pop()
        return self.grids


def _html_tables(markdown: str) -> list[Grid]:
    parser = _HtmlTables()
    for m in _HTML_TABLE.finditer(markdown):
        parser.feed(m.group())
    return parser.finish()


def _pipe_cells(line: str) -> list[str]:
    line = line.strip()
    line = line[1:] if line.startswith("|") else line
    line = line[:-1] if line.endswith("|") else line
    return [c.strip() for c in line.split("|")]


def _pipe_tables(markdown: str) -> list[Grid]:
    grids: list[Grid] = []
    current: Grid = []
    for line in _HTML_TABLE.sub("\n", markdown).splitlines():
        if line.strip().startswith("|"):
            cells = _pipe_cells(line)
            if not all(_SEPARATOR_CELL.fullmatch(c) for c in cells):
                current.append(cells)
        elif current:
            grids.append(current)
            current = []
    if current:
        grids.append(current)
    return grids


def parse_tables(markdown: str) -> list[Grid]:
    """HTML tables first, then pipe tables, each as a grid of cell texts (spans expanded)."""
    return _html_tables(markdown) + _pipe_tables(markdown)


def amount_columns(grids: list[Grid]) -> dict[str, list[int]]:
    """Map each amount's digits to its column indexes counted from the right edge of its table."""
    out: dict[str, list[int]] = {}
    for grid in grids:
        width = max((len(r) for r in grid), default=0)
        for row in grid:
            for col, cell in enumerate(row):
                for key, n in extract_amounts(cell).items():
                    out.setdefault(key, []).extend([width - 1 - col] * n)
    return out


@dataclass
class ColumnScore:
    reference: int
    found_in_table: int
    placed_correctly: int


def score_columns(reference: list[Grid], ocr_markdown: str) -> ColumnScore | None:
    """Column placement of reference table amounts in the OCR tables; None without reference amounts."""
    ref = amount_columns(reference)
    total = sum(len(v) for v in ref.values())
    if not total:
        return None
    ocr = amount_columns(parse_tables(ocr_markdown))
    found = placed = 0
    for key, cols in ref.items():
        got = ocr.get(key, [])
        found += min(len(cols), len(got))
        placed += (Counter(cols) & Counter(got)).total()
    return ColumnScore(reference=total, found_in_table=found, placed_correctly=placed)
