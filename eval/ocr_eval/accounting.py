"""Accounting sanity checks on parsed financial statements: column totals and the balance equation."""
import re
from dataclasses import dataclass, field

from ocr_eval.numbers import _NUM, _TAG, _normalize
from ocr_eval.tables import Grid

_TOTAL_LABEL = re.compile(r"^(jumlah|total|sub ?total)\b", re.IGNORECASE)
_CELL_NUMBER = re.compile(
    r"(\()?\s*([-−])?\s*(?:rp\.?\s*)?(\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)\s*(\))?",
    re.IGNORECASE,
)
_TOLERANCE = 1
_ASSETS = {"jumlah aset", "total assets"}
_LIABILITIES_EQUITY = {"jumlah liabilitas dan ekuitas", "total liabilities and equity"}


@dataclass
class CheckSummary:
    passed: int = 0
    failed: int = 0
    uncheckable: int = 0
    failures: list[str] = field(default_factory=list)


def parse_number(cell: str) -> int | None:
    """Any numeric cell as a signed int (parentheses or a leading minus are negative); None if not numeric."""
    m = _CELL_NUMBER.fullmatch(cell.strip())
    if not m or bool(m.group(1)) != bool(m.group(4)):
        return None
    value = int(_normalize(m.group(3)))
    return -value if m.group(1) or m.group(2) else value


def _label(row: list[str]) -> str | None:
    return next((c for c in row if c.strip() and parse_number(c) is None), None)


def _cell(row: list[str], col: int) -> int | None:
    return parse_number(row[col]) if col < len(row) else None


def _check_grid(grid: Grid, summary: CheckSummary) -> None:
    start = 0
    for i, row in enumerate(grid):
        label = _label(row)
        if label is None or not _TOTAL_LABEL.match(label.strip()):
            continue
        # rows with no number at all are headings; a number elsewhere but not in the column is a missing cell
        details = [r for r in grid[start:i] if any(parse_number(c) is not None for c in r)]
        start = i + 1
        for col in range(max(len(r) for r in [row, *details])):
            total = _cell(row, col)
            values = [_cell(r, col) for r in details]
            if total is None and all(v is None for v in values):
                continue
            if total is None or not values or any(v is None for v in values):
                summary.uncheckable += 1
                continue
            diff = total - sum(values)
            if abs(diff) <= _TOLERANCE:
                summary.passed += 1
            else:
                summary.failed += 1
                summary.failures.append(f"{label.strip()} (kolom {col}): total {total} vs jumlah rincian {sum(values)}, selisih {diff}")


def check_totals(grids: list[Grid]) -> CheckSummary:
    """Each total row (Jumlah/Total/Subtotal) against the sum of the detail rows since the previous total.

    Every numeric cell counts (any parseable number, not only amounts of 7+ digits); parentheses are negative.
    """
    summary = CheckSummary()
    for grid in grids:
        _check_grid(grid, summary)
    return summary


def _first_amounts(markdown: str, labels: set[str]) -> list[int] | None:
    for line in markdown.splitlines():
        line = _TAG.sub(" | ", line)
        first = re.search(r"\d", line)
        head = line[: first.start()] if first else line
        parts = [re.sub(r"[^a-z ]", "", p.lower()).strip() for p in re.split(r"[/|]", head)]
        parts = [p for p in parts if p]
        if parts and all(p in labels for p in parts):
            nums = [int(_normalize(n)) for n in _NUM.findall(line[len(head):])]
            if nums:
                return nums
    return None


def check_balance(markdown_pages: dict[int, str]) -> bool | None:
    """Total assets equals total liabilities and equity in every amount column; None when either row is absent."""
    assets = liabilities = None
    for _, md in sorted(markdown_pages.items()):
        assets = assets or _first_amounts(md, _ASSETS)
        liabilities = liabilities or _first_amounts(md, _LIABILITIES_EQUITY)
    if not assets or not liabilities:
        return None
    return all(abs(a - b) <= _TOLERANCE for a, b in zip(assets, liabilities))
