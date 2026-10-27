"""Accounting sanity checks on parsed financial statements: column totals and the balance equation."""
import re
from dataclasses import dataclass, field

from ocr_eval.numbers import AMOUNT_RE, normalize_amount, strip_tags
from ocr_eval.tables import Grid, parse_tables

_TOTAL_LABEL = re.compile(r"^(jumlah|total|sub ?total)\b", re.IGNORECASE)
_AMOUNT_BODY = r"\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?|\d{7,}(?:[.,]\d{1,2})?"
_CELL_AMOUNT = re.compile(
    rf"(\()?\s*([-−])?\s*(?:rp\.?\s*)?({_AMOUNT_BODY})\s*(\))?", re.IGNORECASE
)
_DASH = frozenset("-–—−")
_ASSETS = {"jumlah aset", "total assets"}
_LIABILITIES_EQUITY = {"jumlah liabilitas dan ekuitas", "total liabilities and equity"}


@dataclass
class CheckSummary:
    passed: int = 0
    failed: int = 0
    uncheckable: int = 0
    failures: list[str] = field(default_factory=list)


def parse_amount(cell: str) -> int | None:
    """An amount-like cell as a signed int; "-" is 0. Years, note refs and other small integers give None."""
    text = cell.strip()
    if text in _DASH:
        return 0
    m = _CELL_AMOUNT.fullmatch(text)
    if not m or bool(m.group(1)) != bool(m.group(4)):
        return None
    value = int(normalize_amount(m.group(3)))
    return -value if m.group(1) or m.group(2) else value


def _label(row: list[str]) -> str | None:
    return next((c for c in row if c.strip() and parse_amount(c) is None), None)


def _is_total(row: list[str]) -> bool:
    label = _label(row)
    return label is not None and bool(_TOTAL_LABEL.match(label.strip()))


def _cell(row: list[str], col: int) -> int | None:
    return parse_amount(row[col]) if col < len(row) else None


def _has_amount(row: list[str]) -> bool:
    return any(parse_amount(c) is not None for c in row)


def _check_column(grid: Grid, t: int, col: int, cover: dict[tuple[int, int], int], summary: CheckSummary) -> None:
    """Walk upward from total row t: a preceding total is one block (jumping over the rows it covered)."""
    label = (_label(grid[t]) or "").strip()
    total = _cell(grid[t], col)
    blocks: list[tuple[int, int]] = []  # (value, first row index covered by the block)
    i = t - 1
    prev_total = next((j for j in range(t - 1, -1, -1) if _is_total(grid[j]) and _has_amount(grid[j])), -1)
    cover[(t, col)] = prev_total + 1
    while i >= 0:
        row = grid[i]
        if not _has_amount(row):
            i -= 1
            continue
        value = _cell(row, col)
        if value is None:  # a missing cell in this column
            summary.uncheckable += 1
            return
        if _is_total(row):
            start = cover.get((i, col), 0)
            blocks.append((value, start))
            i = start - 1
        else:
            blocks.append((value, i))
            i -= 1
    if total is None or not blocks:
        summary.uncheckable += 1
        return
    running, sums = 0, []
    for k, (value, start) in enumerate(blocks, 1):
        running += value
        sums.append(running)
        if abs(total - running) <= k:
            cover[(t, col)] = start
            summary.passed += 1
            return
    closest = min(sums, key=lambda s: abs(total - s))
    summary.failed += 1
    summary.failures.append(f"{label} (kolom {col}): total {total}, jumlah terdekat {closest}, selisih {total - closest}")


def check_totals(grids: list[Grid]) -> CheckSummary:
    """Check each total row (Jumlah/Total/Subtotal) per amount column.

    Only amount-like cells count (thousands separator or 7+ digits; "-" is 0; parentheses negative), so years and
    note references are ignored. Walking up from a total, a preceding total counts as one block and the rows it
    covered are skipped; the total passes if the first k blocks sum to it within k (decimal truncation slack).
    """
    summary = CheckSummary()
    for grid in grids:
        cover: dict[tuple[int, int], int] = {}
        for t, row in enumerate(grid):
            if not (_is_total(row) and _has_amount(row)):
                continue
            above = [r for r in grid[:t] if _has_amount(r)]
            cols = {c for r in [row, *above] for c in range(len(r)) if parse_amount(r[c]) is not None}
            for col in sorted(cols):
                if _cell(row, col) is None and not any(_cell(r, col) is not None for r in above):
                    continue
                _check_column(grid, t, col, cover, summary)
    return summary


def _amounts_only(text: str) -> list[int] | None:
    """The amounts of a cell or line that holds nothing but amounts ("-" is 0), else None."""
    text = text.strip()
    if not text:
        return None
    whole = parse_amount(text)
    if whole is not None:
        return [whole]
    values = [parse_amount(token) for token in re.split(r"[\s|]+", text) if token]
    return values if values and all(v is not None for v in values) else None


def _is_label(text: str, labels: set[str]) -> bool:
    """Every part of a (possibly bilingual "ID / EN") label is one of labels."""
    parts = [re.sub(r"[^a-z ]", "", p.lower()).strip() for p in re.split(r"[/|\n]", text)]
    parts = [" ".join(p.split()) for p in parts if p.strip()]
    return bool(parts) and all(p in labels for p in parts)


def _grid_amounts(markdown: str, labels: set[str]) -> list[int] | None:
    for grid in parse_tables(markdown):
        for row in grid:
            amounts: list[int] = []
            label_cells: list[str] = []
            for cell in row:
                values = _amounts_only(cell)
                if values is not None:
                    amounts += values
                elif cell.strip() and not re.fullmatch(r"\s*\d{1,4}[a-z]?\s*", cell):  # note refs are not labels
                    label_cells.append(cell)
            if amounts and label_cells and all(_is_label(c, labels) for c in label_cells):
                return amounts
    return None


def _line_amounts(markdown: str, labels: set[str]) -> tuple[list[int], bool] | None:
    """(amounts, fixed): amounts on the label's own line (fixed), else the amount-only lines that follow it."""
    lines = [re.sub(r"<[^>]*>", " | ", line) for line in markdown.splitlines()]
    for i, line in enumerate(lines):
        first = re.search(r"\d", line)
        head = line[: first.start()] if first else line
        if not _is_label(head, labels):
            continue
        nums = [int(normalize_amount(n)) for n in AMOUNT_RE.findall(strip_tags(line[len(head):]))]
        if nums:
            return nums, True
        following: list[int] = []
        for nxt in lines[i + 1 :]:
            if not nxt.strip(" |\t"):
                continue
            values = _amounts_only(nxt.strip(" |\t"))
            if values is None:
                break
            following += values
        if following:
            return following, False
    return None


def _row_amounts(markdown: str, labels: set[str]) -> tuple[list[int], bool] | None:
    amounts = _grid_amounts(markdown, labels)
    return (amounts, True) if amounts else _line_amounts(markdown, labels)


def check_balance(markdown_pages: dict[int, str]) -> bool | None:
    """Total assets equals total liabilities and equity in every amount column.

    The rows are looked up in parsed tables (HTML or pipe), else in the text: amounts on the label's line, or
    the amount-only lines that follow it (text layers and one-cell-per-line output). Amounts taken from following
    lines are cut to the column count of the other row when that one is known. None when either row is absent or
    the two rows have a different number of amount columns.
    """
    assets = liabilities = None
    for _, md in sorted(markdown_pages.items()):
        assets = assets or _row_amounts(md, _ASSETS)
        liabilities = liabilities or _row_amounts(md, _LIABILITIES_EQUITY)
    if not assets or not liabilities:
        return None
    (a, a_fixed), (b, b_fixed) = assets, liabilities
    if a_fixed and not b_fixed:
        b = b[: len(a)]
    elif b_fixed and not a_fixed:
        a = a[: len(b)]
    if len(a) != len(b):
        return None
    return all(abs(x - y) <= 1 for x, y in zip(a, b))
