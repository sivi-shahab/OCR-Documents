"""Per-page scoring of cached engine output, and per engine × column summaries.

Columns: `digital` (original PDF, digital pages), `synthetic_scan` (degraded copies of those pages) — both scored
against the original text layer — and `real_scan` (scanned pages, no answer key: consensus and accounting checks).
"""

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

from ocr_eval.accounting import CheckSummary, check_balance, check_totals
from ocr_eval.agreement import consensus, score_against_consensus
from ocr_eval.classify import PageKind
from ocr_eval.numbers import NumberScore, extract_amounts, score_numbers
from ocr_eval.store import RunStore
from ocr_eval.tables import ColumnScore, Grid, parse_tables, score_columns
from ocr_eval.text_metrics import english_coverage, word_dice

COLUMNS = ("digital", "synthetic_scan", "real_scan")
ORIGINAL, SYNTHETIC = "original", "synthetic"
REFERENCE_ENGINE = "reference"  # pseudo engine: the reference tables, used to validate the accounting checker
TABLE_REFERENCE_ENGINE = "docling"  # its own tables are the column reference, so it is not scored on `digital`


@dataclass
class PageRow:
    column: str
    document: str
    page: int  # for synthetic_scan rows: the original page the synthetic page was made from
    engine: str
    kind: str
    number: NumberScore | None
    dice: float | None
    english: float | None
    columns: ColumnScore | None
    totals: CheckSummary | None
    consensus: NumberScore | None
    empty: bool
    truncated: bool
    error: str | None
    seconds: float | None


@dataclass
class Summary:
    column: str
    engine: str
    pages: int
    amount_recall: float | None
    misread_rate: float | None
    negative_recall: float | None
    median_dice: float | None
    english_skipped_pages: int
    column_placement: float | None  # placed in the right column / found in an OCR table
    column_end_to_end: float | None  # placed in the right column / all reference table amounts
    totals_pass_rate: float | None
    consensus_recall: float | None
    empty_pages: int
    truncated_pages: int
    error_pages: int
    pages_per_minute: float | None
    median_seconds: float | None
    p95_seconds: float | None
    cost_usd: float | None
    totals_uncheckable: int = 0


def variant_of(column: str) -> str:
    return SYNTHETIC if column == "synthetic_scan" else ORIGINAL


def load_json(path: Path, default):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def clean_reference_grids(grids: list[Grid]) -> tuple[list[Grid], int]:
    """Blank reference cells that hold more than one amount; their column is ambiguous (garbled Docling cells).

    Returns the cleaned grids and the number of amounts excluded that way.
    """
    excluded = 0
    cleaned: list[Grid] = []
    for grid in grids:
        new_grid = []
        for row in grid:
            new_row = []
            for cell in row:
                n = extract_amounts(cell).total()
                if n > 1:
                    excluded += n
                    cell = ""
                new_row.append(cell)
            new_grid.append(new_row)
        cleaned.append(new_grid)
    return cleaned, excluded


def reference_grids(entries: list[dict]) -> dict[int, list[Grid]]:
    """reference_tables.json entries ([{page, grid}]) grouped by page."""
    by_page: dict[int, list[Grid]] = {}
    for entry in entries:
        by_page.setdefault(int(entry["page"]), []).append(entry["grid"])
    return by_page


def _text_layer(pdf: Path) -> dict[int, str]:
    with pymupdf.open(pdf) as doc:
        return {i + 1: page.get_text() for i, page in enumerate(doc)}


def _engines(store_root: Path, variant: str, doc: str) -> list[str]:
    directory = store_root / variant / doc
    return sorted(p.name for p in directory.iterdir() if p.is_dir()) if directory.exists() else []


def _row(column, doc, page, engine, kind, out, **scores) -> PageRow:
    fields = {"number": None, "dice": None, "english": None, "columns": None, "totals": None, "consensus": None}
    fields.update(scores)
    return PageRow(
        column=column, document=doc, page=page, engine=engine, kind=kind, **fields,
        empty=not out.error and not out.markdown.strip(), truncated=out.truncated, error=out.error,
        seconds=out.seconds,
    )  # fmt: skip


def _text_scored(column, doc, page, engine, kind, out, reference_text, grids) -> PageRow:
    md = out.markdown
    no_columns = column == "digital" and engine == TABLE_REFERENCE_ENGINE
    return _row(
        column, doc, page, engine, kind, out,
        number=score_numbers(reference_text, md), dice=word_dice(md, reference_text),
        english=english_coverage(md, reference_text),
        columns=None if no_columns or not grids else score_columns(grids, md),
    )  # fmt: skip


def score_run(run_dir: Path, pdfs: list[Path]) -> list[PageRow]:
    run_dir = Path(run_dir)
    classification = load_json(run_dir / "classification.json", {})
    synthetic_map = load_json(run_dir / "synthetic_map.json", {})
    references = load_json(run_dir / "reference_tables.json", {})
    store_root = run_dir / "runs"
    store = RunStore(store_root)
    rows: list[PageRow] = []
    for pdf in pdfs:
        doc = Path(pdf).stem
        kinds = {int(p): k for p, k in classification.get(doc, {}).items()}
        texts = _text_layer(pdf)
        raw_grids = reference_grids(references.get(doc, []))
        clean = {page: clean_reference_grids(grids)[0] for page, grids in raw_grids.items()}

        outputs = {e: {} for e in _engines(store_root, ORIGINAL, doc)}
        for engine, pages in outputs.items():
            for page in kinds:
                out = store.load_page(ORIGINAL, doc, engine, page)
                if out is not None:
                    pages[page] = out
        for page, kind in sorted(kinds.items()):
            if kind == PageKind.DIGITAL:
                for engine, pages in outputs.items():
                    if page in pages:
                        rows.append(_text_scored("digital", doc, page, engine, kind, pages[page], texts[page],
                                                 clean.get(page)))  # fmt: skip
                if page in raw_grids:
                    rows.append(PageRow("digital", doc, page, REFERENCE_ENGINE, kind, None, None, None, None,
                                        check_totals(raw_grids[page]), None, False, False, None, None))  # fmt: skip
                continue
            on_page = {engine: pages[page] for engine, pages in outputs.items() if page in pages}
            # A failed page has no reading to vote with; it is still scored (as empty) against the others.
            agreed = consensus({engine: out.markdown for engine, out in on_page.items() if not out.error})
            for engine, out in on_page.items():
                rows.append(_row(
                    "real_scan", doc, page, engine, kind, out,
                    totals=check_totals(parse_tables(out.markdown)),
                    consensus=score_against_consensus(out.markdown, agreed) if agreed else None,
                ))  # fmt: skip

        mapping = {int(s): int(o) for s, o in synthetic_map.get(doc, {}).items()}
        for engine in _engines(store_root, SYNTHETIC, doc):
            for synthetic_page, page in sorted(mapping.items()):
                out = store.load_page(SYNTHETIC, doc, engine, synthetic_page)
                if out is not None:
                    rows.append(_text_scored("synthetic_scan", doc, page, engine, kinds.get(page, PageKind.DIGITAL),
                                             out, texts[page], clean.get(page)))  # fmt: skip
    return rows


def balance_checks(run_dir: Path, pdfs: list[Path]) -> dict[str, dict[str, bool | None]]:
    """Total assets == total liabilities + equity per document and engine (original PDFs).

    The pseudo engine "reference" runs the check on the text layer of the digital pages, to validate the checker;
    there the amounts follow the label line one per line, which check_balance reads with its line fallback.
    """
    run_dir = Path(run_dir)
    classification = load_json(run_dir / "classification.json", {})
    store = RunStore(run_dir / "runs")
    result: dict[str, dict[str, bool | None]] = {}
    for pdf in pdfs:
        doc = Path(pdf).stem
        kinds = {int(p): k for p, k in classification.get(doc, {}).items()}
        checks: dict[str, bool | None] = {}
        digital = [p for p, k in kinds.items() if k == PageKind.DIGITAL]
        if digital:
            texts = _text_layer(pdf)
            checks[REFERENCE_ENGINE] = check_balance({p: texts[p] for p in digital})
        for engine in _engines(run_dir / "runs", ORIGINAL, doc):
            pages = {p: out.markdown for p in kinds if (out := store.load_page(ORIGINAL, doc, engine, p))}
            checks[engine] = check_balance(pages)
        result[doc] = checks
    return result


def coverage_gaps(rows: list[PageRow], classification: dict, synthetic_map: dict) -> list[tuple[str, str, int, int]]:
    """(engine, column, scored pages, expected pages) wherever an engine was scored on fewer pages than exist.

    Expected pages count every scored document: digital and other pages from classification, synthetic pages
    from synthetic_map. Failed pages are scored (as empty), so only pages without any cached output are gaps.
    """
    docs = {r.document for r in rows}
    expected = dict.fromkeys(COLUMNS, 0)
    for doc in docs:
        kinds = classification.get(doc, {}).values()
        expected["digital"] += sum(k == PageKind.DIGITAL for k in kinds)
        expected["real_scan"] += sum(k != PageKind.DIGITAL for k in kinds)
        expected["synthetic_scan"] += len(synthetic_map.get(doc, {}))
    scored: dict[tuple[str, str], set[tuple[str, int]]] = {}
    for r in rows:
        scored.setdefault((r.engine, r.column), set()).add((r.document, r.page))
    engines = sorted({r.engine for r in rows} - {REFERENCE_ENGINE})
    return [
        (engine, column, len(scored.get((engine, column), ())), expected[column])
        for engine in engines
        for column in COLUMNS
        if len(scored.get((engine, column), ())) < expected[column]
    ]


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]  # nearest rank


def _text_scored_row(r: PageRow) -> bool:
    return r.column != "real_scan" and r.engine != REFERENCE_ENGINE


def _summarize_group(column: str, engine: str, rows: list[PageRow], metas: dict, gpu_hour_rate: float) -> Summary:
    numbers = [r.number for r in rows if r.number is not None]
    agreed = [r.consensus for r in rows if r.consensus is not None]
    columns = [r.columns for r in rows if r.columns is not None]
    totals = [r.totals for r in rows if r.totals is not None]
    dice = [r.dice for r in rows if r.dice is not None]
    seconds = [r.seconds for r in rows if r.seconds is not None]
    passed, failed = sum(t.passed for t in totals), sum(t.failed for t in totals)

    # Each page gets an equal share of its runs' wall-clock time. Paid engines are billed per scored page, so
    # their cost needs only the price (an interrupted run may lack wall_seconds); self-hosted cost is GPU time.
    wall = cost = 0.0
    timed = costed = 0
    for r in rows:
        meta = metas.get((variant_of(column), r.document, engine)) or {}
        # timed_pages = pages yielded while the clock ran (meta.json written before it existed: every cached page).
        timed_pages = meta.get("timed_pages", len(meta.get("pages", {})))
        share = meta["wall_seconds"] / timed_pages if meta.get("wall_seconds") and timed_pages else None
        if share is not None:
            wall += share
            timed += 1
        price = meta.get("price_per_page")
        if price is not None:
            cost += price
            costed += 1
        elif share is not None:
            cost += share / 3600 * gpu_hour_rate
            costed += 1

    return Summary(
        column=column,
        engine=engine,
        pages=len(rows),
        amount_recall=_ratio(sum(n.matched for n in numbers), sum(n.reference for n in numbers)),
        misread_rate=_ratio(sum(n.misread for n in numbers), sum(n.reference for n in numbers)),
        negative_recall=_ratio(sum(n.neg_matched for n in numbers), sum(n.neg_reference for n in numbers)),
        median_dice=statistics.median(dice) if dice else None,
        english_skipped_pages=sum(r.english is None for r in rows if _text_scored_row(r)),
        column_placement=_ratio(sum(c.placed_correctly for c in columns), sum(c.found_in_table for c in columns)),
        column_end_to_end=_ratio(sum(c.placed_correctly for c in columns), sum(c.reference for c in columns)),
        totals_pass_rate=_ratio(passed, passed + failed),
        consensus_recall=_ratio(sum(n.matched for n in agreed), sum(n.reference for n in agreed)),
        empty_pages=sum(r.empty for r in rows),
        truncated_pages=sum(r.truncated for r in rows),
        error_pages=sum(r.error is not None for r in rows),
        pages_per_minute=timed / wall * 60 if wall else None,
        median_seconds=statistics.median(seconds) if seconds else None,
        p95_seconds=_p95(seconds) if seconds else None,
        cost_usd=cost if costed else None,
        totals_uncheckable=sum(t.uncheckable for t in totals),
    )


def summarize(rows: list[PageRow], metas: dict, gpu_hour_rate: float) -> list[Summary]:
    """One Summary per column × engine; ratios are Σ over pages / Σ over pages, not means of page ratios.

    metas maps (variant, document, engine) to that run's meta.json.
    """
    groups: dict[tuple[str, str], list[PageRow]] = {}
    for r in rows:
        groups.setdefault((r.column, r.engine), []).append(r)
    order = sorted(groups, key=lambda key: (COLUMNS.index(key[0]), key[1]))
    return [
        _summarize_group(column, engine, groups[(column, engine)], metas, gpu_hour_rate) for column, engine in order
    ]


def rows_to_json(rows: list[PageRow]) -> list[dict]:
    return [asdict(r) for r in rows]


def rows_from_json(data: list[dict]) -> list[PageRow]:
    def build(cls, value):
        return cls(**value) if value is not None else None

    return [
        PageRow(**{
            **d,
            "number": build(NumberScore, d["number"]),
            "consensus": build(NumberScore, d["consensus"]),
            "columns": build(ColumnScore, d["columns"]),
            "totals": build(CheckSummary, d["totals"]),
        })
        for d in data
    ]  # fmt: skip
