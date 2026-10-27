import json

import pymupdf
import pytest

from ocr_eval.accounting import CheckSummary
from ocr_eval.numbers import NumberScore
from ocr_eval.score import (
    PageRow,
    clean_reference_grids,
    coverage_gaps,
    rows_from_json,
    rows_to_json,
    score_run,
    summarize,
)
from ocr_eval.store import PageOutput, RunStore
from ocr_eval.tables import ColumnScore

TABLE = "| Kas | 1.234.567 |\n| Piutang | 2.345.678 |\n| Jumlah | 3.580.245 |\n"


def _make_doc(path):
    doc = pymupdf.open()
    page = doc.new_page()
    lines = ["Kas 1.234.567", "Piutang 2.345.678", "Jumlah 3.580.245"]
    lines += [f"the total of assets and liabilities for the year {i}" for i in range(12)]
    for i, line in enumerate(lines):
        page.insert_text((50, 60 + i * 20), line, fontsize=10)
    doc.new_page()  # page 2 stands in for a scan; its kind comes from classification.json
    doc.save(path)


@pytest.fixture
def run(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _make_doc(pdf)
    out = tmp_path / "run"
    out.mkdir()
    (out / "classification.json").write_text(json.dumps({"doc": {"1": "digital", "2": "scan"}}))
    (out / "synthetic_map.json").write_text(json.dumps({"doc": {"1": 1}}))
    grid = [["Kas", "1.234.567"], ["Piutang", "2.345.678"], ["Jumlah", "3.580.245"]]
    (out / "reference_tables.json").write_text(json.dumps({"doc": [{"page": 1, "grid": grid}]}))
    store = RunStore(out / "runs")
    for engine in ("unlimited", "docling"):
        store.save_page("original", "doc", engine, PageOutput(1, TABLE, 1.0))
        store.save_page("original", "doc", engine, PageOutput(2, "| Kas | 1.234.567 |\n| Jumlah | 1.234.567 |", 2.0))
        store.save_page("synthetic", "doc", engine, PageOutput(1, TABLE, 1.5))
    return out, pdf, store


def _by(rows, column, engine, page=None):
    return [r for r in rows if r.column == column and r.engine == engine and (page is None or r.page == page)]


def test_score_run_columns(run):
    out, pdf, _ = run
    rows = score_run(out, [pdf])
    assert {(r.column, r.engine) for r in rows} == {
        (c, e) for c in ("digital", "synthetic_scan", "real_scan") for e in ("unlimited", "docling")
    } | {("digital", "reference")}
    scan = _by(rows, "real_scan", "unlimited")[0]
    assert scan.page == 2 and scan.kind == "scan"
    assert scan.number is None and scan.consensus is not None
    assert scan.consensus.matched == 2
    assert scan.totals is not None
    digital = _by(rows, "digital", "unlimited")[0]
    assert digital.number.matched == 3 and digital.number.reference == 3
    assert digital.columns == ColumnScore(3, 3, 3)
    assert _by(rows, "digital", "docling")[0].columns is None
    synthetic = _by(rows, "synthetic_scan", "docling")[0]
    assert synthetic.page == 1 and synthetic.columns is not None and synthetic.number.matched == 3
    reference = _by(rows, "digital", "reference")[0]
    assert reference.totals.passed == 1 and reference.number is None


def test_failed_page_is_recorded_and_scored_empty(run):
    out, pdf, store = run
    store.save_page("original", "doc", "unlimited", PageOutput(1, "", None, error="boom"))
    rows = score_run(out, [pdf])
    failed = _by(rows, "digital", "unlimited")[0]
    assert failed.error == "boom"
    assert failed.number.matched == 0 and failed.number.dropped == 3
    summary = next(s for s in summarize(rows, {}, 0.95) if (s.column, s.engine) == ("digital", "unlimited"))
    assert summary.error_pages == 1 and summary.amount_recall == 0.0


def test_reference_cells_with_several_amounts_are_excluded():
    grids = [[["Kas", "1.234.567"], ["Lain", "2.345.678 3.456.789"]]]
    cleaned, excluded = clean_reference_grids(grids)
    assert cleaned == [[["Kas", "1.234.567"], ["Lain", ""]]]
    assert excluded == 2


def test_merged_reference_cell_is_not_scored_for_columns(run):
    out, pdf, _ = run
    grid = [["Kas", "1.234.567"], ["Piutang Jumlah", "2.345.678 3.580.245"]]
    (out / "reference_tables.json").write_text(json.dumps({"doc": [{"page": 1, "grid": grid}]}))
    rows = score_run(out, [pdf])
    assert _by(rows, "digital", "unlimited")[0].columns.reference == 1


def _row(column="digital", engine="unlimited", page=1, **kw):
    fields = dict(
        document="doc", kind="digital", number=None, dice=None, english=None, columns=None, totals=None,
        consensus=None, empty=False, truncated=False, error=None, seconds=None,
    )  # fmt: skip
    fields.update(kw)
    return PageRow(column=column, engine=engine, page=page, **fields)


def _score(reference, matched, misread=0, neg_reference=0, neg_matched=0):
    dropped = reference - matched - misread
    return NumberScore(reference, matched + misread, matched, misread, dropped, 0, neg_reference, neg_matched)


def test_summary_aggregates_ratios():
    rows = [_row(page=1, number=_score(10, 9), dice=0.9), _row(page=2, number=_score(10, 1), dice=0.5)]
    [summary] = summarize(rows, {}, 0.95)
    assert summary.pages == 2
    assert summary.amount_recall == 0.5
    assert summary.median_dice == pytest.approx(0.7)


def test_summary_excludes_pages_without_amounts_and_aggregates_other_metrics():
    rows = [
        _row(page=1, number=_score(4, 2, misread=1, neg_reference=2, neg_matched=1), english=0.5,
             columns=ColumnScore(4, 3, 2), totals=CheckSummary(1, 1, 2)),
        _row(page=2, number=None, empty=True, truncated=True, columns=ColumnScore(4, 4, 4), totals=CheckSummary(1, 0, 1)),
    ]  # fmt: skip
    [s] = summarize(rows, {}, 0.95)
    assert s.amount_recall == 0.5 and s.misread_rate == 0.25 and s.negative_recall == 0.5
    assert s.english_skipped_pages == 1
    assert s.column_placement == pytest.approx(6 / 7)  # placed / found in an OCR table
    assert s.column_end_to_end == 0.75  # placed / all reference amounts
    assert s.totals_pass_rate == pytest.approx(2 / 3) and s.totals_uncheckable == 3
    assert s.empty_pages == 1 and s.truncated_pages == 1 and s.error_pages == 0
    assert s.consensus_recall is None and s.median_dice is None


def test_summary_time_and_cost_from_metas():
    rows = [
        _row(page=1, seconds=10.0), _row(page=2, seconds=30.0),
        _row(column="synthetic_scan", engine="azure", page=1, seconds=4.0),
    ]  # fmt: skip
    metas = {
        ("original", "doc", "unlimited"): {"wall_seconds": 60.0, "pages": {"1": {}, "2": {}}, "price_per_page": None},
        ("synthetic", "doc", "azure"): {"wall_seconds": 10.0, "pages": {"1": {}}, "price_per_page": 0.01},
    }
    by = {(s.column, s.engine): s for s in summarize(rows, metas, 0.95)}
    local = by[("digital", "unlimited")]
    assert local.pages_per_minute == pytest.approx(2.0)
    assert local.cost_usd == pytest.approx(60 / 3600 * 0.95)
    assert local.median_seconds == 20.0 and local.p95_seconds == 30.0
    paid = by[("synthetic_scan", "azure")]
    assert paid.cost_usd == pytest.approx(0.01) and paid.pages_per_minute == pytest.approx(6.0)


def test_rows_round_trip_through_json(run):
    out, pdf, _ = run
    rows = score_run(out, [pdf])
    assert rows_from_json(json.loads(json.dumps(rows_to_json(rows)))) == rows


def test_paid_cost_does_not_need_wall_seconds():
    rows = [_row(column="synthetic_scan", engine="azure", page=p) for p in (1, 2, 3)]
    metas = {("synthetic", "doc", "azure"): {"pages": {"1": {}, "2": {}, "3": {}}, "price_per_page": 0.01}}
    [s] = summarize(rows, metas, 0.95)
    assert s.cost_usd == pytest.approx(0.03)
    assert s.pages_per_minute is None


def test_failed_page_does_not_vote_in_consensus(run):
    out, pdf, store = run
    store.save_page("original", "doc", "unlimited", PageOutput(2, "| Kas | 1.234.567 |\n| Lain | 7.654.321 |", 1.0))
    store.save_page("original", "doc", "mistral", PageOutput(2, "partial 7.654.321", None, error="timeout"))
    rows = score_run(out, [pdf])
    docling = _by(rows, "real_scan", "docling")[0]
    assert docling.consensus.reference == 1  # only 1.234.567; the failed page's 7.654.321 does not count
    failed = _by(rows, "real_scan", "mistral")[0]
    assert failed.error == "timeout" and failed.consensus.matched == 0


def test_coverage_gaps_flags_engines_scored_on_fewer_pages():
    rows = [_row(page=1), _row(page=2), _row(engine="azure", page=1)]
    rows += [_row(column="synthetic_scan", engine=e, page=1) for e in ("unlimited", "azure")]
    rows += [_row(engine="reference", page=1)]
    classification = {"doc": {"1": "digital", "2": "digital"}}
    gaps = coverage_gaps(rows, classification, {"doc": {"1": 1}})
    assert gaps == [("azure", "digital", 1, 2)]
