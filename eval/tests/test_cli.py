import json

import openpyxl

from ocr_eval.cli import main
from ocr_eval.store import PageOutput


class FakeEngine:
    def __init__(self, name, price=None, reason=None):
        self.name = name
        self.paid_price_per_page = price
        self.reason = reason
        self.calls = []

    def available(self):
        return self.reason

    def meta_info(self):
        return {"model": f"{self.name}-v1"}

    async def run(self, pdf, pages, *, is_scan_variant):
        self.calls.append((pdf.name, list(pages), is_scan_variant))
        for page in pages:
            yield PageOutput(page, "Row total assets 1.234.567.000 and 2.345.678.000", 0.5)


class FakeReference:
    name = "docling"

    def __init__(self):
        self.calls = 0

    def available(self):
        return None

    def reference_tables(self, pdf, out):
        self.calls += 1
        return [{"page": 1, "grid": [["Kas", "1.234.567.000"], ["Jumlah", "1.234.567.000"]]}]


def _args(command, out, pdf, *extra):
    return [command, "--out", str(out), "--pdf", str(pdf), *extra]


def test_run_without_yes_stops_before_paid_calls(tmp_path, digital_pdf, capsys):
    out = tmp_path / "out"
    free, paid = FakeEngine("free"), FakeEngine("paid", price=0.01)
    factory = lambda names: [e for e in (free, paid) if e.name in names]
    reference = FakeReference()
    assert main(_args("classify", out, digital_pdf)) == 0
    assert main(_args("degrade", out, digital_pdf)) == 0
    code = main(_args("run", out, digital_pdf, "--engines", "free,paid"), engine_factory=factory, reference=reference)
    assert code == 2
    assert paid.calls == []
    assert sorted(free.calls) == [("digital.pdf", [1, 2], False), ("digital.pdf", [1, 2], True)]
    printed = capsys.readouterr().out
    assert "paid" in printed and "0.04" in printed and "--yes" in printed
    assert json.loads((out / "reference_tables.json").read_text())["digital"][0]["page"] == 1

    main(_args("run", out, digital_pdf, "--engines", "free"), engine_factory=factory, reference=reference)
    assert reference.calls == 1  # cached in reference_tables.json
    assert len(free.calls) == 2  # cached in runs/


def test_all_with_yes_writes_report_and_lists_skipped_engines(tmp_path, digital_pdf):
    out = tmp_path / "out"
    engines = [FakeEngine("free"), FakeEngine("paid", price=0.01), FakeEngine("gone", reason="key kosong")]
    factory = lambda names: [e for e in engines if e.name in names]
    args = _args("all", out, digital_pdf, "--engines", "free,paid,gone", "--yes")
    assert main(args, engine_factory=factory, reference=FakeReference()) == 0
    assert engines[1].calls and not engines[2].calls
    assert json.loads((out / "synthetic_map.json").read_text()) == {"digital": {"1": 1, "2": 2}}
    assert (out / "synthetic" / "digital.pdf").exists()
    book = openpyxl.load_workbook(out / "ocr_eval.xlsx")
    engines_in_summary = {r[1] for r in book["Summary"].iter_rows(min_row=2, values_only=True)}
    assert engines_in_summary == {"free", "paid", "reference"}
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "gone: key kosong" in md and "paid-v1" in md
