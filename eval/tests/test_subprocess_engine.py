import json
import sys
from pathlib import Path

from ocr_eval.engines.subprocess_engine import SubprocessEngine

# Mimics the worker CLI contract: one page_NNNN.md per page plus worker_meta.json.
_FAKE_WORKER = """
import argparse, json, os, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--pdf"); p.add_argument("--pages"); p.add_argument("--out"); p.add_argument("--mode")
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
(out / "call.json").write_text(json.dumps({"mode": a.mode, "pdf": a.pdf, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}))
Path(os.environ["FAKE_LOG"]).write_text((out / "call.json").read_text())
with open(os.environ["FAKE_LOG"] + ".calls", "a") as calls:
    calls.write(a.pages + " ")
if os.environ.get("FAKE_SLEEP"):
    import time
    time.sleep(float(os.environ["FAKE_SLEEP"]))
if os.environ.get("FAKE_FAIL") or os.environ.get("FAKE_FAIL_PAGE") in a.pages.split(","):
    print("worker exploded", file=sys.stderr)
    sys.exit(3)
if a.mode == "reference-tables":
    (out / "tables.json").write_text(json.dumps([{"page": 1, "grid": [["Kas", "1.234.567"]]}]))
else:
    for n in a.pages.split(","):
        if n != os.environ.get("FAKE_SKIP"):
            (out / f"page_{int(n):04d}.md").write_text("ok")
version = "9.9-" + a.pages.split(",")[0] if os.environ.get("FAKE_TAG_VERSION") else "9.9"
(out / "worker_meta.json").write_text(json.dumps({"model": "fake", "version": version, "seconds_total": 4.0}))
"""


def _engine(tmp_path: Path, monkeypatch, chunk_size: int = 25, timeout: float = 1800.0, **env) -> SubprocessEngine:
    worker = tmp_path / "fake_worker.py"
    worker.write_text(_FAKE_WORKER)
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "log.json"))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return SubprocessEngine("docling", Path(sys.executable), worker, gpu="7", chunk_size=chunk_size, timeout=timeout)


def _call(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "log.json").read_text())


async def _collect(engine, pdf, pages, scan=False):
    return [out async for out in engine.run(pdf, pages, is_scan_variant=scan)]


async def test_reads_page_markdown(tmp_path, monkeypatch, digital_pdf):
    engine = _engine(tmp_path, monkeypatch)
    outs = await _collect(engine, digital_pdf, [1, 2])
    assert [(o.page, o.markdown, o.error) for o in outs] == [(1, "ok", None), (2, "ok", None)]
    assert all(o.seconds == 2.0 for o in outs)  # seconds_total split evenly
    call = _call(tmp_path)
    assert call["mode"] == "auto" and call["gpu"] == "7"
    assert engine.meta_info() == {"model": "fake", "version": "9.9", "mode": "auto", "gpu": "7"}


async def test_scan_variant_uses_ocr_mode(tmp_path, monkeypatch, scan_pdf):
    engine = _engine(tmp_path, monkeypatch)
    await _collect(engine, scan_pdf, [1], scan=True)
    assert _call(tmp_path)["mode"] == "ocr"
    assert engine.meta_info()["mode"] == "ocr"


async def test_nonzero_exit_is_error_on_every_page(tmp_path, monkeypatch, digital_pdf):
    engine = _engine(tmp_path, monkeypatch, FAKE_FAIL="1")
    outs = await _collect(engine, digital_pdf, [1, 2])
    assert [o.page for o in outs] == [1, 2]
    assert all(o.markdown == "" and o.seconds is None for o in outs)
    assert all("exploded" in o.error and "3" in o.error for o in outs)


async def test_missing_page_file_is_page_error(tmp_path, monkeypatch, digital_pdf):
    engine = _engine(tmp_path, monkeypatch, FAKE_SKIP="2")
    first, second = await _collect(engine, digital_pdf, [1, 2])
    assert first.error is None and first.markdown == "ok"
    assert second.error and second.markdown == ""


def test_available_reports_missing_interpreter(tmp_path):
    engine = SubprocessEngine("mineru", tmp_path / "nope" / "bin" / "python", tmp_path / "w.py")
    reason = engine.available()
    assert reason and "nope" in reason
    assert engine.paid_price_per_page is None and engine.name == "mineru"


def test_available_when_interpreter_and_worker_exist(tmp_path, monkeypatch):
    assert _engine(tmp_path, monkeypatch).available() is None


def test_reference_tables(tmp_path, monkeypatch, digital_pdf):
    engine = _engine(tmp_path, monkeypatch)
    tables = engine.reference_tables(digital_pdf, tmp_path / "ref")
    assert tables == [{"page": 1, "grid": [["Kas", "1.234.567"]]}]
    assert _call(tmp_path)["mode"] == "reference-tables"
    assert (tmp_path / "ref" / "tables.json").exists()


def _calls(tmp_path: Path) -> list[str]:
    return (tmp_path / "log.json.calls").read_text().split()


async def test_pages_run_in_chunks(tmp_path, monkeypatch, scan_pdf):
    engine = _engine(tmp_path, monkeypatch, chunk_size=2, FAKE_TAG_VERSION="1")
    outs = await _collect(engine, scan_pdf, [1, 2, 3])
    assert _calls(tmp_path) == ["1,2", "3"]
    assert [(o.page, o.error, o.seconds) for o in outs] == [(1, None, 2.0), (2, None, 2.0), (3, None, 4.0)]
    assert engine.meta_info()["version"] == "9.9-3"  # from the latest chunk


async def test_failed_chunk_only_fails_its_pages(tmp_path, monkeypatch, scan_pdf):
    engine = _engine(tmp_path, monkeypatch, chunk_size=2, FAKE_TAG_VERSION="1", FAKE_FAIL_PAGE="3")
    outs = await _collect(engine, scan_pdf, [1, 2, 3])
    assert [(o.page, o.error is None) for o in outs] == [(1, True), (2, True), (3, False)]
    assert engine.meta_info()["version"] == "9.9-1"  # the latest successful chunk


async def test_chunk_timeout_kills_worker_and_fails_its_pages(tmp_path, monkeypatch, digital_pdf):
    import time

    engine = _engine(tmp_path, monkeypatch, timeout=0.5, FAKE_SLEEP="30")
    start = time.perf_counter()
    outs = await _collect(engine, digital_pdf, [1, 2])
    assert time.perf_counter() - start < 10
    assert all(o.markdown == "" and "timeout" in o.error.lower() for o in outs)


async def test_failed_run_resets_stale_worker_meta(tmp_path, monkeypatch, digital_pdf):
    engine = _engine(tmp_path, monkeypatch)
    await _collect(engine, digital_pdf, [1])
    assert engine.meta_info()["model"] == "fake"
    monkeypatch.setenv("FAKE_FAIL", "1")
    await _collect(engine, digital_pdf, [1])
    assert engine.meta_info()["model"] is None and engine.meta_info()["version"] is None


async def test_worker_without_meta_is_page_error(tmp_path, monkeypatch, digital_pdf):
    worker = tmp_path / "silent.py"
    worker.write_text("import sys; sys.exit(0)")
    engine = SubprocessEngine("docling", Path(sys.executable), worker)
    outs = await _collect(engine, digital_pdf, [1, 2])
    assert all(o.error and o.markdown == "" for o in outs)
