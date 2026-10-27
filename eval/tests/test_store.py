from pathlib import Path

from ocr_eval.store import PageOutput, RunStore, run_engine


class FakeEngine:
    name = "fake"
    paid_price_per_page = None

    def __init__(self, fail_pages=()):
        self.calls: list[list[int]] = []
        self._fail = set(fail_pages)

    def available(self):
        return None

    def meta_info(self):
        return {"model": "fake-1", "concurrency": 3}

    async def run(self, pdf: Path, pages: list[int], *, is_scan_variant: bool):
        self.calls.append(list(pages))
        for page in pages:
            if page in self._fail:
                yield PageOutput(page, "", None, error="HTTP 500")
            else:
                yield PageOutput(page, f"text {page}", 0.5)


async def test_cached_pages_are_not_rerun(tmp_path):
    store = RunStore(tmp_path)
    engine = FakeEngine()
    await run_engine(engine, store, "original", Path("doc.pdf"), [1, 2], False)
    await run_engine(engine, store, "original", Path("doc.pdf"), [1, 2], False)
    assert engine.calls == [[1, 2]]
    assert store.load_page("original", "doc", "fake", 1).markdown == "text 1"
    assert store.meta("original", "doc", "fake")["wall_seconds"] >= 0


async def test_only_missing_pages_are_run(tmp_path):
    store = RunStore(tmp_path)
    engine = FakeEngine()
    await run_engine(engine, store, "original", Path("doc.pdf"), [1], False)
    await run_engine(engine, store, "original", Path("doc.pdf"), [1, 2], False)
    assert engine.calls == [[1], [2]]


async def test_failed_page_is_recorded_and_scored_empty(tmp_path):
    store = RunStore(tmp_path)
    await run_engine(FakeEngine(fail_pages=[2]), store, "original", Path("doc.pdf"), [1, 2], False)
    page = store.load_page("original", "doc", "fake", 2)
    assert page.error == "HTTP 500"
    assert page.markdown == ""
    assert store.meta("original", "doc", "fake")["pages"]["2"]["error"] == "HTTP 500"
    assert store.load_page("original", "doc", "fake", 1).error is None


def test_missing_pages_and_meta_defaults(tmp_path):
    store = RunStore(tmp_path)
    assert store.missing_pages("v", "d", "e", [1, 2]) == [1, 2]
    assert store.load_page("v", "d", "e", 1) is None
    assert store.meta("v", "d", "e") == {}
    store.update_meta("v", "d", "e", model="m")
    store.update_meta("v", "d", "e", concurrency=4)
    assert store.meta("v", "d", "e") == {"model": "m", "concurrency": 4}


async def test_meta_contains_engine_info(tmp_path):
    store = RunStore(tmp_path)
    await run_engine(FakeEngine(), store, "original", Path("doc.pdf"), [1], False)
    meta = store.meta("original", "doc", "fake")
    assert meta["model"] == "fake-1" and meta["concurrency"] == 3


async def test_failed_page_is_retried_on_resume(tmp_path):
    store = RunStore(tmp_path)
    await run_engine(FakeEngine(fail_pages=[2]), store, "original", Path("doc.pdf"), [1, 2], False)
    retry = FakeEngine()
    await run_engine(retry, store, "original", Path("doc.pdf"), [1, 2], False)
    assert retry.calls == [[2]]
    page = store.load_page("original", "doc", "fake", 2)
    assert page.error is None and page.markdown == "text 2"
    assert store.meta("original", "doc", "fake")["pages"]["2"]["error"] is None


def test_meta_write_leaves_no_temp_file(tmp_path):
    store = RunStore(tmp_path)
    store.update_meta("v", "d", "e", model="m")
    assert [p.name for p in (tmp_path / "v" / "d" / "e").iterdir()] == ["meta.json"]


class CrashingEngine(FakeEngine):
    """Yields the first pages, then raises (an interrupted run)."""

    def __init__(self, crash_after: int, exc: BaseException):
        super().__init__()
        self._crash_after = crash_after
        self._exc = exc

    async def run(self, pdf: Path, pages: list[int], *, is_scan_variant: bool):
        self.calls.append(list(pages))
        for i, page in enumerate(pages):
            if i == self._crash_after:
                raise self._exc
            yield PageOutput(page, f"text {page}", 0.5)


async def test_wall_time_recorded_when_engine_raises(tmp_path):
    import asyncio

    import pytest

    store = RunStore(tmp_path)
    for exc in (RuntimeError("boom"), asyncio.CancelledError()):
        with pytest.raises(type(exc)):
            await run_engine(CrashingEngine(2, exc), store, "original", Path(f"{type(exc).__name__}.pdf"),
                             [1, 2, 3], False)  # fmt: skip
        meta = store.meta("original", type(exc).__name__, "fake")
        assert meta["timed_pages"] == 2
        assert meta["wall_seconds"] >= 0
        assert meta["model"] == "fake-1"


async def test_resume_accumulates_wall_time_and_timed_pages(tmp_path):
    import pytest

    store = RunStore(tmp_path)
    with pytest.raises(RuntimeError):
        await run_engine(CrashingEngine(1, RuntimeError("x")), store, "original", Path("doc.pdf"), [1, 2, 3], False)
    first = store.meta("original", "doc", "fake")["wall_seconds"]
    await run_engine(FakeEngine(), store, "original", Path("doc.pdf"), [1, 2, 3], False)
    meta = store.meta("original", "doc", "fake")
    assert meta["timed_pages"] == 3
    assert meta["wall_seconds"] >= first
    assert sorted(meta["pages"]) == ["1", "2", "3"]


async def test_retry_counts_retried_pages_as_timed(tmp_path):
    store = RunStore(tmp_path)
    await run_engine(FakeEngine(fail_pages=[2]), store, "original", Path("doc.pdf"), [1, 2], False)
    await run_engine(FakeEngine(), store, "original", Path("doc.pdf"), [1, 2], False)
    assert store.meta("original", "doc", "fake")["timed_pages"] == 3  # page 2 ran twice, so its time counts twice
