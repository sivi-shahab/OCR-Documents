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
