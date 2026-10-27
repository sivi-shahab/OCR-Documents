"""On-disk cache of engine output, so interrupted or repeated runs never re-call an engine."""

import json
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


@dataclass
class PageOutput:
    page: int
    markdown: str
    seconds: float | None
    error: str | None = None
    truncated: bool = False


class Engine(Protocol):
    name: str
    paid_price_per_page: float | None  # None for self-hosted engines

    def available(self) -> str | None:
        """Reason the engine cannot be used, or None when it is ready."""

    def meta_info(self) -> dict:
        """Fields (model, concurrency, ...) recorded in meta.json after a run."""

    def run(self, pdf: Path, pages: list[int], *, is_scan_variant: bool) -> AsyncIterator[PageOutput]: ...


class RunStore:
    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def _dir(self, variant: str, doc: str, engine: str) -> Path:
        return self._root / variant / doc / engine

    def _page_path(self, variant: str, doc: str, engine: str, page: int) -> Path:
        return self._dir(variant, doc, engine) / f"page_{page:04d}.md"

    def missing_pages(self, variant: str, doc: str, engine: str, pages: list[int]) -> list[int]:
        recorded = self.meta(variant, doc, engine).get("pages", {})
        # A page that failed is retried on resume, so errors are never cached permanently.
        return [
            p
            for p in pages
            if not self._page_path(variant, doc, engine, p).exists() or recorded.get(str(p), {}).get("error")
        ]

    def save_page(self, variant: str, doc: str, engine: str, out: PageOutput) -> None:
        path = self._page_path(variant, doc, engine, out.page)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(out.markdown, encoding="utf-8")
        meta = self.meta(variant, doc, engine)
        meta.setdefault("pages", {})[str(out.page)] = {
            "seconds": out.seconds,
            "error": out.error,
            "truncated": out.truncated,
        }
        self._write_meta(variant, doc, engine, meta)

    def load_page(self, variant: str, doc: str, engine: str, page: int) -> PageOutput | None:
        path = self._page_path(variant, doc, engine, page)
        if not path.exists():
            return None
        info = self.meta(variant, doc, engine).get("pages", {}).get(str(page), {})
        return PageOutput(
            page,
            path.read_text(encoding="utf-8"),
            info.get("seconds"),
            error=info.get("error"),
            truncated=info.get("truncated", False),
        )

    def update_meta(self, variant: str, doc: str, engine: str, **fields) -> None:
        meta = self.meta(variant, doc, engine)
        meta.update(fields)
        self._write_meta(variant, doc, engine, meta)

    def meta(self, variant: str, doc: str, engine: str) -> dict:
        path = self._dir(variant, doc, engine) / "meta.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_meta(self, variant: str, doc: str, engine: str, meta: dict) -> None:
        directory = self._dir(variant, doc, engine)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "meta.json"
        tmp = directory / "meta.json.tmp"
        tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, target)


async def run_engine(
    engine: Engine, store: RunStore, variant: str, pdf: Path, pages: list[int], is_scan_variant: bool
) -> None:
    """Runs only the pages not yet cached and records the wall-clock time of this run."""
    doc = Path(pdf).stem
    todo = store.missing_pages(variant, doc, engine.name, pages)
    if not todo:
        return
    store.update_meta(variant, doc, engine.name, started=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    start = time.perf_counter()
    async for out in engine.run(pdf, todo, is_scan_variant=is_scan_variant):
        store.save_page(variant, doc, engine.name, out)
    wall = time.perf_counter() - start
    store.update_meta(variant, doc, engine.name, **engine.meta_info())
    # Accumulate across resumed runs so wall_seconds covers all pages.
    previous = store.meta(variant, doc, engine.name).get("wall_seconds", 0.0)
    store.update_meta(variant, doc, engine.name, wall_seconds=round(previous + wall, 3))
