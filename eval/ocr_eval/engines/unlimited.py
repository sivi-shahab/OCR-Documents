"""Unlimited-OCR served by vLLM, reusing the API's engine client and parser."""

import asyncio
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pymupdf
from app.domain.models import PageImage
from app.domain.parser import parse_blocks, to_markdown
from app.infrastructure.vllm_engine import GenerationConfig, VllmOcrEngine

from ocr_eval.store import PageOutput

_CONFIG = GenerationConfig("Unlimited-OCR", 8192, 35, 128, 1024)
_TIMEOUT = httpx.Timeout(600.0, connect=10.0)


class UnlimitedEngine:
    name = "unlimited"
    paid_price_per_page = None

    def __init__(
        self,
        base_url: str = "http://localhost:8100/v1",
        concurrency: int = 32,
        dpi: int = 200,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url
        self._concurrency = concurrency
        self._dpi = dpi
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self._base_url, timeout=_TIMEOUT, transport=self._transport)

    def available(self) -> str | None:
        async def check() -> bool:
            async with self._client() as client:
                return await VllmOcrEngine(client, _CONFIG).is_healthy()

        if asyncio.run(check()):
            return None
        return f"vLLM tidak merespons di {self._base_url}"

    async def run(self, pdf: Path, pages: list[int], *, is_scan_variant: bool) -> AsyncIterator[PageOutput]:
        semaphore = asyncio.Semaphore(self._concurrency)
        render_lock = threading.Lock()  # PyMuPDF documents are not thread-safe
        document = pymupdf.open(pdf)

        def render(number: int) -> PageImage:
            with render_lock:
                png = document[number - 1].get_pixmap(dpi=self._dpi).tobytes("png")
            return PageImage(png, "image/png")

        async with self._client() as client:
            engine = VllmOcrEngine(client, _CONFIG)

            async def one(number: int) -> PageOutput:
                async with semaphore:
                    start = time.perf_counter()
                    try:
                        image = await asyncio.to_thread(render, number)
                        output = await engine.recognize([image])
                    except Exception as exc:  # one bad page must not stop the others
                        return PageOutput(number, "", None, error=repr(exc))
                    markdown = to_markdown(parse_blocks(output.text))
                    return PageOutput(number, markdown, time.perf_counter() - start, truncated=output.truncated)

            tasks = [asyncio.create_task(one(n)) for n in pages]
            try:
                for finished in asyncio.as_completed(tasks):
                    yield await finished
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                document.close()
