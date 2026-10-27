"""Unlimited-OCR served by vLLM, reusing the API's engine client and parser."""

import asyncio
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pymupdf
from app.core.config import Settings
from app.domain.models import PageImage
from app.domain.parser import parse_blocks, to_markdown
from app.infrastructure.vllm_engine import GenerationConfig, VllmOcrEngine

from ocr_eval.store import PageOutput

_TIMEOUT = httpx.Timeout(600.0, connect=10.0)


class UnlimitedEngine:
    name = "unlimited"
    paid_price_per_page = None

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        base_url: str | None = None,
        concurrency: int = 32,
        dpi: int = 200,
        transport: httpx.AsyncBaseTransport | httpx.BaseTransport | None = None,
    ) -> None:
        """Model and generation parameters come from the API's Settings (OCR_* in .env), as in production.

        base_url defaults to Settings.vllm_base_url, whose own default is the host's vLLM, http://localhost:8100/v1.
        """
        settings = settings or Settings()
        self.config = GenerationConfig(
            settings.model_name,
            settings.max_tokens,
            settings.ngram_size,
            settings.single_image_window,
            settings.multi_image_window,
        )
        self.base_url = base_url or settings.vllm_base_url
        self._concurrency = concurrency
        self._dpi = dpi
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=_TIMEOUT, transport=self._transport)

    def available(self) -> str | None:
        # Synchronous httpx so it is safe to call from inside a running event loop.
        try:
            with httpx.Client(base_url=self.base_url, timeout=10.0, transport=self._transport) as client:
                healthy = client.get("/models").is_success
        except httpx.HTTPError:
            healthy = False
        return None if healthy else f"vLLM tidak merespons di {self.base_url}"

    def meta_info(self) -> dict:
        return {"model": self.config.model_name, "concurrency": self._concurrency, "dpi": self._dpi}

    async def run(self, pdf: Path, pages: list[int], *, is_scan_variant: bool) -> AsyncIterator[PageOutput]:
        semaphore = asyncio.Semaphore(self._concurrency)
        render_lock = threading.Lock()  # PyMuPDF documents are not thread-safe
        document = pymupdf.open(pdf)

        def render(number: int) -> PageImage:
            with render_lock:
                png = document[number - 1].get_pixmap(dpi=self._dpi).tobytes("png")
            return PageImage(png, "image/png")

        async with self._client() as client:
            engine = VllmOcrEngine(client, self.config)

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
