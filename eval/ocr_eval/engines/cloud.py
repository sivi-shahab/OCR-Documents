"""Azure Document Intelligence and Mistral OCR, reusing the API's adapters.

PDFs are sent in sequential chunks (not in parallel) to stay under provider rate limits.
"""

import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

import httpx
import pymupdf
from app.core.config import Settings
from app.core.errors import EngineResponseError, EngineUnavailableError
from app.domain.models import OcrResult
from app.infrastructure.azure_document_intelligence import AzureConfig, AzureDocumentIntelligence
from app.infrastructure.mistral_ocr import MistralConfig, _document, to_results

from ocr_eval.store import PageOutput

_TIMEOUT = httpx.Timeout(600.0, connect=10.0)


class _Recognizer(Protocol):
    async def recognize(self, content: bytes) -> list[OcrResult]: ...


def split_pdf(pdf: Path, pages: list[int], chunk_pages: int) -> list[tuple[bytes, list[int]]]:
    """Cuts the requested 1-based pages into PDFs of at most chunk_pages, keeping the original numbers."""
    chunks = []
    with pymupdf.open(pdf) as source:
        for start in range(0, len(pages), chunk_pages):
            numbers = pages[start : start + chunk_pages]
            with pymupdf.open() as part:
                for number in numbers:
                    part.insert_pdf(source, from_page=number - 1, to_page=number - 1)
                chunks.append((part.tobytes(), numbers))
    return chunks


class _ChunkedEngine:
    name: str
    paid_price_per_page: float

    def __init__(self, chunk_pages: int, transport: httpx.AsyncBaseTransport | None) -> None:
        self._chunk_pages = chunk_pages
        self._transport = transport

    async def _recognize(self, client: httpx.AsyncClient, content: bytes) -> list[OcrResult]:
        raise NotImplementedError

    async def run(self, pdf: Path, pages: list[int], *, is_scan_variant: bool) -> AsyncIterator[PageOutput]:
        async with httpx.AsyncClient(timeout=_TIMEOUT, transport=self._transport) as client:
            for content, numbers in split_pdf(pdf, pages, self._chunk_pages):
                start = time.perf_counter()
                try:
                    results = await self._recognize(client, content)
                except Exception as exc:  # the whole chunk is lost, but later chunks may still work
                    for number in numbers:
                        yield PageOutput(number, "", None, error=repr(exc))
                    continue
                seconds = (time.perf_counter() - start) / len(numbers)
                by_page = {result.page: result.markdown for result in results}
                for index, number in enumerate(numbers, start=1):
                    if index in by_page:
                        yield PageOutput(number, by_page[index], seconds)
                    else:
                        yield PageOutput(number, "", None, error="engine returned no result for this page")


class AzureEngine(_ChunkedEngine):
    name = "azure"
    paid_price_per_page = 0.010

    def __init__(
        self,
        settings: Settings,
        chunk_pages: int = 25,
        transport: httpx.AsyncBaseTransport | None = None,
        adapter: _Recognizer | None = None,
    ) -> None:
        super().__init__(chunk_pages, transport)
        self._settings = settings
        self._adapter = adapter

    def available(self) -> str | None:
        if not self._settings.azure_di_api_key:
            return "OCR_AZURE_DI_API_KEY kosong"
        if not self._settings.azure_di_endpoint:
            return "OCR_AZURE_DI_ENDPOINT kosong"
        return None

    def meta_info(self) -> dict:
        s = self._settings
        return {"model": s.azure_di_model_id, "api_version": s.azure_di_api_version, "chunk_pages": self._chunk_pages}

    async def _recognize(self, client: httpx.AsyncClient, content: bytes) -> list[OcrResult]:
        if self._adapter is not None:
            return await self._adapter.recognize(content)
        s = self._settings
        config = AzureConfig(
            s.azure_di_endpoint, s.azure_di_api_key, s.azure_di_model_id, s.azure_di_api_version, s.azure_di_timeout_seconds
        )
        return await AzureDocumentIntelligence(client, config).recognize(content)


class MistralEngine(_ChunkedEngine):
    name = "mistral"
    paid_price_per_page = 0.004

    def __init__(
        self, settings: Settings, chunk_pages: int = 25, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        super().__init__(chunk_pages, transport)
        self._settings = settings
        self._model_seen: str | None = None

    def available(self) -> str | None:
        return None if self._settings.mistral_api_key else "OCR_MISTRAL_API_KEY kosong"

    def meta_info(self) -> dict:
        return {"model": self._model_seen or self._settings.mistral_ocr_model, "chunk_pages": self._chunk_pages}

    async def _recognize(self, client: httpx.AsyncClient, content: bytes) -> list[OcrResult]:
        # MistralOcr.recognize drops the response's "model", so the POST is repeated here to keep it.
        config = MistralConfig(self._settings.mistral_api_key, self._settings.mistral_base_url, self._settings.mistral_ocr_model)
        payload = {"model": config.model, "document": _document(content), "include_image_base64": False}
        try:
            response = await client.post(
                f"{config.base_url.rstrip('/')}/ocr",
                json=payload,
                headers={"Authorization": f"Bearer {config.api_key}"},
            )
        except httpx.HTTPError as exc:
            raise EngineUnavailableError(f"Mistral OCR unreachable: {exc!r}") from exc
        if not response.is_success:
            raise EngineResponseError(f"Mistral OCR returned {response.status_code}: {response.text[:500]}")
        body = response.json()
        self._model_seen = body.get("model") or self._model_seen
        return to_results(body)
