import asyncio
import time
from dataclasses import dataclass

import httpx

from app.core.errors import EngineResponseError, EngineUnavailableError, InvalidDocumentError
from app.domain.models import Block, OcrResult

# Paragraph roles reported by the layout model; paragraphs without a role are body text.
_CATEGORY_BY_ROLE = {
    "title": "title",
    "sectionHeading": "sub_title",
    "pageHeader": "header",
    "pageFooter": "footer",
    "pageNumber": "page_number",
    "footnote": "footnote",
    "formulaBlock": "formula",
}


@dataclass(frozen=True, slots=True)
class AzureConfig:
    endpoint: str  # e.g. https://<resource>.cognitiveservices.azure.com
    api_key: str
    model_id: str = "prebuilt-layout"
    api_version: str = "2024-11-30"
    timeout_seconds: float = 600
    poll_interval_seconds: float = 1


class AzureDocumentIntelligence:
    """Recognises images and PDFs with Azure AI Document Intelligence's REST API."""

    def __init__(self, client: httpx.AsyncClient, config: AzureConfig) -> None:
        self._client = client
        self._config = config

    async def recognize(self, content: bytes) -> list[OcrResult]:
        operation_url = await self._submit(content)
        result = await self._wait_for(operation_url)
        return to_results(result)

    async def _submit(self, content: bytes) -> str:
        config = self._config
        url = f"{config.endpoint.rstrip('/')}/documentintelligence/documentModels/{config.model_id}:analyze"
        response = await self._request(
            "POST",
            url,
            params={"api-version": config.api_version, "outputContentFormat": "markdown"},
            content=content,
            headers={"Content-Type": "application/octet-stream"},
        )
        if response.status_code != 202 or "operation-location" not in response.headers:
            raise EngineResponseError(f"Azure Document Intelligence returned {response.status_code}")
        return response.headers["operation-location"]

    async def _wait_for(self, operation_url: str) -> dict:
        deadline = time.monotonic() + self._config.timeout_seconds
        while True:
            response = await self._request("GET", operation_url)
            body = response.json()
            match body.get("status"):
                case "succeeded":
                    return body.get("analyzeResult") or {}
                case "failed" | "canceled":
                    error = body.get("error") or {}
                    raise EngineResponseError(f"Azure analysis failed: {error.get('message', body)}")
            if time.monotonic() > deadline:
                raise EngineUnavailableError("Azure Document Intelligence timed out")
            delay = float(response.headers.get("retry-after", self._config.poll_interval_seconds))
            await asyncio.sleep(min(delay, self._config.poll_interval_seconds * 5))

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        headers = {"Ocp-Apim-Subscription-Key": self._config.api_key, **kwargs.pop("headers", {})}
        try:
            response = await self._client.request(method, url, headers=headers, **kwargs)
        except httpx.TimeoutException as exc:
            raise EngineUnavailableError("Azure Document Intelligence timed out") from exc
        except httpx.HTTPError as exc:
            raise EngineUnavailableError(f"Azure Document Intelligence unreachable: {exc}") from exc
        if response.status_code in (400, 415):
            raise InvalidDocumentError(f"Azure rejected the document: {_error_message(response)}")
        if not response.is_success:
            raise EngineResponseError(
                f"Azure Document Intelligence returned {response.status_code}: {_error_message(response)}"
            )
        return response


def to_results(analyze_result: dict) -> list[OcrResult]:
    """Splits the analysed document into one result per page, keeping the markdown and paragraphs."""
    content = analyze_result.get("content", "")
    pages = analyze_result.get("pages", [])
    blocks_by_page: dict[int, list[Block]] = {page["pageNumber"]: [] for page in pages}
    sizes = {page["pageNumber"]: (page.get("width"), page.get("height")) for page in pages}
    for paragraph in analyze_result.get("paragraphs", []):
        regions = paragraph.get("boundingRegions") or []
        if not regions or regions[0]["pageNumber"] not in blocks_by_page:
            continue
        number = regions[0]["pageNumber"]
        blocks_by_page[number].append(
            Block(
                category=_CATEGORY_BY_ROLE.get(paragraph.get("role"), "text"),
                bbox=_normalise(regions[0].get("polygon"), *sizes[number]),
                text=paragraph.get("content", ""),
            )
        )
    results = []
    for page in pages:
        text = "".join(
            content[span["offset"] : span["offset"] + span["length"]] for span in page.get("spans", [])
        )
        number = page["pageNumber"]
        results.append(OcrResult(page=number, markdown=text.strip(), blocks=blocks_by_page[number], raw=text))
    return results


def _normalise(polygon: list[float] | None, width: float | None, height: float | None) -> tuple[int, ...] | None:
    """Converts Azure's polygon (inches or pixels) to [x1, y1, x2, y2] on a 0-999 grid."""
    if not polygon or not width or not height:
        return None
    xs, ys = polygon[0::2], polygon[1::2]

    def scale(value: float, size: float) -> int:
        return max(0, min(999, round(value / size * 999)))

    return (scale(min(xs), width), scale(min(ys), height), scale(max(xs), width), scale(max(ys), height))


def _error_message(response: httpx.Response) -> str:
    try:
        error = response.json().get("error", {})
        return error.get("innererror", {}).get("message") or error.get("message") or response.text[:500]
    except ValueError:
        return response.text[:500]
