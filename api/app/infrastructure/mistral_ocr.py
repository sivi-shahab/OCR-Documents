import base64
from dataclasses import dataclass

import httpx

from app.core.errors import EngineResponseError, EngineUnavailableError, InvalidDocumentError
from app.domain.models import Block, OcrResult
from app.infrastructure.image_loader import load_image


@dataclass(frozen=True, slots=True)
class MistralConfig:
    api_key: str
    base_url: str = "https://api.mistral.ai/v1"
    model: str = "mistral-ocr-latest"


class MistralOcr:
    """Recognises images and PDFs with Mistral's OCR API (POST /v1/ocr)."""

    def __init__(self, client: httpx.AsyncClient, config: MistralConfig) -> None:
        self._client = client
        self._config = config

    async def recognize(self, content: bytes) -> list[OcrResult]:
        config = self._config
        payload = {"model": config.model, "document": _document(content), "include_image_base64": False}
        try:
            response = await self._client.post(
                f"{config.base_url.rstrip('/')}/ocr",
                json=payload,
                headers={"Authorization": f"Bearer {config.api_key}"},
            )
        except httpx.TimeoutException as exc:
            raise EngineUnavailableError("Mistral OCR timed out") from exc
        except httpx.HTTPError as exc:
            raise EngineUnavailableError(f"Mistral OCR unreachable: {exc}") from exc
        if response.status_code in (400, 422):
            raise InvalidDocumentError(f"Mistral rejected the document: {_error_message(response)}")
        if not response.is_success:
            raise EngineResponseError(
                f"Mistral OCR returned {response.status_code}: {_error_message(response)}"
            )
        try:
            return to_results(response.json())
        except (ValueError, KeyError, TypeError) as exc:
            raise EngineResponseError("Mistral OCR returned an unexpected payload") from exc


def _document(content: bytes) -> dict:
    """Builds the data-URL document chunk; PDFs pass through, images are validated like /v1/ocr/image."""
    if content.startswith(b"%PDF"):
        data_url = "data:application/pdf;base64," + base64.b64encode(content).decode()
        return {"type": "document_url", "document_url": data_url}
    image = load_image(content)
    data_url = f"data:{image.mime_type};base64," + base64.b64encode(image.content).decode()
    return {"type": "image_url", "image_url": data_url}


def to_results(body: dict) -> list[OcrResult]:
    """Maps Mistral's pages (0-based index, markdown, image regions in pixels) to one result per page."""
    results = []
    for page in body["pages"]:
        dimensions = page.get("dimensions") or {}
        width, height = dimensions.get("width"), dimensions.get("height")
        blocks = [
            Block(
                category="image",
                bbox=_normalise(image, width, height),
                text=image.get("id", ""),
            )
            for image in page.get("images") or []
        ]
        markdown = page.get("markdown", "")
        results.append(
            OcrResult(page=page["index"] + 1, markdown=markdown.strip(), blocks=blocks, raw=markdown)
        )
    return results


def _normalise(image: dict, width: float | None, height: float | None) -> tuple[int, ...] | None:
    """Converts Mistral's pixel corners to [x1, y1, x2, y2] on a 0-999 grid."""
    corners = [image.get(key) for key in ("top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y")]
    if not width or not height or any(value is None for value in corners):
        return None

    def scale(value: float, size: float) -> int:
        return max(0, min(999, round(value / size * 999)))

    x1, y1, x2, y2 = corners
    return (scale(x1, width), scale(y1, height), scale(x2, width), scale(y2, height))


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:500]
    message = body.get("message") or body.get("detail") or response.text[:500]
    return message if isinstance(message, str) else str(message)[:500]
