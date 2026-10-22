import base64
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

from app.core.errors import EngineResponseError, EngineUnavailableError
from app.domain.models import EngineOutput, PageImage

# The prompt must start with the <image> token, otherwise the model returns nothing.
SINGLE_IMAGE_PROMPT = "<image>document parsing."
MULTI_IMAGE_PROMPT = "<image>Multi page parsing."


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    model_name: str
    max_tokens: int
    ngram_size: int
    single_image_window: int
    multi_image_window: int


class VllmOcrEngine:
    """Calls Unlimited-OCR through vLLM's OpenAI-compatible chat completions API."""

    def __init__(self, client: httpx.AsyncClient, config: GenerationConfig) -> None:
        self._client = client
        self._config = config

    async def recognize(
        self, images: Sequence[PageImage], *, max_tokens: int | None = None, temperature: float = 0.0
    ) -> EngineOutput:
        if not images:
            raise ValueError("At least one image is required")
        payload = self._build_payload(images, max_tokens or self._config.max_tokens, temperature)
        response = await self._post("/chat/completions", payload)
        return self._extract_content(response)

    async def is_healthy(self) -> bool:
        try:
            response = await self._client.get("/models")
        except httpx.HTTPError:
            return False
        return response.is_success

    def _build_payload(self, images: Sequence[PageImage], max_tokens: int, temperature: float) -> dict:
        is_multi = len(images) > 1
        prompt = MULTI_IMAGE_PROMPT if is_multi else SINGLE_IMAGE_PROMPT
        window = self._config.multi_image_window if is_multi else self._config.single_image_window
        content = [{"type": "text", "text": prompt}]
        content += [{"type": "image_url", "image_url": {"url": _to_data_url(image)}} for image in images]
        return {
            "model": self._config.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            # Special tokens carry the layout markers; skipping them empties the output.
            "skip_special_tokens": False,
            "vllm_xargs": {"ngram_size": self._config.ngram_size, "window_size": window},
        }

    async def _post(self, path: str, payload: dict) -> dict:
        try:
            response = await self._client.post(path, json=payload)
        except httpx.TimeoutException as exc:
            raise EngineUnavailableError("OCR engine timed out") from exc
        except httpx.HTTPError as exc:
            raise EngineUnavailableError(f"OCR engine unreachable: {exc}") from exc
        if not response.is_success:
            raise EngineResponseError(f"OCR engine returned {response.status_code}: {response.text[:500]}")
        return response.json()

    @staticmethod
    def _extract_content(response: dict) -> EngineOutput:
        try:
            choice = response["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise EngineResponseError("Unexpected response shape from OCR engine") from exc
        return EngineOutput(text=text, truncated=choice.get("finish_reason") == "length")


def _to_data_url(image: PageImage) -> str:
    encoded = base64.b64encode(image.content).decode("ascii")
    return f"data:{image.mime_type};base64,{encoded}"
