import json

import httpx
import pytest

from app.core.errors import EngineResponseError, EngineUnavailableError
from app.domain.models import PageImage
from app.infrastructure.vllm_engine import GenerationConfig, VllmOcrEngine

CONFIG = GenerationConfig(
    model_name="Unlimited-OCR",
    max_tokens=100,
    ngram_size=35,
    single_image_window=128,
    multi_image_window=1024,
)
IMAGE = PageImage(content=b"img", mime_type="image/png")


def make_engine(handler) -> VllmOcrEngine:
    client = httpx.AsyncClient(base_url="http://vllm/v1", transport=httpx.MockTransport(handler))
    return VllmOcrEngine(client, CONFIG)


def completion(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


async def test_single_image_payload_uses_recipe_parameters():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return completion("ok")

    assert await make_engine(handler).recognize([IMAGE]) == "ok"
    assert captured["skip_special_tokens"] is False
    assert captured["vllm_xargs"] == {"ngram_size": 35, "window_size": 128}
    text, image = captured["messages"][0]["content"]
    assert text["text"].startswith("<image>")
    assert image["image_url"]["url"] == "data:image/png;base64,aW1n"


async def test_multi_image_payload_uses_multi_page_window():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return completion("ok")

    await make_engine(handler).recognize([IMAGE, IMAGE])
    assert captured["vllm_xargs"]["window_size"] == 1024
    assert len(captured["messages"][0]["content"]) == 3


async def test_upstream_error_status_raises_response_error():
    with pytest.raises(EngineResponseError):
        await make_engine(lambda _: httpx.Response(500, text="boom")).recognize([IMAGE])


async def test_unexpected_payload_raises_response_error():
    with pytest.raises(EngineResponseError):
        await make_engine(lambda _: httpx.Response(200, json={})).recognize([IMAGE])


async def test_connection_failure_raises_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(EngineUnavailableError):
        await make_engine(handler).recognize([IMAGE])
    assert await make_engine(handler).is_healthy() is False
