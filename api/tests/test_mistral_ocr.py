import base64
import io
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.dependencies import get_mistral_ocr
from app.core.config import Settings, get_settings
from app.core.errors import EngineResponseError, InvalidDocumentError
from app.infrastructure.mistral_ocr import MistralConfig, MistralOcr, to_results
from app.main import create_app
from tests.fakes import make_pdf

OCR_RESPONSE = {
    "model": "mistral-ocr-latest",
    "pages": [
        {
            "index": 0,
            "markdown": "# Laporan Keuangan\n\n![img-0.jpeg](img-0.jpeg)\n",
            "images": [
                {
                    "id": "img-0.jpeg",
                    "top_left_x": 170,
                    "top_left_y": 220,
                    "bottom_right_x": 850,
                    "bottom_right_y": 440,
                }
            ],
            "dimensions": {"dpi": 200, "width": 1700, "height": 2200},
        },
        {
            "index": 1,
            "markdown": "Halaman dua",
            "images": [],
            "dimensions": {"dpi": 200, "width": 1700, "height": 2200},
        },
    ],
}


def make_mistral(handler) -> MistralOcr:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return MistralOcr(client, MistralConfig(api_key="secret", base_url="https://mistral.example.com/v1/"))


def png_bytes(image_format: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format=image_format)
    return buffer.getvalue()


def test_to_results_maps_pages_and_normalises_bbox():
    results = to_results(OCR_RESPONSE)
    assert [result.page for result in results] == [1, 2]
    assert results[0].markdown == "# Laporan Keuangan\n\n![img-0.jpeg](img-0.jpeg)"
    (image,) = results[0].blocks
    assert (image.category, image.bbox, image.text) == ("image", (100, 100, 500, 200), "img-0.jpeg")
    assert results[1].blocks == []


async def test_recognize_sends_pdf_as_document_url():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=OCR_RESPONSE)

    results = await make_mistral(handler).recognize(b"%PDF-1.7 data")

    (request,) = requests
    assert str(request.url) == "https://mistral.example.com/v1/ocr"
    assert request.headers["Authorization"] == "Bearer secret"
    body = json.loads(request.content)
    assert body["model"] == "mistral-ocr-latest"
    assert body["document"] == {
        "type": "document_url",
        "document_url": "data:application/pdf;base64," + base64.b64encode(b"%PDF-1.7 data").decode(),
    }
    assert len(results) == 2


async def test_recognize_converts_bmp_to_png_image_url():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"pages": [{"index": 0, "markdown": "x"}]})

    await make_mistral(handler).recognize(png_bytes("BMP"))

    document = json.loads(requests[0].content)["document"]
    assert document["type"] == "image_url"
    assert document["image_url"].startswith("data:image/png;base64,")


async def test_unreadable_file_is_rejected_before_calling_mistral():
    with pytest.raises(InvalidDocumentError):
        await make_mistral(lambda _: pytest.fail("Mistral should not be called")).recognize(b"junk")


async def test_rejected_document_raises_invalid_document():
    mistral = make_mistral(lambda _: httpx.Response(400, json={"object": "error", "message": "corrupt pdf"}))
    with pytest.raises(InvalidDocumentError, match="corrupt pdf"):
        await mistral.recognize(b"%PDF")


async def test_server_error_raises_engine_error():
    mistral = make_mistral(lambda _: httpx.Response(401, json={"detail": "Unauthorized"}))
    with pytest.raises(EngineResponseError, match="401: Unauthorized"):
        await mistral.recognize(b"%PDF")


def test_endpoint_returns_pages():
    settings = Settings(max_upload_mb=1)
    app = create_app(settings)
    mistral = make_mistral(lambda _: httpx.Response(200, json=OCR_RESPONSE))
    app.dependency_overrides[get_mistral_ocr] = lambda: mistral
    app.dependency_overrides[get_settings] = lambda: settings

    response = TestClient(app).post("/v1/ocr/mistral", files={"file": ("doc.pdf", make_pdf(2))})

    assert response.status_code == 200
    body = response.json()
    assert body["page_count"] == 2
    assert body["pages"][0]["blocks"][0]["category"] == "image"


def test_endpoint_is_unavailable_without_api_key():
    settings = Settings(mistral_api_key="")
    with TestClient(create_app(settings)) as client:
        response = client.post("/v1/ocr/mistral", files={"file": ("doc.pdf", make_pdf(1))})
    assert response.status_code == 503
    assert "OCR_MISTRAL_API_KEY" in response.json()["detail"]
