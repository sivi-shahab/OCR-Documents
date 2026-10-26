import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_azure_ocr
from app.core.config import Settings, get_settings
from app.core.errors import EngineResponseError, InvalidDocumentError
from app.infrastructure.azure_document_intelligence import (
    AzureConfig,
    AzureDocumentIntelligence,
    to_results,
)
from app.main import create_app
from tests.fakes import make_pdf

OPERATION_URL = "https://di.example.com/documentintelligence/documentModels/prebuilt-layout/analyzeResults/1"

ANALYZE_RESULT = {
    "content": "# Laporan Keuangan\n\nBaris pertama\n\nHalaman dua",
    "pages": [
        {"pageNumber": 1, "width": 8.5, "height": 11, "spans": [{"offset": 0, "length": 35}]},
        {"pageNumber": 2, "width": 8.5, "height": 11, "spans": [{"offset": 35, "length": 12}]},
    ],
    "paragraphs": [
        {
            "role": "title",
            "content": "Laporan Keuangan",
            "boundingRegions": [{"pageNumber": 1, "polygon": [1, 1, 4.25, 1, 4.25, 2.2, 1, 2.2]}],
        },
        {"content": "Baris pertama", "boundingRegions": [{"pageNumber": 1, "polygon": []}]},
        {"content": "Halaman dua", "boundingRegions": [{"pageNumber": 2}]},
    ],
}


def make_azure(handler) -> AzureDocumentIntelligence:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return AzureDocumentIntelligence(
        client, AzureConfig(endpoint="https://di.example.com/", api_key="secret", poll_interval_seconds=0)
    )


def succeeding_handler(requests: list[httpx.Request], polls_before_success: int = 1):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(202, headers={"Operation-Location": OPERATION_URL})
        if len(requests) <= polls_before_success:
            return httpx.Response(200, json={"status": "running"})
        return httpx.Response(200, json={"status": "succeeded", "analyzeResult": ANALYZE_RESULT})

    return handler


def test_to_results_splits_pages_and_normalises_bbox():
    results = to_results(ANALYZE_RESULT)
    assert [result.page for result in results] == [1, 2]
    assert results[0].markdown == "# Laporan Keuangan\n\nBaris pertama"
    assert results[1].markdown == "Halaman dua"
    title, text = results[0].blocks
    assert (title.category, title.bbox) == ("title", (118, 91, 500, 200))
    assert (text.category, text.bbox) == ("text", None)


async def test_recognize_submits_then_polls_until_succeeded():
    requests: list[httpx.Request] = []
    azure = make_azure(succeeding_handler(requests, polls_before_success=2))

    results = await azure.recognize(b"%PDF")

    submit = requests[0]
    assert submit.url.path == "/documentintelligence/documentModels/prebuilt-layout:analyze"
    assert submit.url.params["outputContentFormat"] == "markdown"
    assert submit.headers["Ocp-Apim-Subscription-Key"] == "secret"
    assert submit.content == b"%PDF"
    assert all(request.headers["Ocp-Apim-Subscription-Key"] == "secret" for request in requests)
    assert len(requests) == 3
    assert len(results) == 2


async def test_rejected_document_raises_invalid_document():
    azure = make_azure(
        lambda _: httpx.Response(400, json={"error": {"message": "bad", "innererror": {"message": "corrupt"}}})
    )
    with pytest.raises(InvalidDocumentError, match="corrupt"):
        await azure.recognize(b"junk")


async def test_failed_analysis_raises_engine_error():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, headers={"Operation-Location": OPERATION_URL})
        return httpx.Response(200, json={"status": "failed", "error": {"message": "boom"}})

    with pytest.raises(EngineResponseError, match="boom"):
        await make_azure(handler).recognize(b"%PDF")


def test_endpoint_returns_pages():
    settings = Settings(max_upload_mb=1)
    app = create_app(settings)
    app.dependency_overrides[get_azure_ocr] = lambda: make_azure(succeeding_handler([]))
    app.dependency_overrides[get_settings] = lambda: settings

    response = TestClient(app).post("/v1/ocr/azure", files={"file": ("doc.pdf", make_pdf(2))})

    assert response.status_code == 200
    body = response.json()
    assert body["page_count"] == 2
    assert body["pages"][0]["blocks"][0]["category"] == "title"


def test_endpoint_is_unavailable_without_credentials():
    settings = Settings(azure_di_endpoint="", azure_di_api_key="")
    with TestClient(create_app(settings)) as client:
        response = client.post("/v1/ocr/azure", files={"file": ("doc.pdf", make_pdf(1))})
    assert response.status_code == 503
    assert "OCR_AZURE_DI_API_KEY" in response.json()["detail"]
