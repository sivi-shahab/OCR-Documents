import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_ocr_service
from app.core.config import Settings, get_settings
from app.infrastructure.image_loader import load_image
from app.infrastructure.pdf_renderer import PyMuPdfRenderer
from app.main import create_app
from app.services.ocr_service import OcrService
from tests.fakes import FakeEngine, make_pdf, make_png


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def client(engine: FakeEngine) -> TestClient:
    settings = Settings(max_upload_mb=1)
    service = OcrService(engine, PyMuPdfRenderer(dpi=72, max_pages=5), load_image, page_concurrency=2)
    app = create_app(settings)
    app.dependency_overrides[get_ocr_service] = lambda: service
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def test_ocr_image_returns_markdown_and_blocks(client: TestClient):
    response = client.post("/v1/ocr/image", files={"file": ("scan.png", make_png(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "scan.png"
    assert body["page_count"] == 1
    assert body["pages"][0]["blocks"][0] == {
        "category": "title",
        "bbox": [117, 74, 581, 101],
        "text": "Laporan Keuangan",
    }
    assert body["pages"][0]["raw"] is None


def test_include_raw_returns_model_output(client: TestClient):
    response = client.post(
        "/v1/ocr/image?include_raw=true", files={"file": ("scan.png", make_png(), "image/png")}
    )
    assert response.json()["pages"][0]["raw"].startswith("<|det|>")


def test_ocr_pdf_per_page(client: TestClient):
    response = client.post("/v1/ocr/pdf", files={"file": ("doc.pdf", make_pdf(2), "application/pdf")})
    body = response.json()
    assert response.status_code == 200
    assert [page["page"] for page in body["pages"]] == [1, 2]


def test_ocr_pdf_whole_document(client: TestClient, engine: FakeEngine):
    response = client.post(
        "/v1/ocr/pdf?strategy=whole_document",
        files={"file": ("doc.pdf", make_pdf(2), "application/pdf")},
    )
    assert response.status_code == 200
    assert engine.calls == [2]


@pytest.mark.parametrize(
    ("path", "content", "expected_status"),
    [
        ("/v1/ocr/image", b"", 422),
        ("/v1/ocr/image", b"not an image", 422),
        ("/v1/ocr/pdf", b"not a pdf", 422),
        ("/v1/ocr/image", b"x" * (1024 * 1024 + 1), 413),
    ],
)
def test_invalid_uploads_are_rejected(client: TestClient, path: str, content: bytes, expected_status: int):
    response = client.post(path, files={"file": ("f", content)})
    assert response.status_code == expected_status
    assert "detail" in response.json()


def test_pdf_over_page_limit_returns_413(client: TestClient):
    response = client.post("/v1/ocr/pdf", files={"file": ("doc.pdf", make_pdf(6), "application/pdf")})
    assert response.status_code == 413


@pytest.mark.parametrize(("healthy", "expected_status"), [(True, 200), (False, 503)])
def test_health_reflects_engine_state(
    engine: FakeEngine, client: TestClient, healthy: bool, expected_status: int
):
    engine.healthy = healthy
    assert client.get("/health").status_code == expected_status
