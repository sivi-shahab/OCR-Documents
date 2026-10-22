import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import create_app
from tests.fakes import FakeEngine, make_pdf


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = Settings(job_dir=tmp_path, job_max_upload_mb=1, job_max_pdf_pages=400, pdf_dpi=72)
    monkeypatch.setattr("app.main.build_engine", lambda *_: FakeEngine())
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as client:
        yield client


def wait_finished(client: TestClient, job_id: str) -> dict:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        job = client.get(f"/v1/ocr/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise TimeoutError(job_id)


def test_large_pdf_job_lifecycle(client: TestClient):
    response = client.post("/v1/ocr/jobs", files={"file": ("book.pdf", make_pdf(300), "application/pdf")})
    assert response.status_code == 202
    job_id = response.json()["id"]
    assert response.json()["page_count"] == 300

    job = wait_finished(client, job_id)
    assert job["status"] == "completed"
    assert job["pages_done"] == job["pages_succeeded"] == 300
    assert job["needs_review"] is False

    pages = client.get(f"/v1/ocr/jobs/{job_id}/pages?first=101&limit=3&include_raw=true").json()
    assert [page["page"] for page in pages["pages"]] == [101, 102, 103]
    assert pages["pages"][0]["blocks"][0]["category"] == "title"
    assert pages["pages"][0]["raw"].startswith("<|det|>")

    result = client.get(f"/v1/ocr/jobs/{job_id}/result.md")
    assert result.headers["content-type"].startswith("text/markdown")
    assert result.text.count("<!-- page ") == 300

    document = client.get(f"/v1/ocr/jobs/{job_id}/result.json").json()
    assert document["markdown"] == result.text
    assert len(document["pages"]) == 300
    span = document["pages"][100]
    assert span["page"] == 101 and span["status"] == "succeeded"
    assert document["markdown"][span["start"] : span["end"]].startswith("<!-- page 101 -->")

    assert [listed["id"] for listed in client.get("/v1/ocr/jobs").json()] == [job_id]
    assert client.delete(f"/v1/ocr/jobs/{job_id}").status_code == 204
    assert client.get(f"/v1/ocr/jobs/{job_id}").status_code == 404


def test_finished_job_cannot_be_cancelled(client: TestClient):
    job_id = client.post("/v1/ocr/jobs", files={"file": ("doc.pdf", make_pdf(1))}).json()["id"]
    wait_finished(client, job_id)
    assert client.post(f"/v1/ocr/jobs/{job_id}/cancel").status_code == 409


@pytest.mark.parametrize(
    ("content", "expected_status"),
    [(b"not a pdf", 422), (b"", 422), (make_pdf(401), 413), (b"%PDF" + b"x" * 1024 * 1024, 413)],
)
def test_invalid_job_uploads_are_rejected(
    client: TestClient, tmp_path: Path, content: bytes, expected_status: int
):
    response = client.post("/v1/ocr/jobs", files={"file": ("doc.pdf", content)})
    assert response.status_code == expected_status
    assert list((tmp_path / ".staging").iterdir()) == []  # rejected uploads are cleaned up


@pytest.mark.parametrize(("job_id", "expected_status"), [("0" * 32, 404), ("not-a-job-id", 422)])
def test_unknown_or_malformed_job_id(client: TestClient, job_id: str, expected_status: int):
    assert client.get(f"/v1/ocr/jobs/{job_id}").status_code == expected_status
