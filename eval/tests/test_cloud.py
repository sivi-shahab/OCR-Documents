import json

import httpx
import pymupdf
import pytest
from app.core.config import Settings
from app.core.errors import EngineResponseError
from app.domain.models import OcrResult

from ocr_eval.engines.cloud import AzureEngine, MistralEngine, split_pdf


@pytest.fixture
def five_page_pdf(tmp_path):
    path = tmp_path / "five.pdf"
    doc = pymupdf.open()
    for n in range(5):
        doc.new_page().insert_text((50, 60), f"Page {n + 1}")
    doc.save(path)
    return path


class FakeAdapter:
    def __init__(self, fail=False):
        self.calls = []
        self._fail = fail

    async def recognize(self, content: bytes) -> list[OcrResult]:
        with pymupdf.open(stream=content) as doc:
            count = doc.page_count
        self.calls.append(count)
        if self._fail:
            raise EngineResponseError("boom")
        return [OcrResult(page=i + 1, markdown=f"md{i + 1}", blocks=[], raw="") for i in range(count)]


async def _collect(engine, pdf, pages):
    return [out async for out in engine.run(pdf, pages, is_scan_variant=False)]


def test_split_pdf_maps_pages(five_page_pdf):
    chunks = split_pdf(five_page_pdf, [1, 3, 4, 5], 2)
    assert [pages for _, pages in chunks] == [[1, 3], [4, 5]]
    with pymupdf.open(stream=chunks[0][0]) as doc:
        assert [p.get_text().strip() for p in doc] == ["Page 1", "Page 3"]


async def test_chunk_results_are_renumbered(five_page_pdf):
    engine = AzureEngine(Settings(azure_di_api_key="k", azure_di_endpoint="e"), chunk_pages=2, adapter=FakeAdapter())
    outs = await _collect(engine, five_page_pdf, [1, 3, 4, 5])
    assert [o.page for o in outs] == [1, 3, 4, 5]
    assert [o.markdown for o in outs] == ["md1", "md2", "md1", "md2"]
    assert all(o.error is None and o.seconds is not None for o in outs)


async def test_chunk_failure_marks_all_its_pages(five_page_pdf):
    engine = AzureEngine(Settings(azure_di_api_key="k", azure_di_endpoint="e"), chunk_pages=2, adapter=FakeAdapter(fail=True))
    outs = await _collect(engine, five_page_pdf, [1, 2, 3])
    assert [o.page for o in outs] == [1, 2, 3]
    assert all(o.error and o.markdown == "" for o in outs)


def test_missing_key_reports_unavailable():
    assert "OCR_MISTRAL_API_KEY" in MistralEngine(Settings(mistral_api_key="")).available()
    assert "OCR_AZURE_DI_API_KEY" in AzureEngine(Settings(azure_di_api_key="")).available()
    assert MistralEngine(Settings(mistral_api_key="k")).available() is None


async def test_mistral_records_model_from_response(five_page_pdf):
    def handle(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content)
        assert sent["document"]["type"] == "document_url"
        pages = [{"index": i, "markdown": f"m{i}"} for i in range(2)]
        return httpx.Response(200, json={"model": "mistral-ocr-2505", "pages": pages})

    engine = MistralEngine(Settings(mistral_api_key="k"), transport=httpx.MockTransport(handle))
    assert engine.meta_info()["model"] == "mistral-ocr-latest"
    outs = await _collect(engine, five_page_pdf, [2, 4])
    assert [(o.page, o.markdown) for o in outs] == [(2, "m0"), (4, "m1")]
    assert engine.meta_info() == {"model": "mistral-ocr-2505", "chunk_pages": 25}


async def test_mistral_http_error_marks_pages(five_page_pdf):
    transport = httpx.MockTransport(lambda r: httpx.Response(500, text="no"))
    engine = MistralEngine(Settings(mistral_api_key="k"), transport=transport)
    outs = await _collect(engine, five_page_pdf, [1, 2])
    assert all(o.error for o in outs)
