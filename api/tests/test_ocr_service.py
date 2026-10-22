import pytest

from app.core.errors import DocumentTooLargeError, InvalidDocumentError
from app.domain.models import PdfStrategy
from app.infrastructure.image_loader import load_image
from app.infrastructure.pdf_renderer import PyMuPdfRenderer
from app.services.ocr_service import OcrService
from tests.fakes import FakeEngine, make_pdf, make_png


def make_service(engine: FakeEngine, max_pages: int = 10) -> OcrService:
    return OcrService(
        engine=engine,
        pdf_renderer=PyMuPdfRenderer(dpi=72, max_pages=max_pages),
        image_loader=load_image,
        page_concurrency=2,
    )


async def test_recognize_image_returns_parsed_page():
    engine = FakeEngine()
    result = await make_service(engine).recognize_image(make_png())
    assert result.page == 1
    assert result.markdown.startswith("Laporan Keuangan")
    assert engine.calls == [1]


async def test_bmp_is_converted_to_png():
    assert load_image(make_png("BMP")).mime_type == "image/png"


async def test_invalid_image_is_rejected():
    with pytest.raises(InvalidDocumentError):
        await make_service(FakeEngine()).recognize_image(b"not an image")


async def test_pdf_per_page_sends_one_request_per_page_in_order():
    engine = FakeEngine()
    results = await make_service(engine).recognize_pdf(make_pdf(3), PdfStrategy.PER_PAGE)
    assert [result.page for result in results] == [1, 2, 3]
    assert engine.calls == [1, 1, 1]


async def test_pdf_whole_document_sends_all_pages_in_one_request():
    engine = FakeEngine()
    results = await make_service(engine).recognize_pdf(make_pdf(3), PdfStrategy.WHOLE_DOCUMENT)
    assert len(results) == 1 and results[0].page is None
    assert engine.calls == [3]


async def test_pdf_over_page_limit_is_rejected():
    with pytest.raises(DocumentTooLargeError):
        await make_service(FakeEngine(), max_pages=2).recognize_pdf(make_pdf(3), PdfStrategy.PER_PAGE)


async def test_corrupt_pdf_is_rejected():
    with pytest.raises(InvalidDocumentError):
        await make_service(FakeEngine()).recognize_pdf(b"%PDF-garbage", PdfStrategy.PER_PAGE)


async def test_image_uploaded_as_pdf_is_rejected():
    with pytest.raises(InvalidDocumentError):
        await make_service(FakeEngine()).recognize_pdf(make_png(), PdfStrategy.PER_PAGE)
