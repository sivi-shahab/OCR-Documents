from pathlib import Path

import pymupdf

from app.core.errors import DocumentTooLargeError, InvalidDocumentError
from app.domain.models import PageImage

_PDF_SIGNATURE = b"%PDF-"
# The PDF spec lets the header appear anywhere in the first 1024 bytes.
_SIGNATURE_SEARCH_BYTES = 1024


class PyMuPdfRenderer:
    """Rasterises PDF pages to PNG with PyMuPDF."""

    def __init__(self, dpi: int, max_pages: int) -> None:
        self._dpi = dpi
        self._max_pages = max_pages

    def render(self, content: bytes) -> list[PageImage]:
        with _open_validated(content[:_SIGNATURE_SEARCH_BYTES], stream=content) as document:
            _check_page_count(document, self._max_pages)
            return [_render(page, self._dpi) for page in document]

    def count_pages(self, path: Path, max_pages: int) -> int:
        with path.open("rb") as file:
            head = file.read(_SIGNATURE_SEARCH_BYTES)
        with _open_validated(head, filename=path) as document:
            _check_page_count(document, max_pages)
            return document.page_count

    def open_pages(self, path: Path) -> "PyMuPdfPageSource":
        return PyMuPdfPageSource(pymupdf.open(path, filetype="pdf"), self._dpi)


class PyMuPdfPageSource:
    """Keeps one document open and renders requested pages lazily, so memory stays flat."""

    def __init__(self, document: pymupdf.Document, dpi: int) -> None:
        self._document = document
        self._dpi = dpi

    def render_page(self, number: int, dpi: int | None = None) -> PageImage:
        return _render(self._document[number - 1], dpi or self._dpi)

    def extract_text(self, number: int) -> str:
        return self._document[number - 1].get_text().strip()

    def close(self) -> None:
        self._document.close()


def _open_validated(head: bytes, **source) -> pymupdf.Document:
    # PyMuPDF silently opens images and other formats even with filetype="pdf".
    if _PDF_SIGNATURE not in head:
        raise InvalidDocumentError("File is not a PDF")
    try:
        document = pymupdf.open(filetype="pdf", **source)
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise InvalidDocumentError("File is not a readable PDF") from exc
    if document.needs_pass:
        document.close()
        raise InvalidDocumentError("Password-protected PDFs are not supported")
    if document.page_count == 0:
        document.close()
        raise InvalidDocumentError("PDF has no pages")
    return document


def _check_page_count(document: pymupdf.Document, max_pages: int) -> None:
    if document.page_count > max_pages:
        raise DocumentTooLargeError(f"PDF has {document.page_count} pages; the limit is {max_pages}")


def _render(page: pymupdf.Page, dpi: int) -> PageImage:
    return PageImage(content=page.get_pixmap(dpi=dpi).tobytes("png"), mime_type="image/png")
