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
        # PyMuPDF silently opens images and other formats even with filetype="pdf".
        if _PDF_SIGNATURE not in content[:_SIGNATURE_SEARCH_BYTES]:
            raise InvalidDocumentError("File is not a PDF")
        try:
            document = pymupdf.open(stream=content, filetype="pdf")
        except (pymupdf.FileDataError, RuntimeError) as exc:
            raise InvalidDocumentError("File is not a readable PDF") from exc

        with document:
            if document.needs_pass:
                raise InvalidDocumentError("Password-protected PDFs are not supported")
            if document.page_count == 0:
                raise InvalidDocumentError("PDF has no pages")
            if document.page_count > self._max_pages:
                raise DocumentTooLargeError(
                    f"PDF has {document.page_count} pages; the limit is {self._max_pages}"
                )
            return [
                PageImage(content=page.get_pixmap(dpi=self._dpi).tobytes("png"), mime_type="image/png")
                for page in document
            ]
