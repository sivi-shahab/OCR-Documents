import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.errors import (
    DocumentTooLargeError,
    EngineResponseError,
    EngineUnavailableError,
    InvalidDocumentError,
    OcrError,
)

logger = logging.getLogger(__name__)

_STATUS_BY_ERROR: dict[type[OcrError], int] = {
    InvalidDocumentError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    DocumentTooLargeError: status.HTTP_413_CONTENT_TOO_LARGE,
    EngineUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
    EngineResponseError: status.HTTP_502_BAD_GATEWAY,
}


async def _handle_ocr_error(_: Request, exc: Exception) -> JSONResponse:
    code = next(
        (code for error_type, code in _STATUS_BY_ERROR.items() if isinstance(exc, error_type)),
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
    if code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
        logger.error("OCR request failed: %s", exc)
    return JSONResponse({"detail": str(exc)}, status_code=code)


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(OcrError, _handle_ocr_error)
