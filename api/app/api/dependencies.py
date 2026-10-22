from typing import Annotated

from fastapi import Depends, Request, UploadFile

from app.core.config import Settings, get_settings
from app.core.errors import DocumentTooLargeError, InvalidDocumentError
from app.services.ocr_service import OcrService

_READ_CHUNK_BYTES = 1024 * 1024


def get_ocr_service(request: Request) -> OcrService:
    return request.app.state.ocr_service


async def read_upload(file: UploadFile, settings: Annotated[Settings, Depends(get_settings)]) -> bytes:
    """Reads the upload in chunks so oversized files are rejected without buffering them fully."""
    limit = settings.max_upload_bytes
    chunks, size = [], 0
    while chunk := await file.read(_READ_CHUNK_BYTES):
        size += len(chunk)
        if size > limit:
            raise DocumentTooLargeError(f"File exceeds the {settings.max_upload_mb} MB limit")
        chunks.append(chunk)
    if size == 0:
        raise InvalidDocumentError("Uploaded file is empty")
    return b"".join(chunks)


OcrServiceDep = Annotated[OcrService, Depends(get_ocr_service)]
UploadContent = Annotated[bytes, Depends(read_upload)]
