import asyncio
import hashlib
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import Depends, Request, UploadFile

from app.core.config import Settings, get_settings
from app.core.errors import (
    DocumentTooLargeError,
    EngineUnavailableError,
    InvalidDocumentError,
    InvalidRequestError,
)
from app.infrastructure.azure_document_intelligence import AzureDocumentIntelligence
from app.services.job_service import JobService
from app.services.ocr_service import OcrService

_READ_CHUNK_BYTES = 1024 * 1024


def get_ocr_service(request: Request) -> OcrService:
    return request.app.state.ocr_service


def get_job_service(request: Request) -> JobService:
    return request.app.state.job_service


def get_azure_ocr(request: Request) -> AzureDocumentIntelligence:
    azure = getattr(request.app.state, "azure_ocr", None)
    if azure is None:
        raise EngineUnavailableError(
            "Azure Document Intelligence is not configured: set OCR_AZURE_DI_ENDPOINT and OCR_AZURE_DI_API_KEY"
        )
    return azure


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


async def save_upload(file: UploadFile, destination: Path, limit_bytes: int) -> str:
    """Streams the upload to disk so large PDFs never sit fully in memory. Returns its SHA-256."""
    size = 0
    digest = hashlib.sha256()
    try:
        with destination.open("wb") as output:
            while chunk := await file.read(_READ_CHUNK_BYTES):
                size += len(chunk)
                if size > limit_bytes:
                    raise DocumentTooLargeError(f"File exceeds the {limit_bytes // (1024 * 1024)} MB limit")
                digest.update(chunk)
                await asyncio.to_thread(output.write, chunk)
        if size == 0:
            raise InvalidDocumentError("Uploaded file is empty")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return digest.hexdigest()


def validate_callback_url(url: str, settings: Settings) -> str:
    """Only lets jobs call back to configured hosts, so the API cannot be used to probe the network."""
    if not settings.webhooks_enabled:
        raise InvalidRequestError("Webhooks are disabled on this server")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise InvalidRequestError("callback_url must be an absolute http(s) URL")
    if parts.hostname.lower() not in settings.webhook_hosts:
        raise InvalidRequestError(f"callback_url host {parts.hostname} is not allowed")
    return url


OcrServiceDep = Annotated[OcrService, Depends(get_ocr_service)]
JobServiceDep = Annotated[JobService, Depends(get_job_service)]
AzureOcrDep = Annotated[AzureDocumentIntelligence, Depends(get_azure_ocr)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
UploadContent = Annotated[bytes, Depends(read_upload)]
