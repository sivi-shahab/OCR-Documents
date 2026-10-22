from typing import Annotated

from fastapi import APIRouter, File, Query, UploadFile, status
from fastapi.responses import JSONResponse

from app.api.dependencies import OcrServiceDep, UploadContent
from app.api.schemas import ErrorResponse, HealthResponse, OcrResponse
from app.domain.models import PdfStrategy

router = APIRouter()

_ERROR_RESPONSES = {
    code: {"model": ErrorResponse}
    for code in (
        status.HTTP_413_CONTENT_TOO_LARGE,
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        status.HTTP_502_BAD_GATEWAY,
        status.HTTP_503_SERVICE_UNAVAILABLE,
    )
}

IncludeRaw = Annotated[bool, Query(description="Include the unprocessed model output per page")]


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health(service: OcrServiceDep) -> JSONResponse:
    healthy = await service.is_engine_healthy()
    body = HealthResponse(status="ok" if healthy else "degraded", engine="up" if healthy else "down")
    code = status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(body.model_dump(), status_code=code)


@router.post("/v1/ocr/image", response_model=OcrResponse, responses=_ERROR_RESPONSES, tags=["ocr"])
async def ocr_image(
    service: OcrServiceDep,
    content: UploadContent,
    file: Annotated[UploadFile, File(description="PNG, JPEG, WEBP, BMP or TIFF image")],
    include_raw: IncludeRaw = False,
) -> OcrResponse:
    result = await service.recognize_image(content)
    return OcrResponse.from_domain(file.filename, [result], include_raw)


@router.post("/v1/ocr/pdf", response_model=OcrResponse, responses=_ERROR_RESPONSES, tags=["ocr"])
async def ocr_pdf(
    service: OcrServiceDep,
    content: UploadContent,
    file: Annotated[UploadFile, File(description="PDF document")],
    strategy: Annotated[
        PdfStrategy,
        Query(
            description=(
                "per_page: each page is a separate request, run concurrently (fast, per-page results). "
                "whole_document: all pages in one long-horizon request (keeps cross-page context; "
                "bounded by the model's 32k context)."
            )
        ),
    ] = PdfStrategy.PER_PAGE,
    include_raw: IncludeRaw = False,
) -> OcrResponse:
    results = await service.recognize_pdf(content, strategy)
    return OcrResponse.from_domain(file.filename, results, include_raw)
