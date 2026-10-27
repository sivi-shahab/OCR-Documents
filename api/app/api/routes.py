from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, File, Form, Path, Query, Response, UploadFile, status
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from app.api.dependencies import (
    AzureOcrDep,
    JobServiceDep,
    MistralOcrDep,
    OcrServiceDep,
    SettingsDep,
    UploadContent,
    save_upload,
    validate_callback_url,
)
from app.api.schemas import (
    ErrorResponse,
    HealthResponse,
    JobDocumentResponse,
    JobPageEvent,
    JobPageResponse,
    JobPagesResponse,
    JobResponse,
    OcrResponse,
)
from app.domain.jobs import Job
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

_JOB_ERROR_RESPONSES = {
    code: {"model": ErrorResponse} for code in (status.HTTP_404_NOT_FOUND, status.HTTP_409_CONFLICT)
}

IncludeRaw = Annotated[bool, Query(description="Include the unprocessed model output per page")]
JobId = Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")]
Clean = Annotated[
    bool,
    Query(
        description=(
            "Drop running headers, footers and page numbers, rejoin paragraphs and words split by "
            "page breaks, and merge tables continued on the next page"
        )
    ),
]


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


@router.post("/v1/ocr/azure", response_model=OcrResponse, responses=_ERROR_RESPONSES, tags=["ocr"])
async def ocr_azure(
    azure: AzureOcrDep,
    content: UploadContent,
    file: Annotated[UploadFile, File(description="PDF, image (JPEG, PNG, BMP, TIFF, HEIF) or Office document")],
    include_raw: IncludeRaw = False,
) -> OcrResponse:
    """Recognises the document with Azure AI Document Intelligence (prebuilt-layout) instead of
    Unlimited-OCR. Returns one entry per page; bbox uses the same 0-999 grid."""
    results = await azure.recognize(content)
    return OcrResponse.from_domain(file.filename, results, include_raw)


@router.post("/v1/ocr/mistral", response_model=OcrResponse, responses=_ERROR_RESPONSES, tags=["ocr"])
async def ocr_mistral(
    mistral: MistralOcrDep,
    content: UploadContent,
    file: Annotated[UploadFile, File(description="PDF or image (PNG/JPEG/WEBP; BMP/TIFF converted to PNG)")],
    include_raw: IncludeRaw = False,
) -> OcrResponse:
    """Recognises the document with Mistral OCR instead of Unlimited-OCR. Returns one entry per
    page; blocks list the image regions Mistral detects, with bbox on the same 0-999 grid."""
    results = await mistral.recognize(content)
    return OcrResponse.from_domain(file.filename, results, include_raw)


@router.post(
    "/v1/ocr/jobs",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobResponse,
    responses={
        status.HTTP_200_OK: {"model": JobResponse, "description": "An identical PDF already has a job"},
        **{code: {"model": ErrorResponse} for code in (413, 422)},
    },
    tags=["jobs"],
)
async def create_job(
    service: JobServiceDep,
    settings: SettingsDep,
    response: Response,
    file: Annotated[UploadFile, File(description="PDF document, processed in the background")],
    callback_url: Annotated[
        str | None,
        Form(description="Receives a signed POST when the job finishes; the host must be allow-listed"),
    ] = None,
    reuse: Annotated[
        bool,
        Query(
            description="Return the existing job (200) when an identical PDF is queued, running or completed"
        ),
    ] = True,
) -> JobResponse:
    """Queues a PDF of any size up to the job limits. Poll the job or follow its events, then
    fetch pages or the result."""
    if callback_url:
        validate_callback_url(callback_url, settings)
    staged = service.new_upload_path()
    sha256 = await save_upload(file, staged, settings.job_max_upload_bytes)
    job, created = await service.submit(
        staged, file.filename, sha256=sha256 if reuse else None, callback_url=callback_url
    )
    if not created:
        response.status_code = status.HTTP_200_OK
    return JobResponse.from_domain(job)


@router.get("/v1/ocr/jobs", response_model=list[JobResponse], tags=["jobs"])
async def list_jobs(service: JobServiceDep) -> list[JobResponse]:
    return [JobResponse.from_domain(job) for job in service.list_jobs()]


@router.get(
    "/v1/ocr/jobs/{job_id}", response_model=JobResponse, responses=_JOB_ERROR_RESPONSES, tags=["jobs"]
)
async def get_job(service: JobServiceDep, job_id: JobId) -> JobResponse:
    return JobResponse.from_domain(service.get(job_id))


@router.get(
    "/v1/ocr/jobs/{job_id}/pages",
    response_model=JobPagesResponse,
    responses=_JOB_ERROR_RESPONSES,
    tags=["jobs"],
)
async def get_job_pages(
    service: JobServiceDep,
    job_id: JobId,
    first: Annotated[int, Query(ge=1, description="First page number (1-based)")] = 1,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    include_raw: IncludeRaw = False,
) -> JobPagesResponse:
    """Returns finished pages; available while the job is still running."""
    outcomes = await service.pages(job_id, first, first + limit - 1)
    job = service.get(job_id)
    return JobPagesResponse(
        job_id=job.id,
        status=job.status,
        page_count=job.page_count,
        pages=[JobPageResponse.from_domain(outcome, include_raw) for outcome in outcomes],
    )


@router.get(
    "/v1/ocr/jobs/{job_id}/result.md",
    response_class=PlainTextResponse,
    responses={200: {"content": {"text/markdown": {}}}, **_JOB_ERROR_RESPONSES},
    tags=["jobs"],
)
async def get_job_markdown(
    service: JobServiceDep,
    job_id: JobId,
    clean: Clean = True,
) -> PlainTextResponse:
    """The whole document in page order, each page preceded by a `<!-- page N -->` marker."""
    return PlainTextResponse(await service.markdown(job_id, clean), media_type="text/markdown")


@router.get(
    "/v1/ocr/jobs/{job_id}/result.json",
    response_model=JobDocumentResponse,
    responses=_JOB_ERROR_RESPONSES,
    tags=["jobs"],
)
async def get_job_document(service: JobServiceDep, job_id: JobId, clean: Clean = True) -> JobDocumentResponse:
    """The same markdown as result.md, plus the character span of every page so text can be
    traced back to the page it came from."""
    document = await service.document(job_id, clean)
    return JobDocumentResponse.from_domain(service.get(job_id), document)


@router.get(
    "/v1/ocr/jobs/{job_id}/events",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}, 404: {"model": ErrorResponse}},
    tags=["jobs"],
)
async def stream_job_events(service: JobServiceDep, job_id: JobId) -> StreamingResponse:
    """Server-Sent Events: `job` (a JobResponse) on connect and on every status change, `page`
    (a JobPageEvent) as each page finishes. The stream ends when the job finishes."""
    events = service.events(job_id)
    first = await anext(events)  # raises 404 before the stream starts

    async def stream() -> AsyncIterator[str]:
        try:
            yield _sse(first)
            async for event in events:
                yield _sse(event)
        finally:
            await events.aclose()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event) -> str:
    if event is None:
        return ": keep-alive\n\n"
    if isinstance(event, Job):
        return f"event: job\ndata: {JobResponse.from_domain(event).model_dump_json()}\n\n"
    return f"event: page\ndata: {JobPageEvent.from_domain(event).model_dump_json()}\n\n"


@router.post(
    "/v1/ocr/jobs/{job_id}/cancel", response_model=JobResponse, responses=_JOB_ERROR_RESPONSES, tags=["jobs"]
)
async def cancel_job(service: JobServiceDep, job_id: JobId) -> JobResponse:
    """Stops a queued or running job. Pages finished so far stay readable."""
    return JobResponse.from_domain(await service.cancel(job_id))


@router.delete(
    "/v1/ocr/jobs/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}},
    tags=["jobs"],
)
async def delete_job(service: JobServiceDep, job_id: JobId) -> Response:
    """Cancels the job if needed and deletes its PDF and results."""
    await service.delete(job_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
