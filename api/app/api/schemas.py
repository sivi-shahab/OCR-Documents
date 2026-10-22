from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.assembly import AssembledDocument, PageSpan
from app.domain.jobs import Job, JobStatus, PageFlag, PageOutcome, PageStatus
from app.domain.models import Block, OcrResult


class BlockResponse(BaseModel):
    category: str = Field(examples=["title"])
    bbox: list[int] | None = Field(
        description="[x1, y1, x2, y2] normalised to a 0-999 grid of the page", examples=[[117, 74, 581, 101]]
    )
    text: str

    @classmethod
    def from_domain(cls, block: Block) -> "BlockResponse":
        return cls(category=block.category, bbox=list(block.bbox) if block.bbox else None, text=block.text)


class PageResponse(BaseModel):
    page: int | None = Field(
        description="1-based page number; null when the result covers the whole document"
    )
    markdown: str
    blocks: list[BlockResponse]
    raw: str | None = Field(
        default=None, description="Unprocessed model output, included when include_raw=true"
    )
    truncated: bool = Field(
        default=False, description="True when the model hit max_tokens, so the page text may be cut off"
    )

    @classmethod
    def from_domain(cls, result: OcrResult, include_raw: bool) -> "PageResponse":
        return cls(
            page=result.page,
            markdown=result.markdown,
            blocks=[BlockResponse.from_domain(block) for block in result.blocks],
            raw=result.raw if include_raw else None,
            truncated=result.truncated,
        )


class OcrResponse(BaseModel):
    filename: str | None
    page_count: int
    markdown: str = Field(description="Markdown of all pages joined together")
    pages: list[PageResponse]

    @classmethod
    def from_domain(cls, filename: str | None, results: list[OcrResult], include_raw: bool) -> "OcrResponse":
        return cls(
            filename=filename,
            page_count=len(results),
            markdown="\n\n".join(result.markdown for result in results if result.markdown),
            pages=[PageResponse.from_domain(result, include_raw) for result in results],
        )


class JobResponse(BaseModel):
    id: str
    filename: str | None
    status: JobStatus
    page_count: int
    pages_done: int = Field(description="Pages with an outcome: succeeded + fallback + failed")
    pages_succeeded: int
    pages_fallback: int = Field(description="Pages where OCR failed and the PDF text layer was used instead")
    pages_failed: int
    pages_flagged: int = Field(
        description="Pages with quality warnings (truncated, repetition, text_layer_disagreement)"
    )
    needs_review: bool = Field(
        description="Completed, but more pages fell back or failed than OCR_JOB_MAX_FALLBACK_RATIO allows"
    )
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error: str | None = Field(description="Why the job itself failed; per-page errors are on the pages")
    sha256: str | None = Field(description="SHA-256 of the uploaded PDF")
    callback_url: str | None

    @classmethod
    def from_domain(cls, job: Job) -> "JobResponse":
        return cls(
            id=job.id,
            filename=job.filename,
            status=job.status,
            page_count=job.page_count,
            pages_done=job.pages_done,
            pages_succeeded=job.pages_succeeded,
            pages_fallback=job.pages_fallback,
            pages_failed=job.pages_failed,
            pages_flagged=job.pages_flagged,
            needs_review=job.needs_review,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            error=job.error,
            sha256=job.sha256,
            callback_url=job.callback_url,
        )


class JobPageResponse(BaseModel):
    page: int
    status: PageStatus
    attempts: int
    flags: list[PageFlag]
    error: str | None = Field(description="Last error, for failed and fallback pages")
    markdown: str | None
    blocks: list[BlockResponse]
    raw: str | None = Field(
        default=None, description="Unprocessed model output, included when include_raw=true"
    )

    @classmethod
    def from_domain(cls, outcome: PageOutcome, include_raw: bool) -> "JobPageResponse":
        result = outcome.result
        return cls(
            page=outcome.page,
            status=outcome.status,
            attempts=outcome.attempts,
            flags=list(outcome.flags),
            error=outcome.error,
            markdown=result.markdown if result else None,
            blocks=[BlockResponse.from_domain(block) for block in result.blocks] if result else [],
            raw=result.raw if result and include_raw else None,
        )


class JobPageEvent(BaseModel):
    """Compact page summary sent over the event stream; fetch /pages for the text."""

    page: int
    status: PageStatus
    attempts: int
    flags: list[PageFlag]

    @classmethod
    def from_domain(cls, outcome: PageOutcome) -> "JobPageEvent":
        return cls(
            page=outcome.page, status=outcome.status, attempts=outcome.attempts, flags=list(outcome.flags)
        )


class JobPagesResponse(BaseModel):
    job_id: str
    status: JobStatus
    page_count: int
    pages: list[JobPageResponse] = Field(
        description="Finished pages in the requested range; pages still in progress are omitted"
    )


class PageSpanResponse(BaseModel):
    page: int
    status: PageStatus
    start: int = Field(description="Offset of the page's `<!-- page N -->` marker in markdown")
    end: int = Field(description="Exclusive end offset: markdown[start:end] is the page")

    @classmethod
    def from_domain(cls, span: PageSpan) -> "PageSpanResponse":
        return cls(page=span.page, status=span.status, start=span.start, end=span.end)


class JobDocumentResponse(BaseModel):
    job_id: str
    page_count: int
    markdown: str
    pages: list[PageSpanResponse] = Field(description="Where each page sits in markdown, in page order")

    @classmethod
    def from_domain(cls, job: Job, document: AssembledDocument) -> "JobDocumentResponse":
        return cls(
            job_id=job.id,
            page_count=job.page_count,
            markdown=document.markdown,
            pages=[PageSpanResponse.from_domain(span) for span in document.spans],
        )


class HealthResponse(BaseModel):
    status: str
    engine: str


class ErrorResponse(BaseModel):
    detail: str
