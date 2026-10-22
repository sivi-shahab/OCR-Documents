from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from app.domain.jobs import Job, PageOutcome
from app.domain.models import EngineOutput, PageImage


class OcrEngine(Protocol):
    """Anything that can turn one or more page images into raw model output."""

    async def recognize(
        self, images: Sequence[PageImage], *, max_tokens: int | None = None, temperature: float = 0.0
    ) -> EngineOutput:
        """max_tokens=None uses the engine's configured default."""
        ...

    async def is_healthy(self) -> bool: ...


class DocumentRenderer(Protocol):
    """Converts a multi-page document into page images."""

    def render(self, content: bytes) -> list[PageImage]: ...


class PageSource(Protocol):
    """An open document whose pages are rendered one at a time."""

    def render_page(self, number: int, dpi: int | None = None) -> PageImage:
        """dpi=None uses the reader's configured resolution."""
        ...

    def extract_text(self, number: int) -> str:
        """Returns the page's embedded text layer, or an empty string for scanned pages."""
        ...

    def close(self) -> None: ...


class PagedDocumentReader(Protocol):
    """Validates a document on disk and opens it for page-by-page rendering."""

    def count_pages(self, path: Path, max_pages: int) -> int: ...

    def open_pages(self, path: Path) -> PageSource: ...


class JobStore(Protocol):
    """Durable storage for jobs, their source documents and per-page outcomes."""

    def new_staging_path(self) -> Path: ...

    def create(self, job: Job, staged_source: Path) -> None: ...

    def source_path(self, job_id: str) -> Path: ...

    def save_job(self, job: Job) -> None: ...

    def load_jobs(self) -> list[Job]: ...

    def save_page(self, job_id: str, outcome: PageOutcome) -> None: ...

    def load_pages(self, job_id: str, first: int = 1, last: int | None = None) -> list[PageOutcome]: ...

    def delete(self, job_id: str) -> None: ...


class JobNotifier(Protocol):
    """Tells an external system that a job has finished."""

    async def job_finished(self, job: Job) -> None: ...
