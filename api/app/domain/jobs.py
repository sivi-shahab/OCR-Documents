from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from app.domain.models import OcrResult


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"  # every page was attempted; check pages_failed for partial results
    FAILED = "failed"  # the job itself broke, e.g. the stored PDF became unreadable
    CANCELLED = "cancelled"

    @property
    def is_finished(self) -> bool:
        return self in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)


class PageStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FALLBACK = "fallback"  # the model kept failing, so the PDF's embedded text layer was used
    FAILED = "failed"

    @property
    def is_final(self) -> bool:
        """Final pages are kept when a job resumes; failed pages are attempted again."""
        return self is not PageStatus.FAILED


class PageFlag(StrEnum):
    """Quality warnings attached to a page that still produced text."""

    TRUNCATED = "truncated"  # generation hit max_tokens on every attempt
    REPETITION = "repetition"  # output looks like a generation loop
    TEXT_LAYER_DISAGREEMENT = "text_layer_disagreement"  # output shares few words with the PDF text layer


@dataclass(frozen=True, slots=True)
class PageOutcome:
    page: int
    status: PageStatus
    attempts: int
    result: OcrResult | None = None
    error: str | None = None
    flags: tuple[PageFlag, ...] = ()


@dataclass(slots=True)
class Job:
    id: str
    filename: str | None
    page_count: int
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    sha256: str | None = None  # of the uploaded PDF; identical uploads reuse the job
    callback_url: str | None = None  # notified once the job finishes
    # Completed, but more pages fell back to the text layer or failed than the configured ratio allows.
    needs_review: bool = False
    pages_succeeded: int = 0
    pages_fallback: int = 0
    pages_failed: int = 0
    pages_flagged: int = 0

    @property
    def pages_done(self) -> int:
        return self.pages_succeeded + self.pages_fallback + self.pages_failed

    def reset_progress(self, outcomes: list[PageOutcome]) -> None:
        self.pages_succeeded = self.pages_fallback = self.pages_failed = self.pages_flagged = 0
        for outcome in outcomes:
            self.record(outcome)

    def record(self, outcome: PageOutcome) -> None:
        match outcome.status:
            case PageStatus.SUCCEEDED:
                self.pages_succeeded += 1
            case PageStatus.FALLBACK:
                self.pages_fallback += 1
            case PageStatus.FAILED:
                self.pages_failed += 1
        if outcome.flags:
            self.pages_flagged += 1
