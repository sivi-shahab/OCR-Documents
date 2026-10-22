"""Asynchronous OCR jobs for large PDFs.

A job is accepted immediately and processed in the background: pages are rendered one at a
time into a small buffer, each page is sent to the engine as its own request, and every
finished page is persisted. A page that does not come back clean climbs a bounded ladder:

1. engine errors are retried with exponential backoff;
2. output cut off at max_tokens is retried with a larger budget, plus a little temperature
   when it looks like a generation loop;
3. a page still truncated, or empty although the PDF has text, is re-rendered once at a
   higher DPI;
4. a page that still fails falls back to the PDF's embedded text layer when it has one, and
   is otherwise recorded as failed without affecting the rest of the document.

Pages that produced text are also flagged when their output shares few words with the PDF's text
layer. A completed job whose fallback and failed pages exceed max_fallback_ratio is marked
needs_review.

Unfinished jobs resume after a restart and skip pages that already have a final outcome.
Uploading a PDF identical to an active or completed job returns that job instead.

Jobs run inside the API process, so the API must run as a single process (one uvicorn worker).
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from app.core.errors import EngineResponseError, EngineUnavailableError, JobNotFoundError, JobStateError
from app.domain.assembly import AssembledDocument, assemble_document
from app.domain.jobs import Job, JobStatus, PageFlag, PageOutcome, PageStatus
from app.domain.models import EngineOutput, OcrResult, PageImage
from app.domain.ports import JobNotifier, JobStore, OcrEngine, PagedDocumentReader, PageSource
from app.domain.quality import disagrees_with_text_layer, is_repetitive
from app.services.ocr_service import to_result

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _RenderedPage:
    number: int
    image: PageImage | None  # None when rendering failed
    text_layer: str
    error: str | None = None


# None tells a worker to stop.
_PageItem = _RenderedPage | None
_Rerender = Callable[[int], Awaitable[PageImage]]

# What an event stream yields: a job snapshot, a finished page, or None as a keep-alive tick.
JobEvent = Job | PageOutcome | None


class JobService:
    def __init__(
        self,
        engine: OcrEngine,
        reader: PagedDocumentReader,
        store: JobStore,
        *,
        max_pages: int,
        max_running_jobs: int,
        page_concurrency: int,
        max_attempts: int,
        retry_backoff_seconds: float,
        first_pass_max_tokens: int,
        retry_max_tokens: int,
        retry_temperature_step: float,
        escalation_dpi: int = 0,
        max_fallback_ratio: float = 1.0,
        notifier: JobNotifier | None = None,
    ) -> None:
        self._engine = engine
        self._reader = reader
        self._store = store
        self._max_pages = max_pages
        self._max_running_jobs = max_running_jobs
        self._page_concurrency = page_concurrency
        self._max_attempts = max_attempts
        self._retry_backoff_seconds = retry_backoff_seconds
        self._first_pass_max_tokens = first_pass_max_tokens
        self._retry_max_tokens = retry_max_tokens
        self._retry_temperature_step = retry_temperature_step
        self._escalation_dpi = escalation_dpi
        self._max_fallback_ratio = max_fallback_ratio
        self._notifier = notifier
        self._jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._running: dict[str, asyncio.Task] = {}
        self._dispatchers: list[asyncio.Task] = []
        self._subscribers: dict[str, set[asyncio.Queue[Job | PageOutcome]]] = {}
        self._notifications: set[asyncio.Task] = set()

    async def start(self) -> None:
        """Loads stored jobs, re-queues unfinished ones and starts processing."""
        for job in await asyncio.to_thread(self._store.load_jobs):
            self._jobs[job.id] = job
            if not job.status.is_finished:
                job.status = JobStatus.QUEUED
                self._queue.put_nowait(job.id)
        self._dispatchers = [asyncio.create_task(self._dispatch()) for _ in range(self._max_running_jobs)]

    async def stop(self) -> None:
        """Stops processing. Running jobs stay unfinished on disk and resume on the next start."""
        for task in [*self._dispatchers, *self._notifications]:
            task.cancel()
        await asyncio.gather(*self._dispatchers, *self._notifications, return_exceptions=True)

    def new_upload_path(self) -> Path:
        return self._store.new_staging_path()

    async def submit(
        self,
        staged_source: Path,
        filename: str | None,
        *,
        sha256: str | None = None,
        callback_url: str | None = None,
    ) -> tuple[Job, bool]:
        """Validates the uploaded PDF and queues it. The staged file is moved or deleted.

        Returns the job and whether it is new; an upload whose sha256 matches a queued, running
        or completed job returns that job instead.
        """
        try:
            existing = self._find_reusable(sha256)
            if existing:
                staged_source.unlink(missing_ok=True)
                return existing, False
            page_count = await asyncio.to_thread(self._reader.count_pages, staged_source, self._max_pages)
            existing = self._find_reusable(sha256)  # an identical upload may have won the race
            if existing:
                staged_source.unlink(missing_ok=True)
                return existing, False
        except BaseException:
            staged_source.unlink(missing_ok=True)
            raise
        job = Job(
            id=uuid.uuid4().hex,
            filename=filename,
            page_count=page_count,
            status=JobStatus.QUEUED,
            created_at=_now(),
            sha256=sha256,
            callback_url=callback_url,
        )
        self._jobs[job.id] = job
        try:
            await asyncio.to_thread(self._store.create, job, staged_source)
        except BaseException:
            del self._jobs[job.id]
            staged_source.unlink(missing_ok=True)
            raise
        self._queue.put_nowait(job.id)
        return job, True

    def get(self, job_id: str) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise JobNotFoundError(f"Job {job_id} not found") from None

    def list_jobs(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    async def pages(self, job_id: str, first: int, last: int) -> list[PageOutcome]:
        """Returns the finished pages in first..last; pages still in progress are omitted."""
        self.get(job_id)
        return await asyncio.to_thread(self._store.load_pages, job_id, first, last)

    async def document(self, job_id: str, clean: bool = True) -> AssembledDocument:
        job = self.get(job_id)
        if job.status is not JobStatus.COMPLETED:
            raise JobStateError(f"Job {job_id} is {job.status}; the result is available once it is completed")
        outcomes = await asyncio.to_thread(self._store.load_pages, job_id)
        return assemble_document(outcomes, clean=clean)

    async def markdown(self, job_id: str, clean: bool = True) -> str:
        return (await self.document(job_id, clean)).markdown

    async def events(self, job_id: str, heartbeat_seconds: float = 15) -> AsyncIterator[JobEvent]:
        """Yields a job snapshot, then each finished page and job update, until the job finishes.

        None is yielded after heartbeat_seconds without activity so callers can keep the
        connection alive.
        """
        job = self.get(job_id)
        queue: asyncio.Queue[Job | PageOutcome] = asyncio.Queue()
        subscribers = self._subscribers.setdefault(job_id, set())
        subscribers.add(queue)
        try:
            yield replace(job)
            if job.status.is_finished:
                return
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), heartbeat_seconds)
                except TimeoutError:
                    yield None
                    continue
                yield event
                if isinstance(event, Job) and event.status.is_finished:
                    return
        finally:
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(job_id, None)

    async def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.status.is_finished:
            raise JobStateError(f"Job {job_id} is already {job.status}")
        job.status = JobStatus.CANCELLED
        job.finished_at = _now()
        self._store.save_job(job)
        await self._stop_task(job_id)
        self._finished(job)
        return job

    async def delete(self, job_id: str) -> None:
        job = self.get(job_id)
        await self._stop_task(job_id)
        del self._jobs[job_id]
        await asyncio.to_thread(self._store.delete, job_id)
        if not job.status.is_finished:
            job.status = JobStatus.CANCELLED
            self._publish(job.id, job)

    def _find_reusable(self, sha256: str | None) -> Job | None:
        if sha256 is None:
            return None
        reusable = (JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.COMPLETED)
        return next(
            (job for job in self._jobs.values() if job.sha256 == sha256 and job.status in reusable), None
        )

    def _publish(self, job_id: str, event: Job | PageOutcome) -> None:
        snapshot = replace(event) if isinstance(event, Job) else event
        for queue in self._subscribers.get(job_id, ()):
            queue.put_nowait(snapshot)

    def _finished(self, job: Job) -> None:
        self._publish(job.id, job)
        if self._notifier and job.callback_url:
            task = asyncio.create_task(_notify(self._notifier, replace(job)))
            self._notifications.add(task)
            task.add_done_callback(self._notifications.discard)

    async def _stop_task(self, job_id: str) -> None:
        task = self._running.get(job_id)
        if task:
            task.cancel()
            await asyncio.wait([task])

    async def _dispatch(self) -> None:
        while True:
            job_id = await self._queue.get()
            job = self._jobs.get(job_id)
            if job is None or job.status is not JobStatus.QUEUED:
                continue  # cancelled or deleted while waiting
            task = asyncio.create_task(self._run(job))
            self._running[job_id] = task
            try:
                await asyncio.wait([task])
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.wait([task])
                self._running.pop(job_id, None)

    async def _run(self, job: Job) -> None:
        job.status = JobStatus.RUNNING
        job.started_at = job.started_at or _now()
        self._store.save_job(job)
        self._publish(job.id, job)
        try:
            stored = await asyncio.to_thread(self._store.load_pages, job.id)
            # Pages that failed earlier are retried, so only final outcomes count as done.
            final = [outcome for outcome in stored if outcome.status.is_final]
            job.reset_progress(final)
            done = {outcome.page for outcome in final}
            remaining = [number for number in range(1, job.page_count + 1) if number not in done]
            await self._process_pages(job, remaining)
        except asyncio.CancelledError:
            raise  # cancelled by the user (status already set) or by shutdown (resumes later)
        except Exception as exc:
            logger.exception("Job %s failed", job.id)
            job.status, job.error = JobStatus.FAILED, str(exc)
        else:
            job.status = JobStatus.COMPLETED
            job.needs_review = (
                job.pages_fallback + job.pages_failed
            ) > self._max_fallback_ratio * job.page_count
        job.finished_at = _now()
        self._store.save_job(job)
        self._finished(job)
        logger.info(
            "Job %s %s: %d succeeded, %d fallback, %d failed, %d flagged",
            job.id,
            job.status,
            job.pages_succeeded,
            job.pages_fallback,
            job.pages_failed,
            job.pages_flagged,
        )

    async def _process_pages(self, job: Job, numbers: list[int]) -> None:
        if not numbers:
            return
        # One dedicated thread owns the open document: PDF libraries are not thread-safe,
        # and closing on the same thread waits for any render still in progress.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"render-{job.id[:8]}")
        loop = asyncio.get_running_loop()
        source = await loop.run_in_executor(
            executor, self._reader.open_pages, self._store.source_path(job.id)
        )

        async def rerender(number: int) -> PageImage:
            return await loop.run_in_executor(executor, source.render_page, number, self._escalation_dpi)

        queue: asyncio.Queue[_PageItem] = asyncio.Queue(maxsize=self._page_concurrency)
        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(self._render_pages(executor, source, numbers, queue))
                for _ in range(self._page_concurrency):
                    group.create_task(self._recognize_pages(job, queue, rerender))
        finally:
            loop.run_in_executor(executor, source.close)
            executor.shutdown(wait=False)

    async def _render_pages(
        self, executor: ThreadPoolExecutor, source: PageSource, numbers: list[int], queue: asyncio.Queue
    ) -> None:
        loop = asyncio.get_running_loop()
        for number in numbers:
            page = await loop.run_in_executor(executor, _render_page, source, number)
            await queue.put(page)
        for _ in range(self._page_concurrency):
            await queue.put(None)

    async def _recognize_pages(self, job: Job, queue: asyncio.Queue[_PageItem], rerender: _Rerender) -> None:
        while (page := await queue.get()) is not None:
            outcome = await self._recognize_page(page, rerender)
            self._store.save_page(job.id, outcome)
            job.record(outcome)
            self._store.save_job(job)
            self._publish(job.id, outcome)

    async def _recognize_page(self, page: _RenderedPage, rerender: _Rerender) -> PageOutcome:
        if page.image is None:
            return _fallback_or_fail(page, attempts=0, error=page.error or "")
        output, attempts, error = await self._retry_ladder(page.number, page.image)
        if self._escalation_dpi and output is not None and _is_degraded(output, page):
            escalated = await self._recognize_at_escalation_dpi(page.number, rerender)
            if escalated is not None:
                attempts += 1
                if not _is_degraded(escalated, page):
                    output = escalated
        if output is None:
            return _fallback_or_fail(page, attempts, error)
        if page.text_layer and (_is_empty(output) or (output.truncated and is_repetitive(output.text))):
            return _fallback_or_fail(page, attempts, "Model output was empty or kept looping")
        # Partial text from a genuinely dense page beats nothing; it is flagged as truncated.
        return _succeeded(page, output, attempts)

    async def _retry_ladder(self, number: int, image: PageImage) -> tuple[EngineOutput | None, int, str]:
        """Returns the first untruncated output, else the last truncated one (or None), the
        number of attempts made and the last engine error."""
        max_tokens, temperature = self._first_pass_max_tokens, 0.0
        truncated: EngineOutput | None = None
        error = ""
        for attempt in range(1, self._max_attempts + 1):
            try:
                output = await self._engine.recognize([image], max_tokens=max_tokens, temperature=temperature)
            except (EngineUnavailableError, EngineResponseError) as exc:
                error = str(exc)
                logger.warning("Page %d attempt %d/%d failed: %s", number, attempt, self._max_attempts, exc)
                if attempt < self._max_attempts:
                    await asyncio.sleep(self._retry_backoff_seconds * 2 ** (attempt - 1))
                continue
            if not output.truncated:
                return output, attempt, ""
            # Cut off at max_tokens: a dense page needs a larger budget, a loop needs some randomness.
            truncated = output
            looping = is_repetitive(output.text)
            logger.warning(
                "Page %d attempt %d/%d hit max_tokens=%d%s",
                number,
                attempt,
                self._max_attempts,
                max_tokens,
                " (repetitive)" if looping else "",
            )
            max_tokens = max(max_tokens, self._retry_max_tokens)
            if looping:
                temperature = min(temperature + self._retry_temperature_step, 1.0)
        return truncated, self._max_attempts, error

    async def _recognize_at_escalation_dpi(self, number: int, rerender: _Rerender) -> EngineOutput | None:
        try:
            image = await rerender(number)
            return await self._engine.recognize([image], max_tokens=self._retry_max_tokens)
        except (EngineUnavailableError, EngineResponseError) as exc:
            logger.warning("Page %d at %d DPI failed: %s", number, self._escalation_dpi, exc)
        except Exception:
            logger.exception("Page %d could not be re-rendered at %d DPI", number, self._escalation_dpi)
        return None


async def _notify(notifier: JobNotifier, job: Job) -> None:
    try:
        await notifier.job_finished(job)
    except Exception:
        logger.exception("Could not notify %s about job %s", job.callback_url, job.id)


def _render_page(source: PageSource, number: int) -> _RenderedPage:
    """Runs on the document's render thread."""
    try:
        text_layer = source.extract_text(number)
    except Exception:  # noqa: BLE001 - a broken text layer only disables the fallback
        text_layer = ""
    try:
        image = source.render_page(number)
    except Exception as exc:  # noqa: BLE001 - one unrenderable page must not fail the job
        return _RenderedPage(number, None, text_layer, f"Page could not be rendered: {exc}")
    return _RenderedPage(number, image, text_layer)


def _is_empty(output: EngineOutput) -> bool:
    return not to_result(output, None).markdown


def _is_degraded(output: EngineOutput, page: _RenderedPage) -> bool:
    """Truncated, or empty although the PDF shows there is text: worth a sharper render."""
    return output.truncated or (bool(page.text_layer) and _is_empty(output))


def _succeeded(page: _RenderedPage, output: EngineOutput, attempts: int) -> PageOutcome:
    result = to_result(output, page.number)
    flags = []
    if output.truncated:
        flags.append(PageFlag.TRUNCATED)
    if is_repetitive(output.text):
        flags.append(PageFlag.REPETITION)
    if disagrees_with_text_layer(result.markdown, page.text_layer):
        flags.append(PageFlag.TEXT_LAYER_DISAGREEMENT)
    return PageOutcome(
        page=page.number, status=PageStatus.SUCCEEDED, attempts=attempts, result=result, flags=tuple(flags)
    )


def _fallback_or_fail(page: _RenderedPage, attempts: int, error: str) -> PageOutcome:
    if not page.text_layer:
        return PageOutcome(page=page.number, status=PageStatus.FAILED, attempts=attempts, error=error)
    result = OcrResult(page=page.number, markdown=page.text_layer)
    return PageOutcome(
        page=page.number, status=PageStatus.FALLBACK, attempts=attempts, result=result, error=error
    )


def _now() -> datetime:
    return datetime.now(UTC)
