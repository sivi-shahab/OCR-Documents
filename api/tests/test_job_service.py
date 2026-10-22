import asyncio
import secrets
from pathlib import Path

import pytest

from app.core.errors import DocumentTooLargeError, EngineUnavailableError, JobStateError
from app.domain.jobs import Job, JobStatus, PageFlag, PageOutcome, PageStatus
from app.domain.models import EngineOutput
from app.infrastructure.job_store import FileJobStore
from app.infrastructure.pdf_renderer import PyMuPdfRenderer
from app.services.job_service import JobService
from tests.fakes import RAW_PAGE, FakeEngine, make_pdf

LOOP = "<|det|>text [1, 2, 3, 4]<|/det|>" + "Saldo Saldo " * 500
DENSE = "<|det|>text [1, 2, 3, 4]<|/det|>" + secrets.token_hex(2000)


def make_service(
    engine: FakeEngine,
    root: Path,
    concurrency: int = 4,
    max_pages: int = 2000,
    escalation_dpi: int = 0,
    max_fallback_ratio: float = 0.004,
    notifier=None,
) -> JobService:
    return JobService(
        engine,
        PyMuPdfRenderer(dpi=72, max_pages=max_pages),
        FileJobStore(root),
        max_pages=max_pages,
        max_running_jobs=1,
        page_concurrency=concurrency,
        max_attempts=3,
        retry_backoff_seconds=0,
        first_pass_max_tokens=600,
        retry_max_tokens=800,
        retry_temperature_step=0.2,
        escalation_dpi=escalation_dpi,
        max_fallback_ratio=max_fallback_ratio,
        notifier=notifier,
    )


async def submit(service: JobService, root: Path, pdf: bytes, **kwargs):
    staged = service.new_upload_path()
    staged.write_bytes(pdf)
    job, _ = await service.submit(staged, "doc.pdf", **kwargs)
    return job


async def wait_finished(service: JobService, job_id: str) -> None:
    async with asyncio.timeout(30):
        while not service.get(job_id).status.is_finished:
            await asyncio.sleep(0.01)


async def run_job(engine: FakeEngine, root: Path, pdf: bytes, concurrency: int = 1, **kwargs):
    service = make_service(engine, root, concurrency, **kwargs)
    await service.start()
    try:
        job = await submit(service, root, pdf)
        await wait_finished(service, job.id)
        return service, job, await service.pages(job.id, 1, job.page_count)
    finally:
        await service.stop()


async def test_300_page_document_is_processed_page_by_page(tmp_path: Path):
    engine = FakeEngine()
    service, job, pages = await run_job(engine, tmp_path, make_pdf(300), concurrency=16)

    assert job.status is JobStatus.COMPLETED
    assert (job.pages_succeeded, job.pages_failed) == (300, 0)
    assert engine.calls == [1] * 300
    assert [page.page for page in pages] == list(range(1, 301))
    markdown = await service.markdown(job.id)
    assert markdown.count("<!-- page ") == 300
    assert markdown.index("<!-- page 2 -->") < markdown.index("<!-- page 300 -->")


async def test_engine_error_is_retried(tmp_path: Path):
    def script(call: int) -> EngineOutput:
        if call == 1:
            raise EngineUnavailableError("timeout")
        return EngineOutput(RAW_PAGE)

    _, _, pages = await run_job(FakeEngine(script=script), tmp_path, make_pdf(1))
    assert pages[0].status is PageStatus.SUCCEEDED and pages[0].attempts == 2


@pytest.mark.parametrize(
    ("text_layer", "expected_status"), [(True, PageStatus.FALLBACK), (False, PageStatus.FAILED)]
)
async def test_persistent_failure_falls_back_to_text_layer(
    tmp_path: Path, text_layer: bool, expected_status: PageStatus
):
    def script(_: int) -> EngineOutput:
        raise EngineUnavailableError("down")

    _, job, pages = await run_job(FakeEngine(script=script), tmp_path, make_pdf(1, text_layer=text_layer))

    assert job.status is JobStatus.COMPLETED  # a bad page never fails the whole job
    assert pages[0].status is expected_status
    assert pages[0].attempts == 3 and pages[0].error == "down"
    if text_layer:
        assert pages[0].result.markdown == "Page 1"
        assert job.pages_fallback == 1


async def test_truncated_dense_page_is_retried_with_larger_budget(tmp_path: Path):
    engine = FakeEngine(script=lambda _: EngineOutput(DENSE, truncated=True))
    _, _, pages = await run_job(engine, tmp_path, make_pdf(1))

    assert engine.params == [(600, 0.0), (800, 0.0), (800, 0.0)]
    assert pages[0].status is PageStatus.SUCCEEDED
    assert pages[0].flags == (PageFlag.TRUNCATED,)


async def test_truncation_then_success_has_no_flags(tmp_path: Path):
    engine = FakeEngine(script=lambda call: EngineOutput(DENSE, truncated=call == 1))
    _, job, pages = await run_job(engine, tmp_path, make_pdf(1))
    assert pages[0].flags == () and job.pages_flagged == 0


async def test_generation_loop_raises_temperature_then_falls_back(tmp_path: Path):
    engine = FakeEngine(script=lambda _: EngineOutput(LOOP, truncated=True))
    _, _, pages = await run_job(engine, tmp_path, make_pdf(1))

    assert [temperature for _, temperature in engine.params] == pytest.approx([0.0, 0.2, 0.4])
    assert pages[0].status is PageStatus.FALLBACK


async def test_generation_loop_without_text_layer_keeps_flagged_output(tmp_path: Path):
    engine = FakeEngine(script=lambda _: EngineOutput(LOOP, truncated=True))
    _, _, pages = await run_job(engine, tmp_path, make_pdf(1, text_layer=False))

    assert pages[0].status is PageStatus.SUCCEEDED
    assert set(pages[0].flags) == {PageFlag.TRUNCATED, PageFlag.REPETITION}


async def test_unfinished_job_resumes_without_redoing_finished_pages(tmp_path: Path):
    blocked = asyncio.Event()

    async def first_run():
        engine = FakeEngine()
        original = engine.recognize

        async def recognize(images, **kwargs):
            if len(engine.calls) >= 3:
                blocked.set()
                await asyncio.Event().wait()  # hang until the service shuts down
            return await original(images, **kwargs)

        engine.recognize = recognize
        service = make_service(engine, tmp_path, concurrency=1)
        await service.start()
        job = await submit(service, tmp_path, make_pdf(5))
        await blocked.wait()
        await service.stop()
        return job.id

    job_id = await first_run()

    engine = FakeEngine()
    service = make_service(engine, tmp_path, concurrency=1)
    await service.start()
    try:
        await wait_finished(service, job_id)
        job = service.get(job_id)
        assert job.status is JobStatus.COMPLETED and job.pages_succeeded == 5
        assert len(engine.calls) == 2
    finally:
        await service.stop()


async def test_cancel_stops_job_and_markdown_is_unavailable(tmp_path: Path):
    engine = FakeEngine()

    async def hang(images, **kwargs):
        await asyncio.Event().wait()

    engine.recognize = hang
    service = make_service(engine, tmp_path)
    await service.start()
    try:
        job = await submit(service, tmp_path, make_pdf(3))
        await asyncio.sleep(0.05)
        assert (await service.cancel(job.id)).status is JobStatus.CANCELLED
        with pytest.raises(JobStateError):
            await service.markdown(job.id)
        with pytest.raises(JobStateError):
            await service.cancel(job.id)
        await service.delete(job.id)
        assert not (tmp_path / job.id).exists()
    finally:
        await service.stop()


async def test_document_over_job_limit_is_rejected_and_upload_removed(tmp_path: Path):
    service = make_service(FakeEngine(), tmp_path, max_pages=2)
    staged = service.new_upload_path()
    staged.write_bytes(make_pdf(3))
    with pytest.raises(DocumentTooLargeError):
        await service.submit(staged, "doc.pdf")
    assert not staged.exists()


async def test_degraded_page_is_retried_once_at_escalation_dpi(tmp_path: Path):
    engine = FakeEngine(script=lambda call: EngineOutput(DENSE, truncated=call <= 3))
    _, _, pages = await run_job(engine, tmp_path, make_pdf(1), escalation_dpi=144)

    assert pages[0].status is PageStatus.SUCCEEDED and pages[0].flags == ()
    assert pages[0].attempts == 4
    assert engine.image_sizes[3] > engine.image_sizes[0]  # rendered at 144 instead of 72 DPI


async def test_empty_output_on_text_page_escalates_then_falls_back(tmp_path: Path):
    engine = FakeEngine(output="")
    _, _, pages = await run_job(engine, tmp_path, make_pdf(1), escalation_dpi=144)

    assert len(engine.calls) == 2  # first pass, then one try at the escalation DPI
    assert pages[0].status is PageStatus.FALLBACK and pages[0].result.markdown == "Page 1"


async def test_identical_upload_reuses_job_until_it_is_cancelled(tmp_path: Path):
    engine = FakeEngine()

    async def hang(images, **kwargs):
        await asyncio.Event().wait()

    engine.recognize = hang
    service = make_service(engine, tmp_path)
    await service.start()
    try:
        pdf = make_pdf(2)
        first = await submit(service, tmp_path, pdf, sha256="abc")
        staged = service.new_upload_path()
        staged.write_bytes(pdf)
        again, created = await service.submit(staged, "copy.pdf", sha256="abc")
        assert (again.id, created) == (first.id, False)
        assert not staged.exists()

        await service.cancel(first.id)
        assert (await submit(service, tmp_path, pdf, sha256="abc")).id != first.id
    finally:
        await service.stop()


async def test_event_stream_reports_every_page_then_the_finished_job(tmp_path: Path):
    release = asyncio.Event()
    engine = FakeEngine()
    original = engine.recognize

    async def gated(images, **kwargs):
        await release.wait()
        return await original(images, **kwargs)

    engine.recognize = gated
    service = make_service(engine, tmp_path)
    await service.start()
    try:
        job = await submit(service, tmp_path, make_pdf(3))
        events = []
        async with asyncio.timeout(10):
            async for event in service.events(job.id):
                events.append(event)
                release.set()
    finally:
        await service.stop()

    pages = sorted(event.page for event in events if isinstance(event, PageOutcome))
    jobs = [event for event in events if isinstance(event, Job)]
    assert pages == [1, 2, 3]
    assert jobs[-1].status is JobStatus.COMPLETED and jobs[-1].pages_done == 3


async def test_finished_job_with_callback_notifies(tmp_path: Path):
    notified = asyncio.Event()

    class Notifier:
        job = None

        async def job_finished(self, job):
            self.job = job
            notified.set()

    notifier = Notifier()
    service = make_service(FakeEngine(), tmp_path, notifier=notifier)
    await service.start()
    try:
        job = await submit(service, tmp_path, make_pdf(1), callback_url="http://erp/hook")
        async with asyncio.timeout(10):
            await notified.wait()
    finally:
        await service.stop()
    assert notifier.job.id == job.id and notifier.job.status is JobStatus.COMPLETED


PROSE = " ".join(f"kata{n % 40} rekening nasabah bunga" for n in range(30))


@pytest.mark.parametrize(
    ("ocr_text", "flags"),
    [
        (PROSE, ()),
        (" ".join(f"lain{n} sama sekali berbeda" for n in range(30)), (PageFlag.TEXT_LAYER_DISAGREEMENT,)),
    ],
)
async def test_output_unlike_the_text_layer_is_flagged(tmp_path: Path, ocr_text: str, flags: tuple):
    engine = FakeEngine(output=f"<|det|>text [1, 2, 3, 4]<|/det|>{ocr_text}")
    _, job, pages = await run_job(engine, tmp_path, make_pdf(1, text=PROSE))

    assert pages[0].status is PageStatus.SUCCEEDED
    assert pages[0].flags == flags
    assert job.pages_flagged == len(flags)


@pytest.mark.parametrize(("failing_pages", "needs_review"), [(0, False), (1, True)])
async def test_too_many_fallback_pages_mark_the_job_for_review(
    tmp_path: Path, failing_pages: int, needs_review: bool
):
    def script(call: int) -> EngineOutput:
        if call <= failing_pages * 3:  # every attempt of the first pages fails
            raise EngineUnavailableError("down")
        return EngineOutput(RAW_PAGE)

    _, job, _ = await run_job(FakeEngine(script=script), tmp_path, make_pdf(10), max_fallback_ratio=0.05)

    assert job.status is JobStatus.COMPLETED
    assert job.pages_fallback == failing_pages
    assert job.needs_review is needs_review
