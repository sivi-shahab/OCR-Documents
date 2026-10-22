"""Keeps jobs on the local filesystem so they survive restarts.

    <root>/<job_id>/job.json          job metadata and progress counters
    <root>/<job_id>/source.pdf        the uploaded document
    <root>/<job_id>/pages/00001.json  one file per finished page
    <root>/.staging/                  uploads that have not been accepted yet

Every write goes to a temporary file first and is then renamed, so a crash never
leaves a half-written JSON file behind.
"""

import json
import os
import shutil
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from app.domain.jobs import Job, JobStatus, PageFlag, PageOutcome, PageStatus
from app.domain.models import Block, OcrResult

_SOURCE_NAME = "source.pdf"


class FileJobStore:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._staging = root / ".staging"
        self._staging.mkdir(parents=True, exist_ok=True)

    def new_staging_path(self) -> Path:
        return self._staging / f"{uuid.uuid4().hex}.pdf"

    def create(self, job: Job, staged_source: Path) -> None:
        job_dir = self._root / job.id
        (job_dir / "pages").mkdir(parents=True)
        staged_source.rename(job_dir / _SOURCE_NAME)
        self.save_job(job)

    def source_path(self, job_id: str) -> Path:
        return self._root / job_id / _SOURCE_NAME

    def save_job(self, job: Job) -> None:
        _write_json(self._root / job.id / "job.json", _job_to_dict(job))

    def load_jobs(self) -> list[Job]:
        jobs = []
        for path in self._root.glob("*/job.json"):
            jobs.append(_job_from_dict(json.loads(path.read_text())))
        return sorted(jobs, key=lambda job: job.created_at)

    def save_page(self, job_id: str, outcome: PageOutcome) -> None:
        _write_json(self._page_path(job_id, outcome.page), _outcome_to_dict(outcome))

    def load_pages(self, job_id: str, first: int = 1, last: int | None = None) -> list[PageOutcome]:
        """Returns stored outcomes for pages first..last (inclusive), in page order."""
        outcomes = []
        for path in sorted((self._root / job_id / "pages").glob("*.json")):
            number = int(path.stem)
            if number >= first and (last is None or number <= last):
                outcomes.append(_outcome_from_dict(json.loads(path.read_text())))
        return outcomes

    def delete(self, job_id: str) -> None:
        shutil.rmtree(self._root / job_id, ignore_errors=True)

    def _page_path(self, job_id: str, page: int) -> Path:
        return self._root / job_id / "pages" / f"{page:05d}.json"


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False))
    os.replace(temporary, path)


def _job_to_dict(job: Job) -> dict:
    data = asdict(job)
    for key in ("created_at", "started_at", "finished_at"):
        data[key] = data[key].isoformat() if data[key] else None
    return data


def _job_from_dict(data: dict) -> Job:
    for key in ("created_at", "started_at", "finished_at"):
        data[key] = datetime.fromisoformat(data[key]) if data[key] else None
    data["status"] = JobStatus(data["status"])
    return Job(**data)


def _outcome_to_dict(outcome: PageOutcome) -> dict:
    return asdict(outcome)


def _outcome_from_dict(data: dict) -> PageOutcome:
    result = data["result"]
    if result is not None:
        blocks = [
            Block(
                category=block["category"],
                bbox=tuple(block["bbox"]) if block["bbox"] else None,
                text=block["text"],
            )
            for block in result["blocks"]
        ]
        result = OcrResult(**{**result, "blocks": blocks})
    return PageOutcome(
        **{
            **data,
            "status": PageStatus(data["status"]),
            "result": result,
            "flags": tuple(PageFlag(flag) for flag in data.get("flags", ())),
        }
    )
