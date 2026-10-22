"""Signed webhook delivery for finished jobs.

The receiver verifies a request by recomputing

    hex(HMAC-SHA256(secret, f"{X-OCR-Timestamp}.{raw body}"))

and comparing it with X-OCR-Signature (after the "sha256=" prefix). Rejecting old
timestamps protects against replays.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time

import httpx

from app.domain.jobs import Job

logger = logging.getLogger(__name__)


class WebhookNotifier:
    def __init__(
        self, client: httpx.AsyncClient, secret: str, attempts: int = 3, backoff_seconds: float = 2
    ) -> None:
        self._client = client
        self._secret = secret.encode()
        self._attempts = attempts
        self._backoff_seconds = backoff_seconds

    async def job_finished(self, job: Job) -> None:
        if not job.callback_url:
            return
        body = json.dumps({"event": "job.finished", "job": job_payload(job)}).encode()
        for attempt in range(1, self._attempts + 1):
            timestamp = str(int(time.time()))
            headers = {
                "Content-Type": "application/json",
                "X-OCR-Event": "job.finished",
                "X-OCR-Timestamp": timestamp,
                "X-OCR-Signature": f"sha256={sign(self._secret, timestamp, body)}",
            }
            try:
                response = await self._client.post(job.callback_url, content=body, headers=headers)
                if response.is_success:
                    return
                problem = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                problem = str(exc) or type(exc).__name__
            logger.warning(
                "Webhook for job %s attempt %d/%d failed: %s", job.id, attempt, self._attempts, problem
            )
            if attempt < self._attempts:
                await asyncio.sleep(self._backoff_seconds * 2 ** (attempt - 1))


def sign(secret: bytes, timestamp: str, body: bytes) -> str:
    return hmac.new(secret, timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()


def job_payload(job: Job) -> dict:
    return {
        "id": job.id,
        "filename": job.filename,
        "status": job.status.value,
        "page_count": job.page_count,
        "pages_done": job.pages_done,
        "pages_succeeded": job.pages_succeeded,
        "pages_fallback": job.pages_fallback,
        "pages_failed": job.pages_failed,
        "pages_flagged": job.pages_flagged,
        "needs_review": job.needs_review,
        "error": job.error,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }
