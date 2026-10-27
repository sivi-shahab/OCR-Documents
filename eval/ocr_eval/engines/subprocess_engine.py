"""Docling and MinerU, each run as a worker script inside its own venv so their deps never mix with api/."""

import asyncio
import json
import os
import subprocess
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pymupdf

from ocr_eval.store import PageOutput

EVAL_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = EVAL_DIR / ".cache"  # the root filesystem is small; every model download stays under /data
_STDERR_TAIL = 2000


def worker_env(gpu: str) -> dict[str, str]:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    for key, sub in (
        ("PIP_CACHE_DIR", "pip"),
        ("HF_HOME", "hf"),
        ("MODELSCOPE_CACHE", "modelscope"),
        ("XDG_CACHE_HOME", "xdg"),
        ("TORCH_HOME", "torch"),
        ("EASYOCR_MODULE_PATH", "easyocr"),  # Docling's EasyOCR otherwise downloads to ~/.EasyOCR
    ):
        env[key] = str(CACHE_DIR / sub)
    # MinerU writes and reads its model paths here (otherwise ~/mineru.json).
    env["MINERU_TOOLS_CONFIG_JSON"] = str(CACHE_DIR / "mineru.json")
    return env


class SubprocessEngine:
    paid_price_per_page = None

    def __init__(self, name: str, python: Path, worker: Path, gpu: str = "1") -> None:
        self.name = name
        self._python = Path(python)
        self._worker = Path(worker)
        self._gpu = gpu
        self._mode: str | None = None
        self._worker_meta: dict = {}

    def available(self) -> str | None:
        if not self._python.exists():
            return f"interpreter venv {self.name} tidak ada: {self._python}"
        if not self._worker.exists():
            return f"worker {self.name} tidak ada: {self._worker}"
        return None

    def meta_info(self) -> dict:
        info = {key: self._worker_meta.get(key) for key in ("model", "version")}
        return {**info, "mode": self._mode, "gpu": self._gpu}

    def _command(self, pdf: Path, pages: list[int], out: Path, mode: str) -> list[str]:
        return [
            str(self._python), str(self._worker),
            "--pdf", str(pdf), "--pages", ",".join(map(str, pages)), "--out", str(out), "--mode", mode,
        ]  # fmt: skip

    async def run(self, pdf: Path, pages: list[int], *, is_scan_variant: bool) -> AsyncIterator[PageOutput]:
        self._mode = "ocr" if is_scan_variant else "auto"
        with tempfile.TemporaryDirectory(prefix=f"{self.name}_", dir=CACHE_DIR if CACHE_DIR.exists() else None) as tmp:
            out = Path(tmp)
            try:
                process = await asyncio.create_subprocess_exec(
                    *self._command(Path(pdf).resolve(), pages, out, self._mode),
                    env=worker_env(self._gpu),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await process.communicate()
            except OSError as exc:
                error = repr(exc)
            else:
                error = None
                if process.returncode != 0:
                    tail = stderr.decode("utf-8", "replace")[-_STDERR_TAIL:]
                    error = f"{self.name} worker exited with code {process.returncode}: {tail}"
            if error:
                for number in pages:
                    yield PageOutput(number, "", None, error=error)
                return
            self._worker_meta = json.loads((out / "worker_meta.json").read_text(encoding="utf-8"))
            seconds = self._worker_meta["seconds_total"] / len(pages)
            for number in pages:
                path = out / f"page_{number:04d}.md"
                if path.exists():
                    yield PageOutput(number, path.read_text(encoding="utf-8"), seconds)
                else:
                    yield PageOutput(number, "", None, error="worker wrote no markdown for this page")

    def reference_tables(self, pdf: Path, out: Path) -> list[dict]:
        """Docling's tables from the digital text layer (no OCR), used as the reference for table scoring."""
        with pymupdf.open(pdf) as document:
            pages = list(range(1, document.page_count + 1))
        out = Path(out)
        command = self._command(Path(pdf).resolve(), pages, out, "reference-tables")
        result = subprocess.run(command, env=worker_env(self._gpu), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result.returncode != 0:
            tail = result.stderr.decode("utf-8", "replace")[-_STDERR_TAIL:]
            raise RuntimeError(f"{self.name} reference-tables exited with code {result.returncode}: {tail}")
        return json.loads((out / "tables.json").read_text(encoding="utf-8"))
