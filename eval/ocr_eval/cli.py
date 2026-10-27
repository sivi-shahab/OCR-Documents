"""Command line: python -m ocr_eval {classify,degrade,run,score,report,all} --out DIR --pdf PDF [--pdf PDF ...].

Artifacts in --out: classification.json, synthetic/<doc>.pdf + synthetic_map.json, reference_tables.json,
skipped_engines.json, runs/ (RunStore cache), scores.json, ocr_eval.xlsx and summary.md.
"""

import argparse
import asyncio
import json
import tempfile
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import pymupdf

from ocr_eval.classify import PageKind, classify_pdf
from ocr_eval.degrade import degrade_pdf
from ocr_eval.report import standard_notes, write_report
from ocr_eval.score import (
    ORIGINAL,
    SYNTHETIC,
    balance_checks,
    clean_reference_grids,
    load_json,
    rows_from_json,
    rows_to_json,
    score_run,
    summarize,
)  # fmt: skip
from ocr_eval.store import Engine, RunStore, run_engine

ALL_ENGINES = ("unlimited", "azure", "mistral", "docling", "mineru")
_SHARED_GPU = {"docling": "gpu1", "mineru": "gpu1"}  # Docling and MinerU share GPU 1, so they take turns
_REFERENCE_LANE = "gpu1"  # the Docling reference tables run on GPU 1 too

EngineFactory = Callable[[list[str]], list[Engine]]


def _subprocess_engine(name: str):
    from ocr_eval.engines.subprocess_engine import EVAL_DIR, SubprocessEngine

    return SubprocessEngine(name, EVAL_DIR / f".venv-{name}" / "bin" / "python",
                            EVAL_DIR / "ocr_eval" / "workers" / f"{name}_worker.py")  # fmt: skip


def build_engines(names: list[str]) -> list[Engine]:
    from app.core.config import Settings

    from ocr_eval.engines.cloud import AzureEngine, MistralEngine
    from ocr_eval.engines.unlimited import UnlimitedEngine

    unknown = set(names) - set(ALL_ENGINES)
    if unknown:
        raise SystemExit(f"engine tidak dikenal: {', '.join(sorted(unknown))} (pilihan: {','.join(ALL_ENGINES)})")
    settings = Settings()
    factories = {
        "unlimited": lambda: UnlimitedEngine(settings.vllm_base_url),
        "azure": lambda: AzureEngine(settings),
        "mistral": lambda: MistralEngine(settings),
        "docling": lambda: _subprocess_engine("docling"),
        "mineru": lambda: _subprocess_engine("mineru"),
    }
    return [factories[name]() for name in names]


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _require(path: Path, stage: str) -> dict:
    if not path.exists():
        raise SystemExit(f"{path.name} belum ada: jalankan `{stage}` dulu")
    return load_json(path, {})


# --- classify / degrade -------------------------------------------------------------------------------------------


def cmd_classify(args) -> int:
    path = args.out / "classification.json"
    data = load_json(path, {})
    for pdf in args.pdf:
        kinds = classify_pdf(pdf)
        data[pdf.stem] = {str(page): str(kind) for page, kind in kinds.items()}
        counts = {k: sum(v == k for v in kinds.values()) for k in PageKind}
        print(f"{pdf.stem}: " + ", ".join(f"{n} {k}" for k, n in counts.items() if n), flush=True)
    _write_json(path, data)
    return 0


def cmd_degrade(args) -> int:
    classification = _require(args.out / "classification.json", "classify")
    path = args.out / "synthetic_map.json"
    data = load_json(path, {})
    for pdf in args.pdf:
        digital = sorted(int(p) for p, k in classification.get(pdf.stem, {}).items() if k == PageKind.DIGITAL)
        dst = args.out / "synthetic" / f"{pdf.stem}.pdf"
        if not digital or (dst.exists() and pdf.stem in data):
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"{pdf.stem}: mendegradasi {len(digital)} halaman digital", flush=True)
        data[pdf.stem] = {str(s): o for s, o in degrade_pdf(pdf, dst, digital).items()}
    _write_json(path, data)
    return 0


# --- run ----------------------------------------------------------------------------------------------------------


def _jobs(args, synthetic_map: dict) -> list[tuple[str, Path, list[int], bool]]:
    """(variant, pdf, pages, is_scan_variant) for every engine: original PDFs, then their synthetic scans."""
    jobs = []
    for pdf in args.pdf:
        with pymupdf.open(pdf) as doc:
            jobs.append((ORIGINAL, pdf, list(range(1, doc.page_count + 1)), False))
    for pdf in args.pdf:
        if synthetic_map.get(pdf.stem):
            pages = sorted(int(p) for p in synthetic_map[pdf.stem])
            jobs.append((SYNTHETIC, args.out / "synthetic" / f"{pdf.stem}.pdf", pages, True))
    return jobs


def _estimate(paid: list[Engine], store: RunStore, jobs) -> dict[str, tuple[int, float]]:
    estimate = {}
    for engine in paid:
        todo = sum(len(store.missing_pages(v, pdf.stem, engine.name, pages)) for v, pdf, pages, _ in jobs)
        estimate[engine.name] = (todo, todo * engine.paid_price_per_page)
    print("Estimasi biaya engine berbayar (halaman belum di-cache × harga):")
    print(f"  {'engine':<10} {'halaman':>8} {'US$/hlm':>8} {'US$':>8}")
    for engine in paid:
        todo, cost = estimate[engine.name]
        print(f"  {engine.name:<10} {todo:>8} {engine.paid_price_per_page:>8.3f} {cost:>8.2f}")
    print(f"  {'total':<10} {sum(t for t, _ in estimate.values()):>8} {'':>8} {sum(c for _, c in estimate.values()):>8.2f}",
          flush=True)  # fmt: skip
    return estimate


async def _reference_tables(reference, pdf: Path, out: Path) -> None:
    path = out / "reference_tables.json"
    with tempfile.TemporaryDirectory(prefix="reference_", dir=out) as tmp:
        tables = await asyncio.to_thread(reference.reference_tables, pdf, Path(tmp))
    data = load_json(path, {})  # one lane writes this file, so re-reading is enough
    data[pdf.stem] = tables
    _write_json(path, data)


async def _run_lanes(lanes: dict[str, list[tuple[str, Callable[[], Awaitable[None]]]]]) -> list[str]:
    failures: list[str] = []

    async def lane(steps):
        for label, step in steps:
            print(f"[mulai] {label}", flush=True)
            start = time.perf_counter()
            try:
                await step()
            except Exception as exc:  # one failing step must not stop the other lanes
                failures.append(f"{label}: {exc!r}")
                print(f"[gagal] {label}: {exc!r}", flush=True)
            else:
                print(f"[selesai] {label} ({time.perf_counter() - start:.0f} dtk)", flush=True)

    await asyncio.gather(*(lane(steps) for steps in lanes.values()))
    return failures


def cmd_run(args, engine_factory: EngineFactory, reference=None) -> int:
    classification = _require(args.out / "classification.json", "classify")
    synthetic_map = _require(args.out / "synthetic_map.json", "degrade")
    store = RunStore(args.out / "runs")
    jobs = _jobs(args, synthetic_map)

    skipped: dict[str, str] = {}
    ready: list[Engine] = []
    for engine in engine_factory(args.engines):
        reason = engine.available()
        if reason:
            skipped[engine.name] = reason
            print(f"{engine.name} dilewati: {reason}", flush=True)
        else:
            ready.append(engine)

    paid = [e for e in ready if e.paid_price_per_page is not None]
    blocked = False
    if paid:
        estimate = _estimate(paid, store, jobs)
        if not args.yes:
            for engine in paid:
                todo, cost = estimate[engine.name]
                if todo:
                    blocked = True
                    ready.remove(engine)
                    skipped[engine.name] = f"tidak dijalankan tanpa --yes (estimasi US${cost:.2f})"
            if blocked:
                print("Engine berbayar tidak dipanggil; ulangi dengan --yes untuk menjalankannya.", flush=True)
    _write_json(args.out / "skipped_engines.json", skipped)

    lanes: dict[str, list] = {}
    # Docling reference tables: once per document with digital pages, cached in reference_tables.json.
    cached = load_json(args.out / "reference_tables.json", {})
    need_tables = [
        pdf
        for pdf in args.pdf
        if pdf.stem not in cached and PageKind.DIGITAL in classification.get(pdf.stem, {}).values()
    ]
    if need_tables:
        reference = reference or _subprocess_engine("docling")
        reason = reference.available()
        if reason:
            print(f"Tabel acuan dilewati: {reason}", flush=True)
        for pdf in [] if reason else need_tables:
            step = lambda pdf=pdf: _reference_tables(reference, pdf, args.out)
            lanes.setdefault(_REFERENCE_LANE, []).append((f"tabel acuan {pdf.stem}", step))

    for engine in ready:
        for variant, pdf, pages, is_scan in jobs:
            todo = len(store.missing_pages(variant, pdf.stem, engine.name, pages))
            if not todo:
                continue
            # Recorded up front so the cost is known even if this run is interrupted.
            store.update_meta(variant, pdf.stem, engine.name, price_per_page=engine.paid_price_per_page)

            async def step(engine=engine, variant=variant, pdf=pdf, pages=pages, is_scan=is_scan):
                await run_engine(engine, store, variant, pdf, pages, is_scan)

            lane = _SHARED_GPU.get(engine.name, engine.name)
            lanes.setdefault(lane, []).append((f"{engine.name} {variant}/{pdf.stem} ({todo} hlm)", step))

    failures = asyncio.run(_run_lanes(lanes))
    if blocked:
        return 2
    return 1 if failures else 0


# --- score / report -----------------------------------------------------------------------------------------------


def cmd_score(args) -> int:
    rows = score_run(args.out, args.pdf)
    data = {"rows": rows_to_json(rows), "balance": balance_checks(args.out, args.pdf)}
    _write_json(args.out / "scores.json", data)
    print(f"{len(rows)} baris skor ditulis ke {args.out / 'scores.json'}", flush=True)
    return 0


def _metas(runs: Path) -> dict[tuple[str, str, str], dict]:
    return {
        (path.parts[-4], path.parts[-3], path.parts[-2]): load_json(path, {})
        for path in sorted(runs.glob("*/*/*/meta.json"))
    }


def _extra_notes(out: Path, balance: dict) -> list[str]:
    notes = []
    labels = {True: "lolos", False: "gagal", None: "tidak bisa dicek"}
    for doc, checks in sorted(balance.items()):
        if checks:
            results = ", ".join(f"{engine} {labels[result]}" for engine, result in sorted(checks.items()))
            notes.append(f"Cek neraca (jumlah aset = jumlah liabilitas dan ekuitas) {doc}: {results}.")
    classification = load_json(out / "classification.json", {})
    references = load_json(out / "reference_tables.json", {})
    missing = [doc for doc, pages in sorted(classification.items())
               if PageKind.DIGITAL in pages.values() and doc not in references]  # fmt: skip
    if missing:
        notes.append(f"Tabel acuan tidak ada untuk: {', '.join(missing)}; `column_accuracy` tidak dihitung di sana.")
    return notes


def cmd_report(args) -> int:
    scores = _require(args.out / "scores.json", "score")
    rows = rows_from_json(scores["rows"])
    metas = _metas(args.out / "runs")
    scored_engines = {r.engine for r in rows}
    skipped = {e: reason for e, reason in load_json(args.out / "skipped_engines.json", {}).items()
               if e not in scored_engines}  # fmt: skip
    excluded = sum(
        clean_reference_grids([entry["grid"] for entry in entries])[1]
        for entries in load_json(args.out / "reference_tables.json", {}).values()
    )
    notes = standard_notes(metas, skipped, excluded, args.gpu_hour_rate) + _extra_notes(args.out, scores["balance"])
    write_report(rows, summarize(rows, metas, args.gpu_hour_rate), args.out, notes)
    print(f"Laporan: {args.out / 'ocr_eval.xlsx'} dan {args.out / 'summary.md'}", flush=True)
    return 0


def cmd_all(args, engine_factory: EngineFactory, reference=None) -> int:
    cmd_classify(args)
    cmd_degrade(args)
    code = cmd_run(args, engine_factory, reference)
    cmd_score(args)
    cmd_report(args)
    return code


# --- argument parsing ---------------------------------------------------------------------------------------------


def _engines(value: str) -> list[str]:
    return [name.strip() for name in value.split(",") if name.strip()]


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", type=Path, required=True, help="folder hasil, mis. ../reports/ocr_eval_2026-10-27")
    common.add_argument("--pdf", type=Path, action="append", default=[], help="PDF yang dievaluasi (boleh berulang)")
    common.add_argument("--engines", type=_engines, default=list(ALL_ENGINES), help=",".join(ALL_ENGINES))
    common.add_argument("--gpu-hour-rate", type=float, default=0.95, help="US$ per jam GPU untuk engine self-hosted")
    common.add_argument("--yes", action="store_true", help="izinkan pemanggilan API berbayar")
    parser = argparse.ArgumentParser(prog="python -m ocr_eval", description="Evaluasi engine OCR")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("classify", "degrade", "run", "score", "report", "all"):
        commands.add_parser(name, parents=[common])
    return parser


def main(argv: list[str] | None = None, *, engine_factory: EngineFactory = build_engines, reference=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command != "report" and not args.pdf:
        parser.error("--pdf wajib untuk perintah ini")
    args.out.mkdir(parents=True, exist_ok=True)
    commands = {
        "classify": lambda: cmd_classify(args),
        "degrade": lambda: cmd_degrade(args),
        "run": lambda: cmd_run(args, engine_factory, reference),
        "score": lambda: cmd_score(args),
        "report": lambda: cmd_report(args),
        "all": lambda: cmd_all(args, engine_factory, reference),
    }
    return commands[args.command]()
