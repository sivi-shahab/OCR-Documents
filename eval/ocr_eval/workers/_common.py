"""Shared by the worker scripts. Runs inside each engine's own venv: stdlib + pypdfium2 (a dep of both) only."""

import argparse
import json
from pathlib import Path


def parse_args(modes: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--pages", required=True, help="1-based, comma separated")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=modes, required=True)
    args = parser.parse_args()
    args.pages = [int(p) for p in args.pages.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    return args


def subset_pdf(pdf: Path, pages: list[int], target: Path) -> Path:
    """Copies only the requested pages so the engine never processes the rest; index i maps to pages[i]."""
    import pypdfium2

    source = pypdfium2.PdfDocument(pdf)
    part = pypdfium2.PdfDocument.new()
    part.import_pages(source, [p - 1 for p in pages])
    part.save(target)
    part.close()
    source.close()
    return target


def write_page(out: Path, page: int, markdown: str) -> None:
    (out / f"page_{page:04d}.md").write_text(markdown, encoding="utf-8")


def write_meta(out: Path, **fields) -> None:
    (out / "worker_meta.json").write_text(json.dumps(fields, indent=2, ensure_ascii=False), encoding="utf-8")


def anchor_grid(cells: list[tuple[str, int, int, int, int]], num_rows: int, num_cols: int) -> list[list[str]]:
    """Grid from (text, row_start, row_end, col_start, col_end) cells with exclusive ends.

    A spanning cell's text is stored once, at its top row and rightmost column; the rest of its span is "".
    """
    grid = [[""] * num_cols for _ in range(num_rows)]
    for text, r0, _r1, _c0, c1 in cells:
        if 0 <= r0 < num_rows and 0 < c1 <= num_cols:
            grid[r0][c1 - 1] = text
    return grid
