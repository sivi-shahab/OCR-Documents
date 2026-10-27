"""Docling worker; runs in eval/.venv-docling. See ocr_eval/engines/subprocess_engine.py for the CLI contract."""

import json
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

from _common import anchor_grid, parse_args, subset_pdf, write_meta, write_page
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

OCR_LANG = ["id", "en"]


def pipeline_options(mode: str) -> PdfPipelineOptions:
    if mode == "reference-tables":  # tables straight from the digital text layer
        return PdfPipelineOptions(do_table_structure=True, do_ocr=False)
    # auto: bitmaps are OCR'd and the text layer is used; ocr: every page is OCR'd in full.
    ocr = EasyOcrOptions(lang=OCR_LANG, force_full_page_ocr=mode == "ocr")
    return PdfPipelineOptions(do_table_structure=True, do_ocr=True, ocr_options=ocr)


def format_option(mode: str) -> PdfFormatOption:
    options = pipeline_options(mode)
    if mode == "reference-tables":
        return PdfFormatOption(pipeline_options=options)
    # The default docling-parse backend renders image-only pages so poorly that EasyOCR loses most text
    # (synthetic BSD scans: Dice 0.06); pypdfium2 renders them cleanly (Dice ~0.9). Text-layer pages are unaffected.
    return PdfFormatOption(pipeline_options=options, backend=PyPdfiumDocumentBackend)


def _grid(data) -> list[list[str]]:
    cells = [
        (c.text, c.start_row_offset_idx, c.end_row_offset_idx, c.start_col_offset_idx, c.end_col_offset_idx)
        for c in data.table_cells
    ]
    return anchor_grid(cells, data.num_rows, data.num_cols)


def main() -> None:
    args = parse_args(["auto", "ocr", "reference-tables"])
    start = time.perf_counter()
    converter = DocumentConverter(format_options={InputFormat.PDF: format_option(args.mode)})
    with tempfile.TemporaryDirectory(dir=args.out) as tmp:
        part = subset_pdf(args.pdf, args.pages, Path(tmp) / "part.pdf")
        doc = converter.convert(part).document
    if args.mode == "reference-tables":
        tables = [
            {"page": args.pages[table.prov[0].page_no - 1], "grid": _grid(table.data)}
            for table in doc.tables
            if table.prov
        ]
        (args.out / "tables.json").write_text(json.dumps(tables, ensure_ascii=False), encoding="utf-8")
    else:
        for index, page in enumerate(args.pages, start=1):
            write_page(args.out, page, doc.export_to_markdown(page_no=index))
    write_meta(
        args.out,
        model=(
            "docling (EasyOCR + TableFormer, pypdfium2)"
            if args.mode != "reference-tables"
            else "docling (TableFormer, no OCR, docling-parse)"
        ),
        version=version("docling"),
        ocr_lang=OCR_LANG if args.mode != "reference-tables" else None,
        seconds_total=round(time.perf_counter() - start, 3),
    )


if __name__ == "__main__":
    main()
