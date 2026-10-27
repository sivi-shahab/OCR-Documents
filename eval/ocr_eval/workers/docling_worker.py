"""Docling worker; runs in eval/.venv-docling. See ocr_eval/engines/subprocess_engine.py for the CLI contract."""

import json
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

from _common import parse_args, subset_pdf, write_meta, write_page
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


def main() -> None:
    args = parse_args(["auto", "ocr", "reference-tables"])
    start = time.perf_counter()
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options(args.mode))}
    )
    with tempfile.TemporaryDirectory(dir=args.out) as tmp:
        part = subset_pdf(args.pdf, args.pages, Path(tmp) / "part.pdf")
        doc = converter.convert(part).document
    if args.mode == "reference-tables":
        tables = [
            {"page": args.pages[table.prov[0].page_no - 1], "grid": [[cell.text for cell in row] for row in table.data.grid]}
            for table in doc.tables
            if table.prov
        ]
        (args.out / "tables.json").write_text(json.dumps(tables, ensure_ascii=False), encoding="utf-8")
    else:
        for index, page in enumerate(args.pages, start=1):
            write_page(args.out, page, doc.export_to_markdown(page_no=index))
    write_meta(
        args.out,
        model="docling (EasyOCR + TableFormer)" if args.mode != "reference-tables" else "docling (TableFormer, no OCR)",
        version=version("docling"),
        ocr_lang=OCR_LANG if args.mode != "reference-tables" else None,
        seconds_total=round(time.perf_counter() - start, 3),
    )


if __name__ == "__main__":
    main()
