"""MinerU worker; runs in eval/.venv-mineru. See ocr_eval/engines/subprocess_engine.py for the CLI contract."""

import json
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

from _common import parse_args, subset_pdf, write_meta, write_page

# MinerU 3.x has no separate Latin model: "latin" is an alias it maps to the "ch" PP-OCRv5 model,
# whose script coverage is Chinese, English, Japanese and Latin (enough for Indonesian).
OCR_LANG = "latin"
OCR_MODEL_LANG = "ch"


def to_markdown(items: list[dict]) -> str:
    blocks = []
    for item in items:
        if item.get("type") == "table":
            body = item.get("table_body", "")
        else:
            body = item.get("text", "")
            if body and item.get("text_level"):
                body = "#" * item["text_level"] + " " + body
        if body.strip():
            blocks.append(body.strip())
    return "\n\n".join(blocks)


def main() -> None:
    args = parse_args(["auto", "ocr"])
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(dir=args.out) as tmp:
        part = subset_pdf(args.pdf, args.pages, Path(tmp) / "part.pdf")
        mineru = Path(sys.executable).with_name("mineru")
        command = [str(mineru), "-p", str(part), "-o", tmp, "-b", "pipeline", "-m", args.mode, "-l", OCR_LANG]
        subprocess.run(command, check=True, stdout=sys.stderr)
        (content_list,) = Path(tmp).rglob("*_content_list.json")
        items = json.loads(content_list.read_text(encoding="utf-8"))
    by_index: dict[int, list[dict]] = {}
    for item in items:
        by_index.setdefault(item["page_idx"], []).append(item)
    for index, page in enumerate(args.pages):
        write_page(args.out, page, to_markdown(by_index.get(index, [])))
    write_meta(
        args.out,
        model="mineru pipeline",
        version=version("mineru"),
        ocr_lang=OCR_LANG,
        ocr_model_lang=OCR_MODEL_LANG,
        seconds_total=round(time.perf_counter() - start, 3),
    )


if __name__ == "__main__":
    main()
