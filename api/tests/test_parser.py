from app.domain.models import Block
from app.domain.parser import parse_blocks, to_markdown
from tests.fakes import RAW_PAGE


def test_parses_headers_bboxes_and_continuation_lines():
    assert parse_blocks(RAW_PAGE) == [
        Block(category="title", bbox=(117, 74, 581, 101), text="Laporan Keuangan"),
        Block(category="text", bbox=(117, 130, 592, 148), text="Baris pertama\nbaris kedua"),
    ]


def test_markdown_separates_blocks_with_blank_line():
    assert to_markdown(parse_blocks(RAW_PAGE)) == "Laporan Keuangan\n\nBaris pertama\nbaris kedua"


def test_image_blocks_are_kept_as_blocks_but_dropped_from_markdown():
    raw = "<|det|>image [0, 0, 10, 10]<|/det|>\n<|det|>text [0, 20, 10, 30]<|/det|>Caption"
    blocks = parse_blocks(raw)
    assert [block.category for block in blocks] == ["image", "text"]
    assert to_markdown(blocks) == "Caption"


def test_text_without_headers_becomes_single_block():
    assert parse_blocks("plain line\n\nsecond line") == [
        Block(category="text", bbox=None, text="plain line\nsecond line")
    ]


def test_header_without_bbox_and_stray_special_tokens():
    blocks = parse_blocks("<|det|>table<|/det|><table></table>\nfooter<|end▁of▁sentence|>")
    assert blocks == [Block(category="table", bbox=None, text="<table></table>\nfooter")]


def test_empty_output_yields_no_blocks():
    assert parse_blocks("") == []
    assert to_markdown([]) == ""
