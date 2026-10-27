import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "ocr_eval" / "workers"))
from _common import anchor_grid  # noqa: E402


def test_anchor_grid_puts_span_text_at_top_right():
    cells = [("Aset", 0, 1, 0, 2), ("2025", 0, 2, 2, 3), ("Kas", 1, 2, 0, 1), ("3", 1, 2, 1, 2)]
    assert anchor_grid(cells, 2, 3) == [["", "Aset", "2025"], ["Kas", "3", ""]]


def test_anchor_grid_keeps_equal_adjacent_values_and_ignores_out_of_range():
    cells = [("5", 0, 1, 0, 1), ("5", 0, 1, 1, 2), ("z", 9, 10, 0, 1)]
    assert anchor_grid(cells, 1, 2) == [["5", "5"]]
