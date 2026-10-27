from ocr_eval.tables import amount_columns, parse_tables, score_columns


def test_html_table_with_spans_expands():
    md = '<table><tr><td colspan="2">Aset</td><td>2025</td></tr><tr><td>Kas</td><td>3</td><td>1.234.567</td></tr></table>'
    (g,) = parse_tables(md)
    assert g == [["", "Aset", "2025"], ["Kas", "3", "1.234.567"]]


def test_pipe_table():
    assert parse_tables("| a | 1.234.567 |\n|---|---|\n| b | 7.654.321 |") == [[["a", "1.234.567"], ["b", "7.654.321"]]]


def test_rowspan_expands_and_empty_cells_are_filled():
    md = '<table><tr><td rowspan="2">X</td><td>1</td></tr><tr><td>2</td></tr><tr><td></td></tr></table>'
    assert parse_tables(md) == [[["X", "1"], ["", "2"], [""]]]


def test_unclosed_html_does_not_raise():
    (g,) = parse_tables("<table><tr><td>a<td>1.234.567<tr><td>b")
    assert g == [["a", "1.234.567"], ["b"]]


def test_html_table_inside_markdown_and_pipe_table_together():
    md = "teks\n<table><tr><td>a</td></tr></table>\n\n| b | 1 |\n|---|---|"
    assert parse_tables(md) == [[["a"]], [["b", "1"]]]


def test_amount_columns_indexes_from_right():
    grid = [["Kas", "1.234.567", "2.345.678"], ["x"]]
    assert amount_columns([grid]) == {"1234567": [1], "2345678": [0]}


def test_shifted_column_is_wrong():
    ref = [[["Kas", "1.234.567", "2.345.678"]]]
    ocr = "| Kas | 1.234.567 | 2.345.678 | |"
    s = score_columns(ref, ocr)
    assert (s.reference, s.found_in_table, s.placed_correctly) == (2, 2, 0)


def test_correct_placement():
    s = score_columns([[["Kas", "1.234.567"]]], "| Kas | 1.234.567 |")
    assert (s.reference, s.found_in_table, s.placed_correctly) == (1, 1, 1)


def test_amount_outside_table_is_not_found_in_table():
    s = score_columns([[["Kas", "1.234.567"]]], "Kas 1.234.567")
    assert (s.found_in_table, s.placed_correctly) == (0, 0)


def test_no_reference_amounts_gives_none():
    assert score_columns([[["Kas", "12"]]], "| Kas | 12 |") is None




def test_spanning_amount_scores_same_in_html_and_pipe():
    html = (
        '<table><tr><td colspan="2">Total 1.234.567</td><td>x</td></tr>'
        "<tr><td>Kas</td><td>1.234.567</td><td>2.345.678</td></tr>"
        '<tr><td colspan="2"></td><td>3.456.789</td></tr></table>'
    )
    pipe = "| | Total 1.234.567 | x |\n|---|---|---|\n| Kas | 1.234.567 | 2.345.678 |\n| | | 3.456.789 |"
    ref = [[["", "Total 1.234.567", "x"], ["Kas", "1.234.567", "2.345.678"], ["", "", "3.456.789"]]]
    for ocr in (html, pipe):
        s = score_columns(ref, ocr)
        assert (s.reference, s.found_in_table, s.placed_correctly) == (4, 4, 4)


def test_equal_amounts_in_adjacent_columns_count_twice():
    ref = [[["Modal", "5.000.000", "5.000.000"]]]
    s = score_columns(ref, "| Modal | 5.000.000 | 5.000.000 |")
    assert (s.reference, s.found_in_table, s.placed_correctly) == (2, 2, 2)
    s = score_columns(ref, "| Modal | | 5.000.000 |")
    assert (s.reference, s.found_in_table, s.placed_correctly) == (2, 1, 1)


def test_ragged_rows_do_not_crash():
    ref = [[["a", "1.234.567"], ["b"], ["c", "x", "2.345.678"]]]
    ocr = "| a | 1.234.567 |\n| b |\n| c | x | 2.345.678 |"
    assert amount_columns(ref) == {"1234567": [1], "2345678": [0]}
    s = score_columns(ref, ocr)
    assert (s.reference, s.found_in_table, s.placed_correctly) == (2, 2, 2)
    assert parse_tables('<table><tr><td colspan="3">a</td></tr><tr><td>b</td></tr></table>') == [[["", "", "a"], ["b"]]]
