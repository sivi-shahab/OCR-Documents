from ocr_eval.numbers import score_numbers


def test_counts_matched_misread_dropped():
    s = score_numbers("Kas 462.814.893.666 dan 1.234.567", "Kas 492.814.893.666")
    assert (s.reference, s.matched, s.misread, s.dropped) == (2, 0, 1, 1)
    assert s.misread_examples == ["462814893666 -> 492814893666"]


def test_english_separators_match():
    s = score_numbers("462.814.893.666", "462,814,893,666.00")  # decimal ,00 /.00 ignored when normalising
    assert s.matched == 1


def test_split_by_spaces_counts_as_dropped():
    s = score_numbers("462.814.893.666", "462 814 893 666")
    assert (s.matched, s.misread, s.dropped) == (0, 0, 1)


def test_negative_in_parentheses():
    s = score_numbers("(1.234.567)", "-1.234.567")
    assert (s.neg_reference, s.neg_matched) == (1, 1)


def test_page_without_amounts_is_excluded():
    assert score_numbers("Catatan atas laporan keuangan", "apa saja") is None


def test_html_tags_ignored():
    assert score_numbers("1.234.567", "<td>1.234.567</td>").matched == 1
