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


def test_reference_negatives_are_parenthesised_only():
    from ocr_eval.numbers import extract_reference_negatives

    text = "Rugi (1.234.567)\n-\n  \n1.288.047.900.733\nModal dasar - 40.000.000.000\n-7.654.321"
    assert dict(extract_reference_negatives(text)) == {"1234567": 1}


def test_ocr_minus_must_touch_digits():
    from ocr_eval.numbers import extract_negative_amounts

    assert not extract_negative_amounts("\n-\n  \n1.288.047.900.733")
    assert not extract_negative_amounts("Modal dasar - 40.000.000.000")
    assert not extract_negative_amounts("2020-1.234.567")  # glued to a word/number: a range, not a sign
    assert dict(extract_negative_amounts("(1.234.567) dan −7.654.321 dan -9.999.999")) == {
        "1234567": 1, "7654321": 1, "9999999": 1,
    }


def test_minus_in_reference_is_not_a_negative():
    s = score_numbers("Modal dasar - 40.000.000.000\n-\n1.288.047.900.733", "-40.000.000.000")
    assert s.neg_reference == 0 and s.neg_matched == 0
