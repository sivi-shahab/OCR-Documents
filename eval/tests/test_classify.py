from ocr_eval.classify import PageKind, classify_pdf


def test_digital_page(digital_pdf):
    assert classify_pdf(digital_pdf) == {1: PageKind.DIGITAL, 2: PageKind.DIGITAL}


def test_scan_page(scan_pdf):
    assert classify_pdf(scan_pdf) == {1: PageKind.SCAN, 2: PageKind.SCAN}


def test_hidden_text_is_scan_with_hidden_text(hidden_text_pdf):
    assert set(classify_pdf(hidden_text_pdf).values()) == {PageKind.SCAN_WITH_HIDDEN_TEXT}
