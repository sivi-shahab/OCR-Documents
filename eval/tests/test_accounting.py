from ocr_eval.accounting import check_balance, check_totals


def test_total_row_passes():
    s = check_totals([[["Kas", "1.000.000"], ["Piutang", "2.000.000"], ["Jumlah", "3.000.000"]]])
    assert (s.passed, s.failed, s.uncheckable) == (1, 0, 0)


def test_wrong_digit_fails():
    ok = check_totals([[["Kas", "1.000.000"], ["Piutang", "2.000.000"], ["Jumlah", "3.000.001"]]])
    assert (ok.passed, ok.failed) == (1, 0)
    bad = check_totals([[["Kas", "1.000.000"], ["Piutang", "2.000.000"], ["Jumlah", "3.100.000"]]])
    assert bad.failed == 1
    assert "Jumlah" in bad.failures[0] and "100000" in bad.failures[0]


def test_parentheses_are_negative():
    s = check_totals([[["Kas", "3.000.000"], ["Rugi", "(500.000)"], ["Total / Total", "2.500.000"]]])
    assert (s.passed, s.failed) == (1, 0)


def test_missing_cell_is_uncheckable():
    s = check_totals([[["Kas", "1.000.000", "5"], ["Piutang", "2.000.000", ""], ["Jumlah", "3.000.000", "9"]]])
    assert (s.passed, s.failed, s.uncheckable) == (1, 0, 1)
    s = check_totals([[["Kas", "1.000.000"], ["Jumlah", ""]]])
    assert (s.passed, s.uncheckable) == (0, 1)


def test_spanned_label_and_second_total_resets():
    g = [["", "Kas", "1.000"], ["", "Jumlah", "1.000"], ["", "Bank", "2.000"], ["", "Sub total", "2.000"]]
    s = check_totals([g])
    assert (s.passed, s.failed) == (2, 0)


def test_heading_rows_are_skipped():
    s = check_totals([[["Aset lancar", ""], ["Kas", "1.000"], ["Jumlah", "1.000"]]])
    assert (s.passed, s.failed, s.uncheckable) == (1, 0, 0)


def test_balance_equation():
    pages = {1: "| Jumlah aset | 5.000.000 |", 2: "| Jumlah liabilitas dan ekuitas | 5.000.000 |"}
    assert check_balance(pages) is True
    pages[2] = "Total liabilities and equity 4.000.000"
    assert check_balance(pages) is False


def test_balance_ignores_subtotals_and_missing():
    assert check_balance({1: "Jumlah aset lancar 1.000.000\nJumlah aset / Total assets 5.000.000"}) is None
    pages = {1: "Jumlah aset lancar 9.000.000\nJumlah aset / Total assets 5.000.000\nTotal liabilities and equity 5.000.000"}
    assert check_balance(pages) is True
