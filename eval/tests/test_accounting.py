from pathlib import Path

import pymupdf
import pytest

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
    s = check_totals([[["Kas", "1.000.000", "5.000"], ["Piutang", "2.000.000", ""], ["Jumlah", "3.000.000", "9.000"]]])
    assert (s.passed, s.failed, s.uncheckable) == (1, 0, 1)
    s = check_totals([[["Kas", "1.000.000", "5.000"], ["Jumlah", "1.000.000", ""]]])
    assert (s.passed, s.uncheckable) == (1, 1)


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


def test_header_year_row_and_notes_column_ignored():
    g = [["", "Catatan", "2026", "2025"], ["Kas", "4", "1.000.000", "1.000"], ["Bank", "17 18", "2.000.000", "2.000"],
         ["Jumlah", "", "3.000.000", "3.000"]]
    s = check_totals([g])
    assert (s.passed, s.failed, s.uncheckable) == (2, 0, 0)


def test_income_statement_with_non_jumlah_subtotal():
    g = [["Pendapatan", "10.000"], ["Beban", "(6.000)"], ["Laba kotor", "4.000"], ["Beban usaha", "(1.000)"],
         ["Jumlah laba", "3.000"]]
    s = check_totals([g])
    assert s.failed == 0 and s.passed == 1, s.failures


def test_balance_sheet_grand_total_after_subtotals():
    g = [["Aset lancar", ""], ["Kas", "1.000"], ["Piutang", "2.000"], ["Jumlah aset lancar", "3.000"],
         ["Aset tidak lancar", ""], ["Tanah", "5.000"], ["Jumlah aset tidak lancar", "5.000"], ["JUMLAH ASET", "8.000"]]
    s = check_totals([g])
    assert (s.passed, s.failed, s.uncheckable) == (3, 0, 0)


def test_wrong_grand_total_fails_with_closest_sum():
    g = [["Kas", "3.000"], ["Jumlah aset lancar", "3.000"], ["Tanah", "5.000"], ["Jumlah aset tidak lancar", "5.000"],
         ["JUMLAH ASET", "9.000"]]
    s = check_totals([g])
    assert s.failed == 1
    assert "JUMLAH ASET" in s.failures[0] and "8000" in s.failures[0] and "9000" in s.failures[0]


def test_dash_counts_as_zero():
    s = check_totals([[["Kas", "1.000"], ["Bank", "-"], ["Jumlah", "1.000"]]])
    assert (s.passed, s.failed) == (1, 0)


def test_total_without_amounts_above_is_uncheckable():
    s = check_totals([[["Aset", "2026"], ["Jumlah", "1.000"]]])
    assert (s.passed, s.failed, s.uncheckable) == (0, 0, 1)


def test_balance_column_count_mismatch_is_none():
    pages = {1: "Jumlah aset 5.000.000 4.000.000\nJumlah liabilitas dan ekuitas 5.000.000"}
    assert check_balance(pages) is None


_BSD = Path(__file__).resolve().parents[2] / "tests" / "LK Bumi Serpong Damai.pdf"


@pytest.mark.skipif(not _BSD.exists(), reason="BSD test PDF not available")
def test_balance_on_bsd_text_layer():
    # Text layer: label on its own line, each amount on a following line (with blank lines in between).
    with pymupdf.open(_BSD) as doc:
        pages = {n: doc[n - 1].get_text() for n in (4, 5)}
    assert check_balance(pages) is True


def test_balance_in_one_line_html_tables():
    assets = "<table><tr><td>JUMLAH ASET</td><td>81.942.880.583.837</td><td>79.268.402.201.018</td>" \
             "<td>TOTAL ASSETS</td></tr></table>"
    equity = "<table><tr><td>Kas</td><td>1.000.000</td></tr><tr><td>JUMLAH LIABILITAS DAN EKUITAS</td>" \
             "<td>81.942.880.583.837</td><td>79.268.402.201.018</td><td>TOTAL LIABILITIES AND EQUITY</td></tr></table>"
    assert check_balance({4: assets, 5: equity}) is True
    wrong = equity.replace("79.268.402.201.018", "79.268.402.201.999")
    assert check_balance({4: assets, 5: wrong}) is False


def test_balance_with_one_cell_per_line():
    assets = "Aset lain-lain\n1.000.000\nJUMLAH ASET\n81.942.880.583.837\n\n79.268.402.201.018\nTOTAL ASSETS\n"
    equity = "JUMLAH LIABILITAS DAN EKUITAS\n81.942.880.583.837\n79.268.402.201.018\nLihat catatan\n2.000.000"
    assert check_balance({4: assets, 5: equity}) is True


def test_balance_line_fallback_takes_columns_of_the_other_row():
    assets = "| JUMLAH ASET | 5.000.000 | 4.000.000 |"
    equity = "JUMLAH LIABILITAS DAN EKUITAS\n5.000.000\n4.000.000\n3.000.000"  # next row's amounts follow
    assert check_balance({1: assets, 2: equity}) is True
