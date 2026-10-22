from app.domain.quality import disagrees_with_text_layer, is_repetitive

PROSE = " ".join(f"kata{n % 40} rekening nasabah bunga" for n in range(30))


def test_repetitive_output_is_detected():
    assert is_repetitive("Saldo " * 500)
    assert not is_repetitive("Saldo " * 10)


def test_matching_text_layer_agrees_despite_markup_and_case():
    ocr = "<table><tr><td>" + PROSE.upper() + "</td></tr></table>"
    assert not disagrees_with_text_layer(ocr, PROSE)


def test_unrelated_text_layer_disagrees():
    other = " ".join(f"lain{n} sama sekali berbeda" for n in range(30))
    assert disagrees_with_text_layer(PROSE, other)


def test_short_text_layer_is_not_compared():
    assert not disagrees_with_text_layer("Laporan Keuangan", "Page 1")
