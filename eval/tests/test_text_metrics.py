from ocr_eval.text_metrics import english_coverage, word_dice

_PAGE = " ".join(f"kata{i}" for i in range(60))
_ENGLISH = " ".join(["the of and for with"] * 8)  # 40 function words


def test_word_dice_identical_is_one():
    assert word_dice(_PAGE, _PAGE) == 1.0


def test_word_dice_ignores_html_tags_in_ocr():
    assert word_dice("<p>" + _PAGE + "</p>", _PAGE) == 1.0


def test_word_dice_none_below_threshold():
    assert word_dice("a b c", "a b c") is None


def test_english_coverage_zero_when_all_dropped():
    assert english_coverage("kolom indonesia saja", _ENGLISH) == 0.0


def test_english_coverage_full():
    assert english_coverage(_ENGLISH, _ENGLISH) == 1.0


def test_english_coverage_none_below_threshold():
    assert english_coverage("the of", "the of and") is None
