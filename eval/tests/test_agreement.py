from collections import Counter

from ocr_eval.agreement import consensus, score_against_consensus


def test_consensus_majority():
    a, b = "1234567", "7654321"
    outs = {"e1": "1.234.567", "e2": "1.234.567", "e3": "1.234.567 7.654.321", "e4": "x", "e5": "y"}
    assert consensus(outs) == Counter({a: 1})
    assert b not in consensus(outs)


def test_consensus_with_two_engines_requires_both():
    assert consensus({"e1": "1.234.567 7.654.321", "e2": "1.234.567"}) == Counter({"1234567": 1})


def test_empty_outputs_do_not_count_as_engines():
    assert consensus({"e1": "1.234.567", "e2": "1.234.567", "e3": ""}) == Counter({"1234567": 1})


def test_min_engines_override():
    assert consensus({"e1": "1.234.567", "e2": "x"}, min_engines=1) == Counter({"1234567": 1})


def test_score_against_consensus():
    s = score_against_consensus("1.234.567", Counter({"1234567": 1, "7654321": 1}))
    assert (s.matched, s.dropped) == (1, 1)
