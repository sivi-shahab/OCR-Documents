"""Cross-engine agreement: amounts that several engines read identically serve as a reference-free consensus."""
from collections import Counter

from ocr_eval.numbers import NumberScore, extract_amounts, score_numbers_against


def consensus(page_outputs: dict[str, str], min_engines: int | None = None) -> Counter[str]:
    """Amounts found by at least min_engines engines (default: strict majority, at least 2).

    The multiplicity of an amount is how often at least min_engines engines each found it.
    """
    counts = [extract_amounts(t) for t in page_outputs.values() if t.strip()]
    needed = min_engines if min_engines is not None else max(2, len(counts) // 2 + 1)
    result: Counter[str] = Counter()
    for amount in set().union(*counts) if counts else ():
        per_engine = sorted((c[amount] for c in counts), reverse=True)
        if needed > 0 and len(per_engine) >= needed and per_engine[needed - 1] > 0:
            result[amount] = per_engine[needed - 1]
    return result


def score_against_consensus(engine_text: str, consensus: Counter[str]) -> NumberScore:
    return score_numbers_against(consensus, engine_text)
