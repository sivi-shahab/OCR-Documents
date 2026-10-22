from app.domain.assembly import assemble_document, assemble_markdown
from app.domain.jobs import PageOutcome, PageStatus
from app.domain.models import Block, OcrResult


def page(number: int, *blocks: tuple[str, str], status: PageStatus = PageStatus.SUCCEEDED) -> PageOutcome:
    parsed = [Block(category=category, bbox=None, text=text) for category, text in blocks]
    result = OcrResult(page=number, markdown="\n\n".join(text for _, text in blocks), blocks=parsed)
    return PageOutcome(page=number, status=status, attempts=1, result=result)


def bodies(markdown: str) -> list[str]:
    """Page bodies without their markers."""
    return [part.split("-->", 1)[1].strip() for part in markdown.split("<!-- page ")[1:]]


def test_labelled_furniture_is_removed():
    markdown = assemble_markdown(
        [
            page(1, ("header", "M"), ("text", "Isi halaman satu."), ("page_number", "1")),
            page(2, ("text", "Isi halaman dua."), ("footer", "Preprint. Under review.")),
        ]
    )
    assert bodies(markdown) == ["Isi halaman satu.", "Isi halaman dua."]


WORDS = ["aset", "kredit", "bunga", "ekuitas", "dana"]


def test_repeated_edge_lines_are_removed_ignoring_digits():
    pages = [
        page(
            n,
            ("text", f"Laporan Tahunan 2025 | {n}"),
            ("text", f"Isi tentang {WORDS[n - 1]}."),
            ("text", f"_system/dashboard/stats • 2026-07-{n:02d}T10:23Z"),
        )
        for n in range(1, 6)
    ]
    assert bodies(assemble_markdown(pages)) == [f"Isi tentang {word}." for word in WORDS]


def test_lines_repeated_on_few_pages_are_kept():
    pages = [page(n, ("text", "Ringkasan.")) for n in range(1, 3)]
    assert bodies(assemble_markdown(pages)) == ["Ringkasan.", "Ringkasan."]


def test_paragraph_split_by_page_break_is_rejoined():
    markdown = assemble_markdown(
        [page(1, ("text", "Pendapatan bunga naik karena kre-")), page(2, ("text", "dit tumbuh. Lalu"))]
    )
    assert bodies(markdown) == ["Pendapatan bunga naik karena kredit tumbuh. Lalu", ""]

    markdown = assemble_markdown(
        [page(1, ("text", "Dana pihak ketiga")), page(2, ("text", "meningkat tajam."), ("title", "Bab 2"))]
    )
    assert bodies(markdown) == ["Dana pihak ketiga meningkat tajam.", "Bab 2"]


def test_finished_sentence_or_capitalised_start_is_not_joined():
    pages = [page(1, ("text", "Selesai.")), page(2, ("text", "lanjutan"))]
    assert bodies(assemble_markdown(pages)) == ["Selesai.", "lanjutan"]
    pages = [page(1, ("text", "Tanpa titik")), page(2, ("text", "Kalimat baru"))]
    assert bodies(assemble_markdown(pages)) == ["Tanpa titik", "Kalimat baru"]


def test_table_continued_on_next_page_is_merged_without_repeated_header():
    header = "<tr><td>BULAN</td><td>TOTAL</td></tr>"
    markdown = assemble_markdown(
        [
            page(1, ("table", f"<table>{header}<tr><td>Jul</td><td>69</td></tr></table>")),
            page(
                2, ("table", f"<table>{header}<tr><td>Agu</td><td>70</td></tr></table>"), ("text", "Catatan")
            ),
        ]
    )
    assert bodies(markdown) == [
        f"<table>{header}<tr><td>Jul</td><td>69</td></tr><tr><td>Agu</td><td>70</td></tr></table>",
        "Catatan",
    ]


def test_tables_with_different_columns_stay_separate():
    first = "<table><tr><td>a</td><td>b</td></tr></table>"
    second = "<table><tr><td>a</td></tr></table>"
    assert bodies(assemble_markdown([page(1, ("table", first)), page(2, ("table", second))])) == [
        first,
        second,
    ]


def test_failed_and_fallback_pages_are_marked():
    fallback = PageOutcome(
        page=2, status=PageStatus.FALLBACK, attempts=3, result=OcrResult(page=2, markdown="Teks layer")
    )
    failed = PageOutcome(page=3, status=PageStatus.FAILED, attempts=3, error="down")
    markdown = assemble_markdown([page(1, ("text", "Satu.")), fallback, failed])
    assert "<!-- page 2 (fallback) -->\n\nTeks layer" in markdown
    assert "<!-- page 3: OCR failed -->" in markdown


def test_clean_false_keeps_everything():
    markdown = assemble_markdown(
        [page(1, ("text", "kre-"), ("page_number", "1")), page(2, ("text", "dit"))], False
    )
    assert bodies(markdown) == ["kre-\n\n1", "dit"]


def test_page_spans_point_into_the_assembled_markdown():
    outcomes = [
        page(1, ("text", "Paragraf yang terpotong")),
        page(2, ("text", "di tengah halaman."), ("text", "Paragraf baru.")),
        PageOutcome(page=3, status=PageStatus.FAILED, attempts=3, error="timeout"),
    ]
    document = assemble_document(outcomes)
    assert document.markdown == assemble_markdown(outcomes)
    chunks = [document.markdown[span.start : span.end] for span in document.spans]
    assert chunks == [
        "<!-- page 1 -->\n\nParagraf yang terpotong di tengah halaman.",
        "<!-- page 2 -->\n\nParagraf baru.",
        "<!-- page 3: OCR failed -->",
    ]
    assert [(span.page, span.status) for span in document.spans] == [
        (1, PageStatus.SUCCEEDED),
        (2, PageStatus.SUCCEEDED),
        (3, PageStatus.FAILED),
    ]
