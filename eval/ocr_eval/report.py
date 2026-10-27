"""Excel report (Summary, Pages, Worst pages, Classification, Notes) and a markdown summary."""

from dataclasses import fields
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

from ocr_eval.score import COLUMNS, TABLE_REFERENCE_ENGINE, PageRow, Summary, load_json

_WORST_PER_GROUP = 20
_RATIOS = {"amount_recall", "misread_rate", "negative_recall", "median_dice", "column_placement",
           "column_end_to_end", "totals_pass_rate", "consensus_recall"}  # fmt: skip
_NOT_SCORED = "n/a (acuan)"


def _model_versions(metas: dict) -> dict[str, set[str]]:
    versions: dict[str, set[str]] = {}
    for (_variant, _doc, engine), meta in metas.items():
        model = " ".join(str(meta[k]) for k in ("model", "version") if meta.get(k))
        if model:
            versions.setdefault(engine, set()).add(model)
    return versions


def standard_notes(
    metas: dict, skipped: dict[str, str], excluded_reference_amounts: int, gpu_hour_rate: float
) -> list[str]:
    """The notes every report must carry; metas maps (variant, document, engine) to meta.json."""
    skipped_text = "; ".join(f"{e}: {reason}" for e, reason in sorted(skipped.items())) or "tidak ada"
    versions = _model_versions(metas)
    versions_text = "; ".join(f"{e}: {', '.join(sorted(v))}" for e, v in sorted(versions.items())) or "tidak ada"
    concurrency = sorted(
        {str(m["concurrency"]) for (_, _, e), m in metas.items() if e == "unlimited" and m.get("concurrency")}
    )
    return [
        (
            "Kolom `digital`: setiap engine menerima PDF asli seperti di produksi; Docling, MinerU, dan mungkin Azure "
            "membaca text layer sehingga skornya bisa mendekati 100%. Kolom `synthetic_scan`: halaman digital yang sama "
            "dirender ke gambar dan didegradasi (seed 20261027), sehingga semua engine hanya menerima gambar; kolom ini "
            "mengukur akurasi OCR murni. Kunci jawaban kedua kolom adalah text layer PDF asli. Nomor halaman "
            "`synthetic_scan` adalah nomor halaman asal."
        ),
        (
            "Kolom `real_scan` tidak punya kunci jawaban: `consensus_recall` (angka yang dibaca sama oleh mayoritas "
            "engine) dan `totals_pass_rate` (baris Jumlah/Total = jumlah rinciannya) hanya bukti tidak langsung. "
            "Engine pseudo `reference` menjalankan cek akuntansi pada tabel acuan dari text layer untuk memvalidasi "
            "pemeriksanya sendiri."
        ),
        (
            "Rasio dihitung dari total agregat (Σ cocok / Σ acuan), bukan rata-rata per halaman; halaman tanpa angka "
            "≥7 digit dikecualikan dari akurasi angka. `negative_recall` (acuan: angka dalam kurung; di output engine, "
            "kurung atau tanda minus yang menempel pada angka) hanya dihitung "
            "terhadap text layer (kolom `digital` dan `synthetic_scan`). `totals_pass_rate` = lolos / (lolos + gagal); "
            "tabel yang tidak bisa dicek dihitung terpisah di `totals_uncheckable`."
        ),
        (
            "`column_placement` (metrik kolom utama) = angka acuan yang ditaruh engine di kolom tabel yang sama "
            "(dihitung dari kanan) / angka acuan yang ditemukan di tabel engine; mengukur pergeseran kolom saja. "
            "`column_end_to_end` = angka di kolom yang benar / semua angka di tabel acuan; angka yang hilang atau tidak "
            "di dalam tabel ikut menurunkannya. Acuan struktur tabel adalah Docling dari PDF digital dengan isi sel dari text "
            "layer, sehingga metrik ini bergantung pada kualitas struktur tabel Docling. Sel acuan yang berisi lebih dari "
            f"satu angka (sel tergabung, kolomnya ambigu) dikecualikan: {excluded_reference_amounts} angka acuan "
            "dikecualikan."
        ),
        (
            "Docling di kolom tabel `synthetic_scan` kemungkinan bias ke atas, karena acuan struktur tabel dibuat oleh "
            f"Docling dengan model tabel yang sama. Di kolom `digital`, Docling tidak dinilai ({_NOT_SCORED})."
        ),
        (
            f"Biaya self-hosted = jam GPU × US${gpu_hour_rate:.2f}/jam (1 GPU per engine), dengan asumsi utilisasi penuh "
            "dan belum termasuk biaya operasional. Biaya API: Azure Layout US$10/1.000 halaman, Mistral OCR US$4/1.000 "
            "halaman. Waktu per halaman (untuk `pages_per_minute` dan biaya GPU) = total wall-clock semua run engine itu "
            "(termasuk run yang terputus dan dilanjutkan) dibagi jumlah halaman yang diproses selama diukur"
            + (f"; Unlimited-OCR berjalan dengan konkurensi {', '.join(concurrency)}." if concurrency else ".")
        ),
        (
            "Latensi `median_seconds`/`p95_seconds` benar-benar per halaman hanya untuk Unlimited-OCR (satu request per "
            "halaman). Engine lain memproses satu dokumen/chunk per panggilan, sehingga waktu per halamannya adalah "
            "waktu chunk/worker dibagi rata ke halamannya; median dan p95 mereka tidak menggambarkan sebaran latensi."
        ),
        (
            "`truncated` (finish_reason=length, output terpotong di batas token) adalah satu-satunya proksi untuk "
            "generasi berulang (loop); loop yang berhenti sebelum batas token tidak terdeteksi."
        ),
        "MinerU berlisensi AGPL: pemakaian produksi perlu dicek legal terlebih dulu.",
        f"Engine dilewati: {skipped_text}.",
        f"Versi model: {versions_text}.",
    ]


def _fmt(value) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _summary_values(s: Summary) -> list:
    values = []
    for f in fields(Summary):
        value = getattr(s, f.name)
        if (
            f.name in ("column_placement", "column_end_to_end")
            and s.engine == TABLE_REFERENCE_ENGINE
            and s.column == "digital"
        ):
            value = _NOT_SCORED
        values.append(value)
    return values


def _recall(r: PageRow) -> tuple[str, float, object] | None:
    """(basis, recall, score) used to rank worst pages: text-layer amounts, else consensus amounts."""
    for basis, score in (("text layer", r.number), ("consensus", r.consensus)):
        if score is not None and score.reference:
            return basis, score.matched / score.reference, score
    return None


_PAGE_HEADERS = [
    "column", "engine", "document", "page", "kind", "amount_recall", "amounts_reference", "amounts_matched",
    "amounts_misread", "amounts_dropped", "amounts_extra", "negative_reference", "negative_matched", "dice",
    "english", "columns_reference", "columns_found_in_table", "columns_placed_correctly", "totals_passed",
    "totals_failed", "totals_uncheckable", "consensus_reference", "consensus_matched", "consensus_misread",
    "consensus_dropped", "empty", "truncated", "error", "seconds",
]  # fmt: skip


def _page_values(r: PageRow) -> list:
    n, c, t, k = r.number, r.columns, r.totals, r.consensus
    return [
        r.column, r.engine, r.document, r.page, r.kind,
        n.matched / n.reference if n and n.reference else None,
        *(getattr(n, a) if n else None for a in ("reference", "matched", "misread", "dropped", "extra",
                                                  "neg_reference", "neg_matched")),
        r.dice, r.english,
        *(getattr(c, a) if c else None for a in ("reference", "found_in_table", "placed_correctly")),
        *(getattr(t, a) if t else None for a in ("passed", "failed", "uncheckable")),
        *(getattr(k, a) if k else None for a in ("reference", "matched", "misread", "dropped")),
        r.empty, r.truncated, r.error, r.seconds,
    ]  # fmt: skip


def _worst(rows: list[PageRow]) -> list[list]:
    groups: dict[tuple[str, str], list[tuple[str, float, object, PageRow]]] = {}
    for r in rows:
        ranked = _recall(r)
        if ranked:
            groups.setdefault((r.column, r.engine), []).append((*ranked, r))
    out = []
    for column, engine in sorted(groups, key=lambda key: (COLUMNS.index(key[0]), key[1])):
        worst = sorted(groups[(column, engine)], key=lambda item: (item[1], item[3].document, item[3].page))
        for basis, recall, score, r in worst[:_WORST_PER_GROUP]:
            totals_failures = "; ".join(r.totals.failures) if r.totals else ""
            out.append([column, engine, r.document, r.page, basis, recall, score.reference, score.matched,
                        "; ".join(score.misread_examples), ", ".join(score.dropped_examples), totals_failures,
                        r.error or ""])  # fmt: skip
    return out


def _classification(rows: list[PageRow], out_dir: Path) -> list[list]:
    saved = load_json(out_dir / "classification.json", None)
    if saved is not None:
        return [
            [doc, int(page), kind]
            for doc, pages in saved.items()
            for page, kind in sorted(pages.items(), key=lambda i: int(i[0]))
        ]
    seen = {(r.document, r.page, r.kind) for r in rows if r.column != "synthetic_scan"}
    return [list(item) for item in sorted(seen)]


def _cell_value(value):
    """Control characters (e.g. ANSI escapes in an error message) make openpyxl refuse the cell."""
    return ILLEGAL_CHARACTERS_RE.sub("", value) if isinstance(value, str) else value


def _sheet(book: Workbook, title: str, headers: list[str], rows: list[list], first: bool = False) -> None:
    sheet = book.active if first else book.create_sheet()
    sheet.title = title
    sheet.append(headers)
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    for row in rows:
        sheet.append([_cell_value(value) for value in row])
    sheet.freeze_panes = "A2"


def _markdown(summaries: list[Summary], notes: list[str]) -> str:
    headers = [f.name for f in fields(Summary)]
    lines = ["# Evaluasi engine OCR", "", "## Summary", "", "| " + " | ".join(headers) + " |",
             "|" + "---|" * len(headers)]  # fmt: skip
    for s in summaries:
        lines.append("| " + " | ".join(_fmt(v) for v in _summary_values(s)) + " |")
    lines += ["", "## Notes", ""] + [f"- {note}" for note in notes]
    return "\n".join(lines) + "\n"


def write_report(rows: list[PageRow], summaries: list[Summary], out_dir: Path, notes: list[str]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    book = Workbook()
    _sheet(book, "Summary", [f.name for f in fields(Summary)], [_summary_values(s) for s in summaries], first=True)
    for row in book["Summary"].iter_rows(min_row=2):
        for cell, f in zip(row, fields(Summary)):
            if f.name in _RATIOS and isinstance(cell.value, float):
                cell.number_format = "0.00%"
    _sheet(book, "Pages", _PAGE_HEADERS, [_page_values(r) for r in rows])
    _sheet(book, "Worst pages", ["column", "engine", "document", "page", "basis", "amount_recall",
                                 "amounts_reference", "amounts_matched", "misread_examples", "dropped_examples",
                                 "totals_failures", "error"], _worst(rows))  # fmt: skip
    _sheet(book, "Classification", ["document", "page", "kind"], _classification(rows, out_dir))
    _sheet(book, "Notes", ["note"], [[note] for note in notes])
    book.save(out_dir / "ocr_eval.xlsx")
    (out_dir / "summary.md").write_text(_markdown(summaries, notes), encoding="utf-8")
