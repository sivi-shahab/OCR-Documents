# Alat Evaluasi Engine OCR — Rencana Implementasi

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Alat CLI terpisah (`eval/`) yang menjalankan 5 engine OCR pada dokumen uji, lalu menilai akurasi angka,
teks, dan kolom tabel per jenis halaman (digital, scan buatan, scan asli). Hasilnya ditulis ke laporan Excel dan
markdown.

**Architecture:** Paket Python `ocr_eval` di `eval/`, yang memakai ulang kode `api/` (adapter vLLM, Azure, dan
Mistral) lewat instalasi editable. Docling dan MinerU berjalan sebagai subprocess di venv masing-masing dan menulis
output per halaman ke disk. Alurnya tiga tahap yang terpisah:
- `run`: memanggil engine dan meng-cache output mentahnya.
- `score`: offline, menghitung metrik dari cache.
- `report`: menulis Excel dan markdown.

Dengan pemisahan ini, logika penilaian bisa diubah tanpa memanggil ulang API berbayar.

**Tech Stack:**
- Python 3.12
- PyMuPDF (klasifikasi, render, text layer)
- Pillow + NumPy (degradasi)
- openpyxl (laporan)
- httpx dan adapter `api/` (Unlimited-OCR, Azure, Mistral)
- Docling 2.x dan MinerU 2.x (venv terpisah)
- pytest

**Spec:** `docs/superpowers/specs/2026-10-27-ocr-evaluation-design.md`

## Global Constraints

**Isolasi**
- Tidak ada perubahan di `api/app/` maupun `api/pyproject.toml`. `eval/` hanya mengimpor dari `api/`.
- Docling dan MinerU tidak boleh masuk dependensi `api/`. Masing-masing memakai venv sendiri:
  `eval/.venv-docling` dan `eval/.venv-mineru`.

**Reproduksibilitas**
- Seed degradasi tetap: `20261027`. Parameter tingkat "sedang":
  - rotasi acak ±1,0°
  - Gaussian blur radius 0,6
  - noise Gaussian σ=6 (skala 0–255)
  - grayscale
  - JPEG kualitas 60
  - render 200 DPI
- Versi model setiap engine dicatat di `meta.json`. Untuk Mistral, yang dicatat adalah field `model` dari respons
  API, bukan alias yang dikirim.

**Metrik**
- Angka yang dinilai adalah nominal ≥7 digit, dengan regex dari `numcheck2.py`:
  `\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?|\d{7,}`.
- "Salah baca" = panjang sama dan berbeda ≤2 digit.
- Angka dalam kurung `(…)` dinilai sebagai metrik terpisah. Di output OCR, kurung atau tanda minus di depan
  angka sama-sama dihitung benar.

**Engine dan GPU**
- Unlimited-OCR: vLLM di `http://localhost:8100/v1` (GPU 0), konkurensi 32, `max_tokens` 8192, render 200 DPI.
- Docling dan MinerU berjalan dengan `CUDA_VISIBLE_DEVICES=1`.

**Biaya**
- Azure Layout: US$10 per 1.000 halaman.
- Mistral standar: US$4 per 1.000 halaman.
- Engine self-hosted: jam GPU × `--gpu-hour-rate`, default US$0,95.
- Perintah `run` menampilkan estimasi biaya API berbayar dan berhenti, kecuali diberi `--yes`.

**Kredensial**
- Engine Azure atau Mistral yang key-nya kosong di `.env` dilewati dengan pesan jelas, bukan error.
- Output run masuk ke `reports/ocr_eval_<YYYY-MM-DD>/`.

**Varian**
- `original`: kedua PDF di `tests/` apa adanya.
- `synthetic`: halaman BSD yang terklasifikasi `digital`, didegradasi menjadi PDF berisi gambar saja.

**Kolom laporan**
- `digital`: halaman `original` dengan klasifikasi `digital`.
- `synthetic_scan`: semua halaman varian `synthetic`.
- `real_scan`: halaman `original` dengan klasifikasi `scan` atau `scan_with_hidden_text`.

## Review Focus

1. **Halaman digital dengan teks sangat sedikit** (halaman pemisah, halaman kosong) terklasifikasi `scan`. Ini harus
   tetap berjalan tanpa error. Halaman tanpa angka sama sekali dikecualikan dari rata-rata akurasi angka, bukan
   dihitung 0% atau 100%. Test: `test_page_without_amounts_is_excluded` (Task 1).
2. **Engine gagal atau timeout di satu halaman atau chunk** (HTTP 5xx, loop sampai `max_tokens`). Run harus
   melanjutkan halaman lain, mencatat `error` di `meta.json`, dan menghitung halaman tersebut sebagai output kosong
   di skor, bukan melewatinya diam-diam. Test: `test_failed_page_is_recorded_and_scored_empty` (Task 3).
3. **Run diulang setelah terputus.** Halaman yang sudah ada di cache tidak dipanggil ulang, terutama API berbayar.
   Test: `test_cached_pages_are_not_rerun` (Task 3).
4. **Tabel OCR dengan `colspan`/`rowspan` atau baris tidak rata.** Parser tidak boleh crash, dan indeks kolom
   dihitung dari kanan setelah ekspansi span. Test: `test_html_table_with_spans_expands` (Task 6).
5. **Format angka gaya Inggris** (`1,234,567.00`) dan angka yang terpotong spasi (`462 814 893 666`) dari engine
   lain. Format pertama harus cocok dengan kunci `462.814.893.666`. Format kedua dihitung hilang, bukan salah baca.
   Test: `test_english_separators_match` (Task 1).

---

## File Structure

```
eval/
  README.md                      # cara setup venv, menjalankan, membaca laporan
  pyproject.toml                 # paket ocr_eval; deps: pymupdf, pillow, numpy, openpyxl, httpx; pytest
  requirements-docling.txt       # docling (venv terpisah)
  requirements-mineru.txt        # mineru[core] (venv terpisah)
  .gitignore                     # .venv*/, .smoke/
  ocr_eval/
    __init__.py
    numbers.py                   # ekstraksi & skor angka (dari numcheck2.py)
    text_metrics.py              # word Dice, cakupan kolom Inggris (dari engcheck.py)
    classify.py                  # klasifikasi halaman
    degrade.py                   # pembuatan scan buatan
    store.py                     # layout cache run di disk + meta
    engines/
      __init__.py                # Engine protocol, PageOutput, registry
      unlimited.py               # vLLM langsung (seperti bench.py)
      cloud.py                   # Azure & Mistral via adapter api/, dipecah per chunk
      subprocess_engine.py       # pemanggil worker Docling/MinerU
    workers/
      docling_worker.py          # dijalankan di .venv-docling: mode engine & mode reference-tables
      mineru_worker.py           # dijalankan di .venv-mineru
    tables.py                    # parse tabel markdown/HTML, skor penempatan kolom
    accounting.py                # cek Jumlah/Total & persamaan neraca
    agreement.py                 # konsensus angka antar engine (scan asli)
    score.py                     # gabungkan semua metrik -> baris per halaman
    report.py                    # Excel + summary.md
    cli.py                       # python -m ocr_eval {classify,degrade,run,score,report,all}
  tests/
    conftest.py                  # fixture PDF kecil buatan PyMuPDF
    test_*.py                    # satu file per modul
```

---

### Task 1: Paket `eval/` dengan metrik angka dan teks

**Files:**
- Create: `eval/pyproject.toml`, `eval/.gitignore`, `eval/ocr_eval/__init__.py`, `eval/ocr_eval/numbers.py`,
  `eval/ocr_eval/text_metrics.py`, `eval/tests/conftest.py`, `eval/tests/test_numbers.py`,
  `eval/tests/test_text_metrics.py`

**Interfaces:**
- Produces:
  - `extract_amounts(text: str) -> collections.Counter[str]`: kunci berupa string digit saja. Tag HTML dibuang
    dulu.
  - `extract_negative_amounts(text: str) -> Counter[str]`: angka dalam `(…)` atau berawalan `-`/`−`.
  - `@dataclass NumberScore(reference: int, ocr: int, matched: int, misread: int, dropped: int, extra: int,
    neg_reference: int, neg_matched: int, misread_examples: list[str], dropped_examples: list[str])`
  - `score_numbers(reference_text: str, ocr_text: str) -> NumberScore | None`: `None` bila referensi tidak
    berisi angka.
  - `score_numbers_against(reference: Counter[str], ocr_text: str) -> NumberScore`
  - `word_dice(ocr_text: str, reference_text: str) -> float | None`: `None` bila referensi kurang dari 50 kata.
  - `english_coverage(ocr_text: str, reference_text: str) -> float | None`: `None` bila referensi kurang dari 30
    kata fungsi bahasa Inggris. Daftar kata sama dengan `EN` di `engcheck.py`.

- [ ] **Step 1: Setup paket.** Isi `pyproject.toml`: nama `ocr-eval`, Python ≥3.12, deps
  `pymupdf>=1.24 pillow>=10.4 numpy openpyxl>=3.1 httpx>=0.27`, extras dev `pytest pytest-asyncio`, dan
  `[tool.pytest.ini_options] asyncio_mode="auto"`. Paket `api` diinstal editable (`pip install -e ../api`),
  dicatat di README pada Task 9. `.gitignore`: `.venv*/` dan `.smoke/`.
  Run: `cd eval && python3.12 -m venv .venv && .venv/bin/pip install -e ../api -e '.[dev]'`
  Expected: instalasi sukses.

- [ ] **Step 2: Tulis test yang gagal** di `tests/test_numbers.py`:

```python
def test_counts_matched_misread_dropped():
    s = score_numbers("Kas 462.814.893.666 dan 1.234.567", "Kas 492.814.893.666")
    assert (s.reference, s.matched, s.misread, s.dropped) == (2, 0, 1, 1)
    assert s.misread_examples == ["462814893666 -> 492814893666"]

def test_english_separators_match():
    s = score_numbers("462.814.893.666", "462,814,893,666.00")  # desimal ,00 /.00 diabaikan saat menormalkan
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
```

  Normalisasi kunci: buang 2 digit desimal di akhir bila didahului pemisah yang berbeda dari pemisah ribuan
  (`1.234.567,00` → `1234567` dan `1,234,567.00` → `1234567`), lalu ambil digitnya saja.

  Test di `tests/test_text_metrics.py`:
  - `word_dice` dari teks identik = 1.0.
  - `english_coverage` = 0.0 bila semua kata bahasa Inggris hilang dari OCR.
  - Keduanya mengembalikan `None` di bawah ambang.

- [ ] **Step 3: Run** `.venv/bin/pytest tests -q`. Expected: FAIL (modul belum ada).
- [ ] **Step 4: Implementasi** `numbers.py` (logika pairing salah baca dari `numcheck2.py`) dan `text_metrics.py`
  (pakai `_words` dan `_TAG` dari `app.domain.quality`).
- [ ] **Step 5: Run** `.venv/bin/pytest tests -q`. Expected: semua PASS.
- [ ] **Step 6: Commit**: `git add eval && git commit -m "Add OCR eval package with number and text metrics"`

### Task 2: Klasifikasi halaman dan scan buatan

**Files:**
- Create: `eval/ocr_eval/classify.py`, `eval/ocr_eval/degrade.py`, `eval/tests/test_classify.py`,
  `eval/tests/test_degrade.py`
- Modify: `eval/tests/conftest.py`. Tambahkan fixture:
  - `digital_pdf`: 2 halaman teks dan tabel angka via `page.insert_text`.
  - `scan_pdf`: halaman berupa gambar penuh hasil render `digital_pdf`.
  - `hidden_text_pdf`: gambar penuh ditambah `insert_text(..., render_mode=3)`.

**Interfaces:**
- Produces:
  - `class PageKind(StrEnum): DIGITAL="digital"; SCAN="scan"; SCAN_WITH_HIDDEN_TEXT="scan_with_hidden_text"`
  - `classify_page(page: pymupdf.Page) -> PageKind`
  - `classify_pdf(path: Path) -> dict[int, PageKind]` (nomor halaman mulai dari 1)
  - `degrade_pdf(src: Path, dst: Path, pages: list[int], seed: int = 20261027) -> dict[int, int]`: memetakan
    nomor halaman di `dst` ke nomor halaman di `src`.

Aturan klasifikasi:
- Hitung `visible` = jumlah karakter dari span `page.get_texttrace()` dengan `type != 3`.
- Hitung `full_image` = ada gambar yang menutupi ≥80% luas halaman (`page.get_image_info()`).
- `visible ≥ 50` → `DIGITAL`.
- Bila tidak, dan ada teks tak terlihat ≥50 karakter serta `full_image` → `SCAN_WITH_HIDDEN_TEXT`.
- Selain itu → `SCAN`.

- [ ] **Step 1: Tulis test yang gagal.**
  - `test_digital_page`: `classify_pdf(digital_pdf) == {1: DIGITAL, 2: DIGITAL}`.
  - `test_scan_page`: hasilnya `SCAN`.
  - `test_hidden_text_is_scan_with_hidden_text`.
  - `test_degrade_is_image_only_and_deterministic`: output `degrade_pdf` diklasifikasi `SCAN` di setiap halaman.
    Dua run dengan seed sama menghasilkan byte gambar identik, dan seed berbeda menghasilkan byte berbeda.
    Mapping-nya `{1: 1, 2: 2}` untuk `pages=[1, 2]`.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_classify.py tests/test_degrade.py -q`. Expected: FAIL.
- [ ] **Step 3: Implementasi** `classify.py` dan `degrade.py`. Untuk degradasi:
  - Render halaman dengan PyMuPDF di 200 DPI, lalu jalankan dengan Pillow dalam urutan: grayscale → rotate
    (`expand=False`, latar putih) → GaussianBlur 0,6 → noise σ=6 via NumPy `default_rng(seed + page)` → JPEG q60.
  - Sisipkan ke halaman PDF baru berukuran sama dengan `insert_image`.
- [ ] **Step 4: Run** test yang sama, lalu cek dokumen nyata:
  `.venv/bin/python -c "from ocr_eval.classify import classify_pdf; from collections import Counter; [print(p, Counter(classify_pdf(p).values())) for p in ['../tests/LK Bumi Serpong Damai.pdf','../tests/PT Hutama Karya.pdf']]"`
  Expected: BSD sekitar 206 `digital` dan 1 non-digital; Hutama Karya 41 non-digital. Catat angka persisnya di
  pesan commit.
- [ ] **Step 5: Commit** `git commit -m "Add page classifier and synthetic scan generator"`

### Task 3: Cache run dan engine Unlimited-OCR

**Files:**
- Create: `eval/ocr_eval/store.py`, `eval/ocr_eval/engines/__init__.py`, `eval/ocr_eval/engines/unlimited.py`,
  `eval/tests/test_store.py`, `eval/tests/test_unlimited.py`

**Interfaces:**
- Produces:
  - `@dataclass PageOutput(page: int, markdown: str, seconds: float | None, error: str | None = None,
    truncated: bool = False)`
  - `class Engine(Protocol)`:
    - `name: str`
    - `paid_price_per_page: float | None` (`None` untuk engine self-hosted)
    - `def available(self) -> str | None`: alasan bila engine tidak bisa dipakai, `None` bila siap.
    - `async def run(self, pdf: Path, pages: list[int], *, is_scan_variant: bool) -> AsyncIterator[PageOutput]`
  - `class RunStore(root: Path)`. Layout: `root/<variant>/<doc_stem>/<engine>/page_0001.md`, ditambah `meta.json`
    berisi `{model, started, wall_seconds, concurrency, pages: {page: {seconds, error, truncated}}}`.
    - `missing_pages(variant, doc, engine, pages) -> list[int]`
    - `save_page(variant, doc, engine, out: PageOutput) -> None`
    - `load_page(variant, doc, engine, page) -> PageOutput | None`
    - `update_meta(variant, doc, engine, **fields) -> None`
    - `meta(variant, doc, engine) -> dict`
  - `async def run_engine(engine: Engine, store: RunStore, variant: str, pdf: Path, pages: list[int],
    is_scan_variant: bool) -> None`: hanya halaman yang belum ada di cache, mencatat `wall_seconds`.
  - `UnlimitedEngine(base_url: str = "http://localhost:8100/v1", concurrency: int = 32, dpi: int = 200)` dengan
    `name="unlimited"`. Memakai `VllmOcrEngine` dan `GenerationConfig("Unlimited-OCR", 8192, 35, 128, 1024)`,
    lalu `parse_blocks` dan `to_markdown` dari `app.domain.parser`. `truncated` diambil dari `EngineOutput`.
    `available()` mengecek `is_healthy()`.

- [ ] **Step 1: Tulis test yang gagal.**
  - `test_cached_pages_are_not_rerun`: engine palsu yang menghitung panggilan, `run_engine` dua kali untuk halaman
    `[1, 2]` → panggilan kedua 0 kali.
  - `test_failed_page_is_recorded_and_scored_empty`: engine palsu menghasilkan `PageOutput(2, "", None,
    error="HTTP 500")` → `load_page(...,2).error == "HTTP 500"`, `.markdown == ""`, dan `meta` memuat errornya.
  - `test_unlimited_builds_markdown`: `UnlimitedEngine` dengan `httpx.MockTransport` yang mengembalikan konten
    `<|det|>text [0,0,10,10]<|/det|>Halo` dan `finish_reason="stop"` → markdown `"Halo"`.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_store.py tests/test_unlimited.py -q`. Expected: FAIL.
- [ ] **Step 3: Implementasi.** Exception per halaman ditangkap menjadi `PageOutput(error=repr(exc))`. Satu
  halaman gagal tidak boleh menghentikan halaman lain (`asyncio.Semaphore(concurrency)`).
- [ ] **Step 4: Run** test yang sama. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "Add run cache and Unlimited-OCR eval engine"`

### Task 4: Engine Azure dan Mistral

**Files:**
- Create: `eval/ocr_eval/engines/cloud.py`, `eval/tests/test_cloud.py`

**Interfaces:**
- Consumes: `Engine`, `PageOutput` (Task 3); `AzureDocumentIntelligence`, `AzureConfig` dari
  `app.infrastructure.azure_document_intelligence`; `MistralOcr`, `MistralConfig` dari
  `app.infrastructure.mistral_ocr`; `Settings` dari `app.core.config` (membaca `.env` root).
- Produces:
  - `AzureEngine(settings: Settings, chunk_pages: int = 25)` dengan `name="azure"` dan
    `paid_price_per_page=0.010`.
  - `MistralEngine(settings: Settings, chunk_pages: int = 25)` dengan `name="mistral"` dan
    `paid_price_per_page=0.004`.
  - `split_pdf(pdf: Path, pages: list[int], chunk_pages: int) -> list[tuple[bytes, list[int]]]`: setiap chunk
    berisi byte PDF dan nomor halaman aslinya.
  - `available()` mengembalikan `"OCR_AZURE_DI_API_KEY kosong"` atau `"OCR_MISTRAL_API_KEY kosong"` bila key
    kosong.

- [ ] **Step 1: Tulis test yang gagal.**
  - `test_split_pdf_maps_pages`: 5 halaman, `pages=[1,3,4,5]`, `chunk_pages=2` → nomor halaman per chunk
    `[[1,3],[4,5]]`.
  - `test_chunk_results_are_renumbered`: adapter palsu mengembalikan `OcrResult(page=1..2)` per chunk →
    `PageOutput.page` sesuai nomor asli, `seconds` = durasi chunk dibagi jumlah halaman.
  - `test_missing_key_reports_unavailable`: `Settings(mistral_api_key="")` → `available()` berisi
    `"OCR_MISTRAL_API_KEY"`.
  - `test_chunk_failure_marks_all_its_pages`: adapter palsu melempar `EngineResponseError` → setiap halaman di
    chunk itu memiliki `error`.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_cloud.py -q`. Expected: FAIL.
- [ ] **Step 3: Implementasi.** Adapter dibuat dengan `httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10))`.
  Chunk dikirim berurutan, bukan paralel, untuk menghindari rate limit.
  Mistral: catat `model` dari respons ke meta. Adapter saat ini tidak mengembalikannya, jadi
  `MistralEngine` memanggil `to_results` sendiri. Lakukan POST dengan payload yang sama seperti
  `MistralOcr.recognize`, lalu ambil `body["model"]`, tanpa mengubah `api/`.
- [ ] **Step 4: Run** test yang sama. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "Add Azure and Mistral eval engines"`

### Task 5: Worker Docling dan MinerU

**Files:**
- Create: `eval/requirements-docling.txt` (`docling>=2.40`), `eval/requirements-mineru.txt` (`mineru[core]>=2.1`),
  `eval/ocr_eval/workers/docling_worker.py`, `eval/ocr_eval/workers/mineru_worker.py`,
  `eval/ocr_eval/engines/subprocess_engine.py`, `eval/tests/test_subprocess_engine.py`

**Interfaces:**
- Kontrak worker (CLI; berjalan di venv-nya sendiri dan hanya mengimpor stdlib + library-nya):
  `python <worker>.py --pdf PATH --pages 1,2,3 --out DIR --mode {auto,ocr,reference-tables}`.
  - Worker menulis `DIR/page_0001.md` per halaman dan `DIR/worker_meta.json` berisi
    `{model, version, seconds_total}`.
  - Mode `reference-tables` (hanya Docling) menulis `DIR/tables.json`, berisi
    `[{"page": int, "grid": [[str, ...], ...]}]` dari `table.data.grid` (`cell.text`).
- Produces:
  - `SubprocessEngine(name: str, python: Path, worker: Path, gpu: str = "1")` dengan `name` `"docling"` atau
    `"mineru"`. Memanggil worker dengan `mode="ocr"` bila `is_scan_variant`, selain itu `"auto"`. Env
    `CUDA_VISIBLE_DEVICES=gpu`. `seconds` per halaman = total dibagi jumlah halaman.
  - `available()` mengembalikan alasan bila interpreter venv tidak ada.
  - `reference_tables(pdf: Path, out: Path) -> list[dict]`: menjalankan Docling dengan mode `reference-tables`.

Konfigurasi worker:
- **Docling `auto`**: `PdfPipelineOptions(do_table_structure=True, do_ocr=True)` dengan
  `EasyOcrOptions(lang=["id","en"])`. Bitmap di-OCR, text layer dipakai.
- **Docling `ocr`**: sama, ditambah `force_full_page_ocr=True`.
- **Docling `reference-tables`**: `do_ocr=False`.
- **Docling markdown per halaman**: `doc.export_to_markdown(page_no=n)`.
- **MinerU**: `mineru -p PDF -o TMP -b pipeline -m {auto|ocr}` dengan bahasa model OCR Latin. Cek nama opsinya
  dengan `mineru --help`; pilih nilai untuk teks Latin/Indonesia dan catat di `worker_meta.json`. Pecah
  `*_content_list.json` per `page_idx` menjadi markdown: tabel memakai `table_body` (HTML), teks memakai `text`.

- [ ] **Step 1: Tulis test yang gagal** `test_subprocess_engine.py` dengan worker palsu (skrip Python kecil di
  `tmp_path` yang menulis `page_0001.md` berisi `"ok"`):
  - `SubprocessEngine` menghasilkan `PageOutput(1, "ok", ...)`.
  - Mode `ocr` dipakai bila `is_scan_variant=True`.
  - Worker yang keluar dengan kode ≠0 menghasilkan `error` di setiap halaman.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_subprocess_engine.py -q`. Expected: FAIL.
- [ ] **Step 3: Implementasi** `subprocess_engine.py` dan kedua worker.
- [ ] **Step 4: Run** test. Expected: PASS.
- [ ] **Step 5: Smoke test nyata.** Buat venv, lalu jalankan worker pada 2 halaman BSD:
  `python3.12 -m venv .venv-docling && .venv-docling/bin/pip install -r requirements-docling.txt`
  (lakukan hal yang sama untuk MinerU), lalu:
  `CUDA_VISIBLE_DEVICES=1 .venv-docling/bin/python ocr_eval/workers/docling_worker.py --pdf "../tests/LK Bumi Serpong Damai.pdf" --pages 10,11 --out .smoke/docling --mode reference-tables`
  Expected: `tables.json` berisi minimal satu tabel dengan angka nominal. Jalankan juga `--mode ocr` untuk Docling
  dan MinerU. Expected: markdown tidak kosong.
- [ ] **Step 6: Commit** `git commit -m "Add Docling and MinerU eval workers"`

### Task 6: Parsing tabel dan penempatan kolom

**Files:**
- Create: `eval/ocr_eval/tables.py`, `eval/tests/test_tables.py`

**Interfaces:**
- Consumes: `extract_amounts` (Task 1).
- Produces:
  - `Grid = list[list[str]]`
  - `parse_tables(markdown: str) -> list[Grid]`: mengenali `<table>` HTML (ekspansi `colspan`/`rowspan`, sel
    kosong diisi `""`) dan tabel pipe markdown. Baris pemisah `|---|` dibuang.
  - `amount_columns(grids: list[Grid]) -> dict[str, list[int]]`: memetakan digit angka ke daftar indeks kolom
    **dari kanan** (`width - 1 - col`, dengan `width` = panjang baris terpanjang dalam tabel).
  - `@dataclass ColumnScore(reference: int, found_in_table: int, placed_correctly: int)`
  - `score_columns(reference: list[Grid], ocr_markdown: str) -> ColumnScore | None`: `None` bila referensi tidak
    punya angka di tabel. Benar bila salah satu kemunculan angka di OCR punya indeks-dari-kanan yang sama dengan
    referensi.

- [ ] **Step 1: Tulis test yang gagal.**

```python
def test_html_table_with_spans_expands():
    md = '<table><tr><td colspan="2">Aset</td><td>2025</td></tr><tr><td>Kas</td><td>3</td><td>1.234.567</td></tr></table>'
    (g,) = parse_tables(md)
    assert g == [["Aset", "Aset", "2025"], ["Kas", "3", "1.234.567"]]

def test_pipe_table():
    assert parse_tables("| a | 1.234.567 |\n|---|---|\n| b | 7.654.321 |") == [[["a", "1.234.567"], ["b", "7.654.321"]]]

def test_shifted_column_is_wrong():
    ref = [[["Kas", "1.234.567", "2.345.678"]]]
    ocr = "| Kas | 1.234.567 | 2.345.678 | |"          # bergeser satu kolom ke kiri dari kanan
    s = score_columns(ref, ocr)
    assert (s.reference, s.found_in_table, s.placed_correctly) == (2, 2, 0)

def test_amount_outside_table_is_not_found_in_table():
    s = score_columns([[["Kas", "1.234.567"]]], "Kas 1.234.567")
    assert (s.found_in_table, s.placed_correctly) == (0, 0)
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_tables.py -q`. Expected: FAIL.
- [ ] **Step 3: Implementasi** dengan `html.parser.HTMLParser` dari stdlib, tanpa dependensi baru.
- [ ] **Step 4: Run** test. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "Add table parsing and column placement score"`

### Task 7: Cek akuntansi dan konsensus antar engine

**Files:**
- Create: `eval/ocr_eval/accounting.py`, `eval/ocr_eval/agreement.py`, `eval/tests/test_accounting.py`,
  `eval/tests/test_agreement.py`

**Interfaces:**
- Consumes: `Grid`, `parse_tables` (Task 6); `extract_amounts`, `NumberScore`, `score_numbers_against` (Task 1).
- Produces:
  - `@dataclass CheckSummary(passed: int, failed: int, uncheckable: int, failures: list[str])`
  - `check_totals(grids: list[Grid]) -> CheckSummary`. Untuk setiap baris yang label pertamanya cocok dengan
    `^(jumlah|total|sub ?total)\b` (case-insensitive), per kolom angka: jumlah nilai baris-baris di atasnya sejak
    baris total sebelumnya (atau awal tabel). Kurung dibaca negatif. Lolos bila `|selisih| ≤ 1`. Kolom yang tidak
    punya angka di salah satu baris rincian atau di baris total dihitung `uncheckable`.
  - `check_balance(markdown_pages: dict[int, str]) -> bool | None`. Cari angka pertama pada baris berlabel
    `jumlah aset|total assets` dan `jumlah liabilitas dan ekuitas|total liabilities and equity`. Bandingkan untuk
    setiap kolom angka. `None` bila salah satu baris tidak ditemukan.
  - `consensus(page_outputs: dict[str, str], min_engines: int | None = None) -> collections.Counter[str]`:
    angka yang muncul di output minimal `min_engines` engine. Default `max(2, n // 2 + 1)` dengan `n` = jumlah
    engine yang punya output.
  - `score_against_consensus(engine_text: str, consensus: Counter[str]) -> NumberScore`: memakai ulang
    `score_numbers_against`.

- [ ] **Step 1: Tulis test yang gagal.**
  - `test_total_row_passes`: grid `[["Kas","1.000.000"],["Piutang","2.000.000"],["Jumlah","3.000.000"]]` →
    `passed=1`.
  - `test_wrong_digit_fails`: total `3.000.001` dengan selisih 1 → lolos. Total `3.100.000` → `failed=1`, dan
    `failures[0]` memuat label serta selisihnya.
  - `test_parentheses_are_negative`: `(500.000)` dalam rincian dikurangkan.
  - `test_missing_cell_is_uncheckable`.
  - `test_balance_equation`: dua halaman markdown dengan `Jumlah aset 5.000.000` dan
    `Jumlah liabilitas dan ekuitas 5.000.000` → `True`.
  - `test_consensus_majority`: 5 engine, angka `A` muncul di 3 engine dan `B` di 1 → `consensus == Counter({A: 1})`.
  - `test_consensus_with_two_engines_requires_both`.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_accounting.py tests/test_agreement.py -q`. Expected: FAIL.
- [ ] **Step 3: Implementasi.**
- [ ] **Step 4: Run** test. Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "Add accounting checks and cross-engine consensus"`

### Task 8: Penilaian, laporan, dan CLI

**Files:**
- Create: `eval/ocr_eval/score.py`, `eval/ocr_eval/report.py`, `eval/ocr_eval/cli.py`,
  `eval/ocr_eval/__main__.py`, `eval/tests/test_score.py`, `eval/tests/test_report.py`, `eval/tests/test_cli.py`

**Interfaces:**
- Consumes: semua yang di atas.
- Produces:
  - `@dataclass PageRow(column: str, document: str, page: int, engine: str, kind: str, number: NumberScore | None,
    dice: float | None, english: float | None, columns: ColumnScore | None, totals: CheckSummary | None,
    consensus: NumberScore | None, empty: bool, truncated: bool, error: str | None, seconds: float | None)`.
    Nilai `column` ∈ `{"digital","synthetic_scan","real_scan"}`.
  - `score_run(run_dir: Path, pdfs: list[Path]) -> list[PageRow]`. Membaca `classification.json`,
    `synthetic_map.json`, dan `reference_tables.json` dari `run_dir`. Aturan per kolom:
    - **`digital`** dan **`synthetic_scan`**: angka, Dice, dan Inggris dinilai terhadap text layer halaman asal.
      Kolom tabel dinilai terhadap tabel referensi halaman asal; Docling di kolom `digital` diberi `columns=None`.
    - **`real_scan`**: `consensus` dan `totals`.
    - **Validasi pemeriksa akuntansi**: `totals` juga dihitung untuk engine pseudo `"reference"` (grid dari tabel
      referensi BSD).
  - `@dataclass Summary(column: str, engine: str, pages: int, amount_recall: float | None, misread_rate: float | None,
    negative_recall: float | None, median_dice: float | None, english_skipped_pages: int,
    column_accuracy: float | None, totals_pass_rate: float | None, consensus_recall: float | None,
    empty_pages: int, truncated_pages: int, error_pages: int, pages_per_minute: float | None,
    median_seconds: float | None, p95_seconds: float | None, cost_usd: float | None)`
  - `summarize(rows: list[PageRow], metas: dict, gpu_hour_rate: float) -> list[Summary]`. Rasio dihitung dari
    total agregat (misalnya Σmatched/Σreference), bukan rata-rata per halaman. Halaman dengan `number=None`
    dikecualikan.
  - `write_report(rows, summaries, out_dir: Path, notes: list[str]) -> None`. Menulis `ocr_eval.xlsx` dengan sheet
    `Summary`, `Pages`, `Worst pages` (20 halaman dengan `amount_recall` terendah per engine × kolom, berisi contoh
    salah baca/hilang), `Classification`, dan `Notes`. Juga menulis `summary.md` (tabel Summary dan Notes).
  - `Notes` wajib memuat:
    - beda makna kolom `digital` dan `synthetic_scan`;
    - bias Docling pada kolom tabel `synthetic_scan`;
    - asumsi biaya GPU (utilisasi penuh, tanpa biaya operasional);
    - lisensi MinerU AGPL yang perlu dicek legal untuk produksi;
    - engine yang dilewati beserta alasannya;
    - versi model setiap engine.
  - CLI: `python -m ocr_eval all --out ../reports/ocr_eval_2026-10-27 --pdf "../tests/LK Bumi Serpong Damai.pdf" --pdf "../tests/PT Hutama Karya.pdf" [--engines unlimited,azure,mistral,docling,mineru] [--gpu-hour-rate 0.95] [--yes]`.
    Subcommand `classify`, `degrade`, `run`, `score`, dan `report` bisa dijalankan sendiri-sendiri. `all`
    menjalankan semuanya berurutan.
  - Artefak antar-tahap di `--out`:
    - `classify` menulis `classification.json` (`{doc_stem: {page: kind}}`).
    - `degrade` menulis `synthetic/<doc_stem>.pdf` dan `synthetic_map.json` (`{doc_stem: {synthetic_page:
      original_page}}`).
    - `run` juga menjalankan `reference_tables` sekali per dokumen yang punya halaman digital dan menulis
      `reference_tables.json` (`{doc_stem: [{page, grid}]}`). Hasilnya di-cache: tidak dijalankan ulang bila
      file sudah ada.
    - Cache engine ada di `--out/runs/` (`RunStore`).
  - `run` mencetak tabel estimasi biaya per engine berbayar (halaman yang belum di-cache × harga). Tanpa `--yes`,
    perintah keluar dengan kode 2 tanpa memanggil API berbayar. Engine self-hosted tetap berjalan.
  - Unlimited-OCR berjalan bersamaan dengan Docling atau MinerU (beda GPU). Docling dan MinerU berjalan bergantian
    karena berbagi GPU 1.

- [ ] **Step 1: Tulis test yang gagal.**
  - `test_score_run_columns`: gunakan fixture run buatan di `tmp_path`, berisi 1 halaman digital, 1 halaman
    sintetis, dan 1 halaman scan, dengan output dua engine palsu. Hasil yang diharapkan:
    - `PageRow.column` benar;
    - halaman scan punya `consensus` dan tidak punya `number`;
    - Docling di kolom digital punya `columns is None`.
  - `test_summary_aggregates_ratios`: dua halaman 9/10 dan 1/10 menghasilkan `amount_recall == 0.5`.
  - `test_failed_page_is_recorded_and_scored_empty`: halaman dengan `error` menghasilkan `error_pages == 1`, dan
    angkanya dihitung hilang.
  - `test_report_files`: `write_report` menghasilkan `ocr_eval.xlsx` dengan kelima sheet (cek via
    `openpyxl.load_workbook`) dan `summary.md` yang memuat setiap teks wajib di `Notes`.
  - `test_run_without_yes_stops_before_paid_calls`: CLI `run` dengan engine palsu berbayar keluar dengan kode 2,
    dan engine palsu tidak dipanggil.
- [ ] **Step 2: Run** `.venv/bin/pytest tests -q`. Expected: test baru FAIL.
- [ ] **Step 3: Implementasi.**
- [ ] **Step 4: Run** `.venv/bin/pytest tests -q`. Expected: semua PASS.
- [ ] **Step 5: Commit** `git commit -m "Add scoring, report writer and eval CLI"`

### Task 9: README dan run evaluasi sungguhan

**Files:**
- Create: `eval/README.md`. Isinya dalam bahasa Indonesia, mengikuti gaya `api/README.md`:
  - setup tiga venv;
  - menyalakan vLLM (`cd api && docker compose up -d vllm`);
  - mengisi key di `.env`;
  - perintah `all`;
  - arti setiap kolom dan metrik;
  - peringatan bahwa Azure dan Mistral memproses di luar Indonesia, dan persetujuan hanya untuk dokumen di
    `tests/`.
- Output: `reports/ocr_eval_2026-10-27/`. Tidak di-commit, sama seperti laporan uji sebelumnya. Tanyakan pengguna
  bila mereka ingin hasilnya di-commit.

- [ ] **Step 1: Prasyarat.**
  - Jalankan `cd api && docker compose up -d vllm`, lalu tunggu `curl -sf localhost:8100/health`.
  - Cek key: `grep -c '^OCR_AZURE_DI_API_KEY=.\+' .env` dan `grep -c '^OCR_MISTRAL_API_KEY=.\+' .env`.
  - Bila key kosong, **tanyakan pengguna** apakah mau mengisinya dulu atau menjalankan tanpa engine tersebut.
- [ ] **Step 2: Run tanpa `--yes`.**
  `cd eval && .venv/bin/python -m ocr_eval all --out ../reports/ocr_eval_2026-10-27 --pdf "../tests/LK Bumi Serpong Damai.pdf" --pdf "../tests/PT Hutama Karya.pdf"`
  Expected: estimasi biaya tercetak, sekitar US$4–6 untuk Azure dan sekitar US$2 untuk Mistral. Tunjukkan ke
  pengguna dan minta konfirmasi.
- [ ] **Step 3: Run dengan `--yes`** setelah pengguna setuju. Expected: semua engine yang tersedia selesai.
  `meta.json` setiap engine terisi `model` dan `wall_seconds`.
- [ ] **Step 4: Validasi kewarasan sebelum melapor.**
  - Unlimited-OCR di kolom `digital` harus memiliki `amount_recall` dekat 0,942 (uji 23 Oktober). Selisih lebih
    dari 3 poin berarti ada bug penilaian; selidiki sebelum lanjut.
  - Engine `reference` harus memiliki `totals_pass_rate` tinggi. Bila rendah, heuristik akuntansinya salah.
- [ ] **Step 5: Commit** `git add eval/README.md && git commit -m "Document OCR evaluation tool"`
- [ ] **Step 6: Laporkan** ke pengguna isi tabel Summary dari `summary.md`, termasuk keterbatasannya.
