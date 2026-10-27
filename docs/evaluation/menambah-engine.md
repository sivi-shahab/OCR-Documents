# Menambah engine atau metrik

Baca [arsitektur.md](arsitektur.md) lebih dulu untuk protokol `Engine`, cache, dan lane. Semua tes dijalankan dari
`eval/` dengan `.venv/bin/pytest` (lihat [eval/README.md](../../eval/README.md)); tes tidak memanggil engine
sungguhan, API berbayar, atau GPU.

## Menambah engine

### 1. Implementasikan protokol `Engine` (`ocr_eval/store.py`)

Buat kelas di `eval/ocr_eval/engines/` dengan:

- `name`: nama pendek tanpa spasi; menjadi nama folder di `runs/` dan nama engine di laporan. Jangan memakai
  `reference` (dipakai pseudo engine).
- `paid_price_per_page`: harga US$ per halaman untuk API berbayar, atau `None` untuk self-hosted. Nilai bukan `None`
  otomatis membuat engine masuk estimasi biaya dan gerbang `--yes`; biaya self-hosted dihitung dari waktu GPU.
- `available()`: kembalikan alasan singkat (bahasa Indonesia, mis. `"OCR_FOO_API_KEY kosong"`) bila engine tidak
  bisa dipakai, atau `None`. Alasan ini masuk `skipped_engines.json` dan Notes. Jangan melempar exception, dan
  jangan memakai klien async di sini (dipanggil dari dalam event loop; lihat `UnlimitedEngine.available`).
- `meta_info()`: dict yang dicatat di `meta.json` setelah setiap run. Isi `model` (dan `version` bila ada): keduanya
  ditampilkan di Notes sebagai "Versi model".
- `async def run(self, pdf, pages, *, is_scan_variant)`: async generator yang menghasilkan tepat satu `PageOutput`
  per nomor halaman di `pages` (nomor 1-based dari PDF yang diberikan, urutan bebas).
  - Kegagalan dilaporkan sebagai `PageOutput(page, "", None, error=repr(exc))`, bukan exception, supaya halaman lain
    tetap tersimpan dan halaman gagal dicoba lagi pada run berikutnya.
  - `seconds`: waktu per halaman. Bila engine memproses satu chunk per panggilan, bagi waktu chunk rata ke halamannya
    (seperti `_ChunkedEngine` dan `SubprocessEngine`).
  - `truncated=True` bila output terpotong di batas token.
  - `is_scan_variant=True` berarti input adalah scan sintetis (gambar saja); engine yang punya mode OCR paksa
    sebaiknya memakainya, seperti Docling/MinerU (`--mode ocr`).
  - Output berupa markdown; tabel sebaiknya HTML `<table>` atau tabel pipe agar bisa dinilai oleh `parse_tables`.

Pola yang bisa dipakai ulang:

- **API cloud**: turunkan dari `_ChunkedEngine` di `engines/cloud.py` dan implementasikan `_recognize`; pemotongan
  per 25 halaman, pemetaan nomor halaman, dan penanganan chunk gagal sudah ada.
- **Library dengan dependensi berat**: tulis `eval/ocr_eval/workers/<name>_worker.py` yang memakai `_common.py`
  (kontrak CLI `--pdf --pages --out --mode`, tulis `page_NNNN.md` dan `worker_meta.json` berisi `seconds_total`),
  buat `eval/requirements-<name>.txt` dan venv `eval/.venv-<name>`, lalu pakai `SubprocessEngine` lewat
  `_subprocess_engine("<name>")` di `cli.py`. Chunking, timeout, kill process group, dan cache model di
  `eval/.cache` sudah ditangani. Dokumentasikan setup venv-nya di `eval/README.md`.

### 2. Daftarkan di `cli.py`

- Tambahkan nama ke `ALL_ENGINES` (juga menjadi default `--engines`).
- Tambahkan factory di dict `factories` dalam `build_engines`.
- Bila engine memakai GPU yang sama dengan engine lain, petakan ke lane yang sama di `_SHARED_GPU` (mis.
  `"foo": "gpu1"`), supaya tidak berjalan bersamaan. Tanpa entri, engine mendapat lane sendiri dan berjalan paralel
  dengan yang lain.

Cache, resume, pencatatan `wall_seconds`/`timed_pages`, `price_per_page`, dan scoring berlaku otomatis: `score.py`
menilai setiap folder engine yang ada di `runs/`.

### 3. Tes

Tiru tes engine yang paling mirip:

| Jenis engine | Contoh tes | Teknik |
|---|---|---|
| HTTP (vLLM) | `eval/tests/test_unlimited.py` | `httpx.MockTransport` lewat parameter `transport` |
| API cloud | `eval/tests/test_cloud.py` | adapter palsu (`FakeAdapter`) atau `MockTransport`; cek penomoran ulang chunk dan chunk gagal |
| Subprocess | `eval/tests/test_subprocess_engine.py` | worker palsu (`_FAKE_WORKER`) yang dijalankan dengan `sys.executable`; cek chunk, timeout, exit code, file hilang |

Minimal: output normal menjadi markdown per halaman dengan nomor yang benar; error menjadi `PageOutput` dengan
`error` (bukan exception); `available()` memberi alasan saat konfigurasi kosong; `meta_info()` berisi `model`.
Fixture PDF (`digital_pdf`, `scan_pdf`, `hidden_text_pdf`) ada di `eval/tests/conftest.py`. Integrasi dengan CLI
(gerbang biaya, cache) sudah dites di `eval/tests/test_cli.py` dengan `FakeEngine` yang disuntikkan lewat
`main(..., engine_factory=...)`; tambahkan kasus di sana hanya bila engine baru mengubah alur `run`.

## Menambah metrik

1. **Fungsi murni** di modul yang sesuai (`numbers.py`, `text_metrics.py`, `tables.py`, `accounting.py`,
   `agreement.py`) atau modul baru. Masukan: teks acuan dan/atau markdown output; keluaran: dataclass berisi
   **pembilang dan penyebut**, bukan rasio, supaya bisa diagregasi Σ/Σ. Kembalikan `None` bila halaman tidak punya
   bahan untuk dinilai (seperti `score_numbers` tanpa angka acuan).
2. **Per halaman** di `score.py`: tambahkan field ke `PageRow`, isi di `_text_scored` (kolom `digital` dan
   `synthetic_scan`, ada kunci jawaban) dan/atau di jalur `real_scan` dalam `score_run` (tanpa kunci jawaban). Bila
   field berupa dataclass baru, tambahkan juga ke `rows_from_json` supaya `scores.json` bisa dibaca ulang.
3. **Agregasi** di `_summarize_group`: tambahkan field ke `Summary` dan hitung Σ pembilang / Σ penyebut dengan
   `_ratio`. Field `Summary` otomatis menjadi kolom di sheet Summary dan `summary.md`.
4. **Laporan** di `report.py`: tambahkan nama rasio ke `_RATIOS` (format persen), kolom per halaman ke `_PAGE_HEADERS`
   dan `_page_values`, dan bila metrik punya batasan, satu kalimat di `standard_notes`.
5. **Tes**: fungsi murni di `eval/tests/test_<modul>.py` (kasus normal, batas ambang, kasus `None`); agregasi di
   `eval/tests/test_score.py` (lihat `test_summary_aggregates_ratios`); keluaran di `eval/tests/test_report.py`.
6. **Dokumentasi**: tambahkan definisi dan rumusnya di [metodologi.md](metodologi.md) dan baris tabel metrik di
   `eval/README.md`.

Prinsip yang harus dijaga: kunci jawaban hanya dari text layer PDF digital; metrik tanpa kunci jawaban hanya di
`real_scan` dan disebut sebagai bukti tidak langsung; ketiga kolom tidak pernah digabung.
