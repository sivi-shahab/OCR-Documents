# OCR Evaluation

Alat evaluasi mandiri untuk membandingkan engine OCR pada dokumen keuangan: Unlimited-OCR (vLLM), Azure Document
Intelligence, Mistral OCR, Docling, dan MinerU. Alat ini hanya membaca dari `api/` (memakai adapter Azure/Mistral dan
`Settings`); tidak ada file di `api/` yang diubah.

Arsitektur, definisi metrik, dan cara menambah engine: [docs/evaluation/README.md](../docs/evaluation/README.md).

Alur: `classify` (halaman digital / scan / scan dengan text layer tersembunyi) → `degrade` (halaman digital dirender
menjadi scan sintetis) → `run` (semua engine, hasil di-cache per halaman) → `score` → `report`. Perintah `all`
menjalankan semuanya.

## Setup

Ada tiga virtualenv terpisah supaya dependensi Docling dan MinerU tidak bercampur dengan yang lain. Semua dijalankan
dari folder `eval/`.

Python 3.12 mungkin tidak ada di PATH; CPython 3.12 mana pun cukup (mis. yang diinstal oleh `uv`). Di bawah,
`PY312` adalah path interpreter itu.

Disk root kecil, jadi semua cache pip dan model diarahkan ke `eval/.cache`. Set variabel ini di shell **sebelum**
`pip install` (saat evaluasi berjalan, `ocr_eval` mengaturnya sendiri untuk worker Docling dan MinerU):

```bash
cd eval
export PIP_CACHE_DIR=$PWD/.cache/pip HF_HOME=$PWD/.cache/hf MODELSCOPE_CACHE=$PWD/.cache/modelscope \
       XDG_CACHE_HOME=$PWD/.cache/xdg TORCH_HOME=$PWD/.cache/torch EASYOCR_MODULE_PATH=$PWD/.cache/easyocr
PY312=$(ls ~/.local/share/uv/python/cpython-3.12*/bin/python3.12 | head -1)   # atau interpreter 3.12 lain
```

**1. `eval/.venv` (alat utama)**

```bash
$PY312 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pip install pydantic-settings
echo "$PWD/../api" > .venv/lib/python3.12/site-packages/api_path.pth
```

`pip install -e ../api` gagal karena `api/` memiliki dua package top-level, jadi `api/` dibuat dapat di-import lewat
file `.pth` yang menunjuk ke direktori `api/` repo (perintah `echo` di atas), ditambah `pydantic-settings`.

**2. `eval/.venv-docling`** (Docling, juga dipakai untuk tabel acuan)

```bash
$PY312 -m venv .venv-docling
.venv-docling/bin/pip install -r requirements-docling.txt     # docling[easyocr]
```

**3. `eval/.venv-mineru`** (MinerU)

```bash
$PY312 -m venv .venv-mineru
.venv-mineru/bin/pip install -r requirements-mineru.txt       # mineru[core]>=2.1,<4
```

MinerU dibatasi `<4` karena 4.x memakai CLI yang berbeda. Venv yang tidak ada membuat engine terkait dilewati dan
dicatat di Notes laporan.

Cek: `.venv/bin/python -m ocr_eval --help` dan `.venv/bin/pytest`.

## Prasyarat saat menjalankan

**GPU.** vLLM (Unlimited-OCR) di GPU 0; Docling dan MinerU di GPU 1 (bergantian, bukan bersamaan).

```bash
cd api && docker compose up -d vllm
curl -sf localhost:8100/health     # tunggu sampai berhasil
```

**Key API.** Isi di `.env` di root repo (salin dari `.env.example`):

```
OCR_AZURE_DI_ENDPOINT=https://<resource-name>.cognitiveservices.azure.com
OCR_AZURE_DI_API_KEY=...
OCR_MISTRAL_API_KEY=...
```

Engine dengan key kosong dilewati dan alasannya tercantum di Notes. Begitu juga engine yang tidak siap (vLLM tidak
merespons, venv tidak ada).

> **Privasi data.** Azure dan Mistral memproses dokumen di luar Indonesia. Persetujuan yang ada hanya mencakup dua
> laporan keuangan publik di `tests/` (`LK Bumi Serpong Damai.pdf`, `PT Hutama Karya.pdf`). Jangan memakai alat ini
> dengan engine tersebut pada dokumen nasabah tanpa keputusan terpisah; untuk dokumen sensitif gunakan
> `--engines unlimited,docling,mineru`.

## Menjalankan

```bash
cd eval
.venv/bin/python -m ocr_eval all \
  --out ../reports/ocr_eval_2026-10-27 \
  --pdf "../tests/LK Bumi Serpong Damai.pdf" --pdf "../tests/PT Hutama Karya.pdf"
```

Pakai folder `--out` yang **tetap** (seperti di atas), jangan `$(date +%F)`: memakai ulang folder yang sama
melanjutkan run dari cache. Halaman yang sudah tersimpan di `runs/` tidak diproses (dan tidak ditagih) ulang; hanya
halaman yang belum ada atau yang sebelumnya gagal yang dijalankan lagi. Folder baru berarti semua halaman diproses
dan API berbayar ditagih lagi dari awal. Docling dan MinerU dijalankan per 25 halaman (batas waktu 30 menit per
chunk), jadi crash atau timeout paling banyak menghilangkan satu chunk.

Gerbang `--yes`: tanpa `--yes`, `run` mencetak estimasi biaya API berbayar lalu keluar dengan kode 2 **sebelum
menjalankan engine apa pun** (termasuk engine self-hosted) — tetapi hanya jika ada engine berbayar yang siap (key
terisi) dan masih punya halaman yang belum di-cache atau gagal. Jika semua halaman berbayar sudah di-cache, atau
tidak ada engine berbayar yang siap, `run` langsung berjalan tanpa `--yes`. `all` tetap menjalankan `classify` dan
`degrade` lebih dulu (cepat, tanpa engine) sebelum gerbang ini. Ulangi dengan `--yes` setelah estimasi disetujui.

| Perintah | Fungsi |
|---|---|
| `classify` | Klasifikasi halaman, tulis `classification.json` |
| `degrade` | Buat PDF scan sintetis di `synthetic/` dari halaman digital |
| `run` | Jalankan engine dan tabel acuan Docling, cache di `runs/` |
| `score` | Hitung skor per halaman, tulis `scores.json` |
| `report` | Tulis `ocr_eval.xlsx` dan `summary.md` dari `scores.json` |
| `all` | Semua langkah di atas berurutan |

Opsi umum: `--out DIR` (wajib), `--pdf PDF` (boleh berulang; wajib kecuali `report`), `--engines a,b`
(default `unlimited,azure,mistral,docling,mineru`), `--gpu-hour-rate` (US$/jam GPU, default 0.95), `--yes`.
Contoh tanpa engine berbayar: `--engines unlimited,docling,mineru`.

Hasil ada di folder `--out` (mis. `reports/ocr_eval_2026-10-27/`) dan **tidak di-commit** (sama seperti laporan uji
sebelumnya). Isinya: `ocr_eval.xlsx` (sheet Summary, Pages, Worst pages, Classification, Notes), `summary.md`,
`scores.json`, `classification.json`, `synthetic_map.json` (halaman scan sintetis → halaman asal), `skipped_engines.json`,
`reference_tables.json`, `synthetic/`, dan `runs/` (cache per halaman beserta `meta.json`, termasuk `wall_seconds`
dan `timed_pages` yang terakumulasi lintas run yang dilanjutkan).

## Kolom laporan

| Kolom | Isi | Kunci jawaban |
|---|---|---|
| `digital` | Halaman digital pada PDF asli, seperti di produksi. Docling, MinerU, dan mungkin Azure membaca text layer, jadi skornya bisa mendekati 100% | Text layer PDF |
| `synthetic_scan` | Halaman digital yang sama dirender menjadi gambar dan didegradasi (seed 20261027); semua engine hanya menerima gambar sehingga mengukur akurasi OCR murni. Nomor halaman = halaman asal | Text layer PDF |
| `real_scan` | Halaman scan asli | Tidak ada; hanya bukti tidak langsung (`consensus_recall`, `totals_pass_rate`) |

Engine pseudo `reference` menjalankan cek akuntansi pada tabel acuan dari text layer, untuk memvalidasi pemeriksanya
sendiri; nilai `totals_pass_rate`-nya harus tinggi.

## Metrik Summary

Rasio dihitung dari total agregat (Σ cocok / Σ acuan), bukan rata-rata per halaman. Halaman tanpa angka ≥7 digit
dikecualikan dari akurasi angka.

| Metrik | Arti |
|---|---|
| `pages` | Jumlah halaman yang dinilai pada kolom itu |
| `amount_recall` | Angka acuan yang ditemukan di output engine / semua angka acuan |
| `misread_rate` | Angka acuan yang terbaca salah (ada angka serupa tetapi tidak sama) / semua angka acuan |
| `negative_recall` | Recall khusus angka negatif: acuan = angka dalam kurung di text layer; di output engine, kurung atau tanda minus yang menempel pada angka (`-1.234.567`) dihitung. Hanya `digital` dan `synthetic_scan` |
| `median_dice` | Median kemiripan kata (koefisien Dice) halaman terhadap text layer |
| `english_skipped_pages` | Halaman yang tidak dinilai cakupan teks Inggrisnya |
| `column_placement` | Metrik kolom utama: angka acuan yang ditaruh di kolom tabel yang sama (dihitung dari kanan) / angka acuan yang ditemukan di tabel engine. Hanya mengukur pergeseran kolom |
| `column_end_to_end` | Angka di kolom yang benar / semua angka di tabel acuan; angka yang hilang atau tidak ada di dalam tabel ikut menurunkannya |
| `totals_pass_rate` | Baris Jumlah/Total yang sama dengan jumlah rinciannya: lolos / (lolos + gagal) |
| `totals_uncheckable` | Jumlah tabel yang tidak bisa dicek (dihitung terpisah dari `totals_pass_rate`) |
| `consensus_recall` | Khusus `real_scan`: angka yang dibaca sama oleh mayoritas engine yang juga ada di output engine ini |
| `empty_pages`, `truncated_pages`, `error_pages` | Halaman kosong, terpotong, dan gagal |
| `pages_per_minute`, `median_seconds`, `p95_seconds` | Kecepatan; waktu per halaman = total wall-clock run / halaman yang diproses selama diukur. Median/p95 benar-benar per halaman hanya untuk Unlimited-OCR; engine lain = waktu chunk/worker dibagi rata |
| `cost_usd` | Biaya total halaman yang dinilai: API = harga per halaman; self-hosted = jam GPU × `--gpu-hour-rate` |

Metrik yang tidak berlaku pada suatu kolom kosong (–). Docling tidak dinilai kolom tabelnya di `digital`
(`n/a (acuan)`), karena tabel acuannya dibuat oleh Docling sendiri.

Sheet **Pages** berisi nilai per halaman, **Worst pages** 20 halaman terburuk per kolom × engine (dengan contoh angka
salah/hilang), dan **Notes** semua catatan di bawah.

## Keterbatasan (juga ada di Notes)

- **Acuan tabel dari Docling.** Struktur tabel acuan berasal dari Docling pada PDF digital, dengan isi sel dari text
  layer, sehingga metrik kolom bergantung pada kualitas struktur tabelnya. Sel acuan yang berisi lebih dari satu
  angka (sel tergabung) dikecualikan; jumlahnya tercantum di Notes.
- **Bias Docling pada tabel sintetis.** Di `synthetic_scan`, Docling kemungkinan bias ke atas pada metrik kolom karena
  acuannya dibuat oleh model tabel yang sama.
- **Asumsi biaya GPU.** Biaya self-hosted = jam GPU × tarif per jam (1 GPU per engine), dengan utilisasi penuh dan
  tanpa biaya operasional. Harga API: Azure Layout US$10/1.000 halaman, Mistral OCR US$4/1.000 halaman.
- **MinerU berlisensi AGPL.** Pemakaian produksi perlu dicek legal terlebih dulu.
- **EasyOCR lemah pada pemisah ribuan.** OCR bawaan Docling (EasyOCR) cenderung salah membaca pemisah ribuan
  (titik/koma) pada angka, sehingga recall angkanya bisa rendah pada gambar scan.
- **Deteksi loop.** `truncated` (finish_reason=length) adalah satu-satunya proksi generasi berulang; loop yang berhenti
  sebelum batas token tidak terdeteksi.
- `real_scan` tanpa kunci jawaban: `consensus_recall` dan `totals_pass_rate` hanya indikator tidak langsung.
- Engine yang hanya dinilai pada sebagian halaman ditandai "Cakupan tidak lengkap" di Notes; angkanya tidak sebanding
  penuh dengan engine lain.
