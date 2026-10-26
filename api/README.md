# Unlimited-OCR API

REST API (FastAPI) untuk [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) yang dilayani oleh image Docker `vllm/vllm-openai:unlimited-ocr`.

## Menjalankan

```bash
cd api
docker compose up -d --build        # vLLM (GPU 0, port 8100) + API (port 8200)
curl localhost:8200/health
```

Swagger UI: http://localhost:8200/docs

Variabel compose: `VLLM_GPU` (default `0`), `VLLM_PORT` (`8100`), `API_PORT` (`8200`), `HF_CACHE_DIR` (`/data/hf-cache`).
Konfigurasi aplikasi ada di `.env` di root repo (salin dari `.env.example` di root; semua bersifat opsional).

> **Catatan revisi model.** Commit HF `8249d20` (14 Okt 2026) mengganti `model_type` menjadi `unlimited_ocr`,
> sedangkan image vLLM ini mengenali `unlimited-ocr`. Akibatnya vLLM memuat remote code yang tidak
> kompatibel dengan transformers 5.x. Karena itu compose mem-pin revisi `07dea832` (bobot model sama).

## Endpoint

| Method | Path | Keterangan |
|---|---|---|
| GET | `/health` | `200` jika vLLM siap, `503` jika tidak |
| POST | `/v1/ocr/image` | OCR satu gambar (PNG/JPEG/WEBP; BMP/TIFF dikonversi ke PNG) |
| POST | `/v1/ocr/pdf` | OCR PDF secara sinkron (dokumen kecil, ≤ `OCR_MAX_PDF_PAGES`) |
| POST | `/v1/ocr/jobs` | OCR PDF besar (300+ halaman) secara asinkron, lihat [Job untuk dokumen besar](#job-untuk-dokumen-besar) |
| POST | `/v1/ocr/azure` | OCR PDF/gambar dengan Azure AI Document Intelligence, lihat [Azure Document Intelligence](#azure-document-intelligence) |

Query parameter:
- `include_raw=true`: sertakan output mentah model per halaman.
- `strategy` (khusus PDF):
  - `per_page` (default): satu request per halaman, dijalankan paralel. Hasil tersedia per halaman.
  - `whole_document`: semua halaman dikirim dalam satu request long-horizon. Konteks antarhalaman terjaga, tetapi total output dibatasi konteks 32k token.

```bash
curl -F file=@scan.jpg localhost:8200/v1/ocr/image
curl -F file=@dokumen.pdf "localhost:8200/v1/ocr/pdf?strategy=per_page"
```

Respons:

```json
{
  "filename": "scan.jpg",
  "page_count": 1,
  "markdown": "Laporan Keuangan\n\nPendapatan bunga ...",
  "pages": [
    {
      "page": 1,
      "markdown": "...",
      "blocks": [{"category": "title", "bbox": [117, 74, 581, 101], "text": "Laporan Keuangan"}],
      "raw": null
    }
  ]
}
```

`bbox` berformat `[x1, y1, x2, y2]` dan dinormalisasi ke grid 0–999 relatif terhadap halaman.
Contoh `category`: `title`, `text`, `table`, `image`, `image_caption`, `page_number`.

Kode error: `413` (file/halaman melebihi batas), `422` (file kosong/rusak/salah tipe),
`502` (vLLM mengembalikan error), `503` (vLLM tidak dapat dihubungi).

### Azure Document Intelligence

`/v1/ocr/azure` mengirim file ke Azure AI Document Intelligence (model `prebuilt-layout`, output
markdown) sebagai alternatif Unlimited-OCR. Format respons sama dengan `/v1/ocr/pdf`; `bbox` diambil
dari paragraf Azure dan dinormalisasi ke grid 0–999. Endpoint mengembalikan `503` sampai endpoint dan
key diisi di `.env` (root repo):

```bash
OCR_AZURE_DI_ENDPOINT=https://<resource-name>.cognitiveservices.azure.com
OCR_AZURE_DI_API_KEY=<key dari Azure Portal → Keys and Endpoint>
```

```bash
curl -F file=@dokumen.pdf localhost:8200/v1/ocr/azure
```

## Job untuk dokumen besar

`/v1/ocr/pdf` adalah satu request HTTP sinkron, sehingga tidak cocok untuk dokumen ratusan halaman
(timeout, semua halaman di-render ke RAM, satu halaman gagal = seluruh request gagal). Untuk dokumen besar
gunakan job API. Batas default: 2000 halaman / 500 MB (`OCR_JOB_MAX_PDF_PAGES`, `OCR_JOB_MAX_UPLOAD_MB`).

| Method | Path | Keterangan |
|---|---|---|
| POST | `/v1/ocr/jobs` | Upload PDF → `202` berisi `id` dan `page_count`. File di-stream ke disk. PDF identik (SHA-256 sama) dengan job yang `queued`/`running`/`completed` mengembalikan job tersebut dengan `200`; matikan dengan `?reuse=false` |
| GET | `/v1/ocr/jobs` | Daftar job (terbaru dulu) |
| GET | `/v1/ocr/jobs/{id}` | Status dan progres: `pages_done`, `pages_succeeded`, `pages_fallback`, `pages_failed`, `pages_flagged`, `needs_review` |
| GET | `/v1/ocr/jobs/{id}/pages?first=1&limit=50` | Hasil per halaman; sudah bisa dibaca saat job masih berjalan |
| GET | `/v1/ocr/jobs/{id}/events` | Server-Sent Events: `job` saat terhubung dan tiap perubahan status, `page` tiap halaman selesai; stream berakhir saat job selesai |
| GET | `/v1/ocr/jobs/{id}/result.md` | Markdown seluruh dokumen, tiap halaman diawali `<!-- page N -->` (setelah `completed`, selain itu `409`) |
| GET | `/v1/ocr/jobs/{id}/result.json` | Markdown yang sama plus `pages: [{page, status, start, end}]`; `markdown[start:end]` adalah teks halaman tsb. (provenance) |
| POST | `/v1/ocr/jobs/{id}/cancel` | Hentikan job; halaman yang sudah selesai tetap tersedia |
| DELETE | `/v1/ocr/jobs/{id}` | Hapus job beserta PDF dan hasilnya |

```bash
JOB=$(curl -s -F file=@buku.pdf localhost:8200/v1/ocr/jobs | jq -r .id)
curl -s localhost:8200/v1/ocr/jobs/$JOB | jq '{status, pages_done, page_count}'   # poll tiap ≥2 detik
curl -s localhost:8200/v1/ocr/jobs/$JOB/result.md > buku.md
curl -N localhost:8200/v1/ocr/jobs/$JOB/events                                   # alternatif polling
```

`result.md` dan `result.json` secara default merapikan sambungan antarhalaman: header/footer/nomor halaman
yang berulang dibuang, paragraf dan kata ber-tanda hubung yang terpotong page break disambung, dan tabel
lanjutan dengan jumlah kolom sama digabung (header yang diulang dibuang). `?clean=false` mematikannya.

**Webhook (opsional).** Kirim field form `callback_url` saat membuat job untuk menerima `POST` setelah job
selesai (`completed`/`failed`/`cancelled`). Aktif hanya bila `OCR_WEBHOOK_SECRET` dan
`OCR_WEBHOOK_ALLOWED_HOSTS` diisi; host di luar allow-list ditolak `422`. Request ditandatangani:
`X-OCR-Signature: sha256=<hex HMAC-SHA256(secret, "{X-OCR-Timestamp}.{raw body}")>`. Verifikasi
signature dan tolak timestamp lama untuk mencegah replay. Pengiriman di-retry 3× dengan backoff.

Cara kerja (ringkasan dari `reports/Unlimited OCR 300 page documents.md`):

- **Satu request per halaman.** Mode multi-halaman dibatasi konteks 32k (± 11 halaman padat), jadi job selalu
  mengirim halaman satu per satu (mode Gundam, `window_size=128`).
- **Streaming render.** Halaman di-render satu per satu ke buffer kecil; inferensi dimulai setelah halaman
  pertama, dan memori tetap datar berapa pun jumlah halamannya.
- **Konkurensi.** `OCR_JOB_MAX_RUNNING` job berjalan bersamaan, masing-masing `OCR_JOB_PAGE_CONCURRENCY`
  request in-flight. Job lain antre, sehingga satu dokumen besar tidak memonopoli GPU.
- **Retry ladder per halaman.** Error vLLM di-retry dengan backoff. Output yang terpotong di `max_tokens`
  (`finish_reason=length`) di-retry dengan budget lebih besar (`OCR_JOB_FIRST_PASS_MAX_TOKENS` → `OCR_MAX_TOKENS`),
  dan temperature dinaikkan (`OCR_JOB_RETRY_TEMPERATURE_STEP`) bila output terdeteksi berulang (loop).
- **Eskalasi DPI.** Halaman yang tetap terpotong, atau kosong padahal PDF punya text layer, di-render ulang
  sekali di `OCR_JOB_ESCALATION_DPI` (default 300; `0` = nonaktif).
- **Fallback.** Halaman yang tetap gagal (atau kosong/loop padahal ada text layer) memakai text layer PDF
  (`status: fallback`), selain itu `status: failed`. Satu halaman gagal tidak menggagalkan job; job tetap
  `completed`, jadi periksa `pages_failed` dan `pages_fallback`.
- **Batas fallback per dokumen.** Bila `(pages_fallback + pages_failed) / page_count` melebihi
  `OCR_JOB_MAX_FALLBACK_RATIO` (default `0.004` = 1/250, seperti olmOCR), job tetap `completed` tetapi
  `needs_review: true` (juga dikirim di payload webhook). Hasilnya tetap bisa diambil; tinjau sebelum dipakai.
- **Flag kualitas** per halaman: `truncated` (masih terpotong di `max_tokens` pada semua percobaan),
  `repetition` (output terdeteksi loop), dan `text_layer_disagreement` (PDF punya text layer ≥ 50 kata, tetapi
  kemiripan kata dengan output OCR, koefisien Dice, < 0.5: OCR salah baca/halusinasi, atau text layer-nya
  sendiri rusak). Periksa halaman ber-flag secara manual; ambang ini titik awal, belum diukur pada korpus.
- **Tahan restart.** Status job dan tiap halaman yang selesai disimpan di `OCR_JOB_DIR` (volume `ocr-jobs` di
  compose). Setelah restart, job yang belum selesai dilanjutkan dan halaman final tidak diproses ulang.

> Job berjalan di dalam proses API, jadi API harus berjalan dengan **satu worker uvicorn** (default image).

Tuning vLLM di compose (`VLLM_MAX_NUM_SEQS`, default 128; `VLLM_MAX_NUM_BATCHED_TOKENS`, default 8192) adalah
titik awal. Ukur dengan `vllm bench serve` dan sesuaikan `OCR_JOB_PAGE_CONCURRENCY` × `OCR_JOB_MAX_RUNNING`
agar mendekati `max-num-seqs`.

## Arsitektur

```
app/
├── main.py                  # composition root: merangkai adapter → service → router
├── core/                    # config (env OCR_*) & exception domain
├── domain/                  # model, port (Protocol), parser output model; tanpa I/O
├── services/ocr_service.py  # use case: gambar & PDF (per_page / whole_document)
├── services/job_service.py  # use case: job asinkron untuk PDF besar (render streaming, retry, resume)
├── infrastructure/          # adapter: vLLM (httpx), PyMuPDF, Pillow, penyimpanan job di filesystem
└── api/                     # router, schema DTO, dependency, mapping error → HTTP
```

Arah dependensi: `api → services → domain ← infrastructure`. Service hanya bergantung pada
`OcrEngine`/`DocumentRenderer` (Protocol), sehingga engine bisa diganti (mis. SGLang) tanpa mengubah use case.

## Development

```bash
uv venv --python 3.12 && uv pip install -e . --group dev
OCR_VLLM_BASE_URL=http://localhost:8100/v1 uvicorn app.main:app --reload --port 8200
pytest
```
