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
Konfigurasi aplikasi ada di `.env` (lihat `.env.example`; semua bersifat opsional).

> **Catatan revisi model.** Commit HF `8249d20` (14 Okt 2026) mengganti `model_type` menjadi `unlimited_ocr`,
> sedangkan image vLLM ini mengenali `unlimited-ocr`. Akibatnya vLLM memuat remote code yang tidak
> kompatibel dengan transformers 5.x. Karena itu compose mem-pin revisi `07dea832` (bobot model sama).

## Endpoint

| Method | Path | Keterangan |
|---|---|---|
| GET | `/health` | `200` jika vLLM siap, `503` jika tidak |
| POST | `/v1/ocr/image` | OCR satu gambar (PNG/JPEG/WEBP; BMP/TIFF dikonversi ke PNG) |
| POST | `/v1/ocr/pdf` | OCR PDF |

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

## Arsitektur

```
app/
├── main.py                  # composition root: merangkai adapter → service → router
├── core/                    # config (env OCR_*) & exception domain
├── domain/                  # model, port (Protocol), parser output model; tanpa I/O
├── services/ocr_service.py  # use case: gambar & PDF (per_page / whole_document)
├── infrastructure/          # adapter: vLLM (httpx), PyMuPDF, Pillow
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
