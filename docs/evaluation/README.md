# Dokumentasi alat evaluasi OCR

Alat di `eval/` (package `ocr_eval`) membandingkan lima engine OCR pada laporan keuangan berbahasa Indonesia:
Unlimited-OCR (self-hosted di vLLM), Azure AI Document Intelligence, Mistral OCR, Docling, dan MinerU.

## Mengapa alat ini ada

- **Tidak ada benchmark publik** yang membandingkan Unlimited-OCR, Azure Document Intelligence, dan Mistral OCR,
  apalagi pada laporan keuangan berbahasa Indonesia. Keputusan memilih engine harus didasarkan pada dokumen sendiri.
- **Risiko utama adalah angka yang diam-diam salah**: angka yang salah satu-dua digit, kolom tabel yang bergeser,
  atau teks bahasa Inggris (laporan dua bahasa) yang hilang. Output seperti itu terlihat wajar dan lolos dari
  pemeriksaan sekilas. Karena itu sebagian besar metrik di sini berfokus pada angka dan penempatan kolom, bukan
  kemiripan teks secara umum.
- **Library open source bukan kunci jawaban.** Kunci jawaban hanya diambil dari text layer PDF digital. Untuk scan
  asli, yang tidak punya kunci jawaban, alat ini hanya memberi bukti tidak langsung dan menyebutnya demikian.

Belum ada run benchmark lengkap yang hasilnya didokumentasikan di sini; dokumen ini menjelaskan cara kerja dan cara
membaca hasil, bukan hasilnya.

## Isi dokumentasi

| Dokumen | Isi |
|---|---|
| [arsitektur.md](arsitektur.md) | Tahapan pipeline dan file yang ditulis tiap tahap, peta modul, protokol `Engine`, cara tiap engine dipanggil, cache/resume, jalur paralel (lane) GPU, dan gerbang biaya |
| [metodologi.md](metodologi.md) | Klasifikasi halaman, tiga kolom laporan, degradasi, dan definisi tepat setiap metrik beserta contoh; cara membaca hasil dan keterbatasannya |
| [menambah-engine.md](menambah-engine.md) | Cara menambah engine atau metrik baru, beserta tes yang perlu ditambahkan |

## Dokumen terkait

- [eval/README.md](../../eval/README.md): setup tiga virtualenv, prasyarat (GPU, key API), perintah, dan opsi CLI.
  Dokumen di folder ini tidak mengulang isinya.
- [Spesifikasi desain](../superpowers/specs/2026-10-27-ocr-evaluation-design.md) dan
  [rencana implementasi](../superpowers/plans/2026-10-27-ocr-evaluation.md). Kode berkembang lewat beberapa review
  setelah rencana ditulis; bila berbeda, yang berlaku adalah kode dan dokumentasi di folder ini.
- [api/README.md](../../api/README.md): REST API yang adapter Azure, Mistral, dan konfigurasi `Settings`-nya dipakai
  ulang oleh alat evaluasi.
- Laporan riset `reports/Perbandingan tiga engine OCR.md` (opsional): folder `reports/` tidak di-commit, jadi file
  ini hanya ada di salinan lokal tertentu.
