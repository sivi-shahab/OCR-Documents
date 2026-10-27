# Desain: Alat evaluasi engine OCR untuk laporan keuangan

Tanggal: 2026-10-27

## Maksud dan alasan

Repo ini sekarang bisa menjalankan tiga engine OCR: Unlimited-OCR (self-hosted di vLLM), Azure AI Document
Intelligence, dan Mistral OCR. Riset (`reports/Perbandingan tiga engine OCR.md`) menunjukkan bahwa tidak ada
benchmark publik yang membandingkan ketiganya, apalagi pada laporan keuangan berbahasa Indonesia. Uji 23 Oktober
(`reports/ocr_test_2026-10-23/`) juga menunjukkan bahwa risiko terbesar adalah **angka yang diam-diam salah**:
1,8% angka besar salah 1–2 digit, kolom tabel bergeser, dan kolom bahasa Inggris hilang.

Alat ini menjawab satu pertanyaan dengan bukti dari dokumen sendiri: *engine mana yang paling akurat untuk angka,
teks, dan tabel, masing-masing pada halaman PDF digital dan hasil scan?* Hasilnya akan dipakai untuk merancang
endpoint gabungan nanti, yang menjadi proyek terpisah.

Prinsip utama: library open source **bukan kunci jawaban**, karena hasil OCR-nya sendiri bisa salah. Kunci jawaban
hanya diambil dari sumber yang pasti benar, yaitu text layer PDF digital. Untuk scan asli yang tidak punya kunci
jawaban, alat ini memakai bukti tidak langsung (kesepakatan antar engine dan cek akuntansi) dan menyatakannya
dengan jelas sebagai bukti tidak langsung.

## Tujuan

1. **Klasifikasi per halaman**: `digital`, `scan`, atau `scan_with_hidden_text` (gambar halaman penuh dengan teks
   tak terlihat dari OCR scanner). Jenis ketiga diperlakukan sebagai scan dan text layer-nya tidak dipakai sebagai
   acuan. Klasifikasi dilakukan per halaman karena satu dokumen bisa campuran: BSD punya 206 halaman digital dan
   1 halaman gambar.
2. **Menjalankan 5 engine** pada halaman yang sama dan menyimpan output mentahnya:
   - Unlimited-OCR
   - Azure Document Intelligence (`prebuilt-layout`)
   - Mistral OCR
   - Docling
   - MinerU
3. **Menilai halaman digital terhadap text layer**:
   - Angka persis: recall, salah baca (1–2 digit beda), dan hilang. Metodenya seperti `numcheck2.py`.
   - Word Dice untuk teks.
   - Cakupan kolom bahasa Inggris pada halaman bilingual, seperti `engcheck.py`.
   - Penempatan kolom tabel. Acuannya adalah struktur tabel Docling yang dibaca dari PDF digital, dengan isi sel
     dari text layer.
4. **Menilai scan buatan.** Halaman digital BSD dirender ke gambar, diberi degradasi scan, lalu dijadikan PDF
   berisi gambar saja. Kunci jawabannya tetap text layer asli, sehingga akurasi engine pada kondisi scan bisa
   diukur dengan angka nyata.
5. **Menilai scan asli** (PT Hutama Karya, 41 halaman, tanpa text layer) dengan dua bukti tidak langsung:
   - **Kesepakatan antar engine**: untuk setiap angka, berapa engine yang membaca nilai yang sama.
   - **Cek akuntansi**: baris "Jumlah/Total" sama dengan jumlah rinciannya, dan total aset = total liabilitas +
     ekuitas.

   Cek akuntansi juga dijalankan pada text layer BSD untuk memvalidasi pemeriksanya sendiri: di text layer, cek
   seharusnya lolos.
6. **Menulis laporan** per engine × jenis halaman (digital / scan buatan / scan asli) dalam bentuk Excel ditambah
   ringkasan markdown. Isinya: akurasi angka, teks, dan kolom; halaman kosong, terpotong, atau loop; waktu; biaya.
   Laporan juga memuat daftar halaman terburuk untuk dicek manual.

## Bukan tujuan

- Endpoint gabungan atau perubahan apa pun pada API produksi (`api/app`). Alat ini hanya *memanggil* API dan
  adapter yang sudah ada.
- Marker dan Unstructured. Keduanya sengaja dikeluarkan: Marker karena lisensi bobot modelnya membatasi penggunaan
  komersial, Unstructured karena memakai Tesseract yang lemah di tabel.
- Tulisan tangan dan paraf.
- Penggabungan tabel lintas halaman.
- Fine-tuning atau tuning prompt engine.

## Anti-tujuan (gagal walaupun "jalan")

- Memakai output Docling atau MinerU sebagai kunci jawaban teks dan angka. Docling hanya menjadi acuan untuk
  **struktur tabel**, dan itu pun hanya bila dibaca dari halaman digital, tempat isi selnya berasal dari text
  layer. Acuan yang sama boleh dipakai untuk scan buatan dari halaman tersebut.
- Mencampur hasil halaman digital dan scan dalam satu angka akurasi.
- Memanggil ulang API berbayar hanya karena logika penilaian diubah. Output mentah harus di-cache, dan penilaian
  harus bisa diulang secara offline.
- Hasil yang tidak bisa direproduksi. Seed degradasi tetap, versi model setiap engine dicatat, termasuk field
  `model` dari respons Mistral, karena alias `mistral-ocr-latest` bisa berganti.

## Batasan

- **Data**: pengguna menyetujui pengiriman kedua dokumen di `tests/` (laporan keuangan yang sudah dipublikasikan)
  ke Azure dan Mistral, walaupun diproses di luar Indonesia. Persetujuan ini **hanya untuk dokumen tersebut**.
  Alat ini tidak boleh dipakai pada dokumen nasabah lewat Azure atau Mistral tanpa keputusan terpisah.
- **Hardware**: 2× NVIDIA L40S (46 GB). Container vLLM (`api/docker-compose.yml`, service `vllm`, default GPU 0)
  saat ini **mati** dan harus dinyalakan dengan `docker compose up -d vllm` di `api/`. Docling dan MinerU memakai
  GPU 1 agar tidak berebut dengan vLLM.
- **Kredensial**: `OCR_AZURE_DI_API_KEY` dan `OCR_MISTRAL_API_KEY` di `.env` diisi sendiri oleh pengguna dan belum
  terisi saat dokumen ini ditulis. Alat harus memberi pesan jelas, lalu melewati engine yang kredensialnya kosong
  tanpa gagal total.
- **Dependensi**: Docling dan MinerU berat (torch, model besar) dan saling berpotensi konflik versi. Keduanya
  **tidak boleh** masuk ke dependensi `api/` (`api/pyproject.toml`, image Docker API).
- **Lisensi**: MinerU berlisensi AGPL (dari ingatan, belum dicek ulang). Untuk uji internal tidak masalah. Laporan
  harus mencatat bahwa pemakaian produksi perlu dicek legal.
- **Biaya API**: perkiraan untuk 207 + 41 halaman, digandakan untuk set scan buatan, sekitar US$5–6 untuk Azure
  dan sekitar US$2 untuk Mistral. Alat harus menampilkan estimasi biaya sebelum memanggil API berbayar.

## Keputusan yang sudah diambil pengguna

- Evaluasi dibuat sebagai **alat uji terpisah** sebelum membangun endpoint gabungan.
- Metode evaluasi sendiri seperti dijelaskan di atas: library open source sebagai alat bantu, bukan kunci jawaban.
- Dokumen di `tests/` boleh dikirim ke Azure dan Mistral.
- Engine open source yang diuji: **Docling dan MinerU**.
- Sistem nanti perlu mengecek digital vs scan sebelum OCR. Alat ini memvalidasi pemeriksa itu lebih dulu.

## Keputusan saya (belum dikonfirmasi pengguna secara eksplisit)

- Lokasi kode: folder `eval/` di root repo, ikut di-commit. Output run masuk ke
  `reports/ocr_eval_<YYYY-MM-DD>/`.
- Laporan berbentuk Excel ditambah ringkasan markdown, mengikuti format `reports/OCR_Test_Report.xlsx`.
- Degradasi scan buatan dibuat satu tingkat "sedang" yang realistis untuk scanner kantor: rotasi kecil, blur
  ringan, noise, grayscale, kompresi JPEG, sekitar 200 DPI.
- Angka yang dinilai: nominal ≥7 digit seperti uji 23 Oktober, ditambah angka dalam kurung (nilai negatif) sebagai
  metrik terpisah, karena riset menemukan engine pernah menghilangkan tanda kurung negatif.
- **Input per jenis halaman.** Pada kolom "digital", setiap engine menerima **PDF asli** seperti di produksi.
  Docling, MinerU, dan mungkin Azure akan memakai text layer, sehingga skornya mendekati 100%. Itu memang yang
  ingin diketahui: apakah library pembaca text layer mengalahkan OCR untuk halaman digital. Akurasi OCR murni
  diukur di kolom "scan buatan", tempat semua engine hanya menerima gambar (Docling dengan
  `force_full_page_ocr`, MinerU dengan mode `ocr`). Laporan harus menjelaskan perbedaan makna kedua kolom ini.
- **Docling pada metrik kolom tabel.** Di kolom digital, Docling tidak dinilai pada metrik penempatan kolom karena
  ia sendiri acuannya; tampilkan "n/a (acuan)". Di kolom scan buatan, Docling tetap dinilai, dengan catatan bahwa
  acuan dan engine memakai model tabel yang sama sehingga skornya mungkin bias ke atas.
- **Metrik kolom tabel juga berlaku untuk scan buatan.** Acuannya adalah struktur tabel Docling dari halaman
  digital asalnya. Ini tidak melanggar anti-tujuan, karena isi sel acuan tetap berasal dari text layer. Kolom
  bergeser adalah temuan utama uji 23 Oktober, jadi metrik ini termasuk yang terpenting untuk scan.
- **Waktu dan biaya.** Waktu dicatat sebagai throughput (halaman/menit) dan latensi per halaman (median dan p95),
  dengan konkurensi yang dicatat. Unlimited-OCR dijalankan pada konkurensi 32, sesuai uji 23 Oktober. Biaya API
  dihitung dari harga per halaman. Biaya engine self-hosted = jam GPU × tarif per jam, dengan tarif sebagai
  parameter (default US$0,95/jam untuk L40S sewaan, dari riset). Laporan menyebut bahwa angka ini mengasumsikan GPU
  terpakai penuh dan belum termasuk biaya operasional.

## Diserahkan ke builder

- Cara memanggil engine. Unlimited-OCR boleh lewat `/v1/ocr/jobs` atau langsung ke vLLM seperti `bench.py`.
  Azure dan Mistral boleh lewat endpoint API atau langsung memakai kelas adapter. Endpoint sinkron memiliki batas
  upload 50 MB dan timeout 600 detik, jadi pecah dokumen per halaman atau per potongan bila perlu.
- Isolasi lingkungan Docling dan MinerU: venv terpisah, container, atau cara lain.
- Heuristik klasifikasi halaman dan ambang batasnya, divalidasi pada kedua dokumen uji.
- Algoritma pencocokan angka ke kolom tabel dan heuristik cek akuntansi. Untuk cek akuntansi, laporkan juga
  berapa tabel yang **tidak bisa dicek**, tidak hanya yang lolos dan gagal.
- Struktur skrip dan CLI, serta penggunaan ulang kode dari `reports/ocr_test_2026-10-23/scripts/`. Folder skrip
  itu belum di-commit, jadi kode yang dipakai ulang dipindahkan ke `eval/`.
