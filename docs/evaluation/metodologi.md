# Metodologi dan metrik

Semua nama, ambang, dan rumus di bawah diambil dari kode di `eval/ocr_eval/`. Alur dan file hasil dijelaskan di
[arsitektur.md](arsitektur.md); setup dan perintah di [eval/README.md](../../eval/README.md).

Belum ada run benchmark yang hasilnya dilaporkan di sini. Angka di contoh bersifat ilustratif.

## 1. Klasifikasi halaman (`classify.py`)

Setiap halaman diklasifikasikan sendiri, karena satu dokumen bisa campuran. Dari `page.get_texttrace()`, karakter
dihitung terpisah: teks tak terlihat (span type 3, render mode 3) dan teks terlihat.

| Urutan | Syarat | Hasil (`PageKind`) |
|---|---|---|
| 1 | karakter terlihat ≥ 50 | `digital` |
| 2 | karakter tak terlihat ≥ 50 **dan** ada gambar yang menutupi ≥ 80% luas halaman | `scan_with_hidden_text` |
| 3 | selain itu | `scan` |

`scan_with_hidden_text` adalah scan yang sudah diberi lapisan teks tak terlihat oleh OCR scanner. Lapisan itu hasil
OCR juga, jadi **tidak** dipakai sebagai kunci jawaban; halaman ini diperlakukan seperti `scan`.

## 2. Tiga kolom laporan

Setiap baris Summary adalah satu kombinasi kolom × engine. Ketiga kolom menjawab pertanyaan yang berbeda dan
**tidak boleh dijumlahkan atau dirata-ratakan menjadi satu skor**.

| Kolom | Halaman | Input ke engine | Kunci jawaban | Yang diukur |
|---|---|---|---|---|
| `digital` | halaman `digital` | PDF asli, seperti di produksi | text layer PDF | hasil nyata pada PDF digital. Engine yang membaca text layer (Docling, MinerU, mungkin Azure) bisa mendekati 100% **karena desainnya**, bukan karena OCR-nya bagus; Unlimited-OCR selalu membaca gambar render |
| `synthetic_scan` | halaman `digital` yang sama, dirender dan didegradasi | PDF berisi gambar saja | text layer PDF halaman asal | akurasi OCR murni dengan kunci jawaban pasti. Nomor halaman di laporan = nomor halaman asal |
| `real_scan` | halaman `scan` dan `scan_with_hidden_text` | PDF asli | **tidak ada** | hanya bukti tidak langsung: `consensus_recall` dan `totals_pass_rate` |

Mengapa tidak boleh dicampur: skor `digital` yang tinggi untuk engine text-layer tidak mengatakan apa pun tentang
kemampuan OCR-nya; skor `synthetic_scan` diukur terhadap kebenaran pasti; `real_scan` bahkan bukan ukuran akurasi.
Rata-rata ketiganya akan menyembunyikan tepat perbedaan yang ingin dilihat.

## 3. Scan sintetis (`degrade.py`)

Untuk setiap halaman digital, berurutan:

1. render pada `DPI = 200`, ubah ke grayscale;
2. rotasi acak seragam dalam ±`ROTATION_DEG` = 1,0° (bicubic, latar putih, ukuran tetap);
3. Gaussian blur radius `BLUR_RADIUS` = 0,6;
4. noise Gaussian aditif σ = `NOISE_SIGMA` = 6,0 (lalu di-clip ke 0–255);
5. simpan sebagai JPEG kualitas `JPEG_QUALITY` = 60, sisipkan sebagai satu-satunya isi halaman PDF baru berukuran
   sama (tanpa text layer).

Generator acak per halaman memakai seed `20261027 + nomor halaman asal` (`numpy.random.default_rng`), jadi hasilnya
deterministik dan sama di setiap run. Degradasi ini meniru scan kantor yang wajar; ia tidak mewakili scan yang sangat
buruk (miring jauh, terlipat, cap, tulisan tangan).

## 4. Akurasi angka (`numbers.py`)

### Ekstraksi

Tag HTML dibuang (diganti spasi), lalu angka dicari dengan pola

```
\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?   |   \d{7,}
```

yaitu angka dengan pemisah ribuan (titik atau koma, opsional diikuti desimal 1–2 digit), atau deretan ≥ 7 digit
tanpa pemisah.

### Normalisasi (`normalize_amount`)

Desimal 1–2 digit di akhir dibuang bila pemisahnya berbeda dari pemisah pertama (atau bila hanya ada satu pemisah),
lalu semua karakter non-digit dihapus. Akibatnya format Indonesia dan Inggris disamakan dan desimal diabaikan:

| Teks | Kunci |
|---|---|
| `1.234.567` | `1234567` |
| `1,234,567` | `1234567` |
| `1.234.567,89` | `1234567` |
| `1,234,567.89` | `1234567` |

Hanya kunci dengan **≥ 7 digit** (≥ 1.000.000) yang dihitung. Angka kecil (tahun, nomor catatan, persentase, nomor
halaman) sengaja diabaikan: risiko yang dicari adalah salah baca pada nilai rupiah besar. Angka dihitung sebagai
multiset (`Counter`): angka yang muncul dua kali di acuan harus muncul dua kali di output.

Acuan per halaman adalah `page.get_text()` dari PDF asli. Halaman yang acuannya tidak punya angka ≥ 7 digit
mendapat `number = None` dan **dikecualikan** dari semua metrik angka.

### Cocok, salah baca, hilang, tambahan

Dengan `R` = multiset angka acuan dan `O` = multiset angka output:

- `matched` = |R ∩ O|.
- Sisa acuan (`R − O`) dan sisa output (`O − R`) dipasangkan secara greedy: satu angka output dipasangkan dengan
  angka acuan sisa pertama yang **panjangnya sama dan berbeda paling banyak 2 digit** (`_MAX_MISREAD_DIFF = 2`).
  Setiap pasangan dihitung `misread`.
- `dropped` = angka acuan yang tidak cocok dan tidak berpasangan.
- `extra` = angka output yang tidak cocok dan tidak berpasangan.

Contoh: acuan `462.814.893.666`, output `492.814.893.666`. Kunci `462814893666` dan `492814893666` sama panjang (12)
dan berbeda 1 digit, jadi dihitung 1 `misread` (dan bukan `matched`). Ini persis jenis kesalahan berbahaya yang
dicari: angkanya terlihat wajar.

Angka yang ditulis dengan spasi sebagai pemisah, mis. `462 814 893 666`, tidak cocok dengan pola (tidak ada titik/
koma dan tidak ada 7 digit berurutan), jadi tidak terekstrak dan acuannya dihitung `dropped`. Ini disengaja: sistem
hilir yang mengurai angka dengan cara standar juga akan gagal membacanya. Hal serupa berlaku bila angka terpotong
menjadi beberapa bagian.

Contoh salah baca dan angka hilang (maksimal 4 per halaman) masuk ke sheet Worst pages.

### Angka negatif

- Di acuan, angka negatif = angka **dalam kurung** saja, mis. `(1.234.567)`, karena laporan keuangan menulisnya
  begitu.
- Di output, angka dalam kurung **atau** dengan tanda minus (`-` atau `−`) yang menempel langsung pada digit dan
  tidak didahului huruf/angka/titik/koma, mis. `-1.234.567`. `Modal dasar - 40.000.000` atau `-` di baris sendiri
  bukan tanda negatif.
- `neg_matched` = |negatif acuan ∩ negatif output|. Hanya dihitung terhadap text layer (kolom `digital` dan
  `synthetic_scan`).

## 5. Metrik teks (`text_metrics.py`)

Kata = `\w+` setelah huruf kecil semua; tag HTML output dibuang.

- **`dice`** (word Dice) = 2·|R ∩ O| / (|R| + |O|) atas multiset kata acuan `R` dan output `O`. `None` bila acuan
  < 50 kata. Mengukur kelengkapan teks secara kasar; tidak sensitif pada urutan atau pada satu digit yang salah.
- **`english`** (English coverage) = bagian dari kata fungsi bahasa Inggris di acuan yang juga ada di output:
  |R_en ∩ O_en| / |R_en|, dengan kata fungsi `the of and for as which with are was were has have on by from this
  that`. `None` bila acuan memiliki < 30 kata fungsi tersebut. Laporan keuangan di sini dua bahasa; metrik ini
  menangkap engine yang menghilangkan atau menerjemahkan bagian bahasa Inggris, yang tidak terlihat dari Dice.
- `english_skipped_pages` di Summary = jumlah halaman (di `digital`/`synthetic_scan`) yang `english`-nya `None`.

## 6. Penempatan kolom tabel (`tables.py`)

### Parsing tabel output (`parse_tables`)

Tabel HTML (`<table>`, termasuk yang tidak ditutup di akhir output) diurai lebih dulu, lalu tabel pipe markdown
(baris yang diawali `|`; baris pemisah `---` dilewati). Hasilnya grid teks sel.

**Konvensi anchor span**: teks sel yang ber-`colspan`/`rowspan` disimpan **sekali**, di baris paling atas dan kolom
paling kanan dari span-nya; sisa span diisi `""`. Worker Docling memakai konvensi yang sama untuk tabel acuan
(`anchor_grid`), jadi acuan dan output sebanding.

Contoh: baris `<tr><td>Kas</td><td>5</td><td colspan="2">1.234.567</td></tr>` menjadi
`["Kas", "5", "", "1.234.567"]`; angkanya berada di kolom terakhir.

### Indeks kolom dari kanan

`amount_columns` memberi setiap angka (≥ 7 digit) indeks kolom **dihitung dari tepi kanan** tabelnya:
`lebar_tabel − 1 − kolom`. Engine sering berbeda di kolom kiri (label digabung atau dipisah, kolom Catatan ada atau
tidak), sedangkan kolom angka (tahun berjalan, tahun lalu) rata kanan. Dalam contoh di atas, `1234567` mendapat
indeks 0.

### Acuan dan skor

- Acuan = tabel Docling mode `reference-tables` untuk halaman itu (struktur TableFormer, isi sel dari text layer).
- Sel acuan yang berisi **lebih dari satu angka** dikosongkan (`clean_reference_grids`): kolomnya ambigu (sering sel
  Docling yang tergabung). Jumlah angka yang dikecualikan dicantumkan di Notes.
- Per halaman, untuk setiap kunci angka dengan `ref_cols` (indeks kolom di acuan) dan `ocr_cols` (di tabel output):
  - `found_in_table` += min(|ref_cols|, |ocr_cols|)
  - `placed_correctly` += |ref_cols ∩ ocr_cols| (multiset)
  - `reference` = jumlah semua angka di tabel acuan
- Angka yang ada di output tetapi di luar tabel tidak dihitung `found_in_table`.

Di Summary:

- **`column_placement`** = Σ `placed_correctly` / Σ `found_in_table`. Metrik kolom utama: dari angka yang memang
  dimasukkan engine ke tabel, berapa yang ada di kolom yang benar. Tidak terpengaruh angka yang hilang.
- **`column_end_to_end`** = Σ `placed_correctly` / Σ `reference`. Angka yang hilang, salah baca, atau tidak di dalam
  tabel ikut menurunkannya.

Docling **tidak dinilai** di `digital` (ditampilkan `n/a (acuan)`): acuannya dibuat oleh Docling sendiri dari file
yang sama. Di `synthetic_scan` Docling tetap dinilai, tetapi kemungkinan bias ke atas karena model tabelnya sama.
Metrik kolom tidak dihitung di `real_scan` dan tidak dihitung untuk dokumen yang tidak punya tabel acuan.

## 7. Cek akuntansi (`accounting.py`)

### Sel yang dihitung (`parse_amount`)

Hanya sel yang **seluruhnya** berupa angka yang dipakai: opsional kurung pembuka, opsional `-`/`−`, opsional `Rp`,
angka dengan pemisah ribuan atau ≥ 7 digit (opsional desimal), opsional kurung penutup (kurung harus berpasangan).
Kurung atau minus = negatif. Sel yang hanya berisi tanda hubung (`-`, `–`, `—`, `−`) = 0. Tahun (`2025`), nomor
catatan (`5`, `12a`), dan teks campuran tidak dihitung. Berbeda dari akurasi angka, di sini angka kecil dengan
pemisah ribuan (`1.000`) ikut dihitung.

### Baris total (`check_totals`)

Baris total = baris yang label-nya (sel pertama yang tidak kosong dan bukan angka) diawali `Jumlah`, `Total`, atau
`Subtotal`/`Sub total` (tidak peka huruf besar) dan memuat angka. Untuk setiap baris total dan setiap kolom angka:

1. Berjalan **ke atas** dari baris total, melewati baris tanpa angka. Setiap baris rincian adalah satu *blok*. Baris
   total sebelumnya adalah satu blok yang mewakili semua baris yang sudah ia jumlahkan; baris-baris itu dilompati.
2. Jumlahkan blok satu per satu dari yang terdekat. Total **lolos** bila untuk suatu k, jumlah k blok pertama
   berselisih ≤ k dengan total (toleransi pembulatan 1 per komponen). Baris yang dijumlahkan total ini dicatat,
   sehingga total di bawahnya bisa melompatinya.
3. **Gagal** bila tidak ada k yang cocok; contoh di Worst pages memuat total, jumlah terdekat, dan selisihnya.
4. **Tidak bisa dicek** (`uncheckable`) bila sel total di kolom itu bukan angka, tidak ada blok di atasnya, atau ada
   baris berangka di atasnya yang selnya di kolom ini kosong/tidak terbaca (misalnya angka bergeser ke kolom lain).

Contoh (satu kolom):

| Baris | Label | Nilai | Hasil |
|---|---|---|---|
| 0 | Kas | 100.000.000 | |
| 1 | Piutang | 50.000.000 | |
| 2 | Jumlah aset lancar | 150.000.000 | k=2 (Piutang + Kas) → lolos, menjumlahkan baris 0–1 |
| 3 | Aset tetap | 300.000.000 | |
| 4 | Aset lain | 20.000.000 | |
| 5 | Jumlah aset tidak lancar | 320.000.000 | k=2 (Aset lain + Aset tetap) → lolos, menjumlahkan baris 3–4 |
| 6 | JUMLAH ASET | 470.000.000 | blok: 320.000.000 (baris 5, melompati 3–4), 150.000.000 (baris 2, melompati 0–1); k=2 → lolos |

`totals_pass_rate` = lolos / (lolos + gagal); `totals_uncheckable` dilaporkan terpisah. Untuk engine, cek ini hanya
dijalankan di `real_scan` (tabel hasil parsing output engine), sebagai bukti tidak langsung tanpa kunci jawaban.

### Engine pseudo `reference`

Di kolom `digital`, `check_totals` dijalankan juga pada tabel acuan Docling (sebelum pembersihan sel multi-angka) di
bawah nama engine `reference`. Ini **validasi pemeriksanya**, bukan engine: bila `totals_pass_rate` `reference`
rendah, aturan cek total tidak cocok dengan tata letak laporan itu dan `totals_pass_rate` engine di `real_scan`
tidak boleh dipercaya.

### Cek neraca (`check_balance`)

Per dokumen dan engine (PDF asli, semua halaman): apakah `Jumlah aset` / `Total assets` sama dengan `Jumlah
liabilitas dan ekuitas` / `Total liabilities and equity` di setiap kolom angka.

- Label dua bahasa (`Jumlah aset / Total assets`) diterima bila setiap bagian (dipisah `/`, `|`, atau baris baru)
  adalah salah satu label tersebut.
- Baris dicari lebih dulu di tabel hasil parsing (sel nomor catatan diabaikan). Bila tidak ada, di teks: angka pada
  baris label itu sendiri, atau baris-baris berikutnya yang hanya berisi angka (bentuk text layer dan output
  satu-sel-per-baris). Angka dari baris berikutnya dipotong ke jumlah kolom baris lawannya bila yang itu pasti.
  Halaman dibaca berurutan; kemunculan pertama yang dipakai.
- Hasil: lolos bila setiap pasangan berselisih ≤ 1; gagal bila tidak; tidak bisa dicek (`null`) bila salah satu baris
  tidak ditemukan atau jumlah kolomnya berbeda.
- Pseudo engine `reference` menjalankan cek ini pada text layer halaman digital, juga sebagai validasi pemeriksa.
- Hasilnya hanya muncul di Notes (satu kalimat per dokumen), bukan di Summary.

## 8. Konsensus untuk scan asli (`agreement.py`)

Untuk setiap halaman `real_scan`:

- Hanya engine yang halamannya tidak `error` dan outputnya tidak kosong yang ikut memilih. Dengan n engine pemilih,
  ambang default = max(2, ⌊n/2⌋ + 1) (mayoritas mutlak, minimal 2): 2 atau 3 engine → 2, 4 atau 5 engine → 3.
- Angka konsensus = angka (≥ 7 digit) yang ditemukan oleh minimal sejumlah ambang engine; multiplisitasnya =
  berapa kali angka itu ditemukan oleh setidaknya sejumlah ambang engine.
- Setiap engine yang punya output di halaman itu, termasuk yang gagal (dinilai sebagai kosong), dinilai terhadap
  konsensus dengan aturan cocok/salah baca/hilang yang sama seperti bagian 4.
- Halaman tanpa angka konsensus tidak dinilai.
- **`consensus_recall`** = Σ cocok / Σ angka konsensus.

Konsensus bukan kebenaran: engine ikut memilih angka yang kemudian dipakai untuk menilainya sendiri, dan beberapa
engine yang salah dengan cara yang sama tetap membentuk konsensus.

## 9. Agregasi (`score.py`)

Satu `Summary` per kolom × engine. **Semua rasio adalah Σ pembilang / Σ penyebut atas halaman**, bukan rata-rata
rasio per halaman, sehingga halaman dengan banyak angka berbobot lebih besar. Pengecualian: `median_dice` adalah
median nilai `dice` per halaman.

| Metrik | Rumus |
|---|---|
| `pages` | jumlah baris halaman (termasuk yang gagal) |
| `amount_recall` | Σ matched / Σ reference |
| `misread_rate` | Σ misread / Σ reference |
| `negative_recall` | Σ neg_matched / Σ neg_reference |
| `median_dice` | median `dice` (halaman dengan `dice = None` diabaikan) |
| `column_placement`, `column_end_to_end` | bagian 6 |
| `totals_pass_rate`, `totals_uncheckable` | bagian 7 |
| `consensus_recall` | bagian 8 |
| `empty_pages` | halaman tanpa error dengan output kosong |
| `truncated_pages` | halaman dengan `truncated` (hanya diisi oleh Unlimited-OCR) |
| `error_pages` | halaman dengan `error` |

Rasio dengan penyebut 0 kosong (`–`). Engine yang dinilai pada lebih sedikit halaman daripada seharusnya (halaman
tanpa output di cache sama sekali) diberi catatan "Cakupan tidak lengkap" di Notes (`coverage_gaps`); halaman gagal
tetap dinilai (sebagai kosong) sehingga tidak termasuk kekurangan cakupan.

### Waktu

Untuk setiap halaman, bagian waktunya = `wall_seconds / timed_pages` dari `meta.json` run (varian `original` untuk
`digital` dan `real_scan`, `synthetic` untuk `synthetic_scan`). Bila `timed_pages` tidak ada (meta lama), dipakai
jumlah halaman di meta.

- `pages_per_minute` = jumlah halaman berwaktu / Σ bagian waktu × 60. Ini throughput wall-clock nyata, termasuk efek
  konkurensi (Unlimited-OCR 32 request bersamaan) dan overhead chunk/worker.
- `median_seconds`, `p95_seconds` (p95 = nearest rank) dari `seconds` per halaman. Hanya untuk Unlimited-OCR ini
  benar-benar latensi per halaman (di bawah beban konkurensi). Engine lain memproses satu chunk per panggilan, jadi
  nilainya waktu chunk dibagi rata dan tidak menggambarkan sebaran latensi.

### Biaya

`cost_usd` = total biaya halaman yang dinilai pada kolom itu:

- engine berbayar: Σ `price_per_page` per halaman (Azure 0,010, Mistral 0,004 US$);
- self-hosted: Σ bagian waktu / 3600 × `--gpu-hour-rate` (default 0,95 US$/jam GPU), dengan asumsi satu GPU per
  engine yang terpakai penuh selama run dan tanpa biaya operasional, listrik, atau idle.

## 10. Cara membaca hasil

Urutan prioritas untuk keputusan bank:

1. **`misread_rate` di `synthetic_scan`**: metrik paling penting. Angka yang salah baca terlihat benar dan lolos
   tanpa terdeteksi; nilai yang tidak nol berarti output engine perlu verifikasi angka (mis. cek total) sebelum
   dipakai. `dropped` lebih aman daripada `misread`, karena angka yang hilang mudah terlihat.
2. **`amount_recall` dan `column_placement` di `synthetic_scan`**: akurasi OCR murni dengan kunci jawaban pasti.
   `column_placement` rendah berarti angka tahun berjalan dan tahun lalu tertukar atau bergeser, sehingga ekstraksi
   per kolom tidak bisa dipercaya walaupun angkanya sendiri benar.
3. **Kolom `digital`**: menjawab "apa yang terjadi bila PDF digital dikirim apa adanya". Nilai mendekati 100% untuk
   Docling/MinerU/Azure menunjukkan mereka membaca text layer; itu informasi berguna untuk desain pipeline (PDF
   digital mungkin tidak perlu OCR sama sekali), bukan bukti kualitas OCR.
4. **`real_scan`**: `consensus_recall` rendah berarti engine sering tidak setuju dengan mayoritas; `totals_pass_rate`
   rendah (sementara `reference` tinggi) berarti tabel hasil engine tidak konsisten secara akuntansi. Keduanya
   petunjuk, bukan ukuran akurasi.
5. **`negative_recall`** rendah berarti tanda negatif hilang: berbahaya untuk laba/rugi dan arus kas.
6. **`english`** rendah berarti bagian bahasa Inggris hilang; **`truncated_pages`**/**`empty_pages`**/
   **`error_pages`** menunjukkan masalah keandalan; **`pages_per_minute`** dan **`cost_usd`** menentukan skala.

Selalu bandingkan engine dalam kolom yang sama, dan baca Notes: engine yang dilewati atau cakupannya tidak lengkap
tidak sebanding penuh.

## 11. Keterbatasan

- **Kualitas tabel acuan Docling.** Metrik kolom bergantung pada struktur tabel Docling. Sel multi-angka
  dikecualikan, tetapi kesalahan struktur lain (baris tergabung, kolom terbelah) tetap masuk ke acuan. Docling juga
  bias di `synthetic_scan` dan tidak dinilai di `digital`.
- **EasyOCR lemah pada pemisah ribuan.** OCR Docling (EasyOCR) sering salah membaca titik/koma ribuan, sehingga
  angka tidak terekstrak dan `amount_recall` Docling pada gambar bisa rendah karena hal ini saja.
- **Deteksi loop.** `truncated` (finish_reason=length) adalah satu-satunya proksi generasi berulang, dan hanya
  Unlimited-OCR yang mengisinya. Loop yang berhenti sebelum batas token tidak terdeteksi.
- **Konsensus bukan kebenaran** (bagian 8); `real_scan` tidak punya kunci jawaban sama sekali.
- **Degradasi sintetis** hanya meniru satu tingkat kualitas scan; scan asli bisa jauh lebih buruk.
- **Angka < 7 digit** tidak dinilai di akurasi angka; desimal diabaikan.
- **Biaya self-hosted** adalah estimasi dengan asumsi utilisasi penuh; harga API dikodekan tetap di engine.
- **Residensi data.** Azure dan Mistral memproses dokumen di luar Indonesia. Persetujuan yang ada hanya mencakup dua
  laporan keuangan publik di `tests/` (`LK Bumi Serpong Damai.pdf`, `PT Hutama Karya.pdf`). Untuk dokumen lain
  gunakan `--engines unlimited,docling,mineru` kecuali ada keputusan terpisah.
- **Lisensi MinerU AGPL**: pemakaian produksi perlu dicek legal terlebih dulu.
- **Sampel kecil.** Dua dokumen tidak mewakili semua format laporan keuangan; hasil perlu dikonfirmasi pada dokumen
  lain sebelum menjadi keputusan akhir.
