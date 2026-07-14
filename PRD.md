# PRODUCT REQUIREMENTS DOCUMENT

## Sistem Forecasting Kunjungan Pasien Rawat Jalan per Poliklinik Menggunakan Prophet

## 1. Ringkasan Produk

Sistem Forecasting Kunjungan Pasien Rawat Jalan merupakan aplikasi berbasis Streamlit yang digunakan untuk memprediksi jumlah kunjungan pasien pada setiap poliklinik berdasarkan data historis kunjungan.

Model forecasting menggunakan algoritma Prophet dan dibangun secara terpisah untuk setiap poliklinik karena masing-masing poli memiliki pola kunjungan, jadwal operasional, tren, dan karakteristik musiman yang berbeda.

Sistem memungkinkan pengguna menentukan poliklinik, tanggal awal forecasting, tanggal akhir forecasting, serta bentuk agregasi hasil berupa harian, mingguan, atau bulanan.

Model utama menghasilkan prediksi dalam granularitas harian. Prediksi mingguan dan bulanan diperoleh melalui agregasi hasil forecasting harian.

---

## 2. Tujuan Sistem

Sistem dikembangkan untuk:

1. Memprediksi jumlah kunjungan pasien rawat jalan pada setiap poliklinik.
2. Membantu rumah sakit mengidentifikasi periode kunjungan tinggi dan rendah.
3. Mendukung perencanaan jumlah dokter, perawat, petugas administrasi, ruang pelayanan, dan persediaan operasional.
4. Menyediakan visualisasi forecasting yang mudah dipahami.
5. Memungkinkan pengguna menentukan periode prediksi secara fleksibel.
6. Menyediakan hasil forecasting dalam bentuk harian, mingguan, dan bulanan.
7. Menampilkan rentang ketidakpastian dari setiap hasil prediksi.
8. Menyediakan hasil prediksi yang dapat diunduh dalam format CSV.

---

## 3. Ruang Lingkup

### 3.1 Ruang Lingkup Utama

Sistem mencakup:

* preprocessing data kunjungan pasien;
* agregasi jumlah kunjungan berdasarkan tanggal dan poliklinik;
* exploratory data analysis;
* pelatihan model Prophet per poliklinik;
* evaluasi model;
* penyimpanan model;
* aplikasi Streamlit;
* pemilihan rentang tanggal forecasting;
* visualisasi hasil forecasting;
* rekap harian, mingguan, dan bulanan;
* ekspor hasil forecasting.

### 3.2 Di Luar Ruang Lingkup

Versi awal sistem belum mencakup:

* prediksi jumlah pasien individual;
* prediksi diagnosis pasien;
* rekomendasi medis;
* integrasi langsung dengan sistem informasi rumah sakit;
* pembaruan model secara real-time;
* autentikasi pengguna;
* manajemen hak akses;
* forecasting berdasarkan dokter;
* forecasting berdasarkan diagnosis atau jenis tindakan.

---

## 4. Pengguna Sistem

### 4.1 Pengguna Utama

Pengguna utama sistem adalah:

* manajemen rumah sakit;
* kepala unit rawat jalan;
* petugas administrasi;
* perencana sumber daya rumah sakit;
* staf teknologi informasi;
* peneliti atau analis data.

### 4.2 Kebutuhan Pengguna

Pengguna membutuhkan kemampuan untuk:

* memilih poliklinik;
* menentukan tanggal awal prediksi;
* menentukan tanggal akhir prediksi;
* memilih tampilan harian, mingguan, atau bulanan;
* melihat total prediksi kunjungan;
* melihat rata-rata prediksi;
* melihat periode dengan prediksi tertinggi dan terendah;
* melihat grafik forecasting;
* melihat tabel hasil prediksi;
* mengunduh hasil forecasting.

---

## 5. Sumber Data

Sistem menggunakan dataset historis kunjungan pasien rawat jalan.

Data utama yang diperlukan:

| Kolom             | Keterangan                             |
| ----------------- | -------------------------------------- |
| Tanggal kunjungan | Tanggal pasien memperoleh pelayanan    |
| Poliklinik        | Nama poliklinik tujuan                 |
| Nomor kunjungan   | Identitas unik kunjungan jika tersedia |

Data pasien seperti nama, nomor rekam medis, keluhan, diagnosis, dan identitas lainnya tidak digunakan dalam proses forecasting setelah proses agregasi selesai.

Format akhir dataset untuk model:

| Kolom  | Keterangan                             |
| ------ | -------------------------------------- |
| `ds`   | Tanggal kunjungan                      |
| `poli` | Nama poliklinik                        |
| `y`    | Jumlah kunjungan pada tanggal tersebut |

Contoh:

```text
ds,poli,y
2024-04-01,POLIKLINIK ANAK,34
2024-04-02,POLIKLINIK ANAK,29
2024-04-03,POLIKLINIK ANAK,31
```

---

## 6. Aturan Pengolahan Data

### 6.1 Data Cleaning

Tahapan pembersihan data mencakup:

1. Mengubah kolom tanggal ke format `datetime`.
2. Menyeragamkan penulisan nama poliklinik.
3. Menghapus baris dengan tanggal atau poliklinik kosong.
4. Memeriksa duplikasi kunjungan.
5. Menghapus data identitas pasien setelah agregasi.
6. Mengurutkan data berdasarkan tanggal.
7. Memeriksa tanggal yang hilang.
8. Membedakan nilai nol dengan data yang tidak tersedia.

### 6.2 Agregasi

Data kunjungan individual diubah menjadi jumlah kunjungan harian per poli.

```python
daily_data = (
    raw_data
    .groupby(["tanggal", "poliklinik"])
    .size()
    .reset_index(name="y")
)
```

### 6.3 Periode Data Hilang

Periode yang tidak memiliki sumber data tidak boleh langsung dianggap sebagai nol kunjungan.

Aturan:

* nilai nol digunakan jika poli diketahui beroperasi tetapi tidak memiliki kunjungan;
* nilai kosong digunakan jika data pada tanggal tersebut tidak tersedia;
* periode data hilang dikeluarkan dari data pelatihan atau diberi status khusus.

### 6.4 Hari Tutup

Hari tutup operasional tetap dipertahankan sebagai nilai nol apabila dapat diverifikasi dari jadwal pelayanan rumah sakit.

---

## 7. Desain Model Forecasting

### 7.1 Algoritma

Algoritma utama adalah Prophet.

Prophet dipilih karena mampu memodelkan:

* tren jangka panjang;
* pola mingguan;
* pola tahunan;
* perubahan tren;
* pengaruh hari libur;
* interval ketidakpastian.

### 7.2 Model per Poliklinik

Setiap poliklinik memiliki model terpisah.

Contoh:

```text
models/
├── prophet_anak.pkl
├── prophet_mata.pkl
├── prophet_obgyn.pkl
└── prophet_penyakit_dalam.pkl
```

Model terpisah diperlukan karena setiap poli memiliki:

* jumlah kunjungan berbeda;
* pola hari kerja berbeda;
* jadwal operasional berbeda;
* efek hari libur berbeda;
* tren pertumbuhan atau penurunan berbeda.

### 7.3 Konfigurasi Awal Prophet

```python
Prophet(
    growth="linear",
    weekly_seasonality=True,
    yearly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode="additive",
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10,
    interval_width=0.95
)
```

Model juga dapat menggunakan hari libur Indonesia:

```python
model.add_country_holidays(country_name="ID")
```

### 7.4 Granularitas Model

Model utama menggunakan data harian.

Hasil mingguan dan bulanan diperoleh melalui agregasi prediksi harian.

Alur:

```text
Prediksi harian
      ↓
Agregasi berdasarkan pilihan pengguna
      ├── Harian
      ├── Mingguan
      └── Bulanan
```

### 7.5 Pembagian Data

Pembagian data wajib menggunakan urutan waktu.

Contoh:

* training: data awal sampai 31 Desember 2025;
* validation: 1 Januari sampai 28 Februari 2026;
* testing: 1 Maret sampai 30 April 2026.

Random train-test split tidak digunakan karena dapat menyebabkan kebocoran informasi masa depan.

### 7.6 Baseline Model

Prophet dibandingkan dengan model sederhana:

* naive forecast;
* seasonal naive tujuh hari;
* moving average tujuh hari;
* moving average 28 hari.

Prophet dianggap layak jika memberikan hasil lebih baik atau lebih stabil dibandingkan baseline.

---

## 8. Evaluasi Model

### 8.1 Metrik

Metrik evaluasi utama:

* Mean Absolute Error;
* Root Mean Squared Error;
* Weighted Absolute Percentage Error;
* Symmetric Mean Absolute Percentage Error;
* Mean Error atau bias.

MAPE tidak dijadikan metrik utama karena data memiliki hari dengan kunjungan nol.

### 8.2 Evaluasi per Poli

Evaluasi dilakukan secara terpisah untuk setiap poliklinik.

Contoh tabel:

| Poliklinik     | MAE | RMSE |  WAPE | Bias |
| -------------- | --: | ---: | ----: | ---: |
| Poli Anak      | 4,2 |  5,8 | 14,6% | -0,3 |
| Poli Mata      | 3,9 |  5,1 | 13,9% |  0,2 |
| Poli OBGYN     | 6,1 |  8,7 | 13,2% |  0,7 |
| Penyakit Dalam | 8,3 | 11,5 | 12,7% | -1,1 |

### 8.3 Interpretasi Evaluasi

MAE digunakan sebagai interpretasi operasional.

Contoh:

> MAE sebesar 5 menunjukkan bahwa hasil prediksi rata-rata memiliki selisih sekitar lima pasien dibandingkan jumlah kunjungan aktual.

### 8.4 Cross-Validation

Model dapat diuji menggunakan rolling time-series cross-validation dengan horizon 30 hari.

---

## 9. Alur Sistem

```text
Dataset kunjungan pasien
          ↓
Data cleaning dan preprocessing
          ↓
Agregasi harian per poli
          ↓
Pelatihan model Prophet per poli
          ↓
Evaluasi model
          ↓
Penyimpanan model
          ↓
Aplikasi Streamlit
          ↓
Pengguna memilih poli dan periode
          ↓
Model menghasilkan prediksi
          ↓
Hasil ditampilkan dalam grafik dan tabel
```

---

## 10. Functional Requirements

### FR-01 Pemilihan Poliklinik

Sistem harus menyediakan pilihan poliklinik yang tersedia pada model.

Contoh:

* Poli Anak;
* Poli Mata;
* Poli OBGYN;
* Poli Penyakit Dalam.

### FR-02 Pemilihan Tanggal Awal

Pengguna dapat menentukan tanggal awal forecasting.

Tanggal awal harus berada setelah tanggal terakhir data aktual untuk mode forecasting masa depan.

### FR-03 Pemilihan Tanggal Akhir

Pengguna dapat menentukan tanggal akhir forecasting.

Tanggal akhir tidak boleh lebih kecil dari tanggal awal.

### FR-04 Pemilihan Granularitas

Pengguna dapat memilih:

* harian;
* mingguan;
* bulanan.

### FR-05 Proses Forecasting

Sistem memuat model berdasarkan poli yang dipilih dan menghasilkan prediksi sampai tanggal akhir.

### FR-06 Ringkasan Forecasting

Sistem harus menampilkan:

* total prediksi;
* rata-rata prediksi;
* prediksi tertinggi;
* prediksi terendah;
* tanggal atau periode tertinggi;
* tanggal atau periode terendah.

### FR-07 Grafik Forecasting

Sistem harus menampilkan:

* garis prediksi;
* batas bawah;
* batas atas;
* rentang ketidakpastian;
* label tanggal;
* jumlah pasien.

### FR-08 Tabel Forecasting

Tabel hasil harus memuat:

| Kolom       | Keterangan                    |
| ----------- | ----------------------------- |
| Periode     | Tanggal, minggu, atau bulan   |
| Prediksi    | Nilai utama hasil forecasting |
| Batas bawah | Perkiraan minimum             |
| Batas atas  | Perkiraan maksimum            |

### FR-09 Download Hasil

Pengguna dapat mengunduh hasil forecasting dalam format CSV.

### FR-10 Informasi Model

Sistem harus menampilkan:

* algoritma yang digunakan;
* tanggal terakhir data pelatihan;
* tanggal terakhir training;
* periode data;
* poliklinik yang tersedia;
* metrik evaluasi;
* keterbatasan forecasting.

### FR-11 Data Historis

Sistem dapat menampilkan data historis per poli untuk dibandingkan dengan hasil prediksi.

### FR-12 Evaluasi Model

Sistem menyediakan halaman yang menampilkan hasil evaluasi model per poli.

---

## 11. Validasi Input

Sistem harus menerapkan validasi:

1. Tanggal akhir tidak boleh lebih kecil dari tanggal awal.
2. Poliklinik wajib dipilih.
3. Model poliklinik harus tersedia.
4. Format tanggal harus valid.
5. Periode forecasting tidak boleh melebihi batas maksimum.
6. Tanggal awal tidak boleh berada sebelum tanggal terakhir data aktual untuk mode forecasting masa depan.
7. Hasil prediksi tidak boleh bernilai negatif.

Nilai negatif ditangani dengan:

```python
forecast["yhat"] = forecast["yhat"].clip(lower=0)
```

Batas awal forecasting direkomendasikan maksimal 365 hari.

---

## 12. Non-Functional Requirements

### NFR-01 Kinerja

Hasil forecasting harus dapat ditampilkan dalam waktu yang wajar karena model tidak dilatih ulang saat pengguna menekan tombol.

### NFR-02 Caching

Model harus dimuat menggunakan:

```python
@st.cache_resource
```

Data dapat dimuat menggunakan:

```python
@st.cache_data
```

### NFR-03 Keamanan Data

Data identitas pasien tidak boleh ditampilkan pada sistem forecasting.

### NFR-04 Usability

Antarmuka harus dapat digunakan oleh pengguna nonteknis.

### NFR-05 Reliability

Sistem harus memberikan pesan kesalahan yang jelas jika:

* model tidak ditemukan;
* input tanggal tidak valid;
* data kosong;
* forecasting gagal diproses.

### NFR-06 Maintainability

Proses preprocessing, training, validation, testing, dan evaluasi disusun sebagai tahapan CRISP-DM di dalam satu notebook. Lapisan inferensi forecasting dan UI tetap dipisahkan dari notebook agar aplikasi tidak melakukan training saat digunakan.

### NFR-07 Reproducibility

Konfigurasi model, periode training, evaluasi, dan versi library harus terdokumentasi.

---

## 13. Teknologi

| Komponen           | Teknologi      |
| ------------------ | -------------- |
| Bahasa pemrograman | Python         |
| Antarmuka          | Streamlit      |
| Forecasting        | Prophet        |
| Pengolahan data    | Pandas, NumPy  |
| Visualisasi        | Plotly         |
| Penyimpanan model  | Joblib         |
| Evaluasi           | Scikit-learn   |
| Input dataset      | Excel atau CSV |
| Ekspor hasil       | CSV            |

---

## 14. Struktur Proyek

```text
forecast_rawat_jalan/
├── app.py
├── notebooks/
│   └── 01_crisp_dm_forecasting.ipynb
├── forecasting.py
├── config.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   │   └── rawat_jalan.xlsx
│   └── processed/
│       └── daily_visits_by_poli.csv
│
├── models/
│   ├── prophet_anak.pkl
│   ├── prophet_mata.pkl
│   ├── prophet_obgyn.pkl
│   └── prophet_penyakit_dalam.pkl
│
├── metrics/
│   └── model_evaluation.csv
│
├── outputs/
│   └── forecast_results.csv
│
└── pages/
    ├── 1_Dashboard.py
    ├── 2_Forecasting.py
    ├── 3_Evaluasi_Model.py
    ├── 4_Data_Historis.py
    └── 5_Informasi_Model.py
```

---

## 15. Desain Halaman Streamlit

### 15.1 Dashboard

Menampilkan:

* total kunjungan historis;
* jumlah poliklinik;
* rata-rata kunjungan harian;
* poli dengan kunjungan tertinggi;
* tren historis;
* distribusi kunjungan per poli;
* periode data.

### 15.2 Forecasting

Input:

* poliklinik;
* tanggal awal;
* tanggal akhir;
* granularitas.

Output:

* total prediksi;
* rata-rata;
* prediksi tertinggi;
* prediksi terendah;
* grafik;
* tabel;
* download CSV.

### 15.3 Evaluasi Model

Menampilkan:

* MAE;
* RMSE;
* WAPE;
* sMAPE;
* bias;
* perbandingan Prophet dan baseline;
* grafik aktual versus prediksi.

### 15.4 Data Historis

Menampilkan:

* filter poli;
* filter tanggal;
* grafik kunjungan historis;
* tabel agregasi;
* download data.

### 15.5 Informasi Model

Menampilkan:

* deskripsi Prophet;
* periode training;
* tanggal data terakhir;
* konfigurasi model;
* jadwal retraining;
* batas penggunaan model;
* catatan interpretasi.

---

## 16. Output Forecasting

### 16.1 Harian

| Tanggal    | Poli      | Prediksi | Minimum | Maksimum |
| ---------- | --------- | -------: | ------: | -------: |
| 1 Mei 2026 | Poli Anak |       28 |      20 |       36 |
| 2 Mei 2026 | Poli Anak |       24 |      17 |       32 |

### 16.2 Mingguan

| Periode      | Poli      | Prediksi | Minimum | Maksimum |
| ------------ | --------- | -------: | ------: | -------: |
| 1–7 Mei 2026 | Poli Anak |      164 |     132 |      197 |

### 16.3 Bulanan

| Bulan    | Poli      | Prediksi | Minimum | Maksimum |
| -------- | --------- | -------: | ------: | -------: |
| Mei 2026 | Poli Anak |      682 |     573 |      794 |

### 16.4 Format CSV

```text
periode,poli,prediksi,batas_bawah,batas_atas
2026-05-01,POLIKLINIK ANAK,28,20,36
```

---

## 17. Strategi Training dan Retraining

Training tidak dilakukan setiap kali pengguna meminta prediksi.

Alur:

```text
Data baru masuk
      ↓
Preprocessing
      ↓
Training ulang model
      ↓
Evaluasi
      ↓
Model disimpan
      ↓
Streamlit memuat model terbaru
```

Rekomendasi retraining:

* satu kali setiap bulan;
* ketika terdapat tambahan data baru;
* ketika performa model menurun;
* ketika jadwal pelayanan berubah;
* ketika terdapat perubahan besar pada pola kunjungan.

Setiap model harus memiliki metadata:

* nama poli;
* periode training;
* tanggal training;
* jumlah observasi;
* konfigurasi;
* metrik evaluasi;
* versi model.

---

## 18. Kriteria Keberhasilan

Sistem dinyatakan berhasil apabila:

1. Model dapat dibangun untuk setiap poliklinik.
2. Pengguna dapat memilih rentang tanggal secara fleksibel.
3. Sistem dapat menghasilkan prediksi harian.
4. Sistem dapat mengagregasikan hasil menjadi mingguan dan bulanan.
5. Hasil menampilkan batas bawah dan batas atas.
6. Sistem dapat menampilkan grafik dan tabel.
7. Hasil dapat diunduh dalam format CSV.
8. Model dapat dimuat tanpa training ulang saat aplikasi digunakan.
9. Evaluasi model tersedia untuk setiap poli.
10. Data pribadi pasien tidak ditampilkan.
11. Input tidak valid ditangani dengan pesan kesalahan.
12. Prophet dibandingkan dengan baseline.

---

## 19. Risiko dan Mitigasi

### Risiko 1: Periode data historis terbatas

Mitigasi:

* menggunakan model harian;
* membatasi horizon prediksi;
* menampilkan interval ketidakpastian;
* menambah data secara berkala.

### Risiko 2: Data tanggal hilang

Mitigasi:

* tidak otomatis mengisi nol;
* memberikan status data hilang;
* mengecualikan periode tersebut dari training jika diperlukan.

### Risiko 3: Perubahan jadwal poli

Mitigasi:

* memperbarui kalender operasional;
* melakukan retraining;
* menambahkan regressor atau kalender khusus.

### Risiko 4: Prediksi jangka panjang tidak stabil

Mitigasi:

* membatasi forecasting maksimal 365 hari;
* memberi peringatan untuk periode lebih dari 90 atau 180 hari;
* menampilkan rentang ketidakpastian.

### Risiko 5: Prophet kalah dari baseline

Mitigasi:

* melakukan tuning parameter;
* memeriksa pola data;
* mencoba seasonality mode berbeda;
* mempertimbangkan model alternatif pada pengembangan berikutnya.

---

## 20. Tahapan Pengembangan

### Tahap 1: Data Preparation

* membaca dataset;
* membersihkan data;
* menghapus identitas pasien;
* mengagregasi data harian;
* memeriksa missing date;
* menyimpan data processed.

### Tahap 2: Exploratory Data Analysis

* distribusi kunjungan per poli;
* pola harian;
* pola mingguan;
* tren bulanan;
* hari tanpa kunjungan;
* outlier;
* periode data hilang.

### Tahap 3: Modeling

* membuat baseline;
* membagi train, validation, dan test;
* melatih Prophet per poli;
* melakukan tuning;
* menambahkan hari libur;
* menyimpan model.

### Tahap 4: Evaluation

* menghitung MAE;
* menghitung RMSE;
* menghitung WAPE;
* menghitung bias;
* membandingkan baseline;
* menyimpan evaluasi.

### Tahap 5: Streamlit Development

* membuat dashboard;
* membuat halaman forecasting;
* membuat grafik;
* membuat tabel;
* membuat download CSV;
* membuat validasi input.

### Tahap 6: Testing

* pengujian fungsi preprocessing;
* pengujian model;
* pengujian rentang tanggal;
* pengujian agregasi;
* pengujian UI;
* pengujian error handling.

### Tahap 7: Deployment

* menyiapkan requirements;
* mengatur path model;
* menyiapkan konfigurasi;
* deployment ke Streamlit Community Cloud atau server internal;
* pengujian aplikasi hasil deployment.

---

## 21. Minimum Viable Product

Versi MVP harus memiliki:

1. Model Prophet untuk semua poliklinik.
2. Halaman forecasting Streamlit.
3. Pilihan poliklinik.
4. Input tanggal awal dan akhir.
5. Pilihan harian, mingguan, dan bulanan.
6. Grafik forecasting.
7. Tabel hasil.
8. Ringkasan total dan rata-rata.
9. Interval prediksi.
10. Download CSV.
11. Halaman evaluasi model.
12. Validasi input.

---

## 22. Pengembangan Lanjutan

Pengembangan berikutnya dapat mencakup:

* upload dataset melalui UI;
* retraining melalui Streamlit;
* autentikasi pengguna;
* integrasi database;
* integrasi sistem informasi rumah sakit;
* kalender jadwal dokter;
* kalender tutup poli;
* notifikasi prediksi lonjakan pasien;
* perbandingan beberapa algoritma;
* forecasting seluruh rawat jalan;
* prediksi kebutuhan tenaga medis;
* deployment internal rumah sakit;
* pencatatan histori forecasting.

---

## 23. Kesimpulan Teknis

Sistem dibangun menggunakan Streamlit sebagai antarmuka sekaligus lapisan aplikasi. Prophet digunakan untuk menghasilkan prediksi kunjungan harian secara terpisah pada setiap poliklinik.

Pengguna dapat memilih poliklinik, tanggal awal, tanggal akhir, dan bentuk penyajian hasil. Sistem memuat model yang sudah dilatih, menghasilkan prediksi sampai tanggal akhir, mengambil hasil sesuai periode yang dipilih, kemudian menampilkannya dalam bentuk ringkasan, grafik, tabel, dan file CSV.

Prediksi mingguan dan bulanan diperoleh dari agregasi prediksi harian. Model tidak dilatih ulang setiap kali forecasting dijalankan. Training ulang dilakukan secara berkala ketika tersedia data kunjungan aktual terbaru.

---

## 24. Addendum Alur Retraining

Implementasi akhir menggunakan `notebooks/01_crisp_dm_forecasting.ipynb` sebagai satu-satunya alur preprocessing, training, validation, testing, evaluasi baseline, dan retraining model dengan framework CRISP-DM.

Aplikasi Streamlit hanya memuat artefak hasil notebook. Retraining tidak dijalankan dari file Python terpisah dan tidak dilakukan dari antarmuka aplikasi.
