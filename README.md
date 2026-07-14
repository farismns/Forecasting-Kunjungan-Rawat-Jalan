# Sistem Forecasting Kunjungan Rawat Jalan

Aplikasi Streamlit untuk memprediksi jumlah kunjungan harian per poliklinik dengan Prophet. Model dibuat terpisah untuk setiap poli, diuji menggunakan pemisahan waktu, dibandingkan dengan empat baseline, dan disimpan agar aplikasi tidak melakukan training saat digunakan.

## Menjalankan proyek

Gunakan Python 3.11 atau 3.12.

```powershell
python -m pip install -r requirements-dev.txt
python -m streamlit run app.py
```

Aplikasi tersedia secara lokal di `http://localhost:8501`.

## Notebook CRISP-DM

Alur pembangunan model lengkap tersedia di `notebooks/01_crisp_dm_forecasting.ipynb`. Notebook tersebut menggabungkan business understanding, data understanding, preprocessing, EDA, time-based train/validation/test split, tuning Prophet, perbandingan baseline, evaluasi, dan penyimpanan artefak aplikasi. Seed `42` digunakan agar proses dapat direproduksi.

```powershell
python -m jupyterlab
```

Buka notebook dari JupyterLab, pilih **Restart Kernel and Run All Cells**, lalu pastikan seluruh sanity check lulus. `dataset.xlsx` harus tersedia di root proyek dan tetap diabaikan oleh Git karena mengandung identitas pasien.

Notebook tersebut adalah **satu-satunya alur resmi untuk preprocessing, training, testing, evaluasi, dan retraining**. Aplikasi Streamlit hanya memuat artefak yang dihasilkan notebook dan tidak melakukan training.

## Alur data

1. Notebook CRISP-DM membaca sheet dengan kolom `tglmasuk` dan `poliklinik`.
2. Notebook membersihkan baris kosong, tanggal tidak valid, dan duplikat persis.
3. Data langsung diagregasi menjadi `ds,poli,y`; identitas pasien tidak disimpan pada artefak terproses.
4. Notebook memilih konfigurasi pada validation split dan mengevaluasinya satu kali pada test split 60 hari.
5. Notebook melatih model final dengan seluruh histori dan menyimpan model, metrik, serta metadata.
6. Aplikasi memuat artefak tersebut melalui cache Streamlit tanpa menjalankan proses training.

Tanggal kalender yang tidak ada pada sumber tidak diisi dengan nol. Hal ini membedakan data yang tidak tersedia dari hari operasional dengan nol kunjungan.

## Struktur utama

```text
app.py                  navigasi aplikasi
notebooks/              satu-satunya alur pembangunan dan retraining model
forecasting.py          validasi dan agregasi forecast
pages/                  lima halaman Streamlit
data/processed/         agregat aman dan laporan kualitas
models/                 model serta metadata training
metrics/                metrik dan prediksi periode uji
tests/                  unit test fungsi utama
```

## Evaluasi

Metrik yang disimpan notebook adalah MAE, RMSE, WAPE, sMAPE, dan bias. Prophet dibandingkan dengan naive, seasonal naive tujuh hari, moving average tujuh hari, dan moving average 28 hari. Pemilihan konfigurasi memakai periode validasi sebelum test; seluruh split mengikuti urutan waktu sehingga data masa depan tidak masuk ke training evaluasi.

## Retraining

Perbarui `dataset.xlsx`, buka `notebooks/01_crisp_dm_forecasting.ipynb`, kemudian jalankan **Restart Kernel and Run All Cells** setiap bulan, setelah data baru ditambahkan, atau ketika jadwal operasional berubah. Setelah seluruh cell berhasil, restart aplikasi Streamlit agar cache model dimuat ulang. Forecast dibatasi 365 hari; horizon di atas 90 hari ditandai di aplikasi karena ketidakpastiannya lebih tinggi.
