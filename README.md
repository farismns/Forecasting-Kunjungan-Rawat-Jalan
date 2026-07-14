# Sistem Forecasting Kunjungan Rawat Jalan

Aplikasi Streamlit untuk memprediksi jumlah kunjungan harian per poliklinik dengan Prophet. Model dibuat terpisah untuk setiap poli, diuji menggunakan pemisahan waktu, dibandingkan dengan empat baseline, dan disimpan agar aplikasi tidak melakukan training saat digunakan.

## Menjalankan proyek

Gunakan Python 3.11 atau 3.12.

```powershell
python -m pip install -r requirements-dev.txt
python preprocessing.py
python train_models.py
python -m streamlit run app.py
```

Aplikasi tersedia secara lokal di `http://localhost:8501`.

## Alur data

1. `preprocessing.py` membaca semua sheet yang memiliki kolom `tglmasuk` dan `poliklinik`.
2. Baris kosong, tanggal tidak valid, dan duplikat persis dibersihkan.
3. Data langsung diagregasi menjadi `ds,poli,y`; identitas pasien tidak disimpan pada artefak terproses.
4. `train_models.py` memilih konfigurasi pada periode validasi, menggunakan 60 hari kalender terakhir sebagai periode uji yang tidak disentuh saat tuning, lalu melatih ulang model final dengan seluruh data.
5. Aplikasi memuat model dari `models/` melalui cache Streamlit.

Tanggal kalender yang tidak ada pada sumber tidak diisi dengan nol. Hal ini membedakan data yang tidak tersedia dari hari operasional dengan nol kunjungan.

## Struktur utama

```text
app.py                  navigasi aplikasi
preprocessing.py        pembersihan dan agregasi
train_models.py         training, evaluasi, penyimpanan model
forecasting.py          validasi dan agregasi forecast
evaluation.py           metrik dan baseline
pages/                  lima halaman Streamlit
data/processed/         agregat aman dan laporan kualitas
models/                 model serta metadata training
metrics/                metrik dan prediksi periode uji
tests/                  unit test fungsi utama
```

## Evaluasi

Metrik yang disimpan adalah MAE, RMSE, WAPE, sMAPE, dan bias. Prophet dibandingkan dengan naive, seasonal naive tujuh hari, moving average tujuh hari, dan moving average 28 hari. Pemilihan konfigurasi memakai periode validasi sebelum test; seluruh split mengikuti urutan waktu sehingga data masa depan tidak masuk ke training evaluasi.

## Retraining

Jalankan kembali preprocessing dan training setiap bulan, setelah data baru ditambahkan, atau ketika jadwal operasional berubah. Forecast dibatasi 365 hari; horizon di atas 90 hari ditandai di aplikasi karena ketidakpastiannya lebih tinggi.
