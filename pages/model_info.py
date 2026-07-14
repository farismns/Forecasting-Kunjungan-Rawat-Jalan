from datetime import datetime

import pandas as pd
import streamlit as st

from ui import load_metadata, load_quality_report, page_header


page_header("Informasi Model", "Konfigurasi, cakupan data, kualitas sumber, dan batas penggunaan model.")
metadata = load_metadata()
quality = load_quality_report()

trained_at = datetime.fromisoformat(metadata["trained_at"])
top = st.columns(4)
top[0].metric("Algoritma", metadata["algorithm"])
top[1].metric("Model tersedia", str(len(metadata["models"])))
top[2].metric("Data terakhir", quality["last_date"])
top[3].metric("Training terakhir", trained_at.strftime("%d %b %Y"))

st.subheader("Cakupan model")
model_rows = []
for poli, info in metadata["models"].items():
    model_rows.append(
        {
            "poliklinik": poli,
            "periode_data": f"{info['data_start']} - {info['data_end']}",
            "observasi": info["observations"],
            "total_kunjungan": info["total_visits"],
            "mae": info["metrics"]["mae"],
            "baseline_terbaik": info["best_baseline"],
            "mode": info["selected_configuration"]["seasonality_mode"],
            "changepoint": info["selected_configuration"]["changepoint_prior_scale"],
        }
    )
st.dataframe(
    pd.DataFrame(model_rows),
    width="stretch",
    hide_index=True,
    column_config={
        "poliklinik": "Poliklinik",
        "periode_data": "Periode data",
        "observasi": "Hari tercatat",
        "total_kunjungan": st.column_config.NumberColumn("Total kunjungan", format="%d"),
        "mae": st.column_config.NumberColumn("MAE", format="%.2f"),
        "baseline_terbaik": "Baseline terbaik",
        "mode": "Mode musiman",
        "changepoint": st.column_config.NumberColumn("Changepoint prior", format="%.2f"),
    },
)

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Konfigurasi default")
    config_rows = [{"parameter": key, "nilai": str(value)} for key, value in metadata["configuration"].items()]
    config_rows.extend(
        [
            {"parameter": "hari_libur", "nilai": metadata["country_holidays"]},
            {"parameter": "prophet_version", "nilai": metadata["prophet_version"]},
            {"parameter": "python_version", "nilai": metadata["python_version"]},
        ]
    )
    st.dataframe(pd.DataFrame(config_rows), width="stretch", hide_index=True)

with right:
    st.subheader("Kualitas sumber data")
    quality_rows = pd.DataFrame(
        [
            ("Baris sumber", quality["source_rows"]),
            ("Baris valid", quality["valid_rows"]),
            ("Duplikat persis dihapus", quality["exact_duplicate_rows"]),
            ("Tanggal tidak valid", quality["invalid_date_rows"]),
            ("Poli kosong", quality["empty_poli_rows"]),
            ("Tanggal kalender tanpa data", quality["missing_calendar_dates"]),
        ],
        columns=["indikator", "nilai"],
    )
    st.dataframe(quality_rows, width="stretch", hide_index=True)

st.subheader("Batas penggunaan")
st.markdown(
    """
- Forecast dibatasi maksimal 365 hari dan ketidakpastian meningkat pada horizon yang lebih panjang.
- Tanggal tanpa sumber data tidak otomatis dianggap sebagai nol kunjungan.
- Perubahan jadwal layanan, kapasitas dokter, atau kebijakan rumah sakit belum menjadi regressor model.
- Hasil ditujukan untuk perencanaan operasional, bukan keputusan atau rekomendasi medis.
- Model perlu dilatih ulang setiap bulan atau setelah data aktual baru tersedia.
    """
)
st.caption(f"Waktu training: {trained_at.strftime('%d %b %Y %H:%M %Z')} · Horizon pengujian: {metadata['test_horizon_days']} hari")
