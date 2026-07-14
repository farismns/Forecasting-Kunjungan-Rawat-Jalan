from datetime import timedelta

import pandas as pd
import streamlit as st

from config import GRANULARITY_LABELS, MAX_FORECAST_DAYS, MODELS_DIR
from forecasting import aggregate_forecast, format_forecast_export, make_daily_forecast, validate_forecast_range
from ui import forecast_figure, format_number, load_daily_data, load_metadata, load_model, page_header


page_header("Forecasting", "Proyeksi kunjungan masa depan per poliklinik dengan interval ketidakpastian 95%.")
data = load_daily_data()
metadata = load_metadata()
available_poli = sorted(metadata["models"])
last_actual = data["ds"].max().date()

with st.container(border=True):
    with st.form("forecast_controls"):
        row = st.columns([1.6, 1, 1, 1.1], gap="medium")
        poli = row[0].selectbox("Poliklinik", available_poli)
        start = row[1].date_input(
            "Tanggal awal", value=last_actual + timedelta(days=1), min_value=last_actual + timedelta(days=1)
        )
        end = row[2].date_input(
            "Tanggal akhir", value=last_actual + timedelta(days=30), min_value=last_actual + timedelta(days=1)
        )
        granularity_label = row[3].segmented_control(
            "Granularitas", list(GRANULARITY_LABELS), default="Harian", selection_mode="single"
        )
        submitted = st.form_submit_button("Buat forecast", icon=":material/query_stats:", type="primary")

if not submitted:
    st.info("Atur periode dan jalankan forecast untuk melihat hasil.", icon=":material/date_range:")
    st.stop()

try:
    start_ts, end_ts = validate_forecast_range(start, end, last_actual)
except ValueError as error:
    st.error(str(error))
    st.stop()

horizon = (end_ts - start_ts).days + 1
if horizon > 180:
    st.warning("Horizon di atas 180 hari memiliki ketidakpastian lebih tinggi.")
elif horizon > 90:
    st.info("Horizon di atas 90 hari perlu ditinjau bersama perubahan jadwal operasional.")

model_info = metadata["models"].get(poli)
if not model_info:
    st.error("Metadata model poliklinik tidak ditemukan.")
    st.stop()

try:
    with st.spinner("Menghitung forecast..."):
        model = load_model(str(MODELS_DIR / model_info["file"]))
        daily_forecast = make_daily_forecast(model, start_ts, end_ts, poli)
        forecast = aggregate_forecast(daily_forecast, GRANULARITY_LABELS[granularity_label])
except Exception as error:
    st.error(f"Forecasting gagal diproses: {error}")
    st.stop()

highest = forecast.loc[forecast["yhat"].idxmax()]
lowest = forecast.loc[forecast["yhat"].idxmin()]
summary = st.columns(4)
summary[0].metric("Total prediksi", format_number(forecast["yhat"].sum(), 1))
summary[1].metric("Rata-rata per periode", format_number(forecast["yhat"].mean(), 1))
summary[2].metric("Prediksi tertinggi", format_number(highest["yhat"], 1), highest["periode"].strftime("%d %b %Y"))
summary[3].metric("Prediksi terendah", format_number(lowest["yhat"], 1), lowest["periode"].strftime("%d %b %Y"))

history = data[(data["poli"] == poli) & (data["ds"] >= data["ds"].max() - pd.Timedelta(days=60))]
chart_history = history if GRANULARITY_LABELS[granularity_label] == "daily" else None
st.plotly_chart(forecast_figure(forecast, chart_history), width="stretch", config={"displayModeBar": False})

export = format_forecast_export(forecast)
st.subheader("Rincian hasil")
st.dataframe(
    export,
    width="stretch",
    hide_index=True,
    column_config={
        "periode": "Periode",
        "poli": "Poliklinik",
        "prediksi": st.column_config.NumberColumn("Prediksi", format="%.1f"),
        "batas_bawah": st.column_config.NumberColumn("Batas bawah", format="%.1f"),
        "batas_atas": st.column_config.NumberColumn("Batas atas", format="%.1f"),
    },
)
st.download_button(
    "Unduh CSV",
    export.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"forecast_{poli.lower().replace(' ', '_')}_{start}_{end}.csv",
    mime="text/csv",
    icon=":material/download:",
)
st.caption(f"Model dilatih hingga {model_info['data_end']} · MAE evaluasi {model_info['metrics']['mae']:.2f} pasien")

