import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui import COLORS, base_layout, load_evaluation_predictions, load_metadata, load_metrics, page_header


page_header("Evaluasi Model", "Pengujian berbasis urutan waktu pada 60 hari kalender terakhir setiap poliklinik.")
metrics = load_metrics()
predictions = load_evaluation_predictions()
metadata = load_metadata()

prophet_metrics = metrics[metrics["model"] == "Prophet"].copy()
average = prophet_metrics[["mae", "rmse", "wape", "smape", "bias"]].mean()
summary = st.columns(5)
summary[0].metric("MAE rata-rata", f"{average['mae']:.2f}")
summary[1].metric("RMSE rata-rata", f"{average['rmse']:.2f}")
summary[2].metric("WAPE rata-rata", f"{average['wape']:.1f}%")
summary[3].metric("sMAPE rata-rata", f"{average['smape']:.1f}%")
summary[4].metric("Bias rata-rata", f"{average['bias']:+.2f}")

st.subheader("Metrik Prophet per poliklinik")
display = prophet_metrics[["poli", "mae", "rmse", "wape", "smape", "bias", "n_test"]]
st.dataframe(
    display,
    width="stretch",
    hide_index=True,
    column_config={
        "poli": "Poliklinik",
        "mae": st.column_config.NumberColumn("MAE", format="%.2f"),
        "rmse": st.column_config.NumberColumn("RMSE", format="%.2f"),
        "wape": st.column_config.NumberColumn("WAPE", format="%.1f%%"),
        "smape": st.column_config.NumberColumn("sMAPE", format="%.1f%%"),
        "bias": st.column_config.NumberColumn("Bias", format="%+.2f"),
        "n_test": "Observasi uji",
    },
)

poli = st.selectbox("Poliklinik", sorted(prophet_metrics["poli"].unique()))
selected_metrics = metrics[metrics["poli"] == poli].sort_values("mae")
selected_predictions = predictions[predictions["poli"] == poli].sort_values("ds")

left, right = st.columns([1, 1.55], gap="large")
with left:
    comparison = px.bar(
        selected_metrics,
        x="mae",
        y="model",
        orientation="h",
        color="model",
        color_discrete_map={
            "Prophet": COLORS["teal"],
            "Naive": "#7A8583",
            "Seasonal Naive 7": COLORS["orange"],
            "Moving Average 7": "#4F78A8",
            "Moving Average 28": "#A05D8C",
        },
        labels={"mae": "MAE", "model": ""},
    )
    comparison.update_layout(**base_layout("Perbandingan kesalahan absolut", ""), showlegend=False)
    comparison.update_xaxes(title="MAE (lebih rendah lebih baik)")
    st.plotly_chart(comparison, width="stretch", config={"displayModeBar": False})

with right:
    actual_chart = go.Figure()
    actual_chart.add_trace(
        go.Scatter(
            x=selected_predictions["ds"], y=selected_predictions["y"],
            mode="lines+markers", line={"color": COLORS["ink"], "width": 2},
            marker={"size": 4}, name="Aktual",
        )
    )
    actual_chart.add_trace(
        go.Scatter(
            x=selected_predictions["ds"], y=selected_predictions["yhat"],
            mode="lines", line={"color": COLORS["teal"], "width": 2.5}, name="Prophet",
        )
    )
    actual_chart.update_layout(**base_layout("Aktual dibanding prediksi Prophet"))
    st.plotly_chart(actual_chart, width="stretch", config={"displayModeBar": False})

info = metadata["models"][poli]
if info["prophet_beats_baseline"]:
    st.success(f"Prophet setara atau lebih baik dari baseline terbaik ({info['best_baseline']}) berdasarkan MAE.")
else:
    st.warning(
        f"Baseline {info['best_baseline']} memiliki MAE lebih rendah. Forecast Prophet perlu ditafsirkan dengan kehati-hatian."
    )
st.caption("MAE menyatakan rata-rata selisih absolut dalam jumlah pasien. Nilai bias positif berarti model cenderung memprediksi terlalu tinggi.")

