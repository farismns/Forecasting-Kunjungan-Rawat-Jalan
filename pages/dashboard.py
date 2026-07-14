import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui import COLORS, base_layout, format_number, load_daily_data, page_header


page_header("Dashboard", "Ringkasan kunjungan rawat jalan berdasarkan data agregat harian.")
data = load_daily_data()

total_visits = int(data["y"].sum())
poli_totals = data.groupby("poli", as_index=False)["y"].sum().sort_values("y", ascending=False)
daily_total = data.groupby("ds", as_index=False)["y"].sum().sort_values("ds")
average_daily = daily_total["y"].mean()

cols = st.columns([1, 0.75, 1, 1.4])
cols[0].metric("Total kunjungan", format_number(total_visits))
cols[1].metric("Poliklinik", str(data["poli"].nunique()))
cols[2].metric("Rata-rata per hari", format_number(average_daily, 1))
cols[3].metric("Kunjungan tertinggi", poli_totals.iloc[0]["poli"].replace("POLIKLINIK ", "").title())

st.caption(
    f"Periode data: {data['ds'].min():%d %b %Y} - {data['ds'].max():%d %b %Y} · "
    f"Terakhir diperbarui: {data['ds'].max():%d %b %Y}"
)

left, right = st.columns([1.7, 1], gap="large")
with left:
    daily_total["rata_rata_14_hari"] = daily_total["y"].rolling(14, min_periods=1).mean()
    trend = go.Figure()
    trend.add_trace(
        go.Scatter(
            x=daily_total["ds"], y=daily_total["y"], mode="lines",
            line={"color": "#B8C1BF", "width": 1}, name="Harian",
        )
    )
    trend.add_trace(
        go.Scatter(
            x=daily_total["ds"], y=daily_total["rata_rata_14_hari"], mode="lines",
            line={"color": COLORS["teal"], "width": 2.5}, name="Rata-rata 14 hari",
        )
    )
    trend.update_layout(**base_layout("Tren kunjungan seluruh poliklinik"))
    st.plotly_chart(trend, width="stretch", config={"displayModeBar": False})

with right:
    distribution = px.bar(
        poli_totals.sort_values("y"), x="y", y="poli", orientation="h",
        color_discrete_sequence=[COLORS["orange"]],
        labels={"y": "Kunjungan", "poli": ""},
    )
    distribution.update_traces(hovertemplate="%{y}<br>%{x:,.0f} kunjungan<extra></extra>")
    distribution.update_layout(**base_layout("Distribusi per poliklinik", ""))
    distribution.update_xaxes(title="Jumlah kunjungan")
    st.plotly_chart(distribution, width="stretch", config={"displayModeBar": False})

st.subheader("Aktivitas terkini")
recent = data[data["ds"] >= data["ds"].max() - pd.Timedelta(days=29)]
recent_summary = (
    recent.groupby("poli", as_index=False)["y"]
    .agg(total_30_hari="sum", rata_rata_harian="mean", puncak_harian="max")
    .sort_values("total_30_hari", ascending=False)
)
st.dataframe(
    recent_summary,
    width="stretch",
    hide_index=True,
    column_config={
        "poli": "Poliklinik",
        "total_30_hari": st.column_config.NumberColumn("Total 30 hari", format="%d"),
        "rata_rata_harian": st.column_config.NumberColumn("Rata-rata", format="%.1f"),
        "puncak_harian": st.column_config.NumberColumn("Puncak harian", format="%d"),
    },
)
