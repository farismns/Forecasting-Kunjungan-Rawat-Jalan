import plotly.graph_objects as go
import streamlit as st

from ui import COLORS, base_layout, format_number, load_daily_data, page_header


page_header("Data Historis", "Eksplorasi agregat kunjungan harian tanpa identitas pasien.")
data = load_daily_data()

controls = st.columns([1.5, 1.2, 1], gap="medium")
poli = controls[0].selectbox("Poliklinik", sorted(data["poli"].unique()))
date_range = controls[1].date_input(
    "Rentang tanggal",
    value=(data["ds"].min().date(), data["ds"].max().date()),
    min_value=data["ds"].min().date(),
    max_value=data["ds"].max().date(),
)
aggregation = controls[2].segmented_control(
    "Tampilan", ["Harian", "Mingguan", "Bulanan"], default="Harian", selection_mode="single"
)

if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.info("Pilih tanggal awal dan tanggal akhir.")
    st.stop()
start, end = date_range
selected = data[
    (data["poli"] == poli)
    & (data["ds"].dt.date >= start)
    & (data["ds"].dt.date <= end)
].copy()
if selected.empty:
    st.warning("Tidak ada data pada rentang yang dipilih.")
    st.stop()

if aggregation == "Mingguan":
    shown = selected.set_index("ds")["y"].resample("W-SUN").sum().reset_index()
elif aggregation == "Bulanan":
    shown = selected.set_index("ds")["y"].resample("MS").sum().reset_index()
else:
    shown = selected[["ds", "y"]]

metrics = st.columns(4)
metrics[0].metric("Total kunjungan", format_number(selected["y"].sum()))
metrics[1].metric("Rata-rata harian", format_number(selected["y"].mean(), 1))
metrics[2].metric("Kunjungan tertinggi", format_number(selected["y"].max()))
metrics[3].metric("Hari tercatat", format_number(selected["ds"].nunique()))

chart = go.Figure(
    go.Scatter(
        x=shown["ds"], y=shown["y"], mode="lines",
        fill="tozeroy", fillcolor="rgba(217,119,43,0.10)",
        line={"color": COLORS["orange"], "width": 2}, name="Kunjungan",
        hovertemplate="%{x|%d %b %Y}<br>%{y:.0f} kunjungan<extra></extra>",
    )
)
chart.update_layout(**base_layout(f"Riwayat {aggregation.lower()}"))
st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})

table = shown.rename(columns={"ds": "tanggal", "y": "jumlah_kunjungan"})
table["poliklinik"] = poli
table = table[["tanggal", "poliklinik", "jumlah_kunjungan"]]
st.dataframe(
    table,
    width="stretch",
    hide_index=True,
    column_config={
        "tanggal": st.column_config.DateColumn("Periode", format="DD MMM YYYY"),
        "poliklinik": "Poliklinik",
        "jumlah_kunjungan": st.column_config.NumberColumn("Jumlah kunjungan", format="%d"),
    },
)
st.download_button(
    "Unduh CSV",
    table.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"historis_{poli.lower().replace(' ', '_')}_{start}_{end}.csv",
    mime="text/csv",
    icon=":material/download:",
)

