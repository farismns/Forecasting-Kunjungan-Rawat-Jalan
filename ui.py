"""Komponen bersama dan akses artefak untuk antarmuka Streamlit."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    EVALUATION_PREDICTIONS_PATH,
    METADATA_PATH,
    METRICS_PATH,
    PROCESSED_DATA_PATH,
    QUALITY_REPORT_PATH,
)


COLORS = {
    "teal": "#087E76",
    "orange": "#D9772B",
    "red": "#C74747",
    "ink": "#172529",
    "muted": "#657276",
    "line": "#DCE2E1",
    "surface": "#FFFFFF",
}


def apply_style() -> None:
    st.markdown(
        """
        <style>
        :root { --accent: #087E76; }
        .stApp { background: #F7F8F8; color: #172529; }
        [data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #DCE2E1; }
        [data-testid="stMetric"] {
            background: #FFFFFF; border: 1px solid #DCE2E1;
            border-radius: 6px; padding: 14px 16px;
        }
        [data-testid="stMetricLabel"] { color: #657276; }
        div[data-testid="stMetricValue"] { color: #172529; font-size: 1.75rem !important; }
        .stButton > button, .stDownloadButton > button { border-radius: 6px; }
        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
            border-radius: 6px;
        }
        h1, h2, h3 { letter-spacing: 0 !important; color: #172529; }
        .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
        .context-line { color: #657276; font-size: 0.9rem; margin-top: -0.7rem; margin-bottom: 1.3rem; }
        @media (max-width: 640px) {
            .block-container { padding-top: 1.2rem; padding-left: 1rem; padding-right: 1rem; }
            div[data-testid="stMetricValue"] { font-size: 1.35rem !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, context: str) -> None:
    st.title(title)
    st.markdown(f'<div class="context-line">{context}</div>', unsafe_allow_html=True)


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        st.error(
            f"{label} belum tersedia. Jalankan seluruh cell pada "
            "`notebooks/01_crisp_dm_forecasting.ipynb`."
        )
        st.stop()


@st.cache_data(show_spinner=False)
def load_daily_data() -> pd.DataFrame:
    _require_file(PROCESSED_DATA_PATH, "Data terproses")
    return pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["ds"])


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    _require_file(METRICS_PATH, "Metrik evaluasi")
    return pd.read_csv(METRICS_PATH)


@st.cache_data(show_spinner=False)
def load_evaluation_predictions() -> pd.DataFrame:
    _require_file(EVALUATION_PREDICTIONS_PATH, "Prediksi evaluasi")
    return pd.read_csv(EVALUATION_PREDICTIONS_PATH, parse_dates=["ds"])


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    _require_file(METADATA_PATH, "Metadata model")
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_quality_report() -> dict:
    _require_file(QUALITY_REPORT_PATH, "Laporan kualitas data")
    return json.loads(QUALITY_REPORT_PATH.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    path = Path(model_path)
    _require_file(path, "Model poliklinik")
    return joblib.load(path)


def base_layout(title: str, y_title: str = "Jumlah kunjungan") -> dict:
    return {
        "title": {"text": title, "font": {"size": 17, "color": COLORS["ink"]}},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": COLORS["surface"],
        "font": {"color": COLORS["ink"], "size": 12},
        "margin": {"l": 16, "r": 16, "t": 52, "b": 20},
        "height": 390,
        "hovermode": "x unified",
        "legend": {"orientation": "h", "y": 1.12, "x": 1, "xanchor": "right"},
        "xaxis": {"title": None, "gridcolor": "#EEF1F0", "showline": False},
        "yaxis": {"title": y_title, "gridcolor": "#E6EAE9", "rangemode": "tozero"},
    }


def forecast_figure(forecast: pd.DataFrame, history: pd.DataFrame | None = None) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=forecast["periode"], y=forecast["yhat_upper"], mode="lines",
            line={"width": 0}, hoverinfo="skip", showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast["periode"], y=forecast["yhat_lower"], mode="lines",
            line={"width": 0}, fill="tonexty", fillcolor="rgba(8,126,118,0.15)",
            name="Interval 95%", hovertemplate="Batas bawah %{y:.1f}<extra></extra>",
        )
    )
    if history is not None and not history.empty:
        figure.add_trace(
            go.Scatter(
                x=history["ds"], y=history["y"], mode="lines",
                line={"color": COLORS["muted"], "width": 1.5}, name="Aktual",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=forecast["periode"], y=forecast["yhat"], mode="lines+markers",
            line={"color": COLORS["teal"], "width": 2.5}, marker={"size": 4}, name="Prediksi",
            hovertemplate="%{x|%d %b %Y}<br>%{y:.1f} kunjungan<extra></extra>",
        )
    )
    figure.update_layout(**base_layout("Prediksi dan rentang ketidakpastian"))
    return figure


def format_number(value: float, decimals: int = 0) -> str:
    formatted = f"{value:,.{decimals}f}"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")
