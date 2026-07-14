"""Validasi, inferensi Prophet, dan agregasi hasil forecast."""

from __future__ import annotations

from datetime import date

import pandas as pd

from config import MAX_FORECAST_DAYS


def validate_forecast_range(
    start_date: date | pd.Timestamp,
    end_date: date | pd.Timestamp,
    last_actual_date: date | pd.Timestamp,
    max_days: int = MAX_FORECAST_DAYS,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    last_actual = pd.Timestamp(last_actual_date).normalize()
    if end < start:
        raise ValueError("Tanggal akhir tidak boleh lebih kecil dari tanggal awal.")
    if start <= last_actual:
        raise ValueError("Tanggal awal harus setelah tanggal terakhir data aktual.")
    horizon = (end - start).days + 1
    if horizon > max_days:
        raise ValueError(f"Periode forecasting maksimal {max_days} hari.")
    return start, end


def make_daily_forecast(model, start_date: date, end_date: date, poli: str) -> pd.DataFrame:
    dates = pd.date_range(start_date, end_date, freq="D")
    predicted = model.predict(pd.DataFrame({"ds": dates}))
    result = predicted[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    for column in ["yhat", "yhat_lower", "yhat_upper"]:
        result[column] = result[column].clip(lower=0)
    result["yhat_lower"] = result[["yhat_lower", "yhat"]].min(axis=1)
    result["yhat_upper"] = result[["yhat_upper", "yhat"]].max(axis=1)
    result["poli"] = poli
    return result


def aggregate_forecast(daily: pd.DataFrame, granularity: str) -> pd.DataFrame:
    required = {"ds", "poli", "yhat", "yhat_lower", "yhat_upper"}
    if daily.empty or not required.issubset(daily.columns):
        raise ValueError("Hasil forecast harian tidak lengkap.")

    frame = daily.copy()
    frame["ds"] = pd.to_datetime(frame["ds"])
    if granularity == "daily":
        result = frame.rename(columns={"ds": "periode"})
    elif granularity == "weekly":
        result = (
            frame.set_index("ds")
            .groupby("poli")[["yhat", "yhat_lower", "yhat_upper"]]
            .resample("W-SUN")
            .sum()
            .reset_index()
            .rename(columns={"ds": "periode"})
        )
    elif granularity == "monthly":
        result = (
            frame.set_index("ds")
            .groupby("poli")[["yhat", "yhat_lower", "yhat_upper"]]
            .resample("MS")
            .sum()
            .reset_index()
            .rename(columns={"ds": "periode"})
        )
    else:
        raise ValueError("Granularitas harus daily, weekly, atau monthly.")

    return result[["periode", "poli", "yhat", "yhat_lower", "yhat_upper"]].sort_values("periode")


def format_forecast_export(forecast: pd.DataFrame) -> pd.DataFrame:
    export = forecast.rename(
        columns={
            "yhat": "prediksi",
            "yhat_lower": "batas_bawah",
            "yhat_upper": "batas_atas",
        }
    ).copy()
    for column in ["prediksi", "batas_bawah", "batas_atas"]:
        export[column] = export[column].round(1)
    export["periode"] = pd.to_datetime(export["periode"]).dt.strftime("%Y-%m-%d")
    return export[["periode", "poli", "prediksi", "batas_bawah", "batas_atas"]]

