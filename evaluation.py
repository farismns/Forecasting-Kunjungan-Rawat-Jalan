"""Metrik dan baseline time-series untuk evaluasi model."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_metrics(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> dict[str, float]:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if len(actual_values) == 0 or len(actual_values) != len(predicted_values):
        raise ValueError("Aktual dan prediksi harus memiliki panjang yang sama dan tidak kosong.")

    error = predicted_values - actual_values
    denominator = np.abs(actual_values) + np.abs(predicted_values)
    smape_terms = np.divide(
        2 * np.abs(error),
        denominator,
        out=np.zeros_like(error, dtype=float),
        where=denominator != 0,
    )
    actual_sum = np.abs(actual_values).sum()
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "wape": float(np.abs(error).sum() / actual_sum * 100) if actual_sum else 0.0,
        "smape": float(np.mean(smape_terms) * 100),
        "bias": float(np.mean(error)),
    }


def create_baseline_forecasts(train: pd.DataFrame, test_dates: pd.Series) -> dict[str, np.ndarray]:
    ordered = train.sort_values("ds")
    values = ordered["y"].astype(float)
    if values.empty:
        raise ValueError("Data training baseline kosong.")

    last_value = float(values.iloc[-1])
    weekday_latest = (
        ordered.assign(weekday=ordered["ds"].dt.weekday)
        .groupby("weekday", observed=True)["y"]
        .last()
        .to_dict()
    )
    seasonal = np.array(
        [float(weekday_latest.get(pd.Timestamp(date).weekday(), last_value)) for date in test_dates]
    )
    return {
        "Naive": np.repeat(last_value, len(test_dates)),
        "Seasonal Naive 7": seasonal,
        "Moving Average 7": np.repeat(float(values.tail(7).mean()), len(test_dates)),
        "Moving Average 28": np.repeat(float(values.tail(28).mean()), len(test_dates)),
    }

