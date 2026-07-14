from datetime import date

import pandas as pd
import pytest

from forecasting import aggregate_forecast, format_forecast_export, validate_forecast_range


def sample_forecast():
    return pd.DataFrame(
        {
            "ds": pd.date_range("2026-05-01", periods=10),
            "poli": "POLIKLINIK ANAK",
            "yhat": 10.0,
            "yhat_lower": 8.0,
            "yhat_upper": 12.0,
        }
    )


def test_validate_forecast_range():
    start, end = validate_forecast_range(date(2026, 5, 1), date(2026, 5, 30), date(2026, 4, 30))
    assert (end - start).days + 1 == 30

    with pytest.raises(ValueError, match="setelah"):
        validate_forecast_range(date(2026, 4, 30), date(2026, 5, 2), date(2026, 4, 30))
    with pytest.raises(ValueError, match="maksimal"):
        validate_forecast_range(date(2026, 5, 1), date(2027, 5, 1), date(2026, 4, 30))


def test_aggregate_forecast_sums_daily_values():
    weekly = aggregate_forecast(sample_forecast(), "weekly")
    assert weekly["yhat"].sum() == 100
    assert weekly["yhat_lower"].sum() == 80
    assert weekly["yhat_upper"].sum() == 120

    monthly = aggregate_forecast(sample_forecast(), "monthly")
    export = format_forecast_export(monthly)
    assert export.iloc[0]["prediksi"] == 100
    assert list(export.columns) == ["periode", "poli", "prediksi", "batas_bawah", "batas_atas"]

