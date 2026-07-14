import numpy as np
import pandas as pd

from evaluation import calculate_metrics, create_baseline_forecasts


def test_calculate_metrics_handles_zero_actuals():
    result = calculate_metrics([0, 10, 20], [0, 12, 18])
    assert result["mae"] == 4 / 3
    assert np.isfinite(result["wape"])
    assert np.isfinite(result["smape"])
    assert result["bias"] == 0


def test_baselines_return_one_value_per_test_date():
    train = pd.DataFrame(
        {"ds": pd.date_range("2025-01-01", periods=35), "y": np.arange(1, 36)}
    )
    dates = pd.Series(pd.date_range("2025-02-05", periods=10))
    forecasts = create_baseline_forecasts(train, dates)
    assert set(forecasts) == {
        "Naive", "Seasonal Naive 7", "Moving Average 7", "Moving Average 28"
    }
    assert all(len(values) == len(dates) for values in forecasts.values())
    assert forecasts["Naive"][0] == 35

