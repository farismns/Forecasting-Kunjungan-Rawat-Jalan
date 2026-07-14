"""Melatih, mengevaluasi, dan menyimpan model Prophet per poliklinik."""

from __future__ import annotations

import argparse
import json
import platform
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import prophet
from prophet import Prophet

from config import (
    APP_TIMEZONE,
    EVALUATION_PREDICTIONS_PATH,
    METADATA_PATH,
    METRICS_PATH,
    MODELS_DIR,
    PROCESSED_DATA_PATH,
    PROPHET_CONFIG,
    PROPHET_TUNING_CANDIDATES,
    TEST_HORIZON_DAYS,
)
from evaluation import calculate_metrics, create_baseline_forecasts
from preprocessing import run_preprocessing


def model_filename(poli: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", poli.lower()).strip("_")
    return f"prophet_{slug}.pkl"


def build_model(configuration: dict | None = None) -> Prophet:
    model = Prophet(**(configuration or PROPHET_CONFIG))
    model.add_country_holidays(country_name="ID")
    return model


def select_configuration(train_pool: pd.DataFrame, poli: str) -> tuple[dict, float]:
    validation_cutoff = train_pool["ds"].max() - pd.Timedelta(days=TEST_HORIZON_DAYS)
    tuning_train = train_pool.loc[train_pool["ds"] <= validation_cutoff, ["ds", "y"]]
    validation = train_pool.loc[train_pool["ds"] > validation_cutoff, ["ds", "y"]]
    if len(tuning_train) < 60 or len(validation) < 7:
        return PROPHET_CONFIG.copy(), float("nan")

    candidates: list[tuple[float, dict]] = []
    for configuration in PROPHET_TUNING_CANDIDATES:
        model = build_model(configuration).fit(tuning_train)
        prediction = model.predict(validation[["ds"]])["yhat"].clip(lower=0)
        mae = calculate_metrics(validation["y"], prediction)["mae"]
        candidates.append((mae, configuration))
    best_mae, best_configuration = min(candidates, key=lambda item: item[0])
    print(f"Tuning: {poli} | MAE validasi {best_mae:.2f}")
    return best_configuration.copy(), best_mae


def evaluate_poli(
    poli_data: pd.DataFrame, poli: str
) -> tuple[list[dict], pd.DataFrame, pd.Timestamp, dict, float]:
    last_date = poli_data["ds"].max()
    cutoff = last_date - pd.Timedelta(days=TEST_HORIZON_DAYS)
    train = poli_data.loc[poli_data["ds"] <= cutoff, ["ds", "y"]]
    test = poli_data.loc[poli_data["ds"] > cutoff, ["ds", "y"]]
    if len(train) < 60 or len(test) < 7:
        raise ValueError(f"Data {poli} tidak cukup untuk evaluasi berbasis waktu.")

    selected_configuration, validation_mae = select_configuration(train, poli)
    evaluation_model = build_model(selected_configuration).fit(train)
    prophet_prediction = evaluation_model.predict(test[["ds"]])[
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ]
    prophet_prediction[["yhat", "yhat_lower", "yhat_upper"]] = prophet_prediction[
        ["yhat", "yhat_lower", "yhat_upper"]
    ].clip(lower=0)
    comparison = test.merge(prophet_prediction, on="ds", how="inner")
    comparison["poli"] = poli

    rows = []
    common = {
        "poli": poli,
        "test_start": test["ds"].min().date().isoformat(),
        "test_end": test["ds"].max().date().isoformat(),
        "n_test": len(test),
    }
    rows.append({**common, "model": "Prophet", **calculate_metrics(comparison["y"], comparison["yhat"])})
    for name, predictions in create_baseline_forecasts(train, test["ds"]).items():
        rows.append({**common, "model": name, **calculate_metrics(test["y"], predictions)})
    return rows, comparison, cutoff, selected_configuration, validation_mae


def train_all(data_path: Path = PROCESSED_DATA_PATH) -> None:
    if not data_path.exists():
        run_preprocessing()
    daily = pd.read_csv(data_path, parse_dates=["ds"])
    required = {"ds", "poli", "y"}
    if daily.empty or not required.issubset(daily.columns):
        raise ValueError("Dataset terproses kosong atau tidak memiliki kolom ds, poli, y.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    trained_at = datetime.now(ZoneInfo(APP_TIMEZONE)).isoformat(timespec="seconds")
    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    metadata: dict = {
        "trained_at": trained_at,
        "algorithm": "Prophet",
        "prophet_version": prophet.__version__,
        "python_version": platform.python_version(),
        "configuration": PROPHET_CONFIG,
        "country_holidays": "ID",
        "test_horizon_days": TEST_HORIZON_DAYS,
        "models": {},
    }

    for poli, poli_data in daily.groupby("poli", sort=True):
        poli_data = poli_data[["ds", "y"]].sort_values("ds").reset_index(drop=True)
        rows, comparison, cutoff, selected_configuration, validation_mae = evaluate_poli(poli_data, poli)
        metric_rows.extend(rows)
        prediction_frames.append(comparison)

        final_model = build_model(selected_configuration).fit(poli_data)
        filename = model_filename(poli)
        joblib.dump(final_model, MODELS_DIR / filename, compress=3)

        prophet_metrics = next(row for row in rows if row["model"] == "Prophet")
        baseline_rows = [row for row in rows if row["model"] != "Prophet"]
        best_baseline = min(baseline_rows, key=lambda row: row["mae"])
        metadata["models"][poli] = {
            "file": filename,
            "data_start": poli_data["ds"].min().date().isoformat(),
            "data_end": poli_data["ds"].max().date().isoformat(),
            "observations": len(poli_data),
            "total_visits": int(poli_data["y"].sum()),
            "evaluation_cutoff": cutoff.date().isoformat(),
            "validation_mae": validation_mae,
            "selected_configuration": selected_configuration,
            "metrics": {key: prophet_metrics[key] for key in ["mae", "rmse", "wape", "smape", "bias"]},
            "best_baseline": best_baseline["model"],
            "best_baseline_mae": best_baseline["mae"],
            "prophet_beats_baseline": bool(prophet_metrics["mae"] <= best_baseline["mae"]),
        }
        print(f"Selesai: {poli} | MAE Prophet {prophet_metrics['mae']:.2f}")

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(METRICS_PATH, index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        EVALUATION_PREDICTIONS_PATH, index=False, date_format="%Y-%m-%d"
    )
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"{len(metadata['models'])} model tersimpan di {MODELS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Training model Prophet per poliklinik.")
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA_PATH)
    args = parser.parse_args()
    train_all(args.data)


if __name__ == "__main__":
    main()
