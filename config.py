from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "dataset.xlsx"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "daily_visits_by_poli.csv"
QUALITY_REPORT_PATH = BASE_DIR / "data" / "processed" / "quality_report.json"
MODELS_DIR = BASE_DIR / "models"
METRICS_PATH = BASE_DIR / "metrics" / "model_evaluation.csv"
EVALUATION_PREDICTIONS_PATH = BASE_DIR / "metrics" / "evaluation_predictions.csv"
METADATA_PATH = MODELS_DIR / "metadata.json"

DATE_COLUMN = "tglmasuk"
POLI_COLUMN = "poliklinik"
MAX_FORECAST_DAYS = 365
TEST_HORIZON_DAYS = 60
APP_TIMEZONE = "Asia/Jakarta"

PROPHET_CONFIG = {
    "growth": "linear",
    "weekly_seasonality": True,
    "yearly_seasonality": True,
    "daily_seasonality": False,
    "seasonality_mode": "additive",
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "interval_width": 0.95,
}

PROPHET_TUNING_CANDIDATES = [
    {**PROPHET_CONFIG},
    {**PROPHET_CONFIG, "changepoint_prior_scale": 0.01},
    {**PROPHET_CONFIG, "changepoint_prior_scale": 0.10},
    {**PROPHET_CONFIG, "seasonality_mode": "multiplicative", "changepoint_prior_scale": 0.05},
]

GRANULARITY_LABELS = {
    "Harian": "daily",
    "Mingguan": "weekly",
    "Bulanan": "monthly",
}
