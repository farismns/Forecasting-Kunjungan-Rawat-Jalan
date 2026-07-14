"""Pembersihan dataset kunjungan dan agregasi aman tanpa data identitas."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from config import (
    DATE_COLUMN,
    POLI_COLUMN,
    PROCESSED_DATA_PATH,
    QUALITY_REPORT_PATH,
    RAW_DATA_PATH,
)


@dataclass
class QualityReport:
    source_rows: int
    valid_rows: int
    invalid_date_rows: int
    empty_poli_rows: int
    exact_duplicate_rows: int
    first_date: str
    last_date: str
    observed_dates: int
    missing_calendar_dates: int
    poliklinik_count: int


def normalize_poli(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.upper()
        .replace("", pd.NA)
    )


def preprocess_frames(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, QualityReport]:
    if not frames:
        raise ValueError("Tidak ada sheet berisi data yang dapat diproses.")

    normalized: list[pd.DataFrame] = []
    source_rows = 0
    duplicate_rows = 0
    invalid_date_rows = 0
    empty_poli_rows = 0

    for frame in frames:
        if DATE_COLUMN not in frame.columns or POLI_COLUMN not in frame.columns:
            continue
        source_rows += len(frame)
        duplicate_mask = frame.duplicated(keep="first")
        duplicate_rows += int(duplicate_mask.sum())
        frame = frame.loc[~duplicate_mask, [DATE_COLUMN, POLI_COLUMN]].copy()
        frame["ds"] = pd.to_datetime(frame[DATE_COLUMN], errors="coerce").dt.normalize()
        frame["poli"] = normalize_poli(frame[POLI_COLUMN])
        invalid_date_rows += int(frame["ds"].isna().sum())
        empty_poli_rows += int(frame["poli"].isna().sum())
        normalized.append(frame[["ds", "poli"]].dropna())

    if not normalized:
        raise ValueError(
            f"Kolom wajib '{DATE_COLUMN}' dan '{POLI_COLUMN}' tidak ditemukan."
        )

    visits = pd.concat(normalized, ignore_index=True)
    if visits.empty:
        raise ValueError("Tidak ada baris valid setelah pembersihan data.")

    daily = (
        visits.groupby(["ds", "poli"], as_index=False, observed=True)
        .size()
        .rename(columns={"size": "y"})
        .sort_values(["poli", "ds"], ignore_index=True)
    )
    calendar = pd.date_range(visits["ds"].min(), visits["ds"].max(), freq="D")
    report = QualityReport(
        source_rows=source_rows,
        valid_rows=len(visits),
        invalid_date_rows=invalid_date_rows,
        empty_poli_rows=empty_poli_rows,
        exact_duplicate_rows=duplicate_rows,
        first_date=visits["ds"].min().date().isoformat(),
        last_date=visits["ds"].max().date().isoformat(),
        observed_dates=int(visits["ds"].nunique()),
        missing_calendar_dates=int(len(calendar.difference(visits["ds"].unique()))),
        poliklinik_count=int(visits["poli"].nunique()),
    )
    return daily, report


def read_source(path: Path) -> list[pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
    workbook = pd.ExcelFile(path)
    frames = []
    for sheet in workbook.sheet_names:
        header = pd.read_excel(path, sheet_name=sheet, nrows=0)
        if DATE_COLUMN in header.columns and POLI_COLUMN in header.columns:
            frames.append(pd.read_excel(path, sheet_name=sheet))
    return frames


def run_preprocessing(
    source: Path = RAW_DATA_PATH,
    output: Path = PROCESSED_DATA_PATH,
    report_path: Path = QUALITY_REPORT_PATH,
) -> tuple[pd.DataFrame, QualityReport]:
    daily, report = preprocess_frames(read_source(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output, index=False, date_format="%Y-%m-%d")
    report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return daily, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Agregasi kunjungan rawat jalan per hari dan poli.")
    parser.add_argument("--source", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--output", type=Path, default=PROCESSED_DATA_PATH)
    args = parser.parse_args()
    daily, report = run_preprocessing(args.source, args.output)
    print(f"Tersimpan {len(daily):,} baris agregat ke {args.output}")
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()

