import pandas as pd

from preprocessing import preprocess_frames


def test_preprocess_normalizes_aggregates_and_removes_exact_duplicates():
    frame = pd.DataFrame(
        {
            "tglmasuk": ["2025-01-01", "2025-01-01", "2025-01-01", "2025-01-03", None],
            "poliklinik": [" Poli Anak ", "POLI ANAK", "POLI ANAK", "Poli Mata", "Poli Mata"],
            "visit_id": [1, 2, 2, 3, 4],
        }
    )

    daily, report = preprocess_frames([frame])

    anak = daily[daily["poli"] == "POLI ANAK"].iloc[0]
    assert anak["y"] == 2
    assert report.source_rows == 5
    assert report.valid_rows == 3
    assert report.exact_duplicate_rows == 1
    assert report.invalid_date_rows == 1
    assert report.missing_calendar_dates == 1


def test_preprocess_rejects_missing_required_columns():
    frame = pd.DataFrame({"tanggal": ["2025-01-01"]})
    try:
        preprocess_frames([frame])
    except ValueError as error:
        assert "Kolom wajib" in str(error)
    else:
        raise AssertionError("Seharusnya menolak frame tanpa kolom wajib")

