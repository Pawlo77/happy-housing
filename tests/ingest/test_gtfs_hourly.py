"""Tests for GTFS stop-hour aggregation."""

import zipfile
from pathlib import Path

from happy_housing.ingest import build_gtfs_stop_hourly


def test_build_gtfs_stop_hourly_from_zip(tmp_path: Path) -> None:
    """Departures roll up by stop_id and hour-of-day."""
    zip_path = tmp_path / "gtfs.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(
            "stop_times.txt",
            "stop_id,departure_time\nA,08:10:00\nA,08:40:00\nA,25:10:00\nB,17:05:00\n",
        )
    out = tmp_path / "hourly.parquet"
    frame = build_gtfs_stop_hourly(zip_path, out)
    assert out.exists()
    a8 = frame.loc[(frame["stop_id"] == "A") & (frame["hour"] == 8), "n_departures"]
    assert int(a8.iloc[0]) == 2
    a1 = frame.loc[(frame["stop_id"] == "A") & (frame["hour"] == 1), "n_departures"]
    assert int(a1.iloc[0]) == 1
