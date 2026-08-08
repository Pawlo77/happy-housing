"""Tests for GIOŚ air-quality index / pollutant parsing and station enrich."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from happy_housing.ingest.gios_aq import (
    enrich_stations_with_air_quality,
    latest_sensor_value,
    load_gios_air_quality_fixture,
    parse_gios_aq_index,
)


def test_parse_gios_aq_index_maps_polish_fields() -> None:
    payload = {
        "AqIndex": {
            "Wartość indeksu": 1,
            "Wartość indeksu dla wskaźnika PM10": 0,
            "Wartość indeksu dla wskaźnika PM2.5": 0,
            "Wartość indeksu dla wskaźnika NO2": 0,
            "Nazwa kategorii indeksu": "Dobry",
        }
    }
    parsed = parse_gios_aq_index(payload)
    assert parsed["aq_index"] == 1.0
    assert parsed["aq_pm10_index"] == 0.0
    assert parsed["aq_pm25_index"] == 0.0
    assert parsed["aq_no2_index"] == 0.0


def test_parse_gios_aq_index_treats_missing_as_nan() -> None:
    parsed = parse_gios_aq_index({"AqIndex": {"Wartość indeksu": -1}})
    assert parsed["aq_index"] != parsed["aq_index"]  # NaN


def test_latest_sensor_value_skips_nulls() -> None:
    rows = [
        {"Data": "2026-08-08 16:00:00", "Wartość": None},
        {"Data": "2026-08-08 15:00:00", "Wartość": None},
        {"Data": "2026-08-08 14:00:00", "Wartość": 5.7},
    ]
    assert latest_sensor_value(rows) == 5.7


def test_load_gios_air_quality_fixture() -> None:
    frame = load_gios_air_quality_fixture(Path("data/fixtures/gios_air_quality.json"))
    assert set(frame["station_id"]) == {1, 2}
    assert float(frame.loc[frame["station_id"] == 1, "pm10_ug_m3"].iloc[0]) == 12.0


def test_enrich_stations_with_air_quality_uses_fixture_on_http_failure(
    tmp_path: Path,
) -> None:
    stations = pd.DataFrame({
        "station_id": [1, 2],
        "lat": [52.23, 52.14],
        "lon": [21.01, 21.03],
        "name": ["A", "B"],
        "city": ["Warszawa", "Warszawa"],
    })
    with patch(
        "happy_housing.ingest.gios_aq._fetch_station_air_quality",
        side_effect=OSError("offline"),
    ):
        out = enrich_stations_with_air_quality(
            stations,
            fixture_path=Path("data/fixtures/gios_air_quality.json"),
        )
    assert float(out.loc[out["station_id"] == 1, "aq_index"].iloc[0]) == 1.0
    assert float(out.loc[out["station_id"] == 2, "pm25_ug_m3"].iloc[0]) == 22.0
    assert "aq_index" in out.columns
