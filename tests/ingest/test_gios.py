"""Tests for GIOŚ fixture loading and GTFS local-zip export."""

from __future__ import annotations

from pathlib import Path

from happy_housing.ingest import (
    _parse_gios_station_row,
    export_gtfs_stops_from_zip,
    fetch_gios_stations,
    load_gios_stations_json,
)


def test_load_gios_stations_json_fixture() -> None:
    frame = load_gios_stations_json(Path("data/fixtures/gios_stations.json"))
    assert len(frame) == 2
    assert set(frame.columns) >= {"station_id", "lat", "lon", "name", "city"}


def test_fetch_gios_falls_back_to_fixture(tmp_path: Path) -> None:
    out = tmp_path / "gios.parquet"
    frame = fetch_gios_stations(
        "http://127.0.0.1:9/not-a-real-gios",
        out,
        fixture_path=Path("data/fixtures/gios_stations.json"),
    )
    assert out.exists()
    assert len(frame) == 2


def test_parse_gios_station_row_reads_top_level_city() -> None:
    row = {
        "Identyfikator stacji": 1,
        "Kod stacji": "Wz-1",
        "Nazwa stacji": "Test",
        "WGS84 φ N": 52.2,
        "WGS84 λ E": 21.0,
        "Nazwa miasta": "Warszawa",
        "Ulica": "Krucza",
        "Województwo": "mazowieckie",
    }
    parsed = _parse_gios_station_row(row)
    assert parsed is not None
    assert parsed["city"] == "Warszawa"
    assert parsed["station_code"] == "Wz-1"
    assert parsed["street"] == "Krucza"
    assert parsed["voivodeship"] == "mazowieckie"


def test_export_gtfs_stops_from_fixture_zip(tmp_path: Path) -> None:
    out = tmp_path / "stops.parquet"
    stops = export_gtfs_stops_from_zip(Path("data/fixtures/gtfs_sample.zip"), out)
    assert out.exists()
    assert len(stops) == 2
    assert (tmp_path / "gtfs_stop_hourly.parquet").exists()
