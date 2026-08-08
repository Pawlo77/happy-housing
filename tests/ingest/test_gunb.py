"""Tests for GUNB point GeoDataFrame helpers."""

from __future__ import annotations

import pandas as pd

from happy_housing.ingest.gunb import _frame_from_wfs_geojson, gunb_points_gdf


def test_gunb_points_gdf_from_lon_lat() -> None:
    """Lon/lat columns become a non-empty GeoDataFrame."""
    frame = pd.DataFrame({
        "lon": [21.0, 21.1],
        "lat": [52.2, 52.3],
        "adres": ["a", "b"],
    })
    gdf = gunb_points_gdf(frame)
    assert len(gdf) == 2
    assert gdf.crs.to_string() == "EPSG:4326"


def test_gunb_points_gdf_empty_without_coords() -> None:
    """Missing coordinate columns yield an empty GeoDataFrame."""
    frame = pd.DataFrame({"adres": ["x"]})
    gdf = gunb_points_gdf(frame)
    assert gdf.empty


def test_frame_from_wfs_geojson() -> None:
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [21.0, 52.2]},
                "properties": {"adres": "x", "numer_decyzji": "1"},
            }
        ],
    }
    frame = _frame_from_wfs_geojson(payload)
    assert len(frame) == 1
    assert float(frame.iloc[0]["lon"]) == 21.0
