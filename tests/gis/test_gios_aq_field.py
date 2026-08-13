"""Tests for continuous GIOŚ air-quality IDW field (model + map)."""

from pathlib import Path

import numpy as np
import pandas as pd

from happy_housing.app.maps import air_quality_bitmap_layer
from happy_housing.gis.gios_aq_field import (
    AQ_FEATURE_COLUMNS,
    AQ_MAP_OVERLAYS,
    AQ_POOR_INDEX_THRESHOLD,
    AqMapOverlay,
    attach_gios_aq_field_features,
    build_gios_aq_idw_grids,
    fetch_gios_aq_field_rasters,
    prepare_aq_overlay_for_key,
    sample_aq_field,
)


def _stations() -> pd.DataFrame:
    return pd.DataFrame({
        "station_id": [1, 2],
        "lat": [52.23, 52.14],
        "lon": [21.01, 21.03],
        "aq_index": [1.0, 4.0],
        "pm10_ug_m3": [10.0, 80.0],
        "pm25_ug_m3": [5.0, 50.0],
        "no2_ug_m3": [12.0, 55.0],
    })


def test_aq_overlay_choices_include_pollutants() -> None:
    keys = {key for _, key in AQ_MAP_OVERLAYS}
    assert "off" in keys
    assert "aq_index" in keys
    assert "pm25_ug_m3" in keys
    assert set(AQ_FEATURE_COLUMNS) >= {
        "air_aq_index",
        "air_pm10_ug_m3",
        "air_pm25_ug_m3",
        "air_no2_ug_m3",
        "in_poor_air",
    }


def test_idw_grids_interpolate_between_stations() -> None:
    bbox = (20.95, 52.10, 21.10, 52.30)
    grids = build_gios_aq_idw_grids(
        _stations(),
        bbox_wgs84=bbox,
        width=40,
        height=32,
        power=2.0,
    )
    assert set(grids) >= {"aq_index", "pm25_ug_m3"}
    aq = grids["aq_index"]
    assert aq.grid.shape == (32, 40)
    # Cell near good station should be closer to 1 than to 4.
    near_good = sample_aq_field(aq, np.array([21.01]), np.array([52.23]))[0]
    near_poor = sample_aq_field(aq, np.array([21.03]), np.array([52.14]))[0]
    mid = sample_aq_field(aq, np.array([21.02]), np.array([52.185]))[0]
    assert near_good < 1.5
    assert near_poor > 3.5
    assert near_good < mid < near_poor


def test_attach_overwrites_nearest_with_continuous_field() -> None:
    listings = pd.DataFrame({
        "listing_id": ["a", "b"],
        "lon": [21.01, 21.03],
        "lat": [52.23, 52.14],
        "air_aq_index": [99.0, 99.0],
        "air_pm25_ug_m3": [99.0, 99.0],
        "air_pm10_ug_m3": [99.0, 99.0],
        "air_no2_ug_m3": [99.0, 99.0],
    })
    bbox = (20.95, 52.10, 21.10, 52.30)
    rasters = build_gios_aq_idw_grids(
        _stations(),
        bbox_wgs84=bbox,
        width=40,
        height=32,
    )
    out = attach_gios_aq_field_features(listings, rasters)
    assert float(out.loc[0, "air_aq_index"]) < 1.5
    assert float(out.loc[1, "air_aq_index"]) > 3.5
    assert int(out.loc[0, "in_poor_air"]) == 0
    assert int(out.loc[1, "in_poor_air"]) == 1
    assert AQ_POOR_INDEX_THRESHOLD == 3.0


def test_fetch_caches_and_loads(tmp_path: Path) -> None:
    bbox = (20.95, 52.10, 21.10, 52.30)
    first = fetch_gios_aq_field_rasters(
        tmp_path,
        _stations(),
        bbox_wgs84=bbox,
        width=20,
        height=16,
        force_refresh=True,
    )
    assert "aq_index" in first
    assert (tmp_path / "gios_aq_field" / "meta.json").exists()
    assert (tmp_path / "gios_aq_field" / "aq_index.npy").exists()
    second = fetch_gios_aq_field_rasters(
        tmp_path,
        _stations(),
        bbox_wgs84=bbox,
        width=20,
        height=16,
        force_refresh=False,
    )
    assert np.allclose(first["aq_index"].grid, second["aq_index"].grid)


def test_prepare_overlay_and_bitmap_layer(tmp_path: Path) -> None:
    bbox = (20.95, 52.10, 21.10, 52.30)
    rasters = fetch_gios_aq_field_rasters(
        tmp_path,
        _stations(),
        bbox_wgs84=bbox,
        width=20,
        height=16,
        force_refresh=True,
    )
    overlay = prepare_aq_overlay_for_key(rasters, "aq_index", bbox_wgs84=bbox)
    assert isinstance(overlay, AqMapOverlay)
    assert overlay.image_data_uri.startswith("data:image/png;base64,")
    layer = air_quality_bitmap_layer(overlay)
    assert layer is not None
    assert layer.id == "gios-aq-bitmap"


def test_empty_stations_yield_empty_rasters(tmp_path: Path) -> None:
    empty = pd.DataFrame(columns=["station_id", "lat", "lon", "aq_index"])
    out = fetch_gios_aq_field_rasters(
        tmp_path,
        empty,
        bbox_wgs84=(20.9, 52.1, 21.1, 52.3),
        force_refresh=True,
    )
    assert out == {}
