"""Tests for Warsaw UM WMS strategic noise rasters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from happy_housing.gis.warsaw_noise import (
    NOISE_FEATURE_COLUMNS,
    NOISE_LAYER_SPECS,
    NoiseRaster,
    attach_warsaw_noise_features,
    color_to_db,
    energy_sum_db,
    raster_sample_db,
)


def test_color_to_db_matches_gios_legend() -> None:
    assert color_to_db((255, 255, 255, 0)) == 0.0
    assert color_to_db((255, 255, 0, 255)) == 45.0
    assert color_to_db((255, 191, 15, 255)) == 50.0
    assert color_to_db((252, 138, 30, 255)) == 55.0
    assert color_to_db((253, 88, 5, 255)) == 60.0
    assert color_to_db((222, 62, 62, 255)) == 65.0
    assert color_to_db((177, 126, 217, 255)) == 70.0
    assert color_to_db((3, 171, 226, 255)) == 75.0
    assert color_to_db((0, 76, 230, 255)) == 80.0
    # Small RGB drift still maps to the nearest legend colour.
    assert color_to_db((251, 139, 29, 255)) == 55.0


def test_energy_sum_db() -> None:
    assert energy_sum_db([0.0, 0.0]) == 0.0
    assert abs(energy_sum_db([60.0]) - 60.0) < 1e-6
    # Two equal sources → +3 dB.
    assert abs(energy_sum_db([60.0, 60.0]) - 63.0103) < 0.02


def test_raster_sample_and_attach(tmp_path: Path) -> None:
    # Tiny synthetic 2x2 raster in a known projected bbox.
    # Pixel (0,0)=top-left → north-west: 55 dB; (1,1)=south-east: 70 dB.
    img = Image.new("RGBA", (2, 2))
    img.putpixel((0, 0), (252, 138, 30, 255))  # 55
    img.putpixel((1, 0), (253, 88, 5, 255))  # 60
    img.putpixel((0, 1), (222, 62, 62, 255))  # 65
    img.putpixel((1, 1), (177, 126, 217, 255))  # 70
    path = tmp_path / "road_lden.png"
    img.save(path)
    # Fake EPSG:2178 bbox: x 0..2, y 0..2 (south=0 north=2).
    raster = NoiseRaster(
        path=path,
        layer_key="road_lden",
        out_col="road_lden_db",
        xmin=0.0,
        ymin=0.0,
        xmax=2.0,
        ymax=2.0,
        width=2,
        height=2,
        crs="EPSG:2178",
    )
    # Point near NW corner → 55; near SE → 70.
    values = raster_sample_db(
        raster,
        lon=np.array([0.25, 1.75]),
        lat=np.array([1.75, 0.25]),
        # Bypass lon/lat transform: pass projected xy via private override.
        xy=np.array([[0.25, 1.75], [1.75, 0.25]], dtype=float),
    )
    assert values[0] == 55.0
    assert values[1] == 70.0

    # Build a minimal rasters dict for attach (only road_lden).
    frame = pd.DataFrame({
        "listing_id": ["a", "b"],
        "lon": [20.9, 21.0],
        "lat": [52.2, 52.3],
    })
    # Monkeypatch sample by attaching via prepared columns path:
    # use empty rasters → zeros / energy 0.
    out = attach_warsaw_noise_features(frame, rasters={})
    for col in NOISE_FEATURE_COLUMNS:
        assert col in out.columns
    assert (out["noise_total_lden_db"] == 0.0).all()


def test_noise_layer_specs_cover_sources() -> None:
    keys = {s.key for s in NOISE_LAYER_SPECS}
    assert "road_lden" in keys
    assert "road_lnight" in keys
    assert "rail_lden" in keys
    assert "tram_lden" in keys
    assert "aircraft_lden" in keys
    assert "industry_lden" in keys
    assert "industry_lnight" in keys
