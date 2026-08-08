"""Tests for snapping listing pins out of park polygons via OSM buildings."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

from happy_housing.gis.location_fix import (
    listing_points_inside_parks,
    snap_listings_out_of_parks,
)


def _park_square() -> gpd.GeoDataFrame:
    # ~200 m square around (21.035, 52.21) in WGS84 (Łazienki-ish).
    poly = box(21.033, 52.2085, 21.037, 52.2115)
    return gpd.GeoDataFrame({"name": ["Test Park"]}, geometry=[poly], crs="EPSG:4326")


def _buildings() -> gpd.GeoDataFrame:
    # One building inside the park, one outside to the east.
    return gpd.GeoDataFrame(
        {"building": ["yes", "apartments"]},
        geometry=[
            Point(21.035, 52.210),  # inside
            Point(21.039, 52.210),  # outside
        ],
        crs="EPSG:4326",
    )


def test_listing_points_inside_parks_detects_containment() -> None:
    parks = _park_square()
    listings = pd.DataFrame({
        "listing_id": ["in", "out"],
        "lat": [52.210, 52.220],
        "lon": [21.035, 21.050],
    })
    mask = listing_points_inside_parks(listings, parks)
    assert bool(mask.iloc[0]) is True
    assert bool(mask.iloc[1]) is False


def test_snap_listings_out_of_parks_moves_interior_pin() -> None:
    listings = pd.DataFrame({
        "listing_id": ["WWA-0028", "OK-1"],
        "lat": [52.210, 52.220],
        "lon": [21.035, 21.050],
        "district": ["mokotow", "wola"],
    })
    out = snap_listings_out_of_parks(
        listings,
        parks=_park_square(),
        buildings=_buildings(),
        max_snap_m=800.0,
    )
    assert out.loc[0, "listing_id"] == "WWA-0028"
    assert float(out.loc[0, "lat"]) == 52.210
    assert float(out.loc[0, "lon"]) == 21.039
    assert bool(out.loc[0, "location_snapped"]) is True
    assert float(out.loc[1, "lat"]) == 52.220
    assert float(out.loc[1, "lon"]) == 21.050
    assert bool(out.loc[1, "location_snapped"]) is False


def test_snap_listings_skips_when_no_building_within_max() -> None:
    listings = pd.DataFrame({
        "listing_id": ["far"],
        "lat": [52.210],
        "lon": [21.035],
    })
    buildings = gpd.GeoDataFrame(
        {"building": ["yes"]},
        geometry=[Point(21.060, 52.210)],  # ~1.7 km east
        crs="EPSG:4326",
    )
    out = snap_listings_out_of_parks(
        listings,
        parks=_park_square(),
        buildings=buildings,
        max_snap_m=200.0,
    )
    assert float(out.loc[0, "lon"]) == 21.035
    assert bool(out.loc[0, "location_snapped"]) is False


def test_snap_listings_fetches_buildings_when_missing(
    tmp_path: Path, monkeypatch
) -> None:
    listings = pd.DataFrame({
        "listing_id": ["WWA-0028"],
        "lat": [52.210],
        "lon": [21.035],
    })

    def _fake_fetch(*_args, **_kwargs):
        return _buildings()

    monkeypatch.setattr(
        "happy_housing.gis.location_fix.fetch_osm_building_centroids",
        _fake_fetch,
    )
    out = snap_listings_out_of_parks(
        listings,
        parks=_park_square(),
        buildings=None,
        cache_dir=tmp_path,
        bbox=(21.0, 52.2, 21.1, 52.3),
        max_snap_m=800.0,
    )
    assert bool(out.loc[0, "location_snapped"]) is True
    assert float(out.loc[0, "lon"]) == 21.039
