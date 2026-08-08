"""Tests for map marker shapes (distinct POI silhouettes)."""

from __future__ import annotations

import json
from urllib.parse import unquote

import geopandas as gpd
import pandas as pd
import pydeck as pdk
from shapely.geometry import Point

from happy_housing.app.map_marker_shapes import (
    MARKER_SHAPE_BY_KIND,
    marker_icon,
    shape_for_kind,
)
from happy_housing.app.maps import amenity_layers, shops_layer


def test_shape_for_kind_differs_across_poi_types() -> None:
    """Shops, police, fire, hospital must not share one silhouette."""
    kinds = ("shop", "police", "fire_station", "hospital", "air", "gunb", "transit")
    shapes = {shape_for_kind(k) for k in kinds}
    assert len(shapes) == len(kinds)
    assert shape_for_kind("unknown_poi") == MARKER_SHAPE_BY_KIND["default"]


def test_marker_icon_is_inline_svg_data_url() -> None:
    icon = marker_icon("police", [29, 78, 216, 210])
    assert icon["url"].startswith("data:image/svg+xml")
    assert "<svg" in unquote(icon["url"].split(",", 1)[1])
    assert icon["width"] == 128
    assert icon["anchorY"] == 64
    assert shape_for_kind("police") == "diamond"


def test_amenity_and_shop_layers_use_icon_layer() -> None:
    """Readable POIs use IconLayer shapes, not identical Scatterplot circles."""
    amenities = gpd.GeoDataFrame(
        {
            "amenity": ["hospital", "police", "fire_station"],
            "name": ["H", "P", "F"],
            "geometry": [Point(21.0, 52.2), Point(21.1, 52.3), Point(21.05, 52.25)],
        },
        crs="EPSG:4326",
    )
    layers = amenity_layers(
        amenities, show_hospital=True, show_fire=True, show_police=True
    )
    assert len(layers) == 3
    assert all(layer.type == "IconLayer" for layer in layers)

    shops = shops_layer(
        pd.DataFrame({
            "name": ["Biedronka"],
            "shop": ["supermarket"],
            "lon": [21.0],
            "lat": [52.2],
        })
    )
    assert shops is not None
    assert shops.type == "IconLayer"


def test_icon_atlas_is_literal_not_expression() -> None:
    """Bare ``icon_atlas`` strings become ``@@=data:…`` → Streamlit crash.

    pydeck must emit a plain ``data:`` atlas (quoted input), with rows
    referencing the shared ``marker`` mapping — not per-row SVG blobs.
    """
    amenities = gpd.GeoDataFrame(
        {
            "amenity": ["hospital"],
            "name": ["H"],
            "geometry": [Point(21.0, 52.2)],
        },
        crs="EPSG:4326",
    )
    layers = amenity_layers(
        amenities, show_hospital=True, show_fire=False, show_police=False
    )
    assert len(layers) == 1
    payload = json.loads(pdk.Deck(layers=layers).to_json())
    layer = payload["layers"][0]
    atlas = layer.get("iconAtlas") or ""
    assert isinstance(atlas, str)
    assert atlas.startswith("data:image/svg+xml")
    assert not atlas.startswith("@@=")
    rows = layer.get("data") or []
    assert rows
    assert rows[0].get("icon") == "marker"
