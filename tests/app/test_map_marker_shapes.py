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
    assert shape_for_kind("") == MARKER_SHAPE_BY_KIND["default"]
    assert shape_for_kind("unknown_poi") != "circle"


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


def test_city_poi_kinds_use_distinct_non_circle_shapes() -> None:
    """UM sports/civic layers must not collapse to the default circle."""
    from happy_housing.app.maps import city_poi_layers

    assert shape_for_kind("gym") != "circle"
    assert shape_for_kind("indoor_pool") != "circle"
    assert shape_for_kind("gym") != shape_for_kind("indoor_pool")
    assert shape_for_kind("library") != shape_for_kind("gym")

    frame = gpd.GeoDataFrame(
        {
            "object_type": ["gym", "indoor_pool"],
            "name": ["G", "P"],
            "lon": [21.0, 21.01],
            "lat": [52.2, 52.21],
            "geometry": [Point(21.0, 52.2), Point(21.01, 52.21)],
        },
        crs="EPSG:4326",
    )
    layers = city_poi_layers(frame, object_types=("gym", "indoor_pool"))
    assert len(layers) == 2
    assert all(layer.type == "IconLayer" for layer in layers)
    payloads = [json.loads(pdk.Deck(layers=[layer]).to_json()) for layer in layers]
    atlases = [p["layers"][0]["iconAtlas"] for p in payloads]
    assert atlases[0] != atlases[1]
    assert all(a.startswith("data:image/svg+xml") for a in atlases)
    svgs = [unquote(a.split(",", 1)[1]) for a in atlases]
    # Gym → rounded square; pool → ring (hollow stroke, not filled r=22 circle).
    assert any('rx="14"' in s for s in svgs)
    assert any('fill="none"' in s for s in svgs)
    assert not any(
        'r="22"' in s and 'fill="none"' not in s and "<ellipse" not in s for s in svgs
    )


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
