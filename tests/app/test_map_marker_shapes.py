"""Tests for map marker emojis (readable POI icons)."""

import json
from urllib.parse import unquote

import geopandas as gpd
import pandas as pd
import pydeck as pdk
from shapely.geometry import Point

from happy_housing.app.map_marker_shapes import (
    emoji_for_kind,
    marker_icon,
)
from happy_housing.app.maps import amenity_layers, shops_layer, shops_layers


def test_emoji_for_kind_differs_across_poi_types() -> None:
    """Shops, police, fire, hospital must not share one emoji."""
    kinds = ("grocery", "police", "fire_station", "hospital", "air", "gunb", "transit")
    emojis = {emoji_for_kind(k) for k in kinds}
    assert len(emojis) == len(kinds)
    assert emoji_for_kind("police") == "👮"
    assert emoji_for_kind("grocery") == "🛒"
    assert emoji_for_kind("") == emoji_for_kind("default")
    assert emoji_for_kind("unknown_poi_xyz") != emoji_for_kind("police")


def test_marker_icon_embeds_emoji_in_svg() -> None:
    icon = marker_icon("police", [29, 78, 216, 210])
    assert icon["url"].startswith("data:image/svg+xml")
    svg = unquote(icon["url"].split(",", 1)[1])
    assert "<svg" in svg
    assert "👮" in svg
    assert icon["width"] == 128
    assert icon["anchorY"] == 64


def test_amenity_and_shop_layers_use_icon_layer() -> None:
    """Readable POIs use IconLayer emoji markers, not identical circles."""
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
    payload = json.loads(pdk.Deck(layers=[shops]).to_json())
    atlas = payload["layers"][0]["iconAtlas"]
    assert "🛒" in unquote(atlas.split(",", 1)[1])


def test_shop_subcategories_use_distinct_emojis() -> None:
    shops = pd.DataFrame({
        "name": ["Biedronka", "Rossmann", "Zara"],
        "shop": ["supermarket", "chemist", "clothes"],
        "lon": [21.0, 21.01, 21.02],
        "lat": [52.2, 52.21, 52.22],
    })
    layers = shops_layers(shops)
    assert len(layers) >= 2
    svgs = []
    for layer in layers:
        payload = json.loads(pdk.Deck(layers=[layer]).to_json())
        atlas = payload["layers"][0]["iconAtlas"]
        svgs.append(unquote(atlas.split(",", 1)[1]))
    assert any("🛒" in s for s in svgs)
    assert any("💄" in s for s in svgs) or any("💊" in s for s in svgs)
    assert len({s for s in svgs}) == len(svgs)


def test_city_poi_kinds_use_distinct_emojis() -> None:
    """UM sports/civic layers must not collapse to one generic glyph."""
    from happy_housing.app.maps import city_poi_layers

    assert emoji_for_kind("gym") != emoji_for_kind("indoor_pool")
    assert emoji_for_kind("library") != emoji_for_kind("gym")

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
    assert any("🏋️" in s or "🏋" in s for s in svgs)
    assert any("🏊" in s for s in svgs)


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
    assert "🏥" in unquote(atlas.split(",", 1)[1])
    rows = layer.get("data") or []
    assert rows
    assert rows[0].get("icon") == "marker"
