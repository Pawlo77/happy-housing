"""Tests for city / military / rail / airport map overlays."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon

from happy_housing.app.map_layers import LayerFlags
from happy_housing.app.maps import (
    build_explore_layers,
    build_spatial_layers,
    zone_polygon_layer,
)
from happy_housing.app.zone_overlays import (
    ZONE_FILL,
    ZONE_LINE,
    prepare_zone_records,
)


def _poly(name: str, x0: float, y0: float) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "name": [name],
            "geometry": [
                Polygon([
                    (x0, y0),
                    (x0 + 0.01, y0),
                    (x0 + 0.01, y0 + 0.01),
                    (x0, y0 + 0.01),
                    (x0, y0),
                ])
            ],
        },
        crs="EPSG:4326",
    )


def test_prepare_zone_records_uses_palette() -> None:
    records = prepare_zone_records(
        _poly("Base", 21.0, 52.2),
        fill_rgba=ZONE_FILL["military"],
        line_rgba=ZONE_LINE["military"],
        detail="Military land",
    )
    assert len(records) == 1
    assert records[0]["fill_color"] == ZONE_FILL["military"]
    assert records[0]["tooltip_name"] == "Base"


def test_zone_polygon_layer_and_explore_defaults() -> None:
    military = _poly("JW", 20.9, 52.2)
    layer = zone_polygon_layer(
        military,
        layer_id="military",
        fill_rgba=ZONE_FILL["military"],
        line_rgba=ZONE_LINE["military"],
        detail="Military",
    )
    assert layer is not None
    assert layer.id == "military"

    flags = LayerFlags()
    assert flags.city_boundary is True
    assert flags.military is True
    assert flags.rail_areas is True
    assert flags.airport is True

    layers = build_explore_layers(
        listings=__import__("pandas").DataFrame(),
        flags=flags,
        oou=None,
        amenities=__import__("geopandas").GeoDataFrame(geometry=[], crs="EPSG:4326"),
        shops=__import__("pandas").DataFrame(),
        stops=__import__("pandas").DataFrame(),
        hourly=__import__("pandas").DataFrame(),
        route_shapes=__import__("pandas").DataFrame(),
        traffic_sample=__import__("pandas").DataFrame(),
        air_stations=__import__("pandas").DataFrame(),
        gunb=__import__("pandas").DataFrame(),
        parks=None,
        city_boundary=_poly("Warszawa", 20.85, 52.1),
        military=military,
        rail_areas=_poly("Yard", 21.0, 52.25),
        airport=_poly("Chopin", 20.96, 52.16),
        commute_lon=20.98,
        commute_lat=52.23,
        hour=8,
    )
    ids = {getattr(layer, "id", None) for layer in layers}
    assert "city-boundary" in ids
    assert "military-areas" in ids
    assert "rail-areas" in ids
    assert "airport-areas" in ids


def test_spatial_layers_include_zone_overlays() -> None:
    layers = build_spatial_layers(
        listings=__import__("pandas").DataFrame({
            "lat": [52.23],
            "lon": [21.0],
            "price_per_m2": [15000.0],
        }),
        metric_key="price_per_m2",
        traffic_sample=__import__("pandas").DataFrame(),
        hour=8,
        show_hex=False,
        city_boundary=_poly("Warszawa", 20.85, 52.1),
        military=_poly("JW", 20.9, 52.2),
        rail_areas=None,
        airport=_poly("Chopin", 20.96, 52.16),
        flags=LayerFlags(),
    )
    ids = {getattr(layer, "id", None) for layer in layers}
    assert "city-boundary" in ids
    assert "military-areas" in ids
    assert "airport-areas" in ids


def test_zone_fixtures_exist() -> None:
    root = Path("data/fixtures")
    for name in (
        "warsaw_boundary.geojson",
        "military_areas.geojson",
        "rail_areas.geojson",
        "airport_areas.geojson",
    ):
        path = root / name
        assert path.exists(), name
        gdf = gpd.read_file(path)
        assert not gdf.empty, name
