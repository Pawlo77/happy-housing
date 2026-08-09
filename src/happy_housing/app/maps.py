"""pydeck map builders for Explore Map and Offer tabs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import pandas as pd
import pydeck as pdk
import streamlit as st

from happy_housing.app.commute_target import CommuteTarget, commute_compare_points
from happy_housing.app.map_layers import (
    MAX_SHOP_MAP_POINTS,
    SHOP_CATEGORY_LABELS,
    LayerFlags,
    filter_shops_by_categories,
    limit_map_points,
    prepare_city_pois_for_map,
    prepare_gunb_for_map,
    prepare_listings_for_map,
    prepare_point_layer,
    prepare_shops_for_map,
    prepare_transit_for_hour,
    prepare_transit_paths,
    prepare_transit_stops_network,
    shop_category_for_tag,
    slim_map_frame,
    traffic_weights_for_hour,
)
from happy_housing.app.map_marker_shapes import marker_icon
from happy_housing.app.spatial_insights import prepare_spatial_points
from happy_housing.app.zone_overlays import (
    ZONE_FILL,
    ZONE_LINE,
    prepare_zone_records,
)
from happy_housing.gis.bike_paths import prepare_bike_paths_for_map
from happy_housing.gis.gios_aq_field import AqMapOverlay
from happy_housing.gis.warsaw_noise import NoiseMapOverlay

logger = logging.getLogger(__name__)

_STREETS_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
"""Carto positron GL basemap URL."""

# Token-free satellite tiles via Esri World Imagery (MapLibre style dict).
_SATELLITE_STYLE: dict[str, Any] = {
    "version": 8,
    "sources": {
        "esri-world": {
            "type": "raster",
            "tiles": [
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}"
            ],
            "tileSize": 256,
            "attribution": "Tiles © Esri",
        }
    },
    "layers": [{"id": "esri-world", "type": "raster", "source": "esri-world"}],
}
"""Esri World Imagery style for satellite basemap without Mapbox token."""

_HOVER_TOOLTIP: dict[str, Any] = {
    "html": "<b>{tooltip_name}</b><br/>{tooltip_detail}",
    "style": {"backgroundColor": "#111827", "color": "#F9FAFB", "fontSize": "12px"},
}
"""Shared hover card — layers fill tooltip_name / tooltip_detail."""

_AMENITY_RGBA: dict[str, list[int]] = {
    "hospital": [185, 28, 28, 210],
    "fire_station": [194, 65, 12, 210],
    "police": [29, 78, 216, 210],
    "charging_station": [16, 185, 129, 210],
    "fuel": [120, 53, 15, 210],
}
"""RGBA fills for amenity scatters."""

ADDRESS_FOCUS_ZOOM: float = 16.0
"""Street-level zoom when the Explore map is centered on a searched address."""


@dataclass(frozen=True, slots=True)
class MapAddressFocus:
    """Active Explore Map address pin (Nominatim pick)."""

    lon: float
    """WGS84 longitude."""
    lat: float
    """WGS84 latitude."""
    label: str
    """Display name from the geocoder."""


def basemap_style(mode: str) -> str | dict[str, Any]:
    """Resolve streets vs satellite basemap style."""
    if mode == "Satellite":
        return _SATELLITE_STYLE
    return _STREETS_STYLE


def deck_map_provider(map_style: str | dict[str, Any] | None) -> str | None:
    """pydeck requires ``mapbox`` whenever ``map_style`` is a custom dict."""
    if isinstance(map_style, dict):
        return "mapbox"
    return None


def _scatter(
    data: pd.DataFrame,
    *,
    layer_id: str,
    fill_color: list[int] | str,
    radius: float | str = 90,
    radius_min_pixels: int = 5,
    radius_max_pixels: int = 18,
    auto_highlight: bool = False,
) -> pdk.Layer | None:
    if data.empty:
        return None
    return pdk.Layer(
        "ScatterplotLayer",
        data=slim_map_frame(data),
        get_position="[lon, lat]",
        get_radius=radius,
        radius_min_pixels=radius_min_pixels,
        radius_max_pixels=radius_max_pixels,
        get_fill_color=fill_color,
        pickable=True,
        auto_highlight=auto_highlight,
        stroked=True,
        get_line_color=[255, 255, 255, 160],
        line_width_min_pixels=1,
        id=layer_id,
    )


def _icon(
    data: pd.DataFrame,
    *,
    layer_id: str,
    kind: str,
    fill_color: list[int],
    size: float | str = 26,
    size_min_pixels: int = 14,
    size_max_pixels: int = 36,
) -> pdk.Layer | None:
    """IconLayer with one shared SVG atlas (not per-row data URLs).

    pydeck treats bare string layer props as deck.gl JSON expressions
    (``@@=…``). A raw ``data:`` / ``http:`` ``icon_atlas`` then hits
    Streamlit's ``convertFunction`` and fails with
    ``Unexpected ':' at character 4``. Quote the atlas URL so pydeck
    emits a plain string (see ``pydeck.bindings.layer.Layer``).
    """
    if data.empty:
        return None
    work = slim_map_frame(data).copy()
    icon = marker_icon(kind, fill_color)
    work["icon"] = "marker"
    # Quoted → literal string in JSON (no ``@@=`` expression wrapper).
    atlas_url = f'"{icon["url"]}"'
    return pdk.Layer(
        "IconLayer",
        data=work,
        get_position="[lon, lat]",
        get_icon="icon",
        get_size=size,
        size_units="pixels",
        size_min_pixels=size_min_pixels,
        size_max_pixels=size_max_pixels,
        icon_atlas=atlas_url,
        icon_mapping={
            "marker": {
                "x": 0,
                "y": 0,
                "width": int(icon["width"]),
                "height": int(icon["height"]),
                "anchorY": int(icon["anchorY"]),
            }
        },
        pickable=True,
        id=layer_id,
    )


def listings_scatter_layer(frame: pd.DataFrame) -> pdk.Layer | None:
    """Scatterplot of filtered offers coloured by listing-model delta."""
    map_df = prepare_listings_for_map(frame)
    if map_df.empty:
        return None
    return _scatter(
        map_df,
        layer_id="listings",
        fill_color="[color_r, color_g, color_b, color_a]",
        radius="radius",
        radius_min_pixels=3,
        radius_max_pixels=10,
        auto_highlight=True,
    )


def amenity_layers(
    amenities: gpd.GeoDataFrame,
    *,
    show_hospital: bool = True,
    show_fire: bool = True,
    show_police: bool = True,
    show_charging: bool = False,
    show_fuel: bool = False,
) -> list[pdk.Layer]:
    """Toggleable OSM amenity markers with distinct silhouettes per type."""
    if not (show_hospital or show_fire or show_police or show_charging or show_fuel):
        return []
    if amenities.empty or "amenity" not in amenities.columns:
        return []
    pts = amenities.copy()
    pts["lon"] = pts.geometry.x
    pts["lat"] = pts.geometry.y
    enabled = {
        "hospital": show_hospital,
        "fire_station": show_fire,
        "police": show_police,
        "charging_station": show_charging,
        "fuel": show_fuel,
    }
    layers: list[pdk.Layer] = []
    for kind, rgba in _AMENITY_RGBA.items():
        if not enabled.get(kind, False):
            continue
        subset = pts.loc[pts["amenity"] == kind].copy()
        if subset.empty:
            continue
        subset["tooltip_name"] = (
            subset["name"].fillna(kind).astype(str)
            if "name" in subset.columns
            else kind
        )
        subset["tooltip_detail"] = f"Amenity · {kind.replace('_', ' ')}"
        layer = _icon(
            slim_map_frame(subset),
            layer_id=f"amenity-{kind}",
            kind=kind,
            fill_color=rgba,
            size=28,
            size_min_pixels=16,
            size_max_pixels=34,
        )
        if layer is not None:
            layers.append(layer)
    return layers


def parks_layer(parks: gpd.GeoDataFrame | None) -> pdk.Layer | None:
    """UM Warszawa PLACE_SKWERY park / square polygons."""
    return zone_polygon_layer(
        parks,
        layer_id="warsaw-parks",
        fill_rgba=[34, 139, 34, 70],
        line_rgba=[20, 100, 20, 200],
        detail="UM Warszawa PLACE_SKWERY (testmapa parks/squares)",
        default_name="Park",
    )


_CITY_POI_RGBA: tuple[list[int], ...] = (
    [250, 85, 45, 210],
    [0, 150, 75, 210],
    [0, 110, 220, 210],
    [180, 60, 200, 210],
    [230, 160, 20, 210],
    [40, 160, 180, 210],
)
"""Rotating fill colours for city / sports POI types."""


def city_poi_layers(
    city_pois: gpd.GeoDataFrame | None,
    *,
    object_types: tuple[str, ...] | frozenset[str] | set[str],
) -> list[pdk.Layer]:
    """UM sports, civic, and OSM commercial fitness markers by catalog key."""
    if city_pois is None or city_pois.empty or not object_types:
        return []
    layers: list[pdk.Layer] = []
    for idx, key in enumerate(sorted({str(t) for t in object_types})):
        prepared = prepare_city_pois_for_map(city_pois, object_types={key})
        if prepared.empty:
            continue
        rgba = _CITY_POI_RGBA[idx % len(_CITY_POI_RGBA)]
        layer = _icon(
            prepared,
            layer_id=f"city-poi-{key}",
            kind=key,
            fill_color=rgba,
            size=24,
            size_min_pixels=12,
            size_max_pixels=30,
        )
        if layer is not None:
            layers.append(layer)
    return layers


def zone_polygon_layer(
    frame: gpd.GeoDataFrame | None,
    *,
    layer_id: str,
    fill_rgba: list[int],
    line_rgba: list[int],
    detail: str,
    default_name: str = "Zone",
    line_width_min_pixels: int = 1,
) -> pdk.Layer | None:
    """Generic polygon overlay (city boundary, military, rail, airport, …)."""
    records = prepare_zone_records(
        frame,
        fill_rgba=fill_rgba,
        line_rgba=line_rgba,
        detail=detail,
        default_name=default_name,
    )
    if not records:
        return None
    return pdk.Layer(
        "PolygonLayer",
        data=records,
        get_polygon="coordinates",
        get_fill_color="fill_color",
        get_line_color="line_color",
        line_width_min_pixels=line_width_min_pixels,
        pickable=True,
        id=layer_id,
    )


def city_boundary_layer(boundary: gpd.GeoDataFrame | None) -> pdk.Layer | None:
    """Warsaw city administrative outline (stroke-forward)."""
    return zone_polygon_layer(
        boundary,
        layer_id="city-boundary",
        fill_rgba=ZONE_FILL["city_boundary"],
        line_rgba=ZONE_LINE["city_boundary"],
        detail="Warsaw administrative boundary (OSM)",
        default_name="Warszawa",
        line_width_min_pixels=2,
    )


def military_areas_layer(frame: gpd.GeoDataFrame | None) -> pdk.Layer | None:
    """Military landuse polygons in gray."""
    return zone_polygon_layer(
        frame,
        layer_id="military-areas",
        fill_rgba=ZONE_FILL["military"],
        line_rgba=ZONE_LINE["military"],
        detail="OSM landuse=military",
        default_name="Military",
    )


def rail_areas_layer(frame: gpd.GeoDataFrame | None) -> pdk.Layer | None:
    """Railway land / yard polygons (distinct cool gray)."""
    return zone_polygon_layer(
        frame,
        layer_id="rail-areas",
        fill_rgba=ZONE_FILL["rail_areas"],
        line_rgba=ZONE_LINE["rail_areas"],
        detail="OSM landuse=railway",
        default_name="Railway land",
    )


def airport_areas_layer(frame: gpd.GeoDataFrame | None) -> pdk.Layer | None:
    """Aerodrome footprints (distinct from aircraft Lden contours)."""
    return zone_polygon_layer(
        frame,
        layer_id="airport-areas",
        fill_rgba=ZONE_FILL["airport"],
        line_rgba=ZONE_LINE["airport"],
        detail="OSM aeroway=aerodrome (footprint, not noise)",
        default_name="Airport",
    )


_SHOP_CATEGORY_COLORS: dict[str, list[int]] = {
    "grocery": [22, 163, 74, 200],
    "health_beauty": [219, 39, 119, 200],
    "fashion": [124, 58, 237, 200],
    "home": [180, 83, 9, 200],
    "mobility": [37, 99, 235, 200],
    "other": [100, 116, 139, 200],
}
"""RGBA fills for shop subcategory IconLayers."""


def shops_layers(
    shops: pd.DataFrame,
    *,
    show_grocery: bool = True,
    show_health_beauty: bool = True,
    show_fashion: bool = True,
    show_home: bool = True,
    show_mobility: bool = True,
    show_other: bool = True,
) -> list[pdk.Layer]:
    """One IconLayer per enabled shop subcategory (square markers)."""
    enabled: set[str] = set()
    if show_grocery:
        enabled.add("grocery")
    if show_health_beauty:
        enabled.add("health_beauty")
    if show_fashion:
        enabled.add("fashion")
    if show_home:
        enabled.add("home")
    if show_mobility:
        enabled.add("mobility")
    if show_other:
        enabled.add("other")
    if not enabled:
        return []
    filtered = filter_shops_by_categories(shops, categories=enabled)
    if filtered.empty:
        return []
    work = filtered.copy()
    if "lat" not in work.columns and "geometry" in work.columns:
        work["lon"] = work.geometry.x
        work["lat"] = work.geometry.y
    work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    work["lon"] = pd.to_numeric(work["lon"], errors="coerce")
    work = work.dropna(subset=["lat", "lon"])
    # Cap once across categories so the map payload stays bounded.
    work = limit_map_points(work, MAX_SHOP_MAP_POINTS)
    if work.empty:
        return []
    if "shop" not in work.columns:
        work["shop"] = "shop"
    work["shop_category"] = work["shop"].map(shop_category_for_tag)
    layers: list[pdk.Layer] = []
    for category in SHOP_CATEGORY_LABELS:
        if category not in enabled:
            continue
        subset = work.loc[work["shop_category"] == category]
        if subset.empty:
            continue
        prepared = prepare_shops_for_map(subset)
        layer = _icon(
            prepared,
            layer_id=f"shops-{category}",
            kind="shop",
            fill_color=_SHOP_CATEGORY_COLORS.get(category, [100, 116, 139, 200]),
            size=22,
            size_min_pixels=12,
            size_max_pixels=28,
        )
        if layer is not None:
            layers.append(layer)
    return layers


def shops_layer(shops: pd.DataFrame) -> pdk.Layer | None:
    """First shop subcategory layer (legacy helper for marker-shape tests)."""
    layers = shops_layers(shops)
    return layers[0] if layers else None


def transit_intensity_layers(
    stops: pd.DataFrame,
    hourly: pd.DataFrame,
    *,
    hour: int,
) -> list[pdk.Layer]:
    """Hourly departure heatmap + top active stops (popularity)."""
    prepared = prepare_transit_for_hour(stops, hourly, hour=hour)
    if prepared.empty:
        return []
    layers: list[pdk.Layer] = [
        pdk.Layer(
            "HeatmapLayer",
            data=prepared,
            get_position="[lon, lat]",
            get_weight="weight",
            radiusPixels=28,
            intensity=1.2,
            threshold=0.05,
            opacity=0.55,
            pickable=False,
            id="transit-heat",
        )
    ]
    top = prepared.head(180).copy()
    top["icon_size"] = (14.0 + top["n_departures"].astype(float) * 0.15).clip(16, 34)
    icons = _icon(
        top,
        layer_id="transit-intensity-stops",
        kind="transit",
        fill_color=[14, 116, 144, 210],
        size="icon_size",
        size_min_pixels=14,
        size_max_pixels=34,
    )
    if icons is not None:
        layers.append(icons)
    return layers


def transit_network_layers(
    route_shapes: pd.DataFrame,
    stops: pd.DataFrame,
    *,
    modes: set[str],
) -> list[pdk.Layer]:
    """Mode-coloured route polylines plus GTFS stop markers."""
    layers: list[pdk.Layer] = []
    paths = prepare_transit_paths(route_shapes, modes=modes)
    if not paths.empty:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=paths,
                get_path="path",
                get_color="[color_r, color_g, color_b, color_a]",
                width_min_pixels=2,
                width_max_pixels=6,
                get_width=3,
                pickable=True,
                id="transit-routes",
            )
        )
    stop_pts = prepare_transit_stops_network(stops)
    if not stop_pts.empty:
        stop_layer = _icon(
            stop_pts,
            layer_id="transit-network-stops",
            kind="transit",
            fill_color=[248, 250, 252, 230],
            size=14,
            size_min_pixels=8,
            size_max_pixels=16,
        )
        if stop_layer is not None:
            layers.append(stop_layer)
    return layers


def traffic_heatmap_layer(sample: pd.DataFrame, *, hour: int) -> pdk.Layer | None:
    """Diurnal street-activity heatmap along densified OSM roads."""
    prepared = traffic_weights_for_hour(sample, hour=hour)
    if prepared.empty:
        return None
    # Dense road sample → larger radius + lower threshold reads as corridors,
    # not scattered blob points.
    n = len(prepared)
    radius = 55 if n >= 5_000 else 40
    return pdk.Layer(
        "HeatmapLayer",
        data=prepared,
        get_position="[lon, lat]",
        get_weight="weight",
        radiusPixels=radius,
        intensity=1.15,
        threshold=0.02,
        opacity=0.55,
        pickable=False,
        id="traffic-heat",
    )


def warsaw_noise_heatmap_layer(points: pd.DataFrame) -> pdk.Layer | None:
    """Legacy point HeatmapLayer; prefer ``warsaw_noise_bitmap_layer``."""
    if points is None or getattr(points, "empty", True):
        return None
    work = points.copy()
    if "weight" not in work.columns and "db" in work.columns:
        db = pd.to_numeric(work["db"], errors="coerce").fillna(0.0)
        work["weight"] = ((db - 40.0) / 40.0).clip(lower=0.05, upper=1.0)
    keep = [c for c in ("lon", "lat", "weight") if c in work.columns]
    if len(keep) < 3:
        return None
    slim = work.loc[:, keep].dropna()
    if slim.empty:
        return None
    return pdk.Layer(
        "HeatmapLayer",
        data=slim,
        get_position="[lon, lat]",
        get_weight="weight",
        radiusPixels=22,
        intensity=1.15,
        threshold=0.04,
        opacity=0.55,
        pickable=False,
        id="warsaw-noise-heat",
    )


def warsaw_noise_bitmap_layer(
    overlay: NoiseMapOverlay | None,
    *,
    opacity: float = 0.72,
) -> pdk.Layer | None:
    """Continuous UM strategic noise GetMap as a georeferenced BitmapLayer."""
    if overlay is None:
        return None
    west, south, east, north = overlay.bounds
    # Quote data URI so pydeck does not wrap ``data:`` as a ``@@=`` expression.
    image = f'"{overlay.image_data_uri}"'
    return pdk.Layer(
        "BitmapLayer",
        image=image,
        bounds=[float(west), float(south), float(east), float(north)],
        opacity=float(opacity),
        pickable=False,
        id="warsaw-noise-bitmap",
    )


def air_quality_bitmap_layer(
    overlay: AqMapOverlay | None,
    *,
    opacity: float = 0.65,
) -> pdk.Layer | None:
    """Continuous GIOŚ IDW air-quality field as a georeferenced BitmapLayer."""
    if overlay is None:
        return None
    west, south, east, north = overlay.bounds
    image = f'"{overlay.image_data_uri}"'
    return pdk.Layer(
        "BitmapLayer",
        image=image,
        bounds=[float(west), float(south), float(east), float(north)],
        opacity=float(opacity),
        pickable=False,
        id="gios-aq-bitmap",
    )


def bike_paths_layer(paths: gpd.GeoDataFrame | pd.DataFrame) -> pdk.Layer | None:
    """OSM bicycle infrastructure as PathLayer polylines."""
    if isinstance(paths, gpd.GeoDataFrame):
        prepared = prepare_bike_paths_for_map(paths)
    elif isinstance(paths, pd.DataFrame) and "path" in paths.columns:
        prepared = paths
    else:
        prepared = pd.DataFrame()
    if prepared.empty:
        return None
    return pdk.Layer(
        "PathLayer",
        data=prepared,
        get_path="path",
        get_color="[color_r, color_g, color_b, color_a]",
        width_min_pixels=2,
        width_max_pixels=5,
        get_width=2.5,
        pickable=True,
        id="bike-paths",
    )


def air_stations_layer(stations: pd.DataFrame) -> pdk.Layer | None:
    """GIOŚ air-quality station markers (hexagon)."""
    prepared = prepare_point_layer(
        stations,
        lat_col="lat",
        lon_col="lon",
        name_col="name",
        detail="GIOŚ sensor (IDW field + dist_air_station_m in model)",
        layer_label="Air",
    )
    return _icon(
        prepared,
        layer_id="air",
        kind="air",
        fill_color=[22, 163, 74, 220],
        size=30,
        size_min_pixels=16,
        size_max_pixels=36,
    )


def gunb_layer(points: pd.DataFrame) -> pdk.Layer | None:
    """GUNB construction / permit markers (star); capped for message size."""
    prepared = prepare_gunb_for_map(points)
    return _icon(
        prepared,
        layer_id="gunb",
        kind="gunb",
        fill_color=[161, 98, 7, 210],
        size=26,
        size_min_pixels=14,
        size_max_pixels=32,
    )


def commute_target_layer(
    lon: float, lat: float, *, label: str = "Commute target"
) -> pdk.Layer:
    """Single commute-target pin."""
    data = pd.DataFrame([
        {
            "lon": lon,
            "lat": lat,
            "tooltip_name": label,
            "tooltip_detail": "Commute target used by the model",
        }
    ])
    layer = _icon(
        data,
        layer_id="commute-target",
        kind="commute",
        fill_color=[15, 23, 42, 230],
        size=36,
        size_min_pixels=22,
        size_max_pixels=44,
    )
    if layer is None:
        msg = "commute target icon unexpectedly empty"
        raise RuntimeError(msg)
    return layer


def address_search_layer(
    lon: float,
    lat: float,
    *,
    label: str = "Address",
) -> pdk.Layer:
    """Pin for an Explore Map address search result."""
    data = pd.DataFrame([
        {
            "lon": float(lon),
            "lat": float(lat),
            "tooltip_name": str(label),
            "tooltip_detail": "Searched address",
        }
    ])
    layer = _icon(
        data,
        layer_id="address-search",
        kind="address",
        fill_color=[220, 38, 38, 240],
        size=40,
        size_min_pixels=24,
        size_max_pixels=48,
    )
    if layer is None:
        msg = "address search icon unexpectedly empty"
        raise RuntimeError(msg)
    return layer


def explore_map_camera(
    listings: pd.DataFrame,
    *,
    focus: MapAddressFocus | None,
    fallback_lat: float,
    fallback_lon: float,
    default_zoom: float = 10.5,
) -> tuple[float, float, float]:
    """Latitude, longitude, zoom for the Explore Map initial view."""
    if focus is not None:
        return float(focus.lat), float(focus.lon), float(ADDRESS_FOCUS_ZOOM)
    if not listings.empty and "lat" in listings.columns and "lon" in listings.columns:
        return (
            float(listings["lat"].mean()),
            float(listings["lon"].mean()),
            float(default_zoom),
        )
    return float(fallback_lat), float(fallback_lon), float(default_zoom)


def explore_map_selection_key(focus: MapAddressFocus | None) -> str:
    """pydeck chart key — remount when address focus changes so the view recenters."""
    if focus is None:
        return "explore_map_deck"
    return f"explore_map_deck_{focus.lon:.5f}_{focus.lat:.5f}"


def commute_compare_layers(
    *,
    saved: CommuteTarget,
    draft: CommuteTarget,
) -> list[pdk.Layer]:
    """Admin pins: slate = saved target, teal = draft (new)."""
    frame = commute_compare_points(saved=saved, draft=draft)
    layers: list[pdk.Layer] = []
    saved_layer = _icon(
        frame.loc[frame["role"] == "saved"],
        layer_id="commute-saved",
        kind="commute",
        fill_color=[15, 23, 42, 230],
        size=34,
        size_min_pixels=20,
        size_max_pixels=42,
    )
    draft_layer = _icon(
        frame.loc[frame["role"] == "draft"],
        layer_id="commute-draft",
        kind="commute",
        fill_color=[15, 107, 92, 230],
        size=34,
        size_min_pixels=20,
        size_max_pixels=42,
    )
    if saved_layer is not None:
        layers.append(saved_layer)
    if draft_layer is not None:
        layers.append(draft_layer)
    return layers


def focus_marker_layer(lat: float, lon: float) -> pdk.Layer:
    """Highlight the selected offer."""
    data = pd.DataFrame([
        {
            "lat": lat,
            "lon": lon,
            "tooltip_name": "Selected offer",
            "tooltip_detail": "Focus pin",
        }
    ])
    layer = _icon(
        data,
        layer_id="focus",
        kind="focus",
        fill_color=[15, 15, 15, 240],
        size=38,
        size_min_pixels=24,
        size_max_pixels=48,
    )
    if layer is None:
        msg = "focus marker icon unexpectedly empty"
        raise RuntimeError(msg)
    return layer


def spatial_heatmap_layer(
    points: pd.DataFrame,
    *,
    layer_id: str = "spatial-heat",
    radius_pixels: int = 40,
) -> pdk.Layer | None:
    """Weighted heatmap for Spatial-tab metrics."""
    if points.empty or "weight" not in points.columns:
        return None
    return pdk.Layer(
        "HeatmapLayer",
        data=points,
        get_position="[lon, lat]",
        get_weight="weight",
        radiusPixels=radius_pixels,
        intensity=1.4,
        threshold=0.03,
        opacity=0.65,
        pickable=False,
        id=layer_id,
    )


def spatial_hex_layer(
    points: pd.DataFrame,
    *,
    layer_id: str = "spatial-hex",
) -> pdk.Layer | None:
    """Hexagon bins with elevation from weights (denser than heat alone)."""
    if points.empty or "weight" not in points.columns:
        return None
    return pdk.Layer(
        "HexagonLayer",
        data=points,
        get_position="[lon, lat]",
        get_elevation_weight="weight",
        get_color_weight="weight",
        elevation_scale=40,
        elevation_range=[0, 800],
        extruded=True,
        coverage=0.85,
        radius=280,
        opacity=0.55,
        pickable=True,
        id=layer_id,
    )


def build_spatial_layers(
    *,
    listings: pd.DataFrame,
    metric_key: str,
    traffic_sample: pd.DataFrame,
    hour: int,
    show_hex: bool,
    city_boundary: gpd.GeoDataFrame | None = None,
    military: gpd.GeoDataFrame | None = None,
    rail_areas: gpd.GeoDataFrame | None = None,
    airport: gpd.GeoDataFrame | None = None,
    flags: LayerFlags | None = None,
) -> list[pdk.Layer]:
    """Assemble Spatial-tab layers for the selected metric."""
    layers: list[pdk.Layer] = []
    zone_flags = flags or LayerFlags()
    layers.extend(
        _zone_overlay_layers(
            flags=zone_flags,
            city_boundary=city_boundary,
            military=military,
            rail_areas=rail_areas,
            airport=airport,
        )
    )
    if metric_key == "traffic":
        heat = traffic_heatmap_layer(traffic_sample, hour=hour)
        if heat is not None:
            layers.append(heat)
        return layers
    points = prepare_spatial_points(listings, metric_key)
    heat = spatial_heatmap_layer(points)
    if heat is not None:
        layers.append(heat)
    if show_hex:
        hex_layer = spatial_hex_layer(points)
        if hex_layer is not None:
            layers.append(hex_layer)
    return layers


def _zone_overlay_layers(
    *,
    flags: LayerFlags,
    city_boundary: gpd.GeoDataFrame | None,
    military: gpd.GeoDataFrame | None,
    rail_areas: gpd.GeoDataFrame | None,
    airport: gpd.GeoDataFrame | None,
) -> list[pdk.Layer]:
    """City boundary + restricted-land polygons drawn under other overlays."""
    layers: list[pdk.Layer] = []
    if flags.city_boundary:
        boundary = city_boundary_layer(city_boundary)
        if boundary is not None:
            layers.append(boundary)
    if flags.military:
        mil = military_areas_layer(military)
        if mil is not None:
            layers.append(mil)
    if flags.rail_areas:
        rail = rail_areas_layer(rail_areas)
        if rail is not None:
            layers.append(rail)
    if flags.airport:
        air = airport_areas_layer(airport)
        if air is not None:
            layers.append(air)
    return layers


def build_explore_layers(
    *,
    listings: pd.DataFrame,
    flags: LayerFlags,
    amenities: gpd.GeoDataFrame,
    shops: pd.DataFrame,
    stops: pd.DataFrame,
    hourly: pd.DataFrame,
    route_shapes: pd.DataFrame,
    traffic_sample: pd.DataFrame,
    air_stations: pd.DataFrame,
    gunb: pd.DataFrame,
    parks: gpd.GeoDataFrame | None,
    commute_lon: float,
    commute_lat: float,
    hour: int,
    commute_label: str = "Commute target",
    city_boundary: gpd.GeoDataFrame | None = None,
    military: gpd.GeoDataFrame | None = None,
    rail_areas: gpd.GeoDataFrame | None = None,
    airport: gpd.GeoDataFrame | None = None,
    city_pois: gpd.GeoDataFrame | None = None,
    bike_paths: gpd.GeoDataFrame | None = None,
    noise_overlay: NoiseMapOverlay | None = None,
    aq_overlay: AqMapOverlay | None = None,
) -> list[pdk.Layer]:
    """Assemble enabled Explore map layers in draw order."""
    layers: list[pdk.Layer] = []
    layers.extend(
        _zone_overlay_layers(
            flags=flags,
            city_boundary=city_boundary,
            military=military,
            rail_areas=rail_areas,
            airport=airport,
        )
    )
    if flags.noise_overlay and flags.noise_overlay != "off":
        noise = warsaw_noise_bitmap_layer(noise_overlay)
        if noise is not None:
            layers.append(noise)
    if flags.aq_overlay and flags.aq_overlay != "off":
        aq = air_quality_bitmap_layer(aq_overlay)
        if aq is not None:
            layers.append(aq)
    if flags.parks:
        park_layer = parks_layer(parks)
        if park_layer is not None:
            layers.append(park_layer)
    layers.extend(city_poi_layers(city_pois, object_types=flags.city_poi_types))
    if flags.traffic:
        heat = traffic_heatmap_layer(traffic_sample, hour=hour)
        if heat is not None:
            layers.append(heat)
    if flags.bike_paths:
        bike = bike_paths_layer(
            bike_paths if bike_paths is not None else gpd.GeoDataFrame(geometry=[])
        )
        if bike is not None:
            layers.append(bike)
    if flags.transit_intensity:
        layers.extend(transit_intensity_layers(stops, hourly, hour=hour))
    if flags.transit_network:
        modes: set[str] = set()
        if flags.transit_show_metro:
            modes.add("metro")
        if flags.transit_show_tram:
            modes.add("tram")
        if flags.transit_show_bus:
            modes.add("bus")
        if flags.transit_show_rail:
            modes.add("rail")
        layers.extend(transit_network_layers(route_shapes, stops, modes=modes))
    if flags.shops:
        layers.extend(
            shops_layers(
                shops,
                show_grocery=flags.shop_show_grocery,
                show_health_beauty=flags.shop_show_health_beauty,
                show_fashion=flags.shop_show_fashion,
                show_home=flags.shop_show_home,
                show_mobility=flags.shop_show_mobility,
                show_other=flags.shop_show_other,
            )
        )
    layers.extend(
        amenity_layers(
            amenities,
            show_hospital=flags.hospitals,
            show_fire=flags.fire_stations,
            show_police=flags.police,
            show_charging=flags.charging_stations,
            show_fuel=flags.gas_stations,
        )
    )
    if flags.air:
        air = air_stations_layer(air_stations)
        if air is not None:
            layers.append(air)
    if flags.gunb:
        gunb_pts = gunb_layer(gunb)
        if gunb_pts is not None:
            layers.append(gunb_pts)
    if flags.commute_target:
        layers.append(
            commute_target_layer(commute_lon, commute_lat, label=commute_label)
        )
    if flags.offers:
        offers = listings_scatter_layer(listings)
        if offers is not None:
            layers.append(offers)
    logger.debug(
        "Built %s map layers (offers=%s hour=%s)", len(layers), flags.offers, hour
    )
    return layers


def render_deck(
    layers: list[pdk.Layer],
    *,
    latitude: float,
    longitude: float,
    zoom: float = 11.0,
    map_style: str | dict[str, Any] | None = None,
    tooltip: dict[str, Any] | None = None,
    selection_key: str | None = None,
    chart_key: str | None = None,
) -> Any:
    """Render a pydeck chart; optional click selection when ``selection_key`` set."""
    view = pdk.ViewState(latitude=latitude, longitude=longitude, zoom=zoom)
    style = map_style or _STREETS_STYLE
    provider = deck_map_provider(style)
    try:
        deck_kwargs: dict[str, Any] = {
            "map_style": style,
            "layers": layers,
            "initial_view_state": view,
            "tooltip": tooltip or _HOVER_TOOLTIP,
        }
        if provider is not None:
            deck_kwargs["map_provider"] = provider
        deck = pdk.Deck(**deck_kwargs)
        if selection_key:
            return st.pydeck_chart(
                deck,
                on_select="rerun",
                selection_mode="single-object",
                key=selection_key,
            )
        if chart_key:
            st.pydeck_chart(deck, key=chart_key)
        else:
            st.pydeck_chart(deck)
        return None
    except Exception as exc:
        logger.exception("pydeck render failed: %s", exc)
        st.warning(f"pydeck unavailable ({exc})")
        st.caption("Map fallback unavailable for mixed layers.")
        return None
