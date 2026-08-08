"""pydeck map builders for Explore Map and Offer tabs."""

from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
import pandas as pd
import pydeck as pdk
import streamlit as st

from happy_housing.app.map_layers import (
    LayerFlags,
    prepare_gunb_for_map,
    prepare_listings_for_map,
    prepare_point_layer,
    prepare_shops_for_map,
    prepare_transit_for_hour,
    slim_map_frame,
    traffic_weights_for_hour,
)
from happy_housing.app.map_marker_shapes import marker_icon
from happy_housing.app.spatial_insights import prepare_spatial_points

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
}
"""RGBA fills for amenity scatters."""


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
    """IconLayer with one shared SVG atlas (not per-row data URLs)."""
    if data.empty:
        return None
    work = slim_map_frame(data).copy()
    icon = marker_icon(kind, fill_color)
    work["icon"] = "marker"
    return pdk.Layer(
        "IconLayer",
        data=work,
        get_position="[lon, lat]",
        get_icon="icon",
        get_size=size,
        size_units="pixels",
        size_min_pixels=size_min_pixels,
        size_max_pixels=size_max_pixels,
        icon_atlas=icon["url"],
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
    )


def amenity_layers(
    amenities: gpd.GeoDataFrame,
    *,
    show_hospital: bool = True,
    show_fire: bool = True,
    show_police: bool = True,
) -> list[pdk.Layer]:
    """Toggleable OSM amenity markers with distinct silhouettes per type."""
    if amenities.empty or "amenity" not in amenities.columns:
        return []
    pts = amenities.copy()
    pts["lon"] = pts.geometry.x
    pts["lat"] = pts.geometry.y
    enabled = {
        "hospital": show_hospital,
        "fire_station": show_fire,
        "police": show_police,
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
    if parks is None or parks.empty:
        return None
    work = parks.copy()
    if work.crs is not None and str(work.crs) != "EPSG:4326":
        work = work.to_crs("EPSG:4326")
    records: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        name = (
            str(row["name"]) if "name" in work.columns and row.get("name") else "Park"
        )
        polys = (
            [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
        )
        for poly in polys:
            if poly.geom_type != "Polygon":
                continue
            coords = [[float(x), float(y)] for x, y in poly.exterior.coords]
            records.append({
                "coordinates": [coords],
                "fill_color": [34, 139, 34, 70],
                "line_color": [20, 100, 20, 200],
                "tooltip_name": name,
                "tooltip_detail": "UM Warszawa PLACE_SKWERY (testmapa parks/squares)",
            })
    if not records:
        return None
    return pdk.Layer(
        "PolygonLayer",
        data=records,
        get_polygon="coordinates",
        get_fill_color="fill_color",
        get_line_color="line_color",
        line_width_min_pixels=1,
        pickable=True,
        id="warsaw-parks",
    )


def aircraft_noise_layers(
    lden: gpd.GeoDataFrame | None,
) -> list[pdk.Layer]:
    """Colored END Lden contour bands (darker = louder)."""
    if lden is None or lden.empty or "lden_db" not in lden.columns:
        return []
    # Nested bands: draw quieter first so louder sits on top.
    work = lden.copy()
    work["lden_db"] = pd.to_numeric(work["lden_db"], errors="coerce")
    work = work.dropna(subset=["lden_db"]).sort_values("lden_db")
    records: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        db = float(row["lden_db"])
        # Map 45→soft amber … 75→deep red.
        t = min(1.0, max(0.0, (db - 45.0) / 30.0))
        fill = [
            int(220 - 40 * t),
            int(160 - 120 * t),
            int(40 - 20 * t),
            int(40 + 50 * t),
        ]
        line = [int(180 - 40 * t), int(80 - 40 * t), 20, 200]
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for poly in polys:
            if poly.geom_type != "Polygon":
                continue
            coords = [[float(x), float(y)] for x, y in poly.exterior.coords]
            records.append({
                "coordinates": [coords],
                "fill_color": fill,
                "line_color": line,
                "tooltip_name": f"Aircraft Lden ≥ {db:.0f} dB",
                "tooltip_detail": "EEA END major-airport contour (not live dB)",
            })
    if not records:
        return []
    return [
        pdk.Layer(
            "PolygonLayer",
            data=records,
            get_polygon="coordinates",
            get_fill_color="fill_color",
            get_line_color="line_color",
            line_width_min_pixels=1,
            pickable=True,
            id="aircraft_lden",
        )
    ]


def shops_layer(shops: pd.DataFrame) -> pdk.Layer | None:
    """Retail / supermarket points (square markers)."""
    prepared = prepare_shops_for_map(shops)
    return _icon(
        prepared,
        layer_id="shops",
        kind="shop",
        fill_color=[124, 58, 237, 200],
        size=22,
        size_min_pixels=12,
        size_max_pixels=28,
    )


def transit_layers(
    stops: pd.DataFrame,
    hourly: pd.DataFrame,
    *,
    hour: int,
) -> list[pdk.Layer]:
    """Heatmap + top active stops for the selected hour."""
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
        layer_id="transit-stops",
        kind="transit",
        fill_color=[14, 116, 144, 210],
        size="icon_size",
        size_min_pixels=14,
        size_max_pixels=34,
    )
    if icons is not None:
        layers.append(icons)
    return layers


def traffic_heatmap_layer(sample: pd.DataFrame, *, hour: int) -> pdk.Layer | None:
    """Diurnal road-activity proxy heatmap (not live vehicle counts)."""
    prepared = traffic_weights_for_hour(sample, hour=hour)
    if prepared.empty:
        return None
    return pdk.Layer(
        "HeatmapLayer",
        data=prepared,
        get_position="[lon, lat]",
        get_weight="weight",
        radiusPixels=32,
        intensity=1.4,
        threshold=0.04,
        opacity=0.5,
        pickable=False,
        id="traffic-heat",
    )


def air_stations_layer(stations: pd.DataFrame) -> pdk.Layer | None:
    """GIOŚ air-quality station markers (hexagon)."""
    prepared = prepare_point_layer(
        stations,
        lat_col="lat",
        lon_col="lon",
        name_col="name",
        detail="air station (distance feature in model)",
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


def commute_target_layer(lon: float, lat: float) -> pdk.Layer:
    """Single commute-target pin (Rondo Daszyńskiego)."""
    data = pd.DataFrame([
        {
            "lon": lon,
            "lat": lat,
            "tooltip_name": "Rondo Daszyńskiego",
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
) -> list[pdk.Layer]:
    """Assemble Spatial-tab layers for the selected metric."""
    layers: list[pdk.Layer] = []
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


def build_explore_layers(
    *,
    listings: pd.DataFrame,
    flags: LayerFlags,
    oou: gpd.GeoDataFrame | None,
    amenities: gpd.GeoDataFrame,
    shops: pd.DataFrame,
    stops: pd.DataFrame,
    hourly: pd.DataFrame,
    traffic_sample: pd.DataFrame,
    air_stations: pd.DataFrame,
    gunb: pd.DataFrame,
    parks: gpd.GeoDataFrame | None,
    commute_lon: float,
    commute_lat: float,
    hour: int,
) -> list[pdk.Layer]:
    """Assemble enabled Explore map layers in draw order."""
    layers: list[pdk.Layer] = []
    if flags.oou:
        layers.extend(aircraft_noise_layers(oou))
    if flags.parks:
        park_layer = parks_layer(parks)
        if park_layer is not None:
            layers.append(park_layer)
    if flags.traffic:
        heat = traffic_heatmap_layer(traffic_sample, hour=hour)
        if heat is not None:
            layers.append(heat)
    if flags.transit:
        layers.extend(transit_layers(stops, hourly, hour=hour))
    if flags.shops:
        shop_layer = shops_layer(shops)
        if shop_layer is not None:
            layers.append(shop_layer)
    layers.extend(
        amenity_layers(
            amenities,
            show_hospital=flags.hospitals,
            show_fire=flags.fire_stations,
            show_police=flags.police,
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
        layers.append(commute_target_layer(commute_lon, commute_lat))
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
) -> None:
    """Render a pydeck chart or fall back with a warning."""
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
        st.pydeck_chart(pdk.Deck(**deck_kwargs))
    except Exception as exc:
        logger.exception("pydeck render failed: %s", exc)
        st.warning(f"pydeck unavailable ({exc})")
        st.caption("Map fallback unavailable for mixed layers.")
