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
    prepare_listings_for_map,
    prepare_point_layer,
    prepare_shops_for_map,
    prepare_transit_for_hour,
    traffic_weights_for_hour,
)
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
        data=data,
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
    """Toggleable OSM amenity scatters with readable pixel radii."""
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
        layer = _scatter(
            subset,
            layer_id=f"amenity-{kind}",
            fill_color=rgba,
            radius=55,
            radius_min_pixels=4,
            radius_max_pixels=9,
        )
        if layer is not None:
            layers.append(layer)
    return layers


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
    """Retail / supermarket points."""
    prepared = prepare_shops_for_map(shops)
    return _scatter(
        prepared,
        layer_id="shops",
        fill_color=[124, 58, 237, 200],
        radius=50,
        radius_min_pixels=3,
        radius_max_pixels=8,
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
    top = prepared.head(180)
    scatter = _scatter(
        top,
        layer_id="transit-stops",
        fill_color=[14, 116, 144, 210],
        radius="radius",
        radius_min_pixels=3,
        radius_max_pixels=11,
    )
    if scatter is not None:
        layers.append(scatter)
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
    """GIOŚ air-quality station markers."""
    prepared = prepare_point_layer(
        stations,
        lat_col="lat",
        lon_col="lon",
        name_col="name",
        detail="air station (distance feature in model)",
        layer_label="Air",
    )
    return _scatter(
        prepared,
        layer_id="air",
        fill_color=[22, 163, 74, 220],
        radius=60,
        radius_min_pixels=4,
        radius_max_pixels=10,
    )


def gunb_layer(points: pd.DataFrame) -> pdk.Layer | None:
    """GUNB construction / permit markers."""
    name_col = (
        "adres"
        if "adres" in points.columns
        else ("numer_decyzji" if "numer_decyzji" in points.columns else "lon")
    )
    work = points.copy()
    if name_col not in work.columns:
        work["label"] = "GUNB"
        name_col = "label"
    prepared = prepare_point_layer(
        work,
        lat_col="lat",
        lon_col="lon",
        name_col=name_col,
        detail="construction / permit proxy",
        layer_label="GUNB",
    )
    return _scatter(
        prepared,
        layer_id="gunb",
        fill_color=[161, 98, 7, 210],
        radius=55,
        radius_min_pixels=3,
        radius_max_pixels=9,
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
    layer = _scatter(
        data,
        layer_id="commute-target",
        fill_color=[15, 23, 42, 230],
        radius=70,
        radius_min_pixels=5,
        radius_max_pixels=12,
    )
    if layer is None:
        msg = "commute target scatter unexpectedly empty"
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
    layer = _scatter(
        data,
        layer_id="focus",
        fill_color=[15, 15, 15, 240],
        radius=65,
        radius_min_pixels=5,
        radius_max_pixels=12,
    )
    if layer is None:
        msg = "focus marker scatter unexpectedly empty"
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
    commute_lon: float,
    commute_lat: float,
    hour: int,
) -> list[pdk.Layer]:
    """Assemble enabled Explore map layers in draw order."""
    layers: list[pdk.Layer] = []
    if flags.oou:
        layers.extend(aircraft_noise_layers(oou))
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
    try:
        st.pydeck_chart(
            pdk.Deck(
                map_style=map_style or _STREETS_STYLE,
                layers=layers,
                initial_view_state=view,
                tooltip=tooltip or _HOVER_TOOLTIP,
            )
        )
    except Exception as exc:
        logger.exception("pydeck render failed: %s", exc)
        st.warning(f"pydeck unavailable ({exc})")
        st.caption("Map fallback unavailable for mixed layers.")
