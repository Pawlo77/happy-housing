"""pydeck map builders for Explore Map and Offer tabs."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd
import pydeck as pdk
import streamlit as st

_AMENITY_COLORS: dict[str, list[int]] = {
    "hospital": [220, 50, 50, 200],
    "fire_station": [230, 120, 20, 200],
    "police": [40, 90, 200, 200],
}
"""RGBA colours for amenity scatter layers."""


def _listing_colors(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["lat"] = out["lat"].astype(float)
    out["lon"] = out["lon"].astype(float)
    delta = out["delta_price_per_m2"].astype(float)
    out["color_r"] = delta.apply(lambda d: 220 if d > 0 else 30)
    out["color_g"] = delta.apply(lambda d: 40 if d > 0 else 170)
    out["radius"] = (out["area_m2"].fillna(50).astype(float) * 2.5).clip(60, 280)
    return out


def listings_scatter_layer(frame: pd.DataFrame) -> pdk.Layer:
    """Scatterplot of filtered offers coloured by listing-model delta."""
    map_df = _listing_colors(frame)
    return pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[lon, lat]",
        get_radius="radius",
        get_fill_color="[color_r, color_g, 70, 190]",
        pickable=True,
        id="listings",
    )


def amenity_layers(amenities: gpd.GeoDataFrame) -> list[pdk.Layer]:
    """One scatter layer per amenity type."""
    layers: list[pdk.Layer] = []
    if amenities.empty:
        return layers
    pts = amenities.copy()
    pts["lon"] = pts.geometry.x
    pts["lat"] = pts.geometry.y
    amenity_col = "amenity" if "amenity" in pts.columns else None
    if amenity_col is None:
        return layers
    for kind, rgba in _AMENITY_COLORS.items():
        subset = pts[pts[amenity_col] == kind]
        if subset.empty:
            continue
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=subset,
                get_position="[lon, lat]",
                get_radius=45,
                get_fill_color=rgba,
                pickable=True,
                id=f"amenity-{kind}",
            )
        )
    return layers


def oou_polygon_layer(oou: gpd.GeoDataFrame | None) -> list[pdk.Layer]:
    """Semi-transparent OOU polygon (aircraft restricted-use proxy)."""
    if oou is None or oou.empty:
        return []
    records: list[dict[str, Any]] = []
    for _, row in oou.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            continue
        for poly in polys:
            coords = [[float(x), float(y)] for x, y in poly.exterior.coords]
            records.append({"coordinates": [coords]})
    if not records:
        return []
    return [
        pdk.Layer(
            "PolygonLayer",
            data=records,
            get_polygon="coordinates",
            get_fill_color=[180, 40, 40, 60],
            get_line_color=[160, 20, 20, 180],
            line_width_min_pixels=1,
            pickable=False,
            id="oou",
        )
    ]


def focus_marker_layer(lat: float, lon: float) -> pdk.Layer:
    """Highlight the selected offer."""
    return pdk.Layer(
        "ScatterplotLayer",
        data=[{"lat": lat, "lon": lon}],
        get_position="[lon, lat]",
        get_radius=120,
        get_fill_color=[20, 20, 20, 230],
        pickable=False,
        id="focus",
    )


def render_deck(
    layers: list[pdk.Layer],
    *,
    latitude: float,
    longitude: float,
    zoom: float = 11.0,
    tooltip: dict[str, Any] | None = None,
) -> None:
    """Render a pydeck chart or fall back to ``st.map``."""
    view = pdk.ViewState(latitude=latitude, longitude=longitude, zoom=zoom)
    try:
        st.pydeck_chart(
            pdk.Deck(
                map_style=(
                    "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
                ),
                layers=layers,
                initial_view_state=view,
                tooltip=tooltip
                or {
                    "html": "<b>{listing_id}</b><br/>{district}<br/>"
                    "Δ {delta_price_per_m2} zł/m²",
                    "style": {"backgroundColor": "#111", "color": "white"},
                },
            )
        )
    except Exception as exc:
        st.warning(f"pydeck unavailable ({exc})")
        # Best-effort fallback from first layer data is not always tabular.
        st.caption("Map fallback unavailable for mixed layers.")
