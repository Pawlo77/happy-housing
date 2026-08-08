"""Pure helpers for Explore map layer payloads and visibility flags."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Relative car-mobility intensity by hour-of-day (not live measured traffic).
_TRAFFIC_DIURNAL: tuple[float, ...] = (
    0.12,
    0.08,
    0.06,
    0.06,
    0.10,
    0.35,
    0.70,
    1.00,
    0.92,
    0.68,
    0.62,
    0.65,
    0.70,
    0.68,
    0.70,
    0.78,
    0.90,
    1.00,
    0.88,
    0.65,
    0.45,
    0.32,
    0.22,
    0.16,
)
"""Typical Warsaw road-activity proxy multipliers (00-23)."""


@dataclass(slots=True)
class LayerFlags:
    """Which Explore map overlays are enabled."""

    offers: bool = True
    """Scored listing pins."""
    oou: bool = True
    """Chopin aircraft / noise-proxy polygon."""
    hospitals: bool = False
    """OSM hospitals."""
    fire_stations: bool = False
    """OSM fire stations."""
    police: bool = False
    """OSM police stations."""
    shops: bool = False
    """OSM retail / supermarket points."""
    transit: bool = False
    """GTFS stops sized by departures at the selected hour."""
    traffic: bool = False
    """Road-activity heatmap proxy for the selected hour."""
    air: bool = False
    """GIOŚ air-quality stations."""
    gunb: bool = False
    """GUNB construction / permit points."""
    commute_target: bool = False
    """Rondo Daszyńskiego commute target."""


def diurnal_traffic_weight(hour: int) -> float:
    """Return typical traffic-intensity multiplier for ``hour`` in ``0..23``."""
    return float(_TRAFFIC_DIURNAL[int(hour) % 24])


def prepare_listings_for_map(frame: pd.DataFrame) -> pd.DataFrame:
    """Listing pins with delta colours and hover fields."""
    if frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out["lat"] = out["lat"].astype(float)
    out["lon"] = out["lon"].astype(float)
    delta = pd.to_numeric(out["delta_price_per_m2"], errors="coerce").fillna(0.0)
    over = delta.to_numpy() > 0
    out["color_r"] = np.where(over, 220, 30)
    out["color_g"] = np.where(over, 40, 170)
    out["color_b"] = 70
    out["color_a"] = 200
    area = pd.to_numeric(out.get("area_m2"), errors="coerce").fillna(50.0)
    out["radius"] = (area * 2.8).clip(90, 320)
    out["tooltip_name"] = out["listing_id"].astype(str)
    ppm = pd.to_numeric(out.get("price_per_m2"), errors="coerce")
    out["tooltip_detail"] = (
        out.get("district", pd.Series("", index=out.index)).astype(str)
        + " · Δ "
        + delta.map(lambda v: f"{v:+.0f}")
        + " zł/m² · "
        + ppm.map(lambda v: f"{v:,.0f}" if v == v else "—")
        + " zł/m²"
    )
    return out


def prepare_point_layer(
    frame: pd.DataFrame,
    *,
    lat_col: str,
    lon_col: str,
    name_col: str,
    detail: str,
    layer_label: str,
) -> pd.DataFrame:
    """Normalize a point table for scatter + hover."""
    if frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out["lat"] = pd.to_numeric(out[lat_col], errors="coerce")
    out["lon"] = pd.to_numeric(out[lon_col], errors="coerce")
    out = out.dropna(subset=["lat", "lon"])
    if out.empty:
        return out
    out["tooltip_name"] = out[name_col].astype(str)
    out["tooltip_detail"] = f"{layer_label} · {detail}"
    out["layer_label"] = layer_label
    return out


def prepare_shops_for_map(frame: pd.DataFrame) -> pd.DataFrame:
    """Shop POIs with type in the hover detail."""
    if frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if "lat" not in out.columns and "geometry" in out.columns:
        out["lon"] = out.geometry.x
        out["lat"] = out.geometry.y
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["lat", "lon"])
    shop = out["shop"].astype(str) if "shop" in out.columns else "shop"
    name = out["name"].fillna("Shop").astype(str) if "name" in out.columns else "Shop"
    out["tooltip_name"] = name
    out["tooltip_detail"] = "Shop · " + shop
    out["layer_label"] = "Shop"
    return out


def prepare_transit_for_hour(
    stops: pd.DataFrame,
    hourly: pd.DataFrame,
    *,
    hour: int,
    min_departures: int = 3,
    max_stops: int = 500,
) -> pd.DataFrame:
    """GTFS stops active at ``hour``, sized by departure count."""
    if stops.empty or hourly.empty:
        return pd.DataFrame()
    hour_rows = hourly.loc[hourly["hour"].astype(int) == int(hour)].copy()
    if hour_rows.empty:
        return pd.DataFrame()
    merged = stops.merge(hour_rows, on="stop_id", how="inner")
    merged = merged.loc[merged["n_departures"] >= min_departures]
    if merged.empty:
        return merged
    merged = merged.sort_values("n_departures", ascending=False).head(max_stops)
    merged["lat"] = pd.to_numeric(merged["stop_lat"], errors="coerce")
    merged["lon"] = pd.to_numeric(merged["stop_lon"], errors="coerce")
    merged = merged.dropna(subset=["lat", "lon"])
    merged["radius"] = (40.0 + merged["n_departures"].astype(float) * 1.8).clip(70, 420)
    merged["tooltip_name"] = merged["stop_name"].astype(str)
    merged["tooltip_detail"] = (
        f"Transit · {int(hour):02d}:00 · "
        + merged["n_departures"].astype(int).astype(str)
        + " departures"
    )
    merged["layer_label"] = "Transit"
    merged["weight"] = merged["n_departures"].astype(float)
    return merged.reset_index(drop=True)


def traffic_weights_for_hour(sample: pd.DataFrame, *, hour: int) -> pd.DataFrame:
    """Scale road-sample base weights by the diurnal traffic curve."""
    if sample.empty:
        return pd.DataFrame()
    out = sample.copy()
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["base_weight"] = pd.to_numeric(out["base_weight"], errors="coerce").fillna(1.0)
    out = out.dropna(subset=["lat", "lon"])
    factor = diurnal_traffic_weight(hour)
    out["weight"] = out["base_weight"] * factor
    name = (
        out["name"].astype(str)
        if "name" in out.columns
        else pd.Series("corridor", index=out.index)
    )
    out["tooltip_name"] = name
    out["tooltip_detail"] = (
        f"Traffic proxy · {int(hour):02d}:00 · intensity {factor:.2f}x"
    )
    out["layer_label"] = "Traffic"
    return out.reset_index(drop=True)
