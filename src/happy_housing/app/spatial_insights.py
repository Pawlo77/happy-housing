"""Spatial market metrics for Explore Spatial heatmaps and district charts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class SpatialMetric:
    """One Spatial-tab metric definition."""

    key: str
    """Stable id used in UI keys and prepare helpers."""
    label: str
    """Human label for selectbox."""
    blurb: str
    """Short caption under the map."""
    value_title: str
    """Y-axis / KPI name for district chart."""


SPATIAL_METRICS: tuple[SpatialMetric, ...] = (
    SpatialMetric(
        key="price_ppm",
        label="Asking zł/m²",
        blurb="Heat where asking prices run hottest.",
        value_title="Median zł/m²",
    ),
    SpatialMetric(
        key="listing_density",
        label="Offer density",
        blurb="Where the ask market piles up listings.",
        value_title="Offers",
    ),
    SpatialMetric(
        key="sold_density",
        label="Sold / transaction density",
        blurb="RCN transaction points — where deals actually closed.",
        value_title="Transactions",
    ),
    SpatialMetric(
        key="avg_area",
        label="Average floor area",
        blurb="Larger flats light up warmer on the map.",
        value_title="Mean area m²",
    ),
    SpatialMetric(
        key="popularity",
        label="Fresh-listing popularity",
        blurb="Weight toward the newest listed months in the filtered set.",
        value_title="Freshness score",
    ),
    SpatialMetric(
        key="commute",
        label="Commute ease",
        blurb="Shorter trips to Daszyńskiego score higher.",
        value_title="Mean commute ease",
    ),
    SpatialMetric(
        key="bargain",
        label="Bargain heat",
        blurb="Underpriced (listing - model) residual intensity.",
        value_title="Mean bargain score",
    ),
    SpatialMetric(
        key="traffic",
        label="Traffic proxy",
        blurb="Uses the OSM traffic sample — not listing pins.",
        value_title="Traffic weight",
    ),
)
"""Catalog of Spatial-tab metrics."""

_METRIC_BY_KEY: dict[str, SpatialMetric] = {m.key: m for m in SPATIAL_METRICS}


def get_spatial_metric(key: str) -> SpatialMetric:
    """Resolve a metric key; unknown → asking zł/m²."""
    return _METRIC_BY_KEY.get(key, _METRIC_BY_KEY["price_ppm"])


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def _with_coords(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["lat"] = _num(work, "lat")
    work["lon"] = _num(work, "lon")
    return work.dropna(subset=["lat", "lon"])


def _point_weights(frame: pd.DataFrame, key: str) -> pd.Series:
    """Per-row heatmap weight for listing-based metrics."""
    if key == "price_ppm":
        return _num(frame, "price_per_m2").clip(lower=0.0)
    if key == "listing_density":
        return pd.Series(1.0, index=frame.index, dtype=float)
    if key == "sold_density":
        source = frame.get("source", pd.Series("", index=frame.index)).astype(str)
        return source.str.casefold().eq("rcn").astype(float)
    if key == "avg_area":
        return _num(frame, "area_m2").clip(lower=0.0)
    if key == "popularity":
        months = frame.get("listed_month", pd.Series("", index=frame.index)).astype(str)
        known = sorted({m for m in months if m and m != "unknown"})
        if not known:
            return pd.Series(1.0, index=frame.index, dtype=float)
        rank = {m: i + 1 for i, m in enumerate(known)}
        return months.map(lambda m: float(rank.get(m, 0))).astype(float)
    if key == "commute":
        commute = _num(frame, "commute_min")
        return (60.0 - commute.clip(lower=0.0, upper=60.0)).clip(lower=0.0)
    if key == "bargain":
        delta = _num(frame, "delta_price_per_m2")
        return (-delta).clip(lower=0.0)
    return pd.Series(1.0, index=frame.index, dtype=float)


def prepare_spatial_points(frame: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    """Lon/lat/weight rows for a HeatmapLayer (listing metrics only)."""
    empty = pd.DataFrame(
        columns=["lon", "lat", "weight", "tooltip_name", "tooltip_detail"]
    )
    if metric_key == "traffic":
        return empty
    work = _with_coords(frame)
    if work.empty:
        return empty
    weights = _point_weights(work, metric_key)
    out = work.loc[weights > 0, ["lon", "lat"]].copy()
    out["weight"] = weights.loc[out.index].astype(float)
    metric = get_spatial_metric(metric_key)
    if "listing_id" in work.columns:
        out["tooltip_name"] = work.loc[out.index, "listing_id"].astype(str)
    else:
        out["tooltip_name"] = metric.label
    out["tooltip_detail"] = (
        metric.label + ": " + out["weight"].map(lambda w: f"{w:,.0f}")
    )
    return out.reset_index(drop=True)


def district_spatial_summary(
    frame: pd.DataFrame,
    metric_key: str,
    *,
    min_offers: int = 3,
) -> pd.DataFrame:
    """District rollup for the active Spatial metric (charts beside the map)."""
    empty = pd.DataFrame(columns=["district", "n", "value"])
    if frame.empty or "district" not in frame.columns or metric_key == "traffic":
        return empty
    work = frame.copy()
    work["district"] = work["district"].astype(str).str.strip().str.casefold()
    work = work.loc[work["district"].ne("") & work["district"].ne("nan")]
    if work.empty:
        return empty
    weights = _point_weights(work, metric_key)
    table = pd.DataFrame({
        "district": work["district"],
        "weight": weights,
        "keep": weights > 0,
    })
    table = table.loc[table["keep"]]
    if table.empty:
        return empty
    if metric_key in {"listing_density", "sold_density"}:
        grouped = (
            table
            .groupby("district", dropna=True)
            .agg(n=("weight", "size"), value=("weight", "sum"))
            .reset_index()
        )
    else:
        grouped = (
            table
            .groupby("district", dropna=True)
            .agg(n=("weight", "size"), value=("weight", "mean"))
            .reset_index()
        )
    out = grouped.loc[grouped["n"] >= min_offers].copy()
    if out.empty:
        return empty
    return out.sort_values("value", ascending=False).reset_index(drop=True)


def spatial_kpis(frame: pd.DataFrame, metric_key: str) -> dict[str, float | str]:
    """Headline KPIs for the Spatial tab header."""
    metric = get_spatial_metric(metric_key)
    points = prepare_spatial_points(frame, metric_key)
    n_points = float(len(points))
    if points.empty:
        return {
            "n_points": 0.0,
            "headline": metric.value_title,
            "headline_value": float("nan"),
            "districts": 0.0,
        }
    summary = district_spatial_summary(frame, metric_key, min_offers=1)
    return {
        "n_points": n_points,
        "headline": metric.value_title,
        "headline_value": float(points["weight"].mean()),
        "districts": float(len(summary)),
    }
