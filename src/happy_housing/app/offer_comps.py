"""Local RCN membership, size-band comps, and loyalty stats for Offer → Comps."""

from typing import Any

import numpy as np
import pandas as pd

from happy_housing.ingest.listing_common import BUYER_LISTING_SOURCES
from happy_housing.ingest.rcn import WARSAW_DISTRICT_NAMES, is_district_centroid
from happy_housing.models.neighborhood import query_rcn_neighbors

AREA_BAND_FRAC: float = 0.15
"""Relative area window for similar-size RCN comps."""

AREA_BAND_ABS_M2: float = 10.0
"""Absolute area window (m²) combined with the relative band."""

_WARSAW_GIS_WEST: float = 20.85
"""West edge of the experiment GIS bbox (metro vs city)."""

_WARSAW_GIS_EAST: float = 21.28
"""East edge of the experiment GIS bbox."""

_WARSAW_GIS_SOUTH: float = 52.10
"""South edge of the experiment GIS bbox."""

_WARSAW_GIS_NORTH: float = 52.37
"""North edge of the experiment GIS bbox."""


def drop_centroid_fallbacks(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop stacked dzielnica-centroid pins (unusable on a comps map)."""
    if frame.empty:
        return frame
    out = frame.copy()
    if "coord_source" in out.columns:
        source = out["coord_source"].astype(str).str.casefold()
        out = out.loc[source.ne("centroid")].copy()
    if out.empty or not {"lon", "lat"} <= set(out.columns):
        return out.reset_index(drop=True)
    lon = pd.to_numeric(out["lon"], errors="coerce")
    lat = pd.to_numeric(out["lat"], errors="coerce")
    stacked = [
        is_district_centroid(float(x), float(y))
        if pd.notna(x) and pd.notna(y)
        else False
        for x, y in zip(lon, lat, strict=True)
    ]
    return out.loc[~pd.Series(stacked, index=out.index)].reset_index(drop=True)


def flag_metro_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Tag deeds outside the 18 dzielnice / GIS bbox as metro / unlabelled."""
    if frame.empty:
        out = frame.copy()
        out["is_metro"] = pd.Series(dtype=bool)
        return out
    out = frame.copy()
    lon = pd.to_numeric(out["lon"], errors="coerce") if "lon" in out.columns else None
    lat = pd.to_numeric(out["lat"], errors="coerce") if "lat" in out.columns else None
    outside = pd.Series(False, index=out.index)
    if lon is not None and lat is not None:
        outside = (
            lon.lt(_WARSAW_GIS_WEST)
            | lon.gt(_WARSAW_GIS_EAST)
            | lat.lt(_WARSAW_GIS_SOUTH)
            | lat.gt(_WARSAW_GIS_NORTH)
        )
    district_metro = pd.Series(False, index=out.index)
    if "district" in out.columns:
        names = out["district"].astype(str).str.casefold()
        district_metro = ~names.isin(WARSAW_DISTRICT_NAMES)
        district_metro |= names.eq("rcn")
    out["is_metro"] = (outside | district_metro).fillna(False)
    return out


def similar_size_band(frame: pd.DataFrame, area_m2: float) -> pd.DataFrame:
    """Keep rows whose area is within ±15% or ±10 m² of ``area_m2``."""
    if frame.empty or "area_m2" not in frame.columns or area_m2 <= 0:
        return frame.iloc[0:0].copy() if not frame.empty else frame
    area = pd.to_numeric(frame["area_m2"], errors="coerce")
    window = max(float(area_m2) * AREA_BAND_FRAC, AREA_BAND_ABS_M2)
    keep = (area - float(area_m2)).abs() <= window
    return frame.loc[keep.fillna(False)].copy()


def membership_stats(ask_ppm: float, neighbors: pd.DataFrame) -> dict[str, Any]:
    """Percentile / gap of an ask vs nearby deed zł/m²."""
    empty: dict[str, Any] = {
        "n": 0,
        "median_ppm": None,
        "idw_ppm": None,
        "percentile": None,
        "gap_ppm": None,
        "nearest_m": None,
    }
    if neighbors.empty or "price_per_m2" not in neighbors.columns:
        return empty
    values = pd.to_numeric(neighbors["price_per_m2"], errors="coerce").dropna()
    if values.empty:
        return empty
    median = float(values.median())
    gap = float(ask_ppm) - median
    percentile = float(np.mean(values.to_numpy(dtype=float) <= float(ask_ppm)) * 100.0)
    nearest = None
    if "distance_m" in neighbors.columns:
        nearest_s = pd.to_numeric(neighbors["distance_m"], errors="coerce").dropna()
        if not nearest_s.empty:
            nearest = float(nearest_s.min())
    idw = None
    if "distance_m" in neighbors.columns:
        dist = pd.to_numeric(neighbors["distance_m"], errors="coerce")
        aligned = pd.DataFrame({"ppm": values, "dist": dist.reindex(values.index)})
        aligned = aligned.dropna()
        if not aligned.empty:
            weights = 1.0 / np.clip(aligned["dist"].to_numpy(dtype=float), 25.0, None)
            idw = float(
                np.average(aligned["ppm"].to_numpy(dtype=float), weights=weights)
            )
    return {
        "n": len(values),
        "median_ppm": median,
        "idw_ppm": idw,
        "percentile": percentile,
        "gap_ppm": gap,
        "nearest_m": nearest,
    }


def _listed_timestamps(frame: pd.DataFrame) -> pd.Series:
    if frame.empty or "listed_at" not in frame.columns:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return pd.to_datetime(frame["listed_at"], errors="coerce", utc=True, format="mixed")


def local_monthly_medians(frame: pd.DataFrame) -> pd.DataFrame:
    """Median zł/m² and volume by listed month for a local neighbor set."""
    empty = pd.DataFrame(columns=["listed_month", "n", "median_ppm"])
    if frame.empty or "price_per_m2" not in frame.columns:
        return empty
    work = pd.DataFrame({
        "ppm": pd.to_numeric(frame["price_per_m2"], errors="coerce"),
        "when": _listed_timestamps(frame),
    }).dropna()
    if work.empty:
        return empty
    work["listed_month"] = work["when"].dt.strftime("%Y-%m")
    return (
        work
        .groupby("listed_month", dropna=True)
        .agg(n=("ppm", "size"), median_ppm=("ppm", "median"))
        .reset_index()
        .sort_values("listed_month")
        .reset_index(drop=True)
    )


def local_loyalty(
    neighbors: pd.DataFrame,
    as_of: object | None = None,
) -> dict[str, Any]:
    """12m vs 36m drift, YoY, and IQR — price-level stickiness, not DOM."""
    out: dict[str, Any] = {
        "n": 0,
        "median_12m": None,
        "median_36m": None,
        "drift_12_vs_36": None,
        "yoy_pct": None,
        "iqr_ppm": None,
        "cv": None,
    }
    if neighbors.empty or "price_per_m2" not in neighbors.columns:
        return out
    work = pd.DataFrame({
        "ppm": pd.to_numeric(neighbors["price_per_m2"], errors="coerce"),
        "when": _listed_timestamps(neighbors),
    }).dropna(subset=["ppm"])
    if work.empty:
        return out
    out["n"] = len(work)
    ppm = work["ppm"]
    q75, q25 = float(ppm.quantile(0.75)), float(ppm.quantile(0.25))
    out["iqr_ppm"] = q75 - q25
    mean = float(ppm.mean())
    std = float(ppm.std(ddof=0)) if len(ppm) > 1 else 0.0
    out["cv"] = (std / mean) if mean else None
    dated = work.dropna(subset=["when"])
    if dated.empty:
        return out
    if as_of is None:
        end = dated["when"].max()
    else:
        end = pd.to_datetime(as_of, utc=True, errors="coerce")
        if pd.isna(end):
            end = dated["when"].max()
    m12 = dated.loc[dated["when"] >= end - pd.DateOffset(months=12), "ppm"]
    m36 = dated.loc[dated["when"] >= end - pd.DateOffset(months=36), "ppm"]
    if not m12.empty:
        out["median_12m"] = float(m12.median())
    if not m36.empty:
        out["median_36m"] = float(m36.median())
    if out["median_12m"] is not None and out["median_36m"] is not None:
        out["drift_12_vs_36"] = float(out["median_12m"] - out["median_36m"])
    prev = dated.loc[
        (dated["when"] >= end - pd.DateOffset(months=24))
        & (dated["when"] < end - pd.DateOffset(months=12)),
        "ppm",
    ]
    if not m12.empty and not prev.empty and float(prev.median()) > 0:
        out["yoy_pct"] = (float(m12.median()) / float(prev.median()) - 1.0) * 100.0
    return out


def filter_recency(frame: pd.DataFrame, months: int) -> pd.DataFrame:
    """Keep rows listed within ``months`` of the newest stamp (0 = all)."""
    if months <= 0 or frame.empty:
        return frame
    when = _listed_timestamps(frame)
    end = when.max()
    if pd.isna(end):
        return frame
    keep = when >= end - pd.DateOffset(months=int(months))
    return frame.loc[keep.fillna(False)].copy()


def nearby_buyer_asks(
    lat: float,
    lon: float,
    asks: pd.DataFrame,
    *,
    radius_m: float,
    exclude_listing_id: str | None = None,
) -> pd.DataFrame:
    """Live buyer-ask pins in the same radius (Otodom / NOL / OLX / fixture)."""
    if asks.empty or "source" not in asks.columns:
        return asks.iloc[0:0].copy() if not asks.empty else asks
    sources = asks["source"].astype(str).str.casefold()
    work = asks.loc[sources.isin(BUYER_LISTING_SOURCES)].copy()
    if work.empty:
        return work
    return query_rcn_neighbors(
        lat,
        lon,
        work,
        radius_m=radius_m,
        exclude_listing_id=exclude_listing_id,
    )


def comps_map_payload(
    neighbors: pd.DataFrame,
    *,
    ask_ppm: float,
    max_pins: int,
) -> pd.DataFrame:
    """Slim pydeck rows coloured cheaper / similar / dearer vs the ask."""
    if neighbors.empty:
        return neighbors
    work = neighbors.copy()
    if max_pins > 0 and len(work) > max_pins:
        if "distance_m" in work.columns:
            work = work.nsmallest(int(max_pins), "distance_m")
        else:
            work = work.head(int(max_pins))
    ppm = pd.to_numeric(work["price_per_m2"], errors="coerce")
    rel = (ppm - float(ask_ppm)) / max(float(ask_ppm), 1.0)
    colors: list[list[int]] = []
    for value in rel.tolist():
        if value != value:
            colors.append([128, 128, 128, 200])
        elif value < -0.05:
            colors.append([46, 125, 50, 220])
        elif value > 0.05:
            colors.append([198, 40, 40, 220])
        else:
            colors.append([245, 166, 35, 220])
    work["color_r"] = [c[0] for c in colors]
    work["color_g"] = [c[1] for c in colors]
    work["color_b"] = [c[2] for c in colors]
    work["color_a"] = [c[3] for c in colors]
    work["radius"] = 28.0
    work["tooltip_name"] = [
        f"{float(p):,.0f} zł/m²" if pd.notna(p) else "RCN" for p in ppm
    ]
    dist = (
        pd.to_numeric(work["distance_m"], errors="coerce")
        if "distance_m" in work.columns
        else pd.Series(np.nan, index=work.index)
    )
    area = (
        pd.to_numeric(work["area_m2"], errors="coerce")
        if "area_m2" in work.columns
        else pd.Series(np.nan, index=work.index)
    )
    when = work["listed_at"].astype(str) if "listed_at" in work.columns else ""
    district = work["district"].astype(str) if "district" in work.columns else ""
    metro = (
        work["is_metro"].map(lambda v: "metro" if bool(v) else "city")
        if "is_metro" in work.columns
        else pd.Series("city", index=work.index)
    )
    work["tooltip_detail"] = [
        f"{d} m · {a:.0f} m² · {w} · {dist_lab} · {m}"
        if pd.notna(d) and pd.notna(a)
        else f"{w} · {dist_lab} · {m}"
        for d, a, w, dist_lab, m in zip(
            dist,
            area,
            when,
            district,
            metro,
            strict=True,
        )
    ]
    if "listing_id" not in work.columns:
        work["listing_id"] = [f"RCN-{i}" for i in range(len(work))]
    return work.reset_index(drop=True)


def prepare_offer_comps(
    *,
    lat: float,
    lon: float,
    ask_ppm: float,
    area_m2: float | None,
    listing_id: str | None,
    rcn: pd.DataFrame,
    asks: pd.DataFrame,
    radius_m: float,
    months: int = 0,
    include_metro: bool = True,
    show_nearby_asks: bool = True,
    max_pins: int = 200,
) -> dict[str, Any]:
    """Assemble neighbor frames + KPIs for the Offer Comps subtab."""
    neighbors = query_rcn_neighbors(
        lat,
        lon,
        rcn,
        radius_m=radius_m,
        exclude_listing_id=listing_id,
    )
    neighbors = drop_centroid_fallbacks(neighbors)
    neighbors = flag_metro_rows(neighbors)
    if not include_metro and not neighbors.empty:
        neighbors = neighbors.loc[~neighbors["is_metro"]].copy()
    neighbors = filter_recency(neighbors, months)
    size_band = (
        similar_size_band(neighbors, float(area_m2))
        if area_m2 is not None and area_m2 > 0
        else neighbors.iloc[0:0].copy()
    )
    stats = membership_stats(ask_ppm, neighbors)
    size_stats = membership_stats(ask_ppm, size_band) if not size_band.empty else None
    loyalty = local_loyalty(neighbors)
    ask_neighbors = pd.DataFrame()
    ask_stats: dict[str, Any] | None = None
    if show_nearby_asks:
        ask_neighbors = nearby_buyer_asks(
            lat,
            lon,
            asks,
            radius_m=radius_m,
            exclude_listing_id=listing_id,
        )
        ask_stats = membership_stats(ask_ppm, ask_neighbors)
    return {
        "neighbors": neighbors,
        "size_band": size_band,
        "ask_neighbors": ask_neighbors,
        "stats": stats,
        "size_stats": size_stats,
        "ask_stats": ask_stats,
        "loyalty": loyalty,
        "monthly_rcn": local_monthly_medians(neighbors),
        "monthly_asks": local_monthly_medians(ask_neighbors),
        "map_rcn": comps_map_payload(neighbors, ask_ppm=ask_ppm, max_pins=max_pins),
        "map_asks": comps_map_payload(
            ask_neighbors,
            ask_ppm=ask_ppm,
            max_pins=max_pins,
        )
        if show_nearby_asks
        else pd.DataFrame(),
    }
