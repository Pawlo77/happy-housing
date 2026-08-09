"""Commute-target resolve, staleness, and feature recompute."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import lightgbm as lgb
import pandas as pd
from shapely.geometry import Point

from happy_housing.app.app_settings import (
    AppSettings,
    load_app_settings,
    save_app_settings,
    settings_path,
)
from happy_housing.config import ExperimentConfig
from happy_housing.gis import nearest_distance_meters
from happy_housing.gis.routing import (
    attach_commute_minutes,
    attach_transit_commute_minutes,
)
from happy_housing.models.train import load_model_meta, predict_price_per_m2
from happy_housing.storage import write_geoparquet

logger = logging.getLogger(__name__)

DEFAULT_COMMUTE_LABEL = "Rondo Daszyńskiego"
"""Default commute destination label (YAML coords)."""

COORD_EPS: float = 1e-5
"""Lon/lat equality tolerance for commute stale / pin-overlap detection."""


@dataclass(frozen=True, slots=True)
class CommuteTarget:
    """Resolved commute destination for enrich / map / UI."""

    lon: float
    """WGS84 longitude."""
    lat: float
    """WGS84 latitude."""
    label: str
    """Human-readable place name."""


def commute_compare_points(
    *,
    saved: CommuteTarget,
    draft: CommuteTarget,
) -> pd.DataFrame:
    """Rows for Admin map: saved target vs draft (new) pin."""
    return pd.DataFrame([
        {
            "lon": float(saved.lon),
            "lat": float(saved.lat),
            "role": "saved",
            "tooltip_name": f"Saved: {saved.label}",
            "tooltip_detail": f"{saved.lon:.5f}, {saved.lat:.5f}",
        },
        {
            "lon": float(draft.lon),
            "lat": float(draft.lat),
            "role": "draft",
            "tooltip_name": f"New: {draft.label}",
            "tooltip_detail": f"{draft.lon:.5f}, {draft.lat:.5f}",
        },
    ])


def commute_compare_view(
    *,
    saved: CommuteTarget,
    draft: CommuteTarget,
) -> tuple[float, float, float]:
    """Map center ``(lat, lon, zoom)`` covering saved and draft pins."""
    lat = 0.5 * (float(saved.lat) + float(draft.lat))
    lon = 0.5 * (float(saved.lon) + float(draft.lon))
    span = max(
        abs(float(saved.lat) - float(draft.lat)),
        abs(float(saved.lon) - float(draft.lon)),
    )
    if span < 1e-5:
        zoom = 13.0
    elif span < 0.01:
        zoom = 12.5
    elif span < 0.03:
        zoom = 12.0
    elif span < 0.08:
        zoom = 11.0
    else:
        zoom = 10.0
    return lat, lon, zoom


def resolve_commute_target(
    settings: AppSettings,
    config: ExperimentConfig,
) -> CommuteTarget:
    """Prefer Admin override; else experiment YAML defaults."""
    if (
        settings.commute_target_lon is not None
        and settings.commute_target_lat is not None
    ):
        label = (settings.commute_target_label or "").strip() or DEFAULT_COMMUTE_LABEL
        return CommuteTarget(
            lon=float(settings.commute_target_lon),
            lat=float(settings.commute_target_lat),
            label=label,
        )
    return CommuteTarget(
        lon=float(config.commute_target_lon),
        lat=float(config.commute_target_lat),
        label=DEFAULT_COMMUTE_LABEL,
    )


def commute_target_stale(settings: AppSettings) -> bool:
    """True when scored features were built for a different destination."""
    if settings.commute_target_lon is None or settings.commute_target_lat is None:
        target_lon = None
        target_lat = None
    else:
        target_lon = float(settings.commute_target_lon)
        target_lat = float(settings.commute_target_lat)
    feat_lon = settings.commute_features_lon
    feat_lat = settings.commute_features_lat
    if target_lon is None or target_lat is None:
        return False
    if feat_lon is None or feat_lat is None:
        return False
    return (
        abs(float(feat_lon) - target_lon) > COORD_EPS
        or abs(float(feat_lat) - target_lat) > COORD_EPS
    )


def mark_commute_features_applied(settings: AppSettings, target: CommuteTarget) -> None:
    """Record that scored data matches ``target``."""
    settings.commute_features_lon = float(target.lon)
    settings.commute_features_lat = float(target.lat)


def recompute_commute_features(
    processed: Path,
    *,
    dest_lon: float,
    dest_lat: float,
    label: str,
    config: ExperimentConfig,
    projected_crs: str,
    osrm_url: str,
    use_otp_transit: bool = False,
    otp_url: str | None = None,
) -> dict[str, Any]:
    """Refresh distance / commute / scores for a new destination.

    Updates ``dist_daszynskiego_m``, ``commute_min``, optional
    ``transit_commute_min``, then re-scores with the saved LightGBM booster
    (no full HPO retrain). Persists commute feature fingerprint in app settings.
    """
    scored_path = processed / "listings_scored.parquet"
    if not scored_path.exists():
        msg = f"Missing scored listings: {scored_path}"
        raise FileNotFoundError(msg)
    frame = pd.read_parquet(scored_path)
    if frame.empty or "lat" not in frame.columns or "lon" not in frame.columns:
        msg = "Scored frame has no lon/lat rows to recompute"
        raise ValueError(msg)

    gdf = gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(
            pd.to_numeric(frame["lon"], errors="coerce"),
            pd.to_numeric(frame["lat"], errors="coerce"),
        ),
        crs="EPSG:4326",
    )
    gdf = gdf.dropna(subset=["geometry"])
    target = gpd.GeoDataFrame(
        geometry=[Point(float(dest_lon), float(dest_lat))],
        crs="EPSG:4326",
    )
    gdf = nearest_distance_meters(gdf, target, projected_crs, "dist_daszynskiego_m")
    work = pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))
    work = attach_commute_minutes(
        work,
        dest_lon=float(dest_lon),
        dest_lat=float(dest_lat),
        osrm_url=osrm_url,
    )
    if use_otp_transit:
        work = attach_transit_commute_minutes(
            work,
            dest_lon=float(dest_lon),
            dest_lat=float(dest_lat),
            otp_url=otp_url or config.routing.otp_url,
        )

    model_path = processed / "lightgbm_price_per_m2.txt"
    if model_path.exists():
        booster = lgb.Booster(model_file=str(model_path))
        meta = load_model_meta(model_path)
        work["model_price_per_m2"] = predict_price_per_m2(
            booster,
            work,
            log_target=bool(meta.get("log_target", True)),
            district_offsets=meta.get("district_offsets") or {},
        )
        work["delta_price_per_m2"] = (
            pd.to_numeric(work["price_per_m2"], errors="coerce")
            - work["model_price_per_m2"]
        )
    else:
        logger.warning("No model at %s — skipped rescore", model_path)

    work.to_parquet(scored_path, index=False)
    enriched_path = processed / "listings_enriched.parquet"
    if enriched_path.exists():
        try:
            enriched = gpd.read_parquet(enriched_path)
            for col in (
                "dist_daszynskiego_m",
                "commute_min",
                "transit_commute_min",
                "model_price_per_m2",
                "delta_price_per_m2",
            ):
                if col in work.columns:
                    enriched[col] = work[col].to_numpy()
            write_geoparquet(enriched, enriched_path)
        except Exception:
            logger.exception(
                "Failed to sync enriched GeoParquet after commute recompute"
            )

    settings_file = settings_path(processed)
    settings = load_app_settings(settings_file)
    settings.commute_target_lon = float(dest_lon)
    settings.commute_target_lat = float(dest_lat)
    settings.commute_target_label = (label or "").strip() or DEFAULT_COMMUTE_LABEL
    mark_commute_features_applied(
        settings,
        CommuteTarget(
            lon=float(dest_lon),
            lat=float(dest_lat),
            label=settings.commute_target_label,
        ),
    )
    save_app_settings(settings_file, settings)
    logger.info(
        "Recomputed commute for %s offers → %s (%.5f, %.5f)",
        len(work),
        settings.commute_target_label,
        dest_lon,
        dest_lat,
    )
    return {
        "n_offers": len(work),
        "scored": scored_path,
        "label": settings.commute_target_label,
        "lon": float(dest_lon),
        "lat": float(dest_lat),
    }
