"""Commute-target resolve, staleness, and feature recompute."""

from __future__ import annotations

import logging
from collections.abc import Callable
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
from happy_housing.gis.routing import attach_active_mode_commutes
from happy_housing.models.train import load_model_meta, predict_price_per_m2
from happy_housing.storage import write_geoparquet

logger = logging.getLogger(__name__)

DEFAULT_COMMUTE_LABEL = "Rondo Daszyńskiego"
"""Default commute destination label (YAML coords)."""

COORD_EPS: float = 1e-5
"""Lon/lat equality tolerance for commute stale / pin-overlap detection."""

ProgressCallback = Callable[[str], None]
"""Optional UI progress hook (Admin ``st.status`` write)."""


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


def _emit_progress(on_progress: ProgressCallback | None, message: str) -> None:
    """Log an INFO milestone and optionally notify the UI."""
    text = str(message).strip()
    if not text:
        return
    logger.info("%s", text)
    if on_progress is not None:
        on_progress(text)


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
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Refresh distance / commute / scores for a new destination.

    Updates ``dist_daszynskiego_m``, ``commute_min``, optional
    ``transit_commute_min``, then re-scores with the saved LightGBM booster
    (no full HPO retrain). Persists commute feature fingerprint in app settings.
    """
    dest_label = (label or "").strip() or DEFAULT_COMMUTE_LABEL
    _emit_progress(
        on_progress,
        f"Starting commute recompute → {dest_label} "
        f"({float(dest_lon):.5f}, {float(dest_lat):.5f})",
    )
    scored_path = processed / "listings_scored.parquet"
    if not scored_path.exists():
        msg = f"Missing scored listings: {scored_path}"
        logger.error("%s", msg)
        raise FileNotFoundError(msg)
    _emit_progress(on_progress, f"Loading scored listings from {scored_path.name}…")
    frame = pd.read_parquet(scored_path)
    if frame.empty or "lat" not in frame.columns or "lon" not in frame.columns:
        msg = "Scored frame has no lon/lat rows to recompute"
        logger.error("%s", msg)
        raise ValueError(msg)
    logger.debug("Loaded %s scored rows for commute recompute", len(frame))

    _emit_progress(on_progress, "Computing straight-line distance to target…")
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

    modes = "car/bike/walk" + (" + OTP transit" if use_otp_transit else "")
    _emit_progress(
        on_progress,
        f"Routing {len(work):,} offers ({modes}) via OSRM"
        f"{'/OTP' if use_otp_transit else ''}…",
    )
    work = attach_active_mode_commutes(
        work,
        dest_lon=float(dest_lon),
        dest_lat=float(dest_lat),
        osrm_url=osrm_url,
        require_osrm=bool(config.routing.require_osrm),
        use_otp_transit=bool(use_otp_transit),
        otp_url=otp_url or config.routing.otp_url,
    )
    for col in (
        "commute_min",
        "bike_commute_min",
        "walk_commute_min",
        "transit_commute_min",
    ):
        if col not in work.columns:
            continue
        filled = int(pd.to_numeric(work[col], errors="coerce").notna().sum())
        logger.debug(
            "Commute column %s filled %s/%s",
            col,
            filled,
            len(work),
        )

    model_path = processed / "lightgbm_price_per_m2.txt"
    if model_path.exists():
        _emit_progress(on_progress, "Rescoring with saved LightGBM model…")
        booster = lgb.Booster(model_file=str(model_path))
        meta = load_model_meta(model_path)
        work["model_price_per_m2"] = predict_price_per_m2(
            booster,
            work,
            log_target=bool(meta.get("log_target", True)),
            district_offsets=meta.get("district_offsets") or {},
            use_engineered=bool(meta.get("use_engineered", False)),
            district_ppm_stats=meta.get("district_ppm_stats") or {},
            district_global_median=meta.get("district_global_median"),
            district_global_mean=meta.get("district_global_mean"),
        )
        work["delta_price_per_m2"] = (
            pd.to_numeric(work["price_per_m2"], errors="coerce")
            - work["model_price_per_m2"]
        )
    else:
        logger.warning("No model at %s — skipped rescore", model_path)
        _emit_progress(
            on_progress,
            "No saved model found — skipped price rescore (distances still updated).",
        )

    _emit_progress(on_progress, f"Writing {scored_path.name}…")
    work.to_parquet(scored_path, index=False)
    enriched_path = processed / "listings_enriched.parquet"
    if enriched_path.exists():
        _emit_progress(on_progress, f"Syncing {enriched_path.name}…")
        try:
            enriched = gpd.read_parquet(enriched_path)
            for col in (
                "dist_daszynskiego_m",
                "commute_min",
                "transit_commute_min",
                "bike_commute_min",
                "walk_commute_min",
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
            _emit_progress(
                on_progress,
                "Warning: scored parquet updated, but enriched sync failed (see logs).",
            )

    settings_file = settings_path(processed)
    settings = load_app_settings(settings_file)
    settings.commute_target_lon = float(dest_lon)
    settings.commute_target_lat = float(dest_lat)
    settings.commute_target_label = dest_label
    mark_commute_features_applied(
        settings,
        CommuteTarget(
            lon=float(dest_lon),
            lat=float(dest_lat),
            label=settings.commute_target_label,
        ),
    )
    save_app_settings(settings_file, settings)
    _emit_progress(
        on_progress,
        f"Done — recomputed commute for {len(work):,} offers → {dest_label}",
    )
    return {
        "n_offers": len(work),
        "scored": scored_path,
        "label": settings.commute_target_label,
        "lon": float(dest_lon),
        "lat": float(dest_lat),
    }
