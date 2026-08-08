"""Commute-time features via local OSRM / OTP, with distance fallback."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_OTP_MAX_WORKERS: int = 8
"""Bound OTP fan-out so large listing sets do not open one socket per row."""


def osrm_table_durations_seconds(
    origins: Iterable[tuple[float, float]],
    destination: tuple[float, float],
    base_url: str = "http://127.0.0.1:5000",
    timeout_s: float = 30.0,
) -> np.ndarray | None:
    """Query OSRM Table API for durations from origins to one destination.

    Args:
        origins: Sequence of ``(lon, lat)`` points.
        destination: ``(lon, lat)`` commute target.
        base_url: Local OSRM HTTP root.
        timeout_s: Request timeout.

    Returns:
        Durations in seconds, or ``None`` if OSRM is unavailable.
    """
    coords = list(origins)
    if not coords:
        return np.array([], dtype=float)
    # OSRM table: all points in one coordinates list; destinations index last.
    all_points = [*coords, destination]
    coord_str = ";".join(f"{lon},{lat}" for lon, lat in all_points)
    dest_index = len(all_points) - 1
    sources = ";".join(str(i) for i in range(len(coords)))
    url = (
        f"{base_url.rstrip('/')}/table/v1/driving/{coord_str}"
        f"?sources={sources}&destinations={dest_index}"
    )
    try:
        response = requests.get(url, timeout=timeout_s)
        response.raise_for_status()
        payload = response.json()
        durations = payload.get("durations")
        if not durations:
            return None
        return np.array([row[0] if row else np.nan for row in durations], dtype=float)
    except Exception as exc:
        logger.warning("OSRM unavailable (%s); using distance proxy", exc)
        return None


def attach_commute_minutes(
    frame: pd.DataFrame,
    *,
    dest_lon: float,
    dest_lat: float,
    osrm_url: str = "http://127.0.0.1:5000",
    distance_col: str = "dist_daszynskiego_m",
    out_col: str = "commute_min",
) -> pd.DataFrame:
    """Add peak-commute minutes via OSRM or 30 km/h distance proxy."""
    out = frame.copy()
    origins = list(zip(out["lon"].astype(float), out["lat"].astype(float), strict=True))
    durations = osrm_table_durations_seconds(
        origins, (dest_lon, dest_lat), base_url=osrm_url
    )
    if durations is not None and len(durations) == len(out):
        out[out_col] = durations / 60.0
        return out
    if distance_col in out.columns:
        # Assume ~30 km/h average urban speed.
        out[out_col] = (out[distance_col].astype(float) / 1000.0) / 30.0 * 60.0
    else:
        out[out_col] = np.nan
    return out


def otp_plan_duration_seconds(
    origin: tuple[float, float],
    destination: tuple[float, float],
    *,
    base_url: str = "http://127.0.0.1:8080",
    timeout_s: float = 20.0,
) -> float | None:
    """Query OTP2 plan API for best TRANSIT+WALK duration in seconds.

    Args:
        origin: ``(lon, lat)`` start.
        destination: ``(lon, lat)`` target.
        base_url: Local OTP HTTP root.
        timeout_s: Request timeout.

    Returns:
        Shortest itinerary duration in seconds, or ``None`` if OTP is down.
    """
    lon0, lat0 = origin
    lon1, lat1 = destination
    url = f"{base_url.rstrip('/')}/otp/routers/default/plan"
    params = {
        "fromPlace": f"{lat0},{lon0}",
        "toPlace": f"{lat1},{lon1}",
        "mode": "TRANSIT,WALK",
        "maxWalkDistance": 1500,
        "arriveBy": "false",
        "numItineraries": 3,
    }
    try:
        response = requests.get(url, params=params, timeout=timeout_s)
        response.raise_for_status()
        payload = response.json()
        itineraries = (payload.get("plan") or {}).get("itineraries") or []
        durations_ms = [
            float(item["duration"])
            for item in itineraries
            if isinstance(item, dict) and "duration" in item
        ]
        if not durations_ms:
            return None
        return min(durations_ms) / 1000.0
    except Exception as exc:
        logger.warning("OTP unavailable (%s); transit minutes left empty", exc)
        return None


def attach_transit_commute_minutes(
    frame: pd.DataFrame,
    *,
    dest_lon: float,
    dest_lat: float,
    otp_url: str = "http://127.0.0.1:8080",
    out_col: str = "transit_commute_min",
    max_workers: int = _OTP_MAX_WORKERS,
) -> pd.DataFrame:
    """Add transit+walk minutes via OTP2; NaN when OTP is unavailable.

    GTFS stop proximity (``dist_transit_stop_m``) remains the offline proxy.
    """
    out = frame.copy()
    if out.empty:
        out[out_col] = pd.Series(dtype=float)
        return out
    origins = list(zip(out["lon"].astype(float), out["lat"].astype(float), strict=True))
    dest = (dest_lon, dest_lat)
    results: list[float] = [float("nan")] * len(origins)
    workers = max(1, min(int(max_workers), len(origins)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(otp_plan_duration_seconds, origin, dest, base_url=otp_url): idx
            for idx, origin in enumerate(origins)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            seconds = fut.result()
            if seconds is not None:
                results[idx] = seconds / 60.0
    out[out_col] = results
    return out
