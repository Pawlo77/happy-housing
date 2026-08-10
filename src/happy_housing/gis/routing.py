"""Commute-time features via local OSRM / OTP, with distance fallback."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_OTP_MAX_WORKERS: int = 8
"""Bound OTP fan-out so large listing sets do not open one socket per row."""

OSRM_CHUNK_SIZE: int = 200
"""Max origins per OSRM Table request (URL / server limit)."""

_OSRM_CHUNK_WORKERS: int = 4
"""Parallel OSRM chunk requests when the origin list is large."""


def _osrm_table_chunk(
    origins: Sequence[tuple[float, float]],
    destination: tuple[float, float],
    *,
    base_url: str,
    timeout_s: float,
) -> np.ndarray | None:
    """Single OSRM Table request for one origin chunk."""
    if not origins:
        return np.array([], dtype=float)
    all_points = [*origins, destination]
    coord_str = ";".join(f"{lon},{lat}" for lon, lat in all_points)
    dest_index = len(all_points) - 1
    sources = ";".join(str(i) for i in range(len(origins)))
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


def osrm_table_durations_seconds(
    origins: Iterable[tuple[float, float]],
    destination: tuple[float, float],
    base_url: str = "http://127.0.0.1:5001",
    timeout_s: float = 30.0,
) -> np.ndarray | None:
    """Query OSRM Table API for durations from origins to one destination.

    Large origin lists are split into ``OSRM_CHUNK_SIZE`` batches (fetched in
    parallel) so URL length and server limits do not fail the whole request.

    Args:
        origins: Sequence of ``(lon, lat)`` points.
        destination: ``(lon, lat)`` commute target.
        base_url: Local OSRM HTTP root.
        timeout_s: Request timeout per chunk.

    Returns:
        Durations in seconds, or ``None`` if OSRM is unavailable.
    """
    coords = list(origins)
    if not coords:
        return np.array([], dtype=float)
    chunk = max(1, int(OSRM_CHUNK_SIZE))
    if len(coords) <= chunk:
        return _osrm_table_chunk(
            coords, destination, base_url=base_url, timeout_s=timeout_s
        )

    slices = [coords[i : i + chunk] for i in range(0, len(coords), chunk)]
    parts: list[np.ndarray | None] = [None] * len(slices)
    workers = min(_OSRM_CHUNK_WORKERS, len(slices))

    def _run(
        idx: int, batch: Sequence[tuple[float, float]]
    ) -> tuple[int, np.ndarray | None]:
        return idx, _osrm_table_chunk(
            batch, destination, base_url=base_url, timeout_s=timeout_s
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run, i, batch) for i, batch in enumerate(slices)]
        for fut in as_completed(futures):
            idx, arr = fut.result()
            if arr is None:
                return None
            parts[idx] = arr
    ordered = [p for p in parts if p is not None]
    if len(ordered) != len(slices):
        return None
    return np.concatenate(ordered)


def attach_commute_minutes(
    frame: pd.DataFrame,
    *,
    dest_lon: float,
    dest_lat: float,
    osrm_url: str = "http://127.0.0.1:5001",
    distance_col: str = "dist_daszynskiego_m",
    out_col: str = "commute_min",
    require_osrm: bool = False,
) -> pd.DataFrame:
    """Add peak-commute minutes via OSRM or 30 km/h distance proxy.

    Args:
        require_osrm: When true, raise if the Table API is unreachable instead
            of silently filling the 30 km/h distance proxy.
    """
    out = frame.copy()
    origins = list(zip(out["lon"].astype(float), out["lat"].astype(float), strict=True))
    durations = osrm_table_durations_seconds(
        origins, (dest_lon, dest_lat), base_url=osrm_url
    )
    if durations is not None and len(durations) == len(out):
        out[out_col] = durations / 60.0
        return out
    if require_osrm:
        raise RuntimeError(
            f"OSRM required but unavailable at {osrm_url!r} "
            "(macOS AirPlay often occupies :5000 — use host port 5001)"
        )
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
