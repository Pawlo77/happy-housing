"""Commute-time features via local OSRM / OTP, with distance fallback."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

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
    profile: str = "driving",
) -> np.ndarray | None:
    """Single OSRM Table request for one origin chunk."""
    if not origins:
        return np.array([], dtype=float)
    all_points = [*origins, destination]
    coord_str = ";".join(f"{lon},{lat}" for lon, lat in all_points)
    dest_index = len(all_points) - 1
    sources = ";".join(str(i) for i in range(len(origins)))
    profile_key = str(profile or "driving").strip() or "driving"
    url = (
        f"{base_url.rstrip('/')}/table/v1/{profile_key}/{coord_str}"
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
        logger.warning(
            "OSRM unavailable (%s profile=%s); using distance proxy",
            exc,
            profile_key,
        )
        return None


def osrm_table_durations_seconds(
    origins: Iterable[tuple[float, float]],
    destination: tuple[float, float],
    base_url: str = "http://127.0.0.1:5001",
    timeout_s: float = 30.0,
    profile: str = "driving",
) -> np.ndarray | None:
    """Query OSRM Table API for durations from origins to one destination.

    Large origin lists are split into ``OSRM_CHUNK_SIZE`` batches (fetched in
    parallel) so URL length and server limits do not fail the whole request.

    Args:
        origins: Sequence of ``(lon, lat)`` points.
        destination: ``(lon, lat)`` commute target.
        base_url: Local OSRM HTTP root.
        timeout_s: Request timeout per chunk.
        profile: OSRM profile segment (``driving`` / ``bike`` / ``foot``).

    Returns:
        Durations in seconds, or ``None`` if OSRM is unavailable.
    """
    coords = list(origins)
    if not coords:
        return np.array([], dtype=float)
    chunk = max(1, int(OSRM_CHUNK_SIZE))
    if len(coords) <= chunk:
        return _osrm_table_chunk(
            coords,
            destination,
            base_url=base_url,
            timeout_s=timeout_s,
            profile=profile,
        )

    slices = [coords[i : i + chunk] for i in range(0, len(coords), chunk)]
    parts: list[np.ndarray | None] = [None] * len(slices)
    workers = min(_OSRM_CHUNK_WORKERS, len(slices))

    def _run(
        idx: int, batch: Sequence[tuple[float, float]]
    ) -> tuple[int, np.ndarray | None]:
        return idx, _osrm_table_chunk(
            batch,
            destination,
            base_url=base_url,
            timeout_s=timeout_s,
            profile=profile,
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
    profile: str = "driving",
    proxy_speed_kmh: float = 30.0,
) -> pd.DataFrame:
    """Add commute minutes via OSRM Table or a constant-speed distance proxy.

    Args:
        require_osrm: When true, raise if the Table API is unreachable instead
            of silently filling the distance proxy.
        profile: OSRM routing profile (``driving``, ``bike``, ``foot``).
        proxy_speed_kmh: Assumed urban speed when OSRM is down (car ≈ 30,
            bike ≈ 15, walk ≈ 5).
    """
    out = frame.copy()
    origins = list(zip(out["lon"].astype(float), out["lat"].astype(float), strict=True))
    durations = osrm_table_durations_seconds(
        origins,
        (dest_lon, dest_lat),
        base_url=osrm_url,
        profile=profile,
    )
    if durations is not None and len(durations) == len(out):
        out[out_col] = durations / 60.0
        logger.info(
            "OSRM %s filled %s/%s rows → %s",
            profile,
            len(out),
            len(out),
            out_col,
        )
        return out
    if require_osrm:
        raise RuntimeError(
            f"OSRM required but unavailable at {osrm_url!r} "
            f"(profile={profile!r}; macOS AirPlay often occupies :5000 — "
            "use host port 5001)"
        )
    logger.warning(
        "OSRM %s unavailable at %s — using %.0f km/h proxy for %s",
        profile,
        osrm_url,
        float(proxy_speed_kmh),
        out_col,
    )
    speed = float(proxy_speed_kmh) if float(proxy_speed_kmh) > 0 else 30.0
    if distance_col in out.columns:
        out[out_col] = (out[distance_col].astype(float) / 1000.0) / speed * 60.0
    else:
        out[out_col] = np.nan
    return out


def attach_active_mode_commutes(
    frame: pd.DataFrame,
    *,
    dest_lon: float,
    dest_lat: float,
    osrm_url: str,
    require_osrm: bool,
    use_otp_transit: bool,
    otp_url: str,
) -> pd.DataFrame:
    """Fill car / transit / bike / walk commute columns for one destination.

    Car uses OSRM ``driving`` (``commute_min``). Bike / walk use OSRM
    ``bike`` / ``foot`` when those graphs exist; otherwise speed proxies.
    Transit uses OTP when enabled.
    """
    logger.info(
        "Attach multi-mode commute for %s rows → (%.5f, %.5f) otp=%s",
        len(frame),
        float(dest_lon),
        float(dest_lat),
        bool(use_otp_transit),
    )
    out = attach_commute_minutes(
        frame,
        dest_lon=dest_lon,
        dest_lat=dest_lat,
        osrm_url=osrm_url,
        require_osrm=require_osrm,
        profile="driving",
        out_col="commute_min",
        proxy_speed_kmh=30.0,
    )
    out = attach_commute_minutes(
        out,
        dest_lon=dest_lon,
        dest_lat=dest_lat,
        osrm_url=osrm_url,
        require_osrm=False,
        profile="bike",
        out_col="bike_commute_min",
        proxy_speed_kmh=15.0,
    )
    out = attach_commute_minutes(
        out,
        dest_lon=dest_lon,
        dest_lat=dest_lat,
        osrm_url=osrm_url,
        require_osrm=False,
        profile="foot",
        out_col="walk_commute_min",
        proxy_speed_kmh=5.0,
    )
    if use_otp_transit:
        logger.info("Querying OTP transit minutes at %s", otp_url)
        out = attach_transit_commute_minutes(
            out,
            dest_lon=dest_lon,
            dest_lat=dest_lat,
            otp_url=otp_url,
        )
        filled = int(
            pd.to_numeric(out.get("transit_commute_min"), errors="coerce").notna().sum()
        )
        logger.info("OTP transit filled %s/%s rows", filled, len(out))
    elif "transit_commute_min" not in out.columns:
        out["transit_commute_min"] = float("nan")
    return out


def otp_plan_payload(
    origin: tuple[float, float],
    destination: tuple[float, float],
    *,
    base_url: str = "http://127.0.0.1:8080",
    timeout_s: float = 20.0,
    hour: int | None = None,
    otp_mode: str = "TRANSIT,WALK",
) -> dict[str, object] | None:
    """Fetch OTP2 plan JSON; optional ``hour`` sets departure time today.

    Args:
        origin: ``(lon, lat)`` start.
        destination: ``(lon, lat)`` target.
        base_url: Local OTP HTTP root.
        timeout_s: Request timeout.
        hour: Local clock hour 0-23 for departure; ``None`` = OTP default now.
        otp_mode: OTP ``mode`` query value.

    Returns:
        Parsed JSON payload, or ``None`` if OTP is down / empty.
    """
    lon0, lat0 = origin
    lon1, lat1 = destination
    url = f"{base_url.rstrip('/')}/otp/routers/default/plan"
    params: dict[str, str] = {
        "fromPlace": f"{lat0},{lon0}",
        "toPlace": f"{lat1},{lon1}",
        "mode": str(otp_mode),
        "maxWalkDistance": "1500",
        "arriveBy": "false",
        "numItineraries": "3",
    }
    if hour is not None:
        params["date"] = date.today().isoformat()
        params["time"] = f"{int(hour) % 24:02d}:00:00"
    try:
        response = requests.get(url, params=params, timeout=timeout_s)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return None
        return payload
    except Exception as exc:
        logger.warning("OTP unavailable (%s); transit route left empty", exc)
        return None


def otp_plan_duration_seconds(
    origin: tuple[float, float],
    destination: tuple[float, float],
    *,
    base_url: str = "http://127.0.0.1:8080",
    timeout_s: float = 20.0,
    hour: int | None = None,
) -> float | None:
    """Query OTP2 plan API for best TRANSIT+WALK duration in seconds.

    OTP2 ``itinerary.duration`` is already seconds (not milliseconds).

    Args:
        origin: ``(lon, lat)`` start.
        destination: ``(lon, lat)`` target.
        base_url: Local OTP HTTP root.
        timeout_s: Request timeout.
        hour: Optional departure hour (0-23).

    Returns:
        Shortest itinerary duration in seconds, or ``None`` if OTP is down.
    """
    payload = otp_plan_payload(
        origin,
        destination,
        base_url=base_url,
        timeout_s=timeout_s,
        hour=hour,
    )
    if payload is None:
        return None
    plan = payload.get("plan")
    if not isinstance(plan, dict):
        return None
    itineraries = plan.get("itineraries") or []
    if not isinstance(itineraries, list):
        return None
    durations = [
        float(item["duration"])
        for item in itineraries
        if isinstance(item, dict) and "duration" in item
    ]
    if not durations:
        return None
    return min(durations)


def otp_plan_best_itinerary(
    origin: tuple[float, float],
    destination: tuple[float, float],
    *,
    base_url: str = "http://127.0.0.1:8080",
    timeout_s: float = 20.0,
    hour: int | None = None,
) -> dict[str, object] | None:
    """Return the shortest OTP itinerary dict (with legs), or ``None``."""
    payload = otp_plan_payload(
        origin,
        destination,
        base_url=base_url,
        timeout_s=timeout_s,
        hour=hour,
    )
    if payload is None:
        return None
    plan = payload.get("plan")
    if not isinstance(plan, dict):
        return None
    itineraries = plan.get("itineraries") or []
    if not isinstance(itineraries, list) or not itineraries:
        return None
    scored: list[tuple[float, dict[str, object]]] = []
    for item in itineraries:
        if not isinstance(item, dict) or "duration" not in item:
            continue
        scored.append((float(item["duration"]), item))
    if not scored:
        return None
    scored.sort(key=lambda pair: pair[0])
    return scored[0][1]


def osrm_route_geojson(
    origin: tuple[float, float],
    destination: tuple[float, float],
    *,
    base_url: str = "http://127.0.0.1:5001",
    profile: str = "driving",
    timeout_s: float = 30.0,
) -> dict[str, object] | None:
    """OSRM ``/route`` GeoJSON for one origin→destination.

    Returns:
        Dict with ``duration`` (s), ``distance`` (m), ``coordinates``
        (``[[lon,lat],…]``), or ``None`` when OSRM is down.
    """
    lon0, lat0 = origin
    lon1, lat1 = destination
    profile_key = str(profile or "driving").strip() or "driving"
    url = (
        f"{base_url.rstrip('/')}/route/v1/{profile_key}/"
        f"{lon0},{lat0};{lon1},{lat1}"
        "?overview=full&geometries=geojson"
    )
    try:
        response = requests.get(url, timeout=timeout_s)
        response.raise_for_status()
        payload = response.json()
        routes = payload.get("routes") or []
        if not routes:
            return None
        route = routes[0]
        if not isinstance(route, dict):
            return None
        geometry = route.get("geometry") or {}
        coords = geometry.get("coordinates") if isinstance(geometry, dict) else None
        if not isinstance(coords, list) or len(coords) < 2:
            return None
        path: list[list[float]] = []
        for point in coords:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            path.append([float(point[0]), float(point[1])])
        if len(path) < 2:
            return None
        return {
            "duration": float(route.get("duration") or 0.0),
            "distance": float(route.get("distance") or 0.0),
            "coordinates": path,
        }
    except Exception as exc:
        logger.warning(
            "OSRM route unavailable (%s profile=%s)",
            exc,
            profile_key,
        )
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
