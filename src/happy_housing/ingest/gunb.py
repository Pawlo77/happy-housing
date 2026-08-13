"""GUNB building-permit features from public bulk zip + GUGiK WFS."""

from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from happy_housing.ingest.http_client import resilient_requests_get
from happy_housing.pipeline_progress import pipeline_tqdm

logger = logging.getLogger(__name__)

_GUNB_MAZ_ZIP = "https://wyszukiwarka.gunb.gov.pl/pliki_pobranie/wynik_mazowieckie.zip"
"""Official Mazowieckie bulk dump (CSV inside zip; no coordinates)."""

_GUNB_WFS = (
    "https://mapy.geoportal.gov.pl/wss/ext/GlownyUrzadNadzoruBudowlanego/RWDZ-WFS"
)
"""GUGiK WFS exposing RWDZ geometries (parcel polygons)."""

_GUNB_WFS_LAYERS_ALL: tuple[str, ...] = (
    "ms:pozwolenia_2026",
    "ms:pozwolenia_2025",
    "ms:pozwolenia_2024",
    "ms:pozwolenia_2023",
    "ms:pozwolenia_2022",
    "ms:pozwolenia_2021",
    "ms:pozwolenia_2020",
    "ms:pozwolenia_pozostale",
)
"""All year-partitioned WFS type names published by GUGiK."""

_GUNB_WFS_LAYERS_RECENT: tuple[str, ...] = (
    "ms:pozwolenia_2026",
    "ms:pozwolenia_2025",
    "ms:pozwolenia_2024",
)
"""Default WFS layers — recent permits only (GUGiK ignores server-side filters)."""

_GUNB_WFS_LAYERS: tuple[str, ...] = _GUNB_WFS_LAYERS_RECENT
"""Default layers for :func:`fetch_gunb_warsaw_points` (recent years)."""

_WFS_TIMEOUT_S: tuple[float, float] = (30.0, 300.0)
"""Connect / read timeout for GUGiK GetFeature (geoportal is often slow)."""

_WFS_RETRIES: int = 3
"""Attempts per WFS page before abandoning the layer page."""

_WFS_PAGE_SIZE: int = 5000
"""Features per GetFeature page (fewer round-trips on slow geoportal)."""

_WFS_MAX_START_INDEX: int = 2_000_000
"""Runaway guard only — national year layers are ~0.5M features."""

# Lon/lat envelope around Warsaw + near suburbs (permits near listings matter).
_WARSAW_METRO_BBOX: tuple[float, float, float, float] = (20.70, 51.95, 21.40, 52.45)
"""``(lon_min, lat_min, lon_max, lat_max)`` client-side keep window."""

_WARSAW_TERC_PREFIX: str = "1465"
"""TERC prefix for m.st. Warszawa (districts share ``1465…``)."""

_MEMBER_RE = re.compile(
    r"<wfs:member>(.*?)</wfs:member>",
    re.IGNORECASE | re.DOTALL,
)
"""Split a FeatureCollection into member payloads."""

_POS_LIST_RE = re.compile(
    r"<gml:posList[^>]*>([^<]+)</gml:posList>",
    re.IGNORECASE,
)
"""Extract the first GML posList body from a member."""

_TAG_RE = re.compile(
    r"<ms:([A-Za-z0-9_]+)[^>]*>([^<]*)</ms:\1>",
    re.IGNORECASE,
)
"""Pull every ``ms:*`` attribute tag from a WFS member blob."""

_SKIP_TAGS = frozenset({"msgeometry", "boundedby"})
"""Geometry / envelope tags skipped when flattening WFS attributes."""


def resolve_gunb_wfs_layers(*, full_history: bool = False) -> tuple[str, ...]:
    """Return WFS layer list for pipeline fetch.

    GUGiK MapServer ignores attribute/bbox filters, so each layer is paginated
    nationally. ``full_history`` adds older year buckets and ``pozostale`` (2M+
    features) — first fetch can take many hours; recent-only is the default.
    """
    if full_history:
        return _GUNB_WFS_LAYERS_ALL
    return _GUNB_WFS_LAYERS_RECENT


def _wfs_get(
    wfs_url: str,
    params: dict[str, object],
    *,
    timeout_s: float | tuple[float, float] = _WFS_TIMEOUT_S,
    retries: int = _WFS_RETRIES,
) -> requests.Response:
    """GET one WFS page with backoff + free-proxy rotation on blocks."""
    return resilient_requests_get(
        wfs_url,
        params=params,
        timeout=timeout_s,
        retries=max(1, int(retries)),
        label="GUNB WFS",
    )


def _keep_warsaw_metro_row(row: dict[str, object]) -> bool:
    """Whether a WFS feature belongs in the Warsaw metro permit set.

    GUGiK's MapServer build ignores ``CQL_FILTER`` / attribute filters, so the
    national feed must be narrowed client-side via city name, TERC, or bbox.
    """
    city = str(row.get("miasto") or "").strip().casefold()
    if "warszawa" in city:
        return True
    terc = str(row.get("terc") or "").strip()
    if terc.startswith(_WARSAW_TERC_PREFIX):
        return True
    try:
        lon = float(str(row.get("lon")))
        lat = float(str(row.get("lat")))
    except (TypeError, ValueError):
        return False
    lo0, la0, lo1, la1 = _WARSAW_METRO_BBOX
    return lo0 <= lon <= lo1 and la0 <= lat <= la1


def _filter_warsaw_metro(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop national WFS rows outside the Warsaw metro keep rules."""
    if frame.empty:
        return frame
    if "miasto" in frame.columns:
        city = frame["miasto"].astype(str).str.casefold()
    else:
        city = pd.Series("", index=frame.index, dtype=str)
    if "terc" in frame.columns:
        terc = frame["terc"].astype(str)
    else:
        terc = pd.Series("", index=frame.index, dtype=str)
    lon = pd.to_numeric(frame.get("lon"), errors="coerce")
    lat = pd.to_numeric(frame.get("lat"), errors="coerce")
    lo0, la0, lo1, la1 = _WARSAW_METRO_BBOX
    mask = (
        city.str.contains("warszawa", regex=False)
        | terc.str.startswith(_WARSAW_TERC_PREFIX)
        | (lon.between(lo0, lo1) & lat.between(la0, la1))
    )
    return frame.loc[mask].reset_index(drop=True)


def _centroid_lon_lat(pos_list: str) -> tuple[float, float] | None:
    """Average ``lat lon`` pairs from a GML posList into ``(lon, lat)``."""
    parts = pos_list.split()
    if len(parts) < 4 or len(parts) % 2 != 0:
        return None
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    lats = nums[0::2]
    lons = nums[1::2]
    return (sum(lons) / len(lons), sum(lats) / len(lats))


def _frame_from_wfs_gml(xml_text: str) -> pd.DataFrame:
    """Parse a WFS FeatureCollection GML payload into lon/lat point rows."""
    rows: list[dict[str, object]] = []
    for member in _MEMBER_RE.findall(xml_text):
        pos_match = _POS_LIST_RE.search(member)
        if pos_match is None:
            continue
        centroid = _centroid_lon_lat(pos_match.group(1).strip())
        if centroid is None:
            continue
        lon, lat = centroid
        props = {
            m.group(1).lower(): m.group(2).strip()
            for m in _TAG_RE.finditer(member)
            if m.group(1).lower() not in _SKIP_TAGS
        }
        street = props.get("ulica", "")
        street_more = props.get("ulica_dalej", "")
        number = props.get("nr_domu", "")
        city = props.get("miasto", "")
        adres = ", ".join(p for p in (street, street_more, number, city) if p)
        row: dict[str, object] = {
            "lon": lon,
            "lat": lat,
            "adres": adres,
            "numer_decyzji": (
                props.get("numer_ewidencyjny_system")
                or props.get("numer_ewidencyjny_urzad")
                or ""
            ),
            "miasto": city,
        }
        for key, value in props.items():
            if key in {"ulica", "ulica_dalej", "nr_domu", "miasto"}:
                continue
            row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _fetch_wfs_city_layer(
    layer: str,
    *,
    city: str = "Warszawa",
    page_size: int = _WFS_PAGE_SIZE,
    wfs_url: str = _GUNB_WFS,
    timeout_s: float | tuple[float, float] = _WFS_TIMEOUT_S,
    retries: int = _WFS_RETRIES,
    max_start_index: int = _WFS_MAX_START_INDEX,
) -> pd.DataFrame:
    """Paginate one GUGiK year layer; keep Warsaw-metro rows client-side.

    ``city`` is retained for API compatibility; the remote CQL filter is a no-op
    on this MapServer, so filtering uses :func:`_keep_warsaw_metro_row`.
    """
    del city  # Server ignores attribute filters; client filter is metro-wide.
    frames: list[pd.DataFrame] = []
    start = 0
    matched: int | None = None
    kept = 0
    page_bar = pipeline_tqdm(
        total=None,
        desc=layer.replace("ms:", ""),
        unit="feat",
        leave=False,
    )
    try:
        while True:
            params: dict[str, object] = {
                "service": "WFS",
                "version": "2.0.0",
                "request": "GetFeature",
                "typeNames": layer,
                "srsName": "EPSG:4326",
                "count": page_size,
                "startIndex": start,
            }
            response = _wfs_get(
                wfs_url,
                params,
                timeout_s=timeout_s,
                retries=retries,
            )
            if matched is None:
                match = re.search(r'numberMatched="([^"]+)"', response.text)
                if match and match.group(1).isdigit():
                    matched = int(match.group(1))
                    page_bar.total = matched
                    page_bar.refresh()
                    logger.info(
                        "GUNB WFS %s numberMatched=%s (national; filtering locally)",
                        layer,
                        matched,
                    )
            chunk = _frame_from_wfs_gml(response.text)
            if chunk.empty:
                break
            page_bar.update(len(chunk))
            local = _filter_warsaw_metro(chunk)
            if not local.empty:
                frames.append(local)
                kept += len(local)
            if len(chunk) < page_size:
                break
            start += page_size
            if matched is not None and start >= matched:
                break
            if start > max_start_index:
                logger.warning(
                    "GUNB WFS pagination stopped at startIndex=%s (runaway guard)",
                    start,
                )
                break
    finally:
        page_bar.close()
    if not frames:
        return pd.DataFrame(columns=["lon", "lat", "adres", "numer_decyzji", "miasto"])
    logger.info("GUNB WFS %s kept %s Warsaw-metro rows", layer, kept)
    return pd.concat(frames, ignore_index=True)


def fetch_gunb_warsaw_points(
    *,
    wfs_url: str = _GUNB_WFS,
    layers: tuple[str, ...] | None = None,
    full_history: bool = False,
    city: str = "Warszawa",
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download geolocated GUNB permits for Warsaw metro from GUGiK WFS."""
    layer_list = (
        layers
        if layers is not None
        else resolve_gunb_wfs_layers(full_history=full_history)
    )
    mode = "full history" if full_history else "recent years"
    logger.info(
        "GUNB WFS fetch (%s): %s layer(s)",
        mode,
        len(layer_list),
    )
    frames: list[pd.DataFrame] = []
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for layer in pipeline_tqdm(layer_list, desc="GUNB WFS layers", unit="layer"):
        cache_path = None
        if cache_dir is not None:
            safe = layer.replace(":", "_").replace("/", "_")
            cache_path = cache_dir / f"gunb_wfs_{safe}.parquet"
        if (
            use_cache
            and cache_path is not None
            and cache_path.exists()
            and cache_path.stat().st_size > 0
        ):
            part = pd.read_parquet(cache_path)
            logger.info(
                "GUNB WFS %s → %s rows (cache %s)",
                layer,
                len(part),
                cache_path.name,
            )
        else:
            try:
                part = _fetch_wfs_city_layer(layer, city=city, wfs_url=wfs_url)
            except Exception as exc:
                logger.warning("GUNB WFS layer %s failed: %s", layer, exc)
                continue
            logger.info("GUNB WFS %s → %s rows", layer, len(part))
            if cache_path is not None:
                part.to_parquet(cache_path, index=False)
        if not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["lon", "lat", "adres", "numer_decyzji", "miasto"])
    frame = pd.concat(frames, ignore_index=True)
    if "numer_decyzji" in frame.columns:
        frame = frame.drop_duplicates(subset=["numer_decyzji", "lon", "lat"])
    else:
        frame = frame.drop_duplicates(subset=["lon", "lat", "adres"])
    return frame.reset_index(drop=True)


def download_gunb_bulk_zip(
    raw_dir: Path,
    url: str = _GUNB_MAZ_ZIP,
) -> Path:
    """Download the official Mazowieckie GUNB zip into ``raw_dir``."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "gunb_mazowieckie.zip"
    if zip_path.exists() and zip_path.stat().st_size > 1_000_000:
        logger.info("Using cached GUNB zip %s", zip_path)
        return zip_path
    logger.info("Downloading GUNB bulk zip %s", url)
    response = resilient_requests_get(
        url,
        timeout=600,
        retries=5,
        label="GUNB bulk zip",
    )
    zip_path.write_bytes(response.content)
    logger.info("Wrote GUNB zip (%s bytes) → %s", len(response.content), zip_path)
    return zip_path


def _frame_from_bulk_zip(zip_path: Path) -> pd.DataFrame:
    """Load the CSV member from a GUNB voivodeship zip (no coordinates)."""
    with zipfile.ZipFile(zip_path) as archive:
        members = [m for m in archive.namelist() if m.lower().endswith(".csv")]
        if not members:
            return pd.DataFrame()
        with archive.open(members[0]) as handle:
            return pd.read_csv(handle, sep=";", dtype=str, low_memory=False)


def download_gunb_mazowieckie(
    raw_dir: Path,
    output_parquet: Path,
    url: str = _GUNB_MAZ_ZIP,
    fixture_path: Path | None = None,
    wfs_url: str = _GUNB_WFS,
    *,
    refresh_wfs_cache: bool = False,
    wfs_full_history: bool = False,
) -> pd.DataFrame:
    """Download full Mazowieckie GUNB dump + Warsaw WFS points for GIS.

    Prefers GUGiK WFS centroids for ``lon``/``lat`` (map + distance features).
    Always attempts the official bulk zip for raw archival under ``raw_dir``.
    Falls back to ``fixture_path`` when both live sources fail.

    By default only recent WFS year layers are fetched (see
    :func:`resolve_gunb_wfs_layers`); set ``wfs_full_history=True`` for all
    buckets including ``pozostale`` (very slow first run).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        zip_path = download_gunb_bulk_zip(raw_dir, url=url)
        bulk = _frame_from_bulk_zip(zip_path)
        if not bulk.empty:
            bulk_out = raw_dir / "gunb_mazowieckie_bulk.parquet"
            bulk.to_parquet(bulk_out, index=False)
            logger.info("Wrote %s GUNB bulk attribute rows → %s", len(bulk), bulk_out)
    except Exception as exc:
        logger.warning("GUNB bulk zip failed: %s", exc)

    frame = pd.DataFrame(columns=["lon", "lat", "adres", "numer_decyzji", "miasto"])
    try:
        frame = fetch_gunb_warsaw_points(
            wfs_url=wfs_url,
            full_history=wfs_full_history,
            cache_dir=raw_dir,
            use_cache=not refresh_wfs_cache,
        )
        logger.info("Loaded %s GUNB Warsaw-metro points from WFS", len(frame))
    except Exception as exc:
        logger.warning("GUNB WFS fetch failed: %s", exc)

    if frame.empty and fixture_path is not None and fixture_path.exists():
        logger.warning("Using GUNB fixture %s", fixture_path)
        frame = pd.read_json(fixture_path)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_parquet, index=False)
    logger.info("Wrote %s GUNB rows to %s", len(frame), output_parquet)
    return frame


def gunb_points_gdf(frame: pd.DataFrame) -> gpd.GeoDataFrame:
    """Build a point GeoDataFrame from GUNB rows when lon/lat columns exist."""
    lon_col = next(
        (c for c in frame.columns if c.lower() in {"lon", "longitude", "x", "dlugosc"}),
        None,
    )
    lat_col = next(
        (
            c
            for c in frame.columns
            if c.lower() in {"lat", "latitude", "y", "szerokosc"}
        ),
        None,
    )
    if lon_col is None or lat_col is None or frame.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    pts = frame.dropna(subset=[lon_col, lat_col]).copy()
    return gpd.GeoDataFrame(
        pts,
        geometry=gpd.points_from_xy(pts[lon_col], pts[lat_col]),
        crs="EPSG:4326",
    )


def _frame_from_wfs_geojson(payload: dict[str, object]) -> pd.DataFrame:
    """Flatten a GeoJSON FeatureCollection into lon/lat rows."""
    features = payload.get("features")
    if not isinstance(features, list):
        return pd.DataFrame(columns=["lon", "lat", "adres", "numer_decyzji"])
    rows: list[dict[str, object]] = []
    for feat in features:
        if not isinstance(feat, dict):
            continue
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") if isinstance(geom, dict) else None
        props_raw = feat.get("properties")
        props = props_raw if isinstance(props_raw, dict) else {}
        if not (isinstance(coords, list) and len(coords) >= 2):
            continue
        if geom.get("type") == "Point":
            lon, lat = float(coords[0]), float(coords[1])
        elif (
            geom.get("type") == "Polygon" and isinstance(coords[0], list) and coords[0]
        ):
            ring = coords[0]
            lons = [float(p[0]) for p in ring if isinstance(p, list) and len(p) >= 2]
            lats = [float(p[1]) for p in ring if isinstance(p, list) and len(p) >= 2]
            if not lons:
                continue
            lon, lat = sum(lons) / len(lons), sum(lats) / len(lats)
        else:
            continue
        rows.append({
            "lon": lon,
            "lat": lat,
            "adres": props.get("adres") or props.get("address") or "",
            "numer_decyzji": props.get("numer_decyzji") or props.get("id") or "",
        })
    return pd.DataFrame(rows)
