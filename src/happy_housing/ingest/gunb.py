"""GUNB building-permit features from public bulk zip + GUGiK WFS."""

from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_GUNB_MAZ_ZIP = "https://wyszukiwarka.gunb.gov.pl/pliki_pobranie/wynik_mazowieckie.zip"
"""Official Mazowieckie bulk dump (CSV inside zip; no coordinates)."""

_GUNB_WFS = (
    "https://mapy.geoportal.gov.pl/wss/ext/GlownyUrzadNadzoruBudowlanego/RWDZ-WFS"
)
"""GUGiK WFS exposing RWDZ geometries (parcel polygons)."""

_GUNB_WFS_LAYERS: tuple[str, ...] = (
    "ms:pozwolenia_2026",
    "ms:pozwolenia_2025",
    "ms:pozwolenia_2024",
    "ms:pozwolenia_2023",
    "ms:pozwolenia_2022",
    "ms:pozwolenia_2021",
    "ms:pozwolenia_2020",
    "ms:pozwolenia_pozostale",
)
"""Year-partitioned WFS type names published by GUGiK."""

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
    page_size: int = 500,
    wfs_url: str = _GUNB_WFS,
) -> pd.DataFrame:
    """Paginate one GUGiK layer filtered to a city via CQL."""
    frames: list[pd.DataFrame] = []
    start = 0
    while True:
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": layer,
            "srsName": "EPSG:4326",
            "count": page_size,
            "startIndex": start,
            "CQL_FILTER": f"miasto='{city}'",
        }
        response = requests.get(wfs_url, params=params, timeout=180)
        response.raise_for_status()
        chunk = _frame_from_wfs_gml(response.text)
        if chunk.empty:
            break
        frames.append(chunk)
        if len(chunk) < page_size:
            break
        start += page_size
        if start > 200_000:
            logger.warning("GUNB WFS pagination stopped at startIndex=%s", start)
            break
    if not frames:
        return pd.DataFrame(columns=["lon", "lat", "adres", "numer_decyzji", "miasto"])
    return pd.concat(frames, ignore_index=True)


def fetch_gunb_warsaw_points(
    *,
    wfs_url: str = _GUNB_WFS,
    layers: tuple[str, ...] = _GUNB_WFS_LAYERS,
    city: str = "Warszawa",
) -> pd.DataFrame:
    """Download geolocated GUNB permits for one city from GUGiK WFS."""
    frames: list[pd.DataFrame] = []
    for layer in layers:
        try:
            part = _fetch_wfs_city_layer(layer, city=city, wfs_url=wfs_url)
        except Exception as exc:
            logger.warning("GUNB WFS layer %s failed: %s", layer, exc)
            continue
        logger.info("GUNB WFS %s → %s rows", layer, len(part))
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
    response = requests.get(url, timeout=600)
    response.raise_for_status()
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
) -> pd.DataFrame:
    """Download full Mazowieckie GUNB dump + Warsaw WFS points for GIS.

    Prefers GUGiK WFS centroids for ``lon``/``lat`` (map + distance features).
    Always attempts the official bulk zip for raw archival under ``raw_dir``.
    Falls back to ``fixture_path`` when both live sources fail.
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
        frame = fetch_gunb_warsaw_points(wfs_url=wfs_url)
        logger.info("Loaded %s GUNB Warsaw points from WFS", len(frame))
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
