"""GUNB building-permit proximity features from public CSV dumps."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Public bulk dumps (voivodeship files) — URL pattern from GUNB wyszukiwarka.
_GUNB_MAZ_CSV = "https://wyszukiwarka.gunb.gov.pl/pliki/Decyzje_mazowieckie.csv"
"""Best-effort Mazowieckie decisions CSV URL (may 404 if renamed)."""

_GUNB_WFS_GEOJSON = (
    "https://wyszukiwarka.gunb.gov.pl/geoserver/wfs"
    "?service=WFS&version=2.0.0&request=GetFeature"
    "&typeNames=gunb:decyzje&outputFormat=application/json"
    "&count=500"
)
"""Best-effort GUNB WFS GeoJSON (may 404; CSV/fixture remain fallbacks)."""


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
        rows.append({
            "lon": float(coords[0]),
            "lat": float(coords[1]),
            "adres": props.get("adres") or props.get("address") or "",
            "numer_decyzji": props.get("numer_decyzji") or props.get("id") or "",
        })
    return pd.DataFrame(rows)


def download_gunb_mazowieckie(
    raw_dir: Path,
    output_parquet: Path,
    url: str = _GUNB_MAZ_CSV,
    fixture_path: Path | None = None,
    wfs_url: str = _GUNB_WFS_GEOJSON,
) -> pd.DataFrame:
    """Download GUNB Mazowieckie decisions via WFS, then CSV, then fixture.

    When the CDN path 404s, fall back to an optional fixture JSON so the
    pipeline stays online for local demos.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "gunb_mazowieckie.csv"
    frame = pd.DataFrame(columns=["lon", "lat", "adres", "numer_decyzji"])

    try:
        wfs_resp = requests.get(wfs_url, timeout=60)
        if wfs_resp.status_code == 200:
            payload = wfs_resp.json()
            if isinstance(payload, dict):
                candidate = _frame_from_wfs_geojson(payload)
                if not candidate.empty:
                    frame = candidate
                    logger.info("Loaded %s GUNB rows from WFS", len(frame))
    except Exception as exc:
        logger.warning("GUNB WFS fetch failed: %s", exc)

    if frame.empty:
        try:
            response = requests.get(url, timeout=120)
            if response.status_code == 404:
                logger.warning("GUNB CSV not found at %s", url)
            else:
                response.raise_for_status()
                csv_path.write_bytes(response.content)
                frame = pd.read_csv(
                    csv_path, sep=";", encoding="utf-8", low_memory=False
                )
        except Exception as exc:
            logger.warning("GUNB download/parse failed: %s", exc)
            if csv_path.exists():
                try:
                    frame = pd.read_csv(
                        csv_path, sep=";", encoding="cp1250", low_memory=False
                    )
                except Exception:
                    frame = pd.DataFrame(
                        columns=["lon", "lat", "adres", "numer_decyzji"]
                    )

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
