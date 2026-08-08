"""UM Warszawa map portal WFS client (testmapa / mapa.um.warszawa.pl).

Official parks & squares live in ``PLACE_SKWERY`` (same layers shown on
https://testmapa.um.warszawa.pl). Strategic noise maps are WMS-only today —
aircraft Lden still comes from EEA END fixtures.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import geopandas as gpd
import requests

logger = logging.getLogger(__name__)

WARSAW_WFS_URL: str = "https://wfs.um.warszawa.pl/serwis"
"""City WFS endpoint behind the public map portal."""

PLACE_SKWERY_TYPE: str = "ns92528565:PLACE_SKWERY"
"""Official parks / squares / skwers layer (PLACE_SKWERY)."""

_DEFAULT_TIMEOUT_S: float = 180.0
"""WFS GetFeature timeout for city-wide parks download."""


def normalize_warsaw_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Assign EPSG:2178 when coordinates look projected and CRS is missing."""
    if gdf.empty:
        return gdf
    out = gdf.copy()
    if out.crs is not None:
        return out
    xs = out.geometry.centroid.x
    if float(xs.abs().median()) > 1000.0:
        return out.set_crs("EPSG:2178")
    return out.set_crs("EPSG:4326")


def fetch_warsaw_parks(
    cache_dir: Path,
    *,
    fixture_path: Path | None = None,
    force_refresh: bool = False,
    wfs_url: str = WARSAW_WFS_URL,
    type_name: str = PLACE_SKWERY_TYPE,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> gpd.GeoDataFrame:
    """Download PLACE_SKWERY polygons via WFS; cache as GeoJSON in EPSG:4326.

    Args:
        cache_dir: Directory for ``warsaw_parks.geojson``.
        fixture_path: Optional offline GeoJSON/JSON fallback.
        force_refresh: Ignore existing cache.
        wfs_url: WFS endpoint.
        type_name: WFS typeName.
        timeout_s: HTTP timeout.

    Returns:
        Park polygons in EPSG:4326 with a ``name`` column when available.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "warsaw_parks.geojson"
    if not force_refresh and cache_path.exists():
        return gpd.read_file(cache_path)

    frame = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    try:
        response = requests.get(
            wfs_url,
            params={
                "service": "WFS",
                "version": "1.1.0",
                "request": "GetFeature",
                "typeName": type_name,
                "maxFeatures": "10000",
            },
            timeout=timeout_s,
            headers={"User-Agent": "happy-housing/1.0"},
        )
        response.raise_for_status()
        loaded = gpd.read_file(io.BytesIO(response.content))
        loaded = normalize_warsaw_crs(loaded)
        if loaded.crs is not None and str(loaded.crs) != "EPSG:4326":
            loaded = loaded.to_crs("EPSG:4326")
        keep = [c for c in loaded.columns if c != "geometry"]
        frame = gpd.GeoDataFrame(
            loaded[[*keep, "geometry"]],
            geometry="geometry",
            crs="EPSG:4326",
        )
        if "NAZWA_PODST" in frame.columns:
            frame["name"] = frame["NAZWA_PODST"].astype(str)
        elif "NAZWA" in frame.columns:
            frame["name"] = frame["NAZWA"].astype(str)
        elif "name" not in frame.columns:
            frame["name"] = "park"
        frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty]
        logger.info("Fetched %s Warsaw parks/squares from WFS", len(frame))
    except Exception as exc:
        logger.warning("Warsaw parks WFS failed: %s", exc)

    if frame.empty and fixture_path is not None and fixture_path.exists():
        logger.warning("Using parks fixture %s", fixture_path)
        frame = gpd.read_file(fixture_path)
        if frame.crs is None:
            frame = frame.set_crs("EPSG:4326")
        elif str(frame.crs) != "EPSG:4326":
            frame = frame.to_crs("EPSG:4326")
        if "name" not in frame.columns:
            frame["name"] = "park"

    if not frame.empty:
        frame.to_file(cache_path, driver="GeoJSON")
        logger.info("Cached %s parks → %s", len(frame), cache_path)
    return frame
