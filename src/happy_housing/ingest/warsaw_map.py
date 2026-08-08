"""UM Warszawa map portal WFS client (testmapa / mapa.um.warszawa.pl).

Parks & squares live in ``PLACE_SKWERY``; education POIs / catchments come from
WFS layers matching school / oświata keywords (see ``SCHOOL_TYPE_NAME_CANDIDATES``).
Strategic noise maps are WMS-only today — aircraft Lden still comes from EEA END
fixtures elsewhere.
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import geopandas as gpd
import requests

logger = logging.getLogger(__name__)

WARSAW_WFS_URL: str = "https://wfs.um.warszawa.pl/serwis"
"""City WFS endpoint behind the public map portal."""

PLACE_SKWERY_TYPE: str = "ns92528565:PLACE_SKWERY"
"""Official parks / squares / skwers layer (PLACE_SKWERY)."""

SCHOOL_TYPE_NAME_CANDIDATES: tuple[str, ...] = (
    "ns92528565:OBWODY_SZKOL_PODSTAWOWYCH",
    "ns92528565:SZKOLY",
    "ns92528565:OSWIATA",
    "ns92528565:PRZEDSZKOLA",
    "ns92528565:PLACOWKI_OSWIATOWE",
    "wfs:SZKOLY",
    "wfs:OSWIATA",
    "wfs:PRZEDSZKOLA",
)
"""Candidate education typeNames (live GetCapabilities + common UM guesses)."""

_EDUCATION_NAME_RE: re.Pattern[str] = re.compile(
    r"SZKOL|OSWIAT|PRZEDSZKOL|PLACOWK.*OSW|EDUK",
    re.IGNORECASE,
)
"""Substring filter for education-related WFS typeNames."""

_DEFAULT_TIMEOUT_S: float = 180.0
"""WFS GetFeature timeout for city-wide layer download."""

WARSAW_LAYER_REGISTRY: dict[str, dict[str, str]] = {
    "parks": {
        "type_name": PLACE_SKWERY_TYPE,
        "cache_name": "warsaw_parks.geojson",
    },
    "schools": {
        "cache_name": "warsaw_schools.geojson",
    },
}
"""Named layers → default cache file (and parks typeName)."""


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


def _ensure_name_column(frame: gpd.GeoDataFrame, default: str) -> gpd.GeoDataFrame:
    """Copy a Polish name attribute into ``name`` when missing."""
    out = frame
    if "name" in out.columns:
        return out
    for col in ("NAZWA_PODST", "NAZWA", "NAZWA_SKRO", "nazwa"):
        if col in out.columns:
            out = out.copy()
            out["name"] = out[col].astype(str)
            return out
    out = out.copy()
    out["name"] = default
    return out


def _load_fixture_geojson(path: Path, *, default_name: str) -> gpd.GeoDataFrame:
    """Read offline GeoJSON into EPSG:4326 with a ``name`` column."""
    frame = gpd.read_file(path)
    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")
    elif str(frame.crs) != "EPSG:4326":
        frame = frame.to_crs("EPSG:4326")
    return _ensure_name_column(frame, default_name)


def discover_education_type_names(
    *,
    wfs_url: str = WARSAW_WFS_URL,
    timeout_s: float = 60.0,
) -> tuple[str, ...]:
    """Parse WFS GetCapabilities for typeNames matching education keywords."""
    try:
        response = requests.get(
            wfs_url,
            params={
                "service": "WFS",
                "version": "1.1.0",
                "request": "GetCapabilities",
            },
            timeout=timeout_s,
            headers={"User-Agent": "happy-housing/1.0"},
        )
        response.raise_for_status()
        text = response.text
    except Exception as exc:
        logger.warning("WFS GetCapabilities failed: %s", exc)
        return ()
    names = re.findall(r"<wfs:Name>([^<]+)</wfs:Name>", text)
    matched = tuple(n for n in names if _EDUCATION_NAME_RE.search(n))
    if matched:
        logger.info("Discovered education WFS typeNames: %s", matched)
    return matched


def fetch_warsaw_wfs_layer(
    cache_dir: Path,
    *,
    type_name: str,
    cache_name: str,
    fixture_path: Path | None = None,
    force_refresh: bool = False,
    wfs_url: str = WARSAW_WFS_URL,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    default_name: str = "feature",
) -> gpd.GeoDataFrame:
    """Download one WFS typeName; cache as GeoJSON in EPSG:4326.

    Args:
        cache_dir: Directory for the cache GeoJSON.
        type_name: WFS typeName.
        cache_name: Filename under ``cache_dir``.
        fixture_path: Optional offline GeoJSON fallback when WFS fails.
        force_refresh: Ignore existing cache.
        wfs_url: WFS endpoint.
        timeout_s: HTTP timeout.
        default_name: Fallback ``name`` value when attributes lack a label.

    Returns:
        Features in EPSG:4326 (possibly empty).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_name
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
        frame = _ensure_name_column(frame, default_name)
        frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty]
        logger.info("Fetched %s features from WFS typeName=%s", len(frame), type_name)
    except Exception as exc:
        logger.warning("Warsaw WFS typeName=%s failed: %s", type_name, exc)

    if frame.empty and fixture_path is not None and fixture_path.exists():
        logger.warning("Using WFS fixture %s for %s", fixture_path, cache_name)
        frame = _load_fixture_geojson(fixture_path, default_name=default_name)

    if not frame.empty:
        frame.to_file(cache_path, driver="GeoJSON")
        logger.info("Cached %s features → %s", len(frame), cache_path)
    return frame


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
    meta = WARSAW_LAYER_REGISTRY["parks"]
    return fetch_warsaw_wfs_layer(
        cache_dir,
        type_name=type_name,
        cache_name=meta["cache_name"],
        fixture_path=fixture_path,
        force_refresh=force_refresh,
        wfs_url=wfs_url,
        timeout_s=timeout_s,
        default_name="park",
    )


def fetch_warsaw_schools(
    cache_dir: Path,
    *,
    fixture_path: Path | None = None,
    force_refresh: bool = False,
    wfs_url: str = WARSAW_WFS_URL,
    type_names: tuple[str, ...] | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> gpd.GeoDataFrame:
    """Fetch education POIs / school catchments; cache as ``warsaw_schools.geojson``.

    Tries ``type_names`` (or discovered + candidate names) until one returns
    features. Falls back to ``fixture_path`` when every WFS attempt fails.

    Args:
        cache_dir: Directory for ``warsaw_schools.geojson``.
        fixture_path: Offline GeoJSON fallback (e.g. ``data/fixtures/schools.geojson``).
        force_refresh: Ignore existing cache.
        wfs_url: WFS endpoint.
        type_names: Explicit typeName trial order; default = discover + candidates.
        timeout_s: HTTP timeout per GetFeature.

    Returns:
        School geometries in EPSG:4326 (empty if WFS and fixture both unavailable).
    """
    cache_name = WARSAW_LAYER_REGISTRY["schools"]["cache_name"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_name
    if not force_refresh and cache_path.exists():
        return gpd.read_file(cache_path)

    if type_names is None:
        discovered = discover_education_type_names(wfs_url=wfs_url, timeout_s=timeout_s)
        ordered: list[str] = []
        for name in (*discovered, *SCHOOL_TYPE_NAME_CANDIDATES):
            if name not in ordered:
                ordered.append(name)
        type_names = tuple(ordered)

    for type_name in type_names:
        frame = fetch_warsaw_wfs_layer(
            cache_dir,
            type_name=type_name,
            cache_name=cache_name,
            fixture_path=None,
            force_refresh=True,
            wfs_url=wfs_url,
            timeout_s=timeout_s,
            default_name="school",
        )
        if not frame.empty:
            logger.info(
                "Schools loaded from typeName=%s (%s rows)", type_name, len(frame)
            )
            return frame

    if fixture_path is not None and fixture_path.exists():
        logger.warning("Using schools fixture %s", fixture_path)
        frame = _load_fixture_geojson(fixture_path, default_name="school")
        if not frame.empty:
            frame.to_file(cache_path, driver="GeoJSON")
            logger.info("Cached %s schools → %s", len(frame), cache_path)
        return frame

    return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
