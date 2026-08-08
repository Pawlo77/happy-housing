"""UM Warszawa map portal WFS + MapViewer client (mapa.um.warszawa.pl).

Parks & squares live in ``PLACE_SKWERY``; education POIs / catchments come from
WFS layers matching school / oświata keywords (see ``SCHOOL_TYPE_NAME_CANDIDATES``).
Sports objects (gyms, pools, …) come from Oracle MapViewer dataserver used by
``/obiekty-sportowe/``. Other civic POIs (libraries, pharmacies, …) are WFS
typeNames. Strategic noise maps are WMS-only — aircraft Lden still comes from
EEA END fixtures elsewhere.
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import shape

from happy_housing.gis.amenity_catalog import sport_key_for_osrid

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
    "city_pois": {
        "cache_name": "warsaw_city_pois.geojson",
    },
}
"""Named layers → default cache file (and parks typeName)."""

WARSAW_MAPVIEWER_DATASERVER: str = (
    "https://testmapa.um.warszawa.pl/mapviewer/dataserver"
)
"""Oracle MapViewer GeoJSON dataserver (sports portal backend)."""

SPORTS_SQL: str = (
    "select ws.ID, ws.SHAPE, ws.DZIELNICA, ws.OSRID, ws.OSRNAZWASKROCONA "
    "from OBIEKTY_SPORTOWE_MAPA_V ws "
    "WHERE ws.ROK_WYLACZENIA IS NULL AND ws.OSRID not in (32,31,66)"
)
"""JDBC theme SQL matching the public sports map portal."""

ICE_RINKS_SQL: str = "select * from LODOWISKA_MAPA_V"
"""Seasonal ice rinks theme (MapViewer ``edytor`` datasource)."""

# Civic WFS layers → amenity catalog keys (point-ish POIs).
WARSAW_CIVIC_WFS_LAYERS: tuple[tuple[str, str, str], ...] = (
    ("library", "ns92528565:BIBLIOTEKI", "library"),
    ("pharmacy", "wfs:APTEKI", "pharmacy"),
    ("bike_share", "wfs:STACJE_ROWEROWE", "bike station"),
    ("park_and_ride", "wfs:PARKINGI_PARK_AND_RIDE", "P+R"),
    ("hotel", "wfs:HOTELE", "hotel"),
    ("theater", "wfs:TEATRY", "theatre"),
    ("police_um", "wfs:POLICJA", "police"),
    ("cemetery", "ns92528565:CMENTARZE", "cemetery"),
    ("dormitory", "wfs:AKADEMIKI", "dormitory"),
    ("metro_entrance", "wfs:METRO_WEJSCIA", "metro entrance"),
    ("senior_club", "ns92528565:KLUBY_SENIORA", "senior club"),
)
"""``(object_type, WFS typeName, default name)`` civic layers."""


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


def _geojson_features_to_gdf(payload: dict[str, object]) -> gpd.GeoDataFrame:
    """Parse MapViewer FeatureCollection JSON into EPSG:4326 GeoDataFrame."""
    features = payload.get("features") or []
    if not isinstance(features, list) or not features:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    rows: list[dict[str, object]] = []
    geoms = []
    for feat in features:
        if not isinstance(feat, dict):
            continue
        raw_props = feat.get("properties") or {}
        props: dict[str, object] = (
            dict(raw_props) if isinstance(raw_props, dict) else {}
        )
        fid = feat.get("_id")
        if fid is not None and "ID" not in props:
            props["ID"] = fid
        geom = feat.get("geometry")
        if not geom:
            continue
        try:
            parsed = shape(geom)
        except (TypeError, ValueError) as exc:
            logger.debug("Skip MapViewer feature geometry: %s", exc)
            continue
        geoms.append(parsed)
        rows.append(props)
    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    frame = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
    return frame[frame.geometry.notna() & ~frame.geometry.is_empty]


def fetch_mapviewer_jdbc_theme(
    *,
    datasource: str,
    theme: str,
    sql: str,
    id_col: str = "ID",
    geom_col: str = "SHAPE",
    jdbc_srid: int = 2178,
    to_srid: int = 4326,
    dataserver_base: str = WARSAW_MAPVIEWER_DATASERVER,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> gpd.GeoDataFrame:
    """Fetch a MapViewer JDBC theme as GeoJSON features (EPSG:4326)."""
    url = f"{dataserver_base.rstrip('/')}/{datasource}"
    try:
        response = requests.get(
            url,
            params={
                "t": theme,
                "sql": sql,
                "id_col": id_col,
                "geom_col": geom_col,
                "jdbc_srid": str(jdbc_srid),
                "to_srid": str(to_srid),
                "include_label_box": "true",
            },
            timeout=timeout_s,
            headers={"User-Agent": "happy-housing/1.0"},
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("mds_error"):
            logger.warning(
                "MapViewer dataserver error theme=%s: %s",
                theme,
                payload.get("mds_error"),
            )
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        frame = _geojson_features_to_gdf(payload)
        logger.info("MapViewer theme=%s returned %s features", theme, len(frame))
        return frame
    except Exception as exc:
        logger.warning("MapViewer theme=%s failed: %s", theme, exc)
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def _normalize_sports_frame(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Attach ``object_type`` / ``name`` from sports OSRID attributes."""
    if frame.empty:
        return gpd.GeoDataFrame(
            columns=["object_type", "name", "source", "geometry"],
            geometry=[],
            crs="EPSG:4326",
        )
    out = frame.copy()
    if "OSRID" in out.columns:
        out["object_type"] = out["OSRID"].map(sport_key_for_osrid)
    else:
        out["object_type"] = None
    name_series = None
    for candidate in ("OSRNAZWASKROCONA", "NAZWA_SERWIS", "LABEL", "name"):
        if candidate in out.columns:
            name_series = out[candidate].astype(str)
            break
    if name_series is None:
        name_series = (
            out["object_type"]
            .fillna("sport")
            .astype(str)
            .str.replace("_", " ", regex=False)
        )
    out["name"] = name_series
    out["source"] = "sports"
    out = out[out["object_type"].notna()]
    keep = ["object_type", "name", "source", "geometry"]
    return gpd.GeoDataFrame(out[keep], geometry="geometry", crs="EPSG:4326")


def fetch_warsaw_sports_objects(
    *,
    force_refresh: bool = False,
    dataserver_base: str = WARSAW_MAPVIEWER_DATASERVER,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    fixture_path: Path | None = None,
) -> gpd.GeoDataFrame:
    """Download sports + ice-rink POIs from the city sports map backend.

    Args:
        force_refresh: Unused here (caller caches combined city POIs).
        dataserver_base: MapViewer dataserver root.
        timeout_s: HTTP timeout.
        fixture_path: Offline GeoJSON fallback.

    Returns:
        Points with ``object_type``, ``name``, ``source`` in EPSG:4326.
    """
    del force_refresh  # combined cache owns refresh; keep API symmetric
    sports = fetch_mapviewer_jdbc_theme(
        datasource="dane_wawa",
        theme="obiekty-sportowe",
        sql=SPORTS_SQL,
        id_col="ID",
        geom_col="SHAPE",
        dataserver_base=dataserver_base,
        timeout_s=timeout_s,
    )
    ice = fetch_mapviewer_jdbc_theme(
        datasource="edytor",
        theme="lodowiska",
        sql=ICE_RINKS_SQL,
        id_col="OBJECTID",
        geom_col="SHAPE",
        dataserver_base=dataserver_base,
        timeout_s=timeout_s,
    )
    parts = [_normalize_sports_frame(sports), _normalize_sports_frame(ice)]
    parts = [p for p in parts if not p.empty]
    if parts:
        return gpd.GeoDataFrame(
            pd.concat(parts, ignore_index=True),
            geometry="geometry",
            crs="EPSG:4326",
        )
    if fixture_path is not None and fixture_path.exists():
        logger.warning("Using sports fixture %s", fixture_path)
        loaded = gpd.read_file(fixture_path)
        if loaded.crs is None:
            loaded = loaded.set_crs("EPSG:4326")
        elif str(loaded.crs) != "EPSG:4326":
            loaded = loaded.to_crs("EPSG:4326")
        if "object_type" not in loaded.columns:
            loaded = _normalize_sports_frame(loaded)
        return loaded
    return gpd.GeoDataFrame(
        columns=["object_type", "name", "source", "geometry"],
        geometry=[],
        crs="EPSG:4326",
    )


def _civic_layer_to_pois(
    frame: gpd.GeoDataFrame,
    *,
    object_type: str,
    default_name: str,
) -> gpd.GeoDataFrame:
    """Normalize one WFS civic layer to city-POI schema (centroids)."""
    if frame.empty:
        return gpd.GeoDataFrame(
            columns=["object_type", "name", "source", "geometry"],
            geometry=[],
            crs="EPSG:4326",
        )
    out = _ensure_name_column(frame, default_name)
    if out.crs is None:
        out = out.set_crs("EPSG:4326")
    elif str(out.crs) != "EPSG:4326":
        out = out.to_crs("EPSG:4326")
    # Centroids for polygons
    geom_types = set(out.geometry.geom_type.unique())
    if not geom_types <= {"Point", "MultiPoint"}:
        projected = out.to_crs("EPSG:2180")
        out = out.copy()
        out["geometry"] = projected.geometry.centroid.to_crs("EPSG:4326")
    out["object_type"] = object_type
    out["source"] = "wfs"
    return gpd.GeoDataFrame(
        out[["object_type", "name", "source", "geometry"]],
        geometry="geometry",
        crs="EPSG:4326",
    )


def fetch_warsaw_civic_pois(
    cache_dir: Path,
    *,
    force_refresh: bool = False,
    wfs_url: str = WARSAW_WFS_URL,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> gpd.GeoDataFrame:
    """Download civic WFS POI layers and normalize to city-POI schema."""
    parts: list[gpd.GeoDataFrame] = []
    for object_type, type_name, default_name in WARSAW_CIVIC_WFS_LAYERS:
        cache_name = f"warsaw_civic_{object_type}.geojson"
        frame = fetch_warsaw_wfs_layer(
            cache_dir,
            type_name=type_name,
            cache_name=cache_name,
            force_refresh=force_refresh,
            wfs_url=wfs_url,
            timeout_s=timeout_s,
            default_name=default_name,
        )
        parts.append(
            _civic_layer_to_pois(
                frame, object_type=object_type, default_name=default_name
            )
        )
    parts = [p for p in parts if not p.empty]
    if not parts:
        return gpd.GeoDataFrame(
            columns=["object_type", "name", "source", "geometry"],
            geometry=[],
            crs="EPSG:4326",
        )
    return gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )


def fetch_warsaw_city_pois(
    cache_dir: Path,
    *,
    fixture_path: Path | None = None,
    force_refresh: bool = False,
    wfs_url: str = WARSAW_WFS_URL,
    dataserver_base: str = WARSAW_MAPVIEWER_DATASERVER,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> gpd.GeoDataFrame:
    """Sports + civic city POIs; cache as ``warsaw_city_pois.geojson``.

    Args:
        cache_dir: Cache directory.
        fixture_path: Offline combined GeoJSON fallback.
        force_refresh: Ignore combined cache.
        wfs_url: Civic WFS endpoint.
        dataserver_base: Sports MapViewer dataserver.
        timeout_s: HTTP timeout.

    Returns:
        Points with ``object_type``, ``name``, ``source`` columns.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / WARSAW_LAYER_REGISTRY["city_pois"]["cache_name"]
    if not force_refresh and cache_path.exists():
        return gpd.read_file(cache_path)

    sports = fetch_warsaw_sports_objects(
        dataserver_base=dataserver_base,
        timeout_s=timeout_s,
        fixture_path=None,
    )
    civic = fetch_warsaw_civic_pois(
        cache_dir,
        force_refresh=force_refresh,
        wfs_url=wfs_url,
        timeout_s=timeout_s,
    )
    parts = [p for p in (sports, civic) if not p.empty]
    if parts:
        frame = gpd.GeoDataFrame(
            pd.concat(parts, ignore_index=True),
            geometry="geometry",
            crs="EPSG:4326",
        )
    else:
        frame = gpd.GeoDataFrame(
            columns=["object_type", "name", "source", "geometry"],
            geometry=[],
            crs="EPSG:4326",
        )

    if frame.empty and fixture_path is not None and fixture_path.exists():
        logger.warning("Using city POI fixture %s", fixture_path)
        frame = gpd.read_file(fixture_path)
        if frame.crs is None:
            frame = frame.set_crs("EPSG:4326")
        elif str(frame.crs) != "EPSG:4326":
            frame = frame.to_crs("EPSG:4326")

    if not frame.empty:
        frame.to_file(cache_path, driver="GeoJSON")
        logger.info("Cached %s city POIs → %s", len(frame), cache_path)
    return frame
