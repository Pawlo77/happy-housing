"""OSM bicycle infrastructure: fetch, densify, and listing features.

Provides nearest bike-path distance and neighbourhood path coverage (metres of
cycle infrastructure within a radius) for the price model and Explore map.
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString

from happy_housing.gis.frame_util import assign_columns
from happy_housing.gis.spatial_query import balltree_radius_weight_sums

logger = logging.getLogger(__name__)

_CACHE_NAME: str = "bike_paths.geojson"
"""Cached OSM bicycle network (line geometries)."""

_SAMPLE_CACHE_NAME: str = "bike_path_sample.parquet"
"""Densified sample points for neighbourhood coverage."""

BIKE_PATH_COVERAGE_RADIUS_M: float = 500.0
"""Neighbourhood radius for ``bike_path_coverage_m`` (metres)."""

_DEFAULT_SAMPLE_STEP_M: float = 40.0
"""Spacing along bike paths when estimating coverage length."""

MAX_BIKE_PATH_MAP_FEATURES: int = 4_000
"""Hard cap for PathLayer rows sent to the browser."""

_OSM_TIMEOUT_S: int = 300
"""OSMnx Overpass timeout."""

_BIKE_TAGS: dict[str, bool | str | list[str]] = {
    "highway": "cycleway",
    "cycleway": True,
    "bicycle": "designated",
}
"""Overpass / OSMnx tags for dedicated or designated bike infrastructure."""


def _lines_from_geometry(geom: object) -> list[LineString]:
    """Normalize geometry to LineString list."""
    if geom is None or getattr(geom, "is_empty", False):
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [line for line in geom.geoms if isinstance(line, LineString)]
    return []


def densify_bike_paths(
    paths: gpd.GeoDataFrame,
    *,
    step_m: float = _DEFAULT_SAMPLE_STEP_M,
    projected_crs: str = "EPSG:2180",
) -> pd.DataFrame:
    """Sample points along bike centre-lines for coverage estimation."""
    if paths.empty:
        return pd.DataFrame(columns=["lon", "lat", "step_m"])
    work = paths.copy()
    if work.crs is None:
        work = work.set_crs("EPSG:4326")
    projected = work.to_crs(projected_crs)
    rows: list[dict[str, float]] = []
    step = max(float(step_m), 1.0)
    for geom in projected.geometry:
        for line in _lines_from_geometry(geom):
            length = float(line.length)
            if length <= 0.0:
                continue
            n_seg = max(1, int(np.ceil(length / step)))
            for dist in np.linspace(0.0, length, n_seg + 1):
                point = line.interpolate(float(dist))
                rows.append({
                    "x": float(point.x),
                    "y": float(point.y),
                    "step_m": step,
                })
    if not rows:
        return pd.DataFrame(columns=["lon", "lat", "step_m"])
    frame = pd.DataFrame(rows)
    pts = gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame["x"], frame["y"]),
        crs=projected_crs,
    ).to_crs("EPSG:4326")
    return pd.DataFrame({
        "lon": pts.geometry.x.astype(float),
        "lat": pts.geometry.y.astype(float),
        "step_m": pts["step_m"].astype(float),
    }).reset_index(drop=True)


def attach_bike_path_features(
    listings: gpd.GeoDataFrame,
    paths: gpd.GeoDataFrame,
    *,
    projected_crs: str,
    coverage_radius_m: float = BIKE_PATH_COVERAGE_RADIUS_M,
    sample_step_m: float = _DEFAULT_SAMPLE_STEP_M,
    sample_points: pd.DataFrame | None = None,
    left_projected: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Attach ``dist_bike_path_m`` and ``bike_path_coverage_m`` to listings.

    Coverage approximates metres of bike infrastructure within
    ``coverage_radius_m`` via densified path samples (count x step).
    """
    out = listings
    if paths.empty:
        return assign_columns(
            out,
            {"dist_bike_path_m": float("nan"), "bike_path_coverage_m": 0.0},
        )

    left = left_projected if left_projected is not None else out.to_crs(projected_crs)
    right = paths.to_crs(projected_crs)[["geometry"]]
    matched = left.sjoin_nearest(right, how="left", distance_col="dist_bike_path_m")
    matched = matched[~matched.index.duplicated(keep="first")]
    aligned = matched.reindex(out.index)
    cols: dict[str, object] = {
        "dist_bike_path_m": aligned["dist_bike_path_m"].to_numpy(dtype=float),
    }

    if sample_points is None:
        sample_points = densify_bike_paths(
            paths, step_m=sample_step_m, projected_crs=projected_crs
        )
    if sample_points.empty:
        cols["bike_path_coverage_m"] = 0.0
        return assign_columns(out, cols)

    pts = gpd.GeoDataFrame(
        sample_points,
        geometry=gpd.points_from_xy(sample_points["lon"], sample_points["lat"]),
        crs="EPSG:4326",
    ).to_crs(projected_crs)
    step = (
        float(sample_points["step_m"].iloc[0])
        if "step_m" in sample_points
        else (float(sample_step_m))
    )
    query_xy = np.column_stack([left.geometry.x.to_numpy(), left.geometry.y.to_numpy()])
    data_xy = np.column_stack([pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy()])
    weights = np.full(len(pts), step, dtype=float)
    coverage = balltree_radius_weight_sums(
        query_xy, data_xy, weights, float(coverage_radius_m)
    )
    cols["bike_path_coverage_m"] = coverage
    return assign_columns(out, cols)


def _download_osm_bike_paths(
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
) -> gpd.GeoDataFrame:
    """Download bicycle infrastructure lines via OSMnx features API."""
    west, south, east, north = bbox
    import osmnx as ox

    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_dir / "osmnx")
    ox.settings.requests_timeout = _OSM_TIMEOUT_S
    features = ox.features_from_bbox(
        bbox=(west, south, east, north),
        tags=_BIKE_TAGS,
    )
    if features.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    work = features.copy()
    if work.crs is None:
        work = work.set_crs("EPSG:4326")
    elif str(work.crs) != "EPSG:4326":
        work = work.to_crs("EPSG:4326")
    # Keep linear bike infrastructure; drop pure points/polygons without lines.
    line_mask = work.geometry.geom_type.isin(["LineString", "MultiLineString"])
    work = work.loc[line_mask].copy()
    if work.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    keep = [
        c
        for c in ("highway", "cycleway", "bicycle", "name", "geometry")
        if c in work.columns
    ]
    roads = gpd.GeoDataFrame(work[keep], geometry="geometry", crs="EPSG:4326")
    logger.info("Fetched %s OSM bike-path features", len(roads))
    return roads


def fetch_bike_paths(
    cache_dir: Path,
    *,
    bbox: tuple[float, float, float, float],
    fixture_path: Path | None = None,
    force_refresh: bool = False,
) -> gpd.GeoDataFrame:
    """Fetch / cache Warsaw OSM bicycle network lines."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _CACHE_NAME
    if not force_refresh and cache_path.exists():
        cached = gpd.read_file(cache_path)
        if not cached.empty:
            return cached

    roads = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    try:
        roads = _download_osm_bike_paths(bbox, cache_dir=cache_dir)
    except Exception as exc:
        logger.warning("OSMnx bike-path fetch failed: %s", exc)

    if roads.empty and fixture_path is not None and fixture_path.exists():
        logger.warning("Using bike-path fixture %s", fixture_path)
        roads = gpd.read_file(fixture_path)

    if not roads.empty:
        roads.to_file(cache_path, driver="GeoJSON")
        sample = densify_bike_paths(roads)
        if not sample.empty:
            sample.to_parquet(cache_dir / _SAMPLE_CACHE_NAME, index=False)
        logger.info("Cached %s bike-path features → %s", len(roads), cache_path)
    return roads


def load_bike_path_sample(cache_dir: Path) -> pd.DataFrame:
    """Load densified coverage sample if present."""
    path = cache_dir / _SAMPLE_CACHE_NAME
    if not path.exists():
        return pd.DataFrame(columns=["lon", "lat", "step_m"])
    return pd.read_parquet(path)


def prepare_bike_paths_for_map(
    paths: gpd.GeoDataFrame,
    max_features: int = MAX_BIKE_PATH_MAP_FEATURES,
) -> pd.DataFrame:
    """Convert bike LineStrings to PathLayer ``path`` payloads."""
    if paths.empty:
        return pd.DataFrame()
    work = paths.copy()
    if work.crs is None:
        work = work.set_crs("EPSG:4326")
    elif str(work.crs) != "EPSG:4326":
        work = work.to_crs("EPSG:4326")
    # Prefer longer corridors when capping for the browser.
    projected = work.to_crs("EPSG:2180")
    work = work.assign(length_m=projected.geometry.length.to_numpy())
    work = work.sort_values("length_m", ascending=False)
    if max_features > 0 and len(work) > max_features:
        work = work.head(int(max_features))
    rows: list[dict[str, object]] = []
    for _, row in work.iterrows():
        for line in _lines_from_geometry(row.geometry):
            coords = [[float(x), float(y)] for x, y in line.coords]
            if len(coords) < 2:
                continue
            name = row.get("name")
            label = (
                str(name)
                if name is not None and str(name) not in {"", "nan"}
                else "bike path"
            )
            rows.append({
                "path": coords,
                "tooltip_name": label,
                "tooltip_detail": "OSM bicycle infrastructure",
                "color_r": 14,
                "color_g": 165,
                "color_b": 233,
                "color_a": 200,
            })
    return pd.DataFrame(rows)
