"""OSM road-network diurnal traffic intensity proxy.

Builds a dense point sample along Warsaw driveable highways (not sparse
corridor centroids), weighted by OSM ``highway`` class and scaled by hour.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString

logger = logging.getLogger(__name__)

HIGHWAY_BASE_WEIGHT: dict[str, float] = {
    "motorway": 10.0,
    "motorway_link": 8.0,
    "trunk": 9.0,
    "trunk_link": 7.0,
    "primary": 7.0,
    "primary_link": 5.5,
    "secondary": 5.0,
    "secondary_link": 4.0,
    "tertiary": 3.5,
    "tertiary_link": 2.5,
    "residential": 1.6,
    "living_street": 1.2,
    "unclassified": 2.0,
    "service": 0.8,
}
"""Relative traffic intensity by OSM highway class (unitless)."""

_MAJOR_HIGHWAYS: frozenset[str] = frozenset({
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
})
"""Roads densified at the fine step."""

_DEFAULT_FINE_STEP_M: float = 50.0
"""Sample spacing along major corridors (metres)."""

_DEFAULT_COARSE_STEP_M: float = 140.0
"""Sample spacing along residential / minor streets (metres)."""

MAX_TRAFFIC_SAMPLE_POINTS: int = 28_000
"""Hard cap for Streamlit / pydeck message size."""

TRAFFIC_NEIGHBOURHOOD_RADIUS_M: float = 300.0
"""Radius for ``traffic_weight_nearby`` (sum of ``base_weight``)."""

_CACHE_NAME: str = "traffic_roads.parquet"
"""Cached densified road sample filename."""

_OSM_TIMEOUT_S: int = 300
"""OSMnx Overpass timeout for the city drive graph."""


def normalize_highway_key(highway: object) -> str:
    """Pick the highest-weight OSM highway tag from a scalar or list."""
    if highway is None or (isinstance(highway, float) and np.isnan(highway)):
        return "unclassified"
    if isinstance(highway, list | tuple):
        parts = [normalize_highway_key(part) for part in highway]
        if not parts:
            return "unclassified"
        return max(parts, key=lambda key: HIGHWAY_BASE_WEIGHT.get(key, 0.0))
    key = str(highway).strip().lower()
    if key.startswith("[") and key.endswith("]"):
        # Defensive: stringified list from a prior serialization.
        inner = key[1:-1].replace("'", "").replace('"', "")
        return normalize_highway_key([part.strip() for part in inner.split(",")])
    if ";" in key:
        return normalize_highway_key(key.split(";"))
    return key if key in HIGHWAY_BASE_WEIGHT else "unclassified"


def highway_weight(highway: object) -> float:
    """Map an OSM highway tag (or list) to a base intensity weight."""
    key = normalize_highway_key(highway)
    return float(HIGHWAY_BASE_WEIGHT.get(key, HIGHWAY_BASE_WEIGHT["unclassified"]))


def _lines_from_geometry(geom: object) -> list[LineString]:
    """Normalize a geometry to a list of LineStrings."""
    if geom is None or getattr(geom, "is_empty", False):
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [line for line in geom.geoms if isinstance(line, LineString)]
    return []


def densify_road_lines(
    roads: gpd.GeoDataFrame,
    *,
    fine_step_m: float = _DEFAULT_FINE_STEP_M,
    coarse_step_m: float = _DEFAULT_COARSE_STEP_M,
    projected_crs: str = "EPSG:2180",
) -> pd.DataFrame:
    """Interpolate points along road centre-lines for a heatmap sample.

    Args:
        roads: Road edges with ``highway`` and LineString geometries.
        fine_step_m: Spacing on major roads.
        coarse_step_m: Spacing on residential / minor roads.
        projected_crs: Metric CRS for length-based sampling.

    Returns:
        Columns ``lon``, ``lat``, ``base_weight``, ``name``, ``highway``.
    """
    if roads.empty:
        return pd.DataFrame(columns=["lon", "lat", "base_weight", "name", "highway"])
    work = roads.copy()
    if work.crs is None:
        work = work.set_crs("EPSG:4326")
    projected = work.to_crs(projected_crs)
    rows: list[dict[str, object]] = []
    for row in projected.itertuples(index=False):
        highway = getattr(row, "highway", None)
        hwy_key = normalize_highway_key(highway)
        # Skip parking-lot / driveway clutter — drown street grid otherwise.
        if hwy_key == "service":
            continue
        weight = highway_weight(hwy_key)
        step = fine_step_m if hwy_key in _MAJOR_HIGHWAYS else coarse_step_m
        name = getattr(row, "name", None)
        label = (
            str(name) if name is not None and str(name) not in {"", "nan"} else hwy_key
        )
        for line in _lines_from_geometry(row.geometry):
            length = float(line.length)
            if length <= 0.0:
                continue
            # Always include endpoints; step along the line in metres.
            n_seg = max(1, int(np.ceil(length / max(step, 1.0))))
            distances = np.linspace(0.0, length, n_seg + 1)
            for dist in distances:
                point = line.interpolate(float(dist))
                rows.append({
                    "x": float(point.x),
                    "y": float(point.y),
                    "base_weight": weight,
                    "name": label,
                    "highway": hwy_key,
                })
    if not rows:
        return pd.DataFrame(columns=["lon", "lat", "base_weight", "name", "highway"])
    frame = pd.DataFrame(rows)
    pts = gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame["x"], frame["y"]),
        crs=projected_crs,
    ).to_crs("EPSG:4326")
    out = pd.DataFrame({
        "lon": pts.geometry.x.astype(float),
        "lat": pts.geometry.y.astype(float),
        "base_weight": pts["base_weight"].astype(float),
        "name": pts["name"].astype(str),
        "highway": pts["highway"].astype(str),
    })
    return out.reset_index(drop=True)


def _cap_sample(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """Downsample while keeping street-grid coverage (not motorway-only).

    Stratify by ``highway`` so residential tertiary network survives the
    browser payload cap. Intensity still comes from ``base_weight``.
    """
    if max_points <= 0 or frame.empty or len(frame) <= max_points:
        return frame.reset_index(drop=True)
    if "highway" not in frame.columns:
        return frame.sample(n=max_points, random_state=42).reset_index(drop=True)

    groups = list(frame.groupby("highway", sort=False))
    n_groups = len(groups)
    # Floor so every class keeps a ribbon of the network.
    floor = max(40, max_points // max(n_groups * 8, 1))
    quotas: dict[str, int] = {}
    # First pass: proportional + floor.
    total = len(frame)
    for key, group in groups:
        share = round(max_points * (len(group) / total))
        quotas[str(key)] = min(len(group), max(floor, share))
    # Scale down if floors overshoot.
    assigned = sum(quotas.values())
    if assigned > max_points:
        scale = max_points / assigned
        quotas = {key: max(1, round(n * scale)) for key, n in quotas.items()}
        # Trim largest groups until exact.
        while sum(quotas.values()) > max_points:
            biggest = max(quotas, key=lambda k: quotas[k])
            if quotas[biggest] <= 1:
                break
            quotas[biggest] -= 1
    remaining = max_points - sum(quotas.values())
    parts: list[pd.DataFrame] = []
    leftovers: list[pd.DataFrame] = []
    for key, group in groups:
        take = min(quotas.get(str(key), 0), len(group))
        if take <= 0:
            leftovers.append(group)
            continue
        if take >= len(group):
            parts.append(group)
        else:
            picked = group.sample(n=take, random_state=42)
            parts.append(picked)
            leftovers.append(group.drop(index=picked.index))
    out = pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0]
    if remaining > 0 and leftovers:
        pool = pd.concat(leftovers, ignore_index=True)
        extra = pool.sample(n=min(remaining, len(pool)), random_state=42)
        out = pd.concat([out, extra], ignore_index=True)
    if len(out) > max_points:
        out = out.sample(n=max_points, random_state=42)
    return out.reset_index(drop=True)


def _load_fixture_points(path: Path) -> pd.DataFrame:
    """Read JSON list of lon/lat/base_weight samples."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(payload)
    for col in ("lon", "lat", "base_weight"):
        if col not in frame.columns:
            return pd.DataFrame(
                columns=["lon", "lat", "base_weight", "name", "highway"]
            )
    if "name" not in frame.columns:
        frame["name"] = "road"
    if "highway" not in frame.columns:
        frame["highway"] = "unclassified"
    return frame[["lon", "lat", "base_weight", "name", "highway"]]


def _download_osm_drive_edges(
    bbox: tuple[float, float, float, float],
    *,
    cache_dir: Path,
) -> gpd.GeoDataFrame:
    """Fetch driveable highway edges via OSMnx for ``bbox``."""
    west, south, east, north = bbox
    import osmnx as ox

    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_dir / "osmnx")
    ox.settings.requests_timeout = _OSM_TIMEOUT_S
    # Omit ``service`` (driveways / parking aisles) — they dominate Overpass
    # counts and turn the heatmap into noise instead of a street grid.
    custom = (
        '["highway"~"motorway|trunk|primary|secondary|tertiary|'
        'residential|living_street|unclassified"]'
    )
    graph = ox.graph_from_bbox(
        bbox=(west, south, east, north),
        network_type="drive",
        custom_filter=custom,
        simplify=True,
        retain_all=False,
    )
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    if edges.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    keep = [c for c in ("highway", "name", "geometry") if c in edges.columns]
    roads = gpd.GeoDataFrame(edges[keep], geometry="geometry", crs=edges.crs)
    if roads.crs is None:
        roads = roads.set_crs("EPSG:4326")
    elif str(roads.crs) != "EPSG:4326":
        roads = roads.to_crs("EPSG:4326")
    logger.info("Fetched %s OSM road edges for traffic proxy", len(roads))
    return roads


def fetch_traffic_road_sample(
    cache_dir: Path,
    *,
    bbox: tuple[float, float, float, float],
    fixture_path: Path | None = None,
    force_refresh: bool = False,
    max_points: int = MAX_TRAFFIC_SAMPLE_POINTS,
    fine_step_m: float = _DEFAULT_FINE_STEP_M,
    coarse_step_m: float = _DEFAULT_COARSE_STEP_M,
) -> pd.DataFrame:
    """Download Warsaw drive edges, densify, cache as parquet.

    Args:
        cache_dir: Directory for ``traffic_roads.parquet``.
        bbox: ``(west, south, east, north)`` WGS84.
        fixture_path: Offline JSON fallback.
        force_refresh: Ignore existing cache.
        max_points: Browser payload cap.
        fine_step_m: Major-road spacing.
        coarse_step_m: Minor-road spacing.

    Returns:
        Point sample with ``lon``, ``lat``, ``base_weight`` (+ metadata).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _CACHE_NAME
    if not force_refresh and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if not cached.empty and {"lon", "lat", "base_weight"}.issubset(cached.columns):
            return cached

    roads = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    try:
        roads = _download_osm_drive_edges(bbox, cache_dir=cache_dir)
    except Exception as exc:
        logger.warning("OSMnx traffic road fetch failed: %s", exc)

    sample = densify_road_lines(
        roads,
        fine_step_m=fine_step_m,
        coarse_step_m=coarse_step_m,
    )
    if sample.empty and fixture_path is not None and fixture_path.exists():
        logger.warning("Using traffic fixture %s", fixture_path)
        sample = _load_fixture_points(fixture_path)

    sample = _cap_sample(sample, max_points)
    if not sample.empty:
        sample.to_parquet(cache_path, index=False)
        logger.info("Cached %s traffic sample points → %s", len(sample), cache_path)
    return sample


def attach_traffic_features(
    listings: gpd.GeoDataFrame,
    sample: pd.DataFrame,
    *,
    projected_crs: str,
    neighbourhood_radius_m: float = TRAFFIC_NEIGHBOURHOOD_RADIUS_M,
) -> gpd.GeoDataFrame:
    """Attach ``dist_traffic_road_m`` and ``traffic_weight_nearby``.

    Distance is to the nearest densified OSM drive sample. Neighbourhood weight
    sums ``base_weight`` of sample points within ``neighbourhood_radius_m``.
    """
    out = listings.copy()
    if sample.empty or not {"lon", "lat", "base_weight"}.issubset(sample.columns):
        out["dist_traffic_road_m"] = float("nan")
        out["traffic_weight_nearby"] = 0.0
        return out

    pts = gpd.GeoDataFrame(
        sample,
        geometry=gpd.points_from_xy(sample["lon"], sample["lat"]),
        crs="EPSG:4326",
    )
    left = out.to_crs(projected_crs)
    right = pts.to_crs(projected_crs)
    matched = left.sjoin_nearest(
        right[["geometry"]], how="left", distance_col="dist_traffic_road_m"
    )
    matched = matched[~matched.index.duplicated(keep="first")]
    aligned = matched.reindex(out.index)
    out["dist_traffic_road_m"] = aligned["dist_traffic_road_m"].to_numpy(dtype=float)

    weight_pts = right[["geometry"]].copy()
    weight_pts["base_weight"] = pts.to_crs(projected_crs)["base_weight"].astype(float)
    buffers = left.copy()
    buffers["geometry"] = buffers.geometry.buffer(float(neighbourhood_radius_m))
    hit = gpd.sjoin(
        weight_pts,
        buffers[["geometry"]],
        how="inner",
        predicate="within",
    )
    if hit.empty:
        out["traffic_weight_nearby"] = 0.0
        return out
    weights = hit.groupby("index_right")["base_weight"].sum()
    out["traffic_weight_nearby"] = (
        weights.reindex(out.index, fill_value=0.0).astype(float).to_numpy()
    )
    return out
