"""Correct listing map pins that fall inside park polygons.

Otodom / fixture / RCN centroids sometimes land in parks (privacy approx or
synthetic coords). Snap those points to the nearest OSM building centroid
that lies outside park polygons so Explore pins match habitable locations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

_CACHE_NAME: str = "osm_building_centroids.parquet"
"""Cached OSM building centroids (lon/lat only)."""

_DEFAULT_MAX_SNAP_M: float = 1_000.0
"""Max distance when snapping a park pin to a building."""

_BUILDING_TAGS: dict[str, bool | list[str]] = {
    "building": [
        "yes",
        "apartments",
        "residential",
        "house",
        "detached",
        "semidetached_house",
        "terrace",
        "dormitory",
        "bungalow",
    ],
}
"""OSM building tags treated as plausible housing locations."""

_OSM_TIMEOUT_S: int = 300
"""OSMnx Overpass timeout for building downloads."""

_PROJECTED_CRS: str = "EPSG:2180"
"""Metric CRS for Warsaw nearest-neighbour distances."""


def listing_points_inside_parks(
    listings: pd.DataFrame,
    parks: gpd.GeoDataFrame,
) -> pd.Series:
    """Boolean mask: listing lat/lon falls inside any park polygon."""
    if listings.empty or parks.empty:
        return pd.Series(False, index=listings.index)
    if not {"lat", "lon"}.issubset(listings.columns):
        return pd.Series(False, index=listings.index)
    pts = gpd.GeoDataFrame(
        listings[[]],
        geometry=gpd.points_from_xy(
            pd.to_numeric(listings["lon"], errors="coerce"),
            pd.to_numeric(listings["lat"], errors="coerce"),
        ),
        crs="EPSG:4326",
        index=listings.index,
    )
    valid = pts.geometry.notna() & ~pts.geometry.is_empty
    if not valid.any():
        return pd.Series(False, index=listings.index)
    parks_m = parks.to_crs(_PROJECTED_CRS)
    park_union = unary_union(parks_m.geometry.values)
    pts_m = pts.loc[valid].to_crs(_PROJECTED_CRS)
    inside = pts_m.geometry.within(park_union)
    out = pd.Series(False, index=listings.index)
    out.loc[inside.index] = inside.to_numpy(dtype=bool)
    return out


def fetch_osm_building_centroids(
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
    *,
    force_refresh: bool = False,
) -> gpd.GeoDataFrame:
    """Download OSM residential/apartment building centroids for ``bbox``."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _CACHE_NAME
    if not force_refresh and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if not cached.empty and {"lon", "lat"}.issubset(cached.columns):
            return gpd.GeoDataFrame(
                cached,
                geometry=gpd.points_from_xy(cached["lon"], cached["lat"]),
                crs="EPSG:4326",
            )

    west, south, east, north = bbox
    import osmnx as ox

    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_dir / "osmnx")
    ox.settings.requests_timeout = _OSM_TIMEOUT_S
    try:
        features = ox.features_from_bbox(
            bbox=(west, south, east, north),
            tags=_BUILDING_TAGS,
        )
    except Exception as exc:
        logger.warning("OSM building fetch failed: %s", exc)
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    if features.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    work = features.copy()
    if work.crs is None:
        work = work.set_crs("EPSG:4326")
    centroids = work.to_crs(_PROJECTED_CRS)
    centroids["geometry"] = centroids.geometry.centroid
    centroids = centroids.to_crs("EPSG:4326")
    frame = pd.DataFrame({
        "lon": centroids.geometry.x.astype(float),
        "lat": centroids.geometry.y.astype(float),
    }).dropna()
    frame = frame.drop_duplicates(subset=["lon", "lat"]).reset_index(drop=True)
    frame.to_parquet(cache_path, index=False)
    logger.info("Cached %s OSM building centroids → %s", len(frame), cache_path)
    return gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame["lon"], frame["lat"]),
        crs="EPSG:4326",
    )


def snap_listings_out_of_parks(
    listings: pd.DataFrame,
    *,
    parks: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame | None = None,
    cache_dir: Path | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    max_snap_m: float = _DEFAULT_MAX_SNAP_M,
    force_refresh_buildings: bool = False,
) -> pd.DataFrame:
    """Move park-interior listing pins to the nearest outside building.

    Args:
        listings: Rows with ``lat`` / ``lon``.
        parks: Park polygons (WGS84 or projected).
        buildings: Optional preloaded building points; fetched via OSM when None.
        cache_dir: Required when ``buildings`` is None.
        bbox: Required when fetching buildings.
        max_snap_m: Skip snap when nearest outside building is farther.
        force_refresh_buildings: Ignore building centroid cache.

    Returns:
        Copy of ``listings`` with updated coords and ``location_snapped``.
    """
    out = listings.copy()
    out["location_snapped"] = False
    if out.empty or parks.empty:
        return out

    inside = listing_points_inside_parks(out, parks)
    n_inside = int(inside.sum())
    if n_inside == 0:
        logger.info("No listing pins inside parks — location snap skipped")
        return out

    bld = buildings
    if bld is None:
        if cache_dir is None or bbox is None:
            logger.warning(
                "Cannot snap %s park pins: buildings/cache/bbox missing",
                n_inside,
            )
            return out
        bld = fetch_osm_building_centroids(
            bbox,
            cache_dir,
            force_refresh=force_refresh_buildings,
        )
    if bld is None or bld.empty:
        logger.warning("No OSM buildings available to snap %s park pins", n_inside)
        return out

    parks_m = parks.to_crs(_PROJECTED_CRS)
    park_union = unary_union(parks_m.geometry.values)
    bld_m = bld.to_crs(_PROJECTED_CRS)
    outside_mask = ~bld_m.geometry.within(park_union)
    candidates = bld_m.loc[outside_mask].copy()
    if candidates.empty:
        logger.warning("All OSM buildings lie inside parks — snap skipped")
        return out
    candidates = candidates.reset_index(drop=True)
    candidates["bld_idx"] = np.arange(len(candidates))

    bad = gpd.GeoDataFrame(
        out.loc[inside, ["lat", "lon"]].copy(),
        geometry=gpd.points_from_xy(
            out.loc[inside, "lon"],
            out.loc[inside, "lat"],
        ),
        crs="EPSG:4326",
        index=out.loc[inside].index,
    ).to_crs(_PROJECTED_CRS)

    joined = bad.sjoin_nearest(
        candidates[["bld_idx", "geometry"]],
        how="left",
        max_distance=float(max_snap_m),
        distance_col="snap_m",
    )
    # One match per listing (sjoin_nearest can duplicate ties).
    joined = joined[~joined.index.duplicated(keep="first")]

    snapped = 0
    for idx, row in joined.iterrows():
        bld_idx = row.get("bld_idx")
        snap_m = row.get("snap_m")
        if bld_idx is None or (isinstance(bld_idx, float) and np.isnan(bld_idx)):
            continue
        if snap_m is None or (isinstance(snap_m, float) and np.isnan(snap_m)):
            continue
        if float(snap_m) > float(max_snap_m):
            continue
        geom = candidates.loc[int(bld_idx)].geometry
        lon_lat = gpd.GeoSeries([geom], crs=_PROJECTED_CRS).to_crs("EPSG:4326").iloc[0]
        out.at[idx, "lon"] = float(lon_lat.x)
        out.at[idx, "lat"] = float(lon_lat.y)
        out.at[idx, "location_snapped"] = True
        snapped += 1

    logger.info(
        "Snapped %s / %s park-interior listing pins onto OSM buildings",
        snapped,
        n_inside,
    )
    return out
