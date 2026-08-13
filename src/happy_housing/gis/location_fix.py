"""Correct listing map pins that miss habitable building footprints.

Otodom / fixture / RCN coords are often privacy-approximate: park interiors,
street centre-lines, or sidewalks. Snap those points to the nearest OSM
residential building (centroid) outside park polygons so Explore pins sit on
buildings rather than roads or greens.
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

_CACHE_NAME: str = "osm_building_centroids.parquet"
"""Cached OSM building centroids (lon/lat only) — legacy / fallback."""

_CACHE_FOOTPRINTS: str = "osm_building_footprints.parquet"
"""Cached OSM building footprint polygons (GeoParquet)."""

_DEFAULT_MAX_SNAP_M: float = 1_000.0
"""Max distance when snapping a misplaced pin to a building."""

_POINT_OUTSIDE_M: float = 25.0
"""When only centroids exist, treat pins farther than this as outside."""

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


def _listing_points_gdf(listings: pd.DataFrame) -> gpd.GeoDataFrame:
    """WGS84 GeoDataFrame of listing lat/lon points."""
    return gpd.GeoDataFrame(
        listings[[]],
        geometry=gpd.points_from_xy(
            pd.to_numeric(listings["lon"], errors="coerce"),
            pd.to_numeric(listings["lat"], errors="coerce"),
        ),
        crs="EPSG:4326",
        index=listings.index,
    )


def listing_points_inside_parks(
    listings: pd.DataFrame,
    parks: gpd.GeoDataFrame,
) -> pd.Series:
    """Boolean mask: listing lat/lon falls inside any park polygon."""
    if listings.empty or parks.empty:
        return pd.Series(False, index=listings.index)
    if not {"lat", "lon"}.issubset(listings.columns):
        return pd.Series(False, index=listings.index)
    pts = _listing_points_gdf(listings)
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


def listing_points_outside_buildings(
    listings: pd.DataFrame,
    buildings: gpd.GeoDataFrame,
) -> pd.Series:
    """Boolean mask: listing is not inside (or near) a building footprint.

    Polygon buildings use geometric ``within``. Point-only caches (centroids)
    treat a pin as outside when nearest centroid is farther than
    ``_POINT_OUTSIDE_M`` (road / sidewalk privacy approx).
    """
    if listings.empty or buildings.empty:
        return pd.Series(False, index=listings.index)
    if not {"lat", "lon"}.issubset(listings.columns):
        return pd.Series(False, index=listings.index)

    pts = _listing_points_gdf(listings)
    valid = pts.geometry.notna() & ~pts.geometry.is_empty
    out = pd.Series(False, index=listings.index)
    if not valid.any():
        return out

    pts_m = pts.loc[valid].to_crs(_PROJECTED_CRS)
    bld_m = buildings.to_crs(_PROJECTED_CRS)
    geom_types = {g for g in bld_m.geometry.geom_type.unique() if g}
    has_area = bool(geom_types & {"Polygon", "MultiPolygon"})

    if has_area:
        area = bld_m.loc[bld_m.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        union = unary_union(area.geometry.values)
        inside = pts_m.geometry.within(union)
        out.loc[inside.index] = (~inside).to_numpy(dtype=bool)
        return out

    # Centroid-only cache: distance heuristic.
    joined = pts_m.sjoin_nearest(
        bld_m[["geometry"]],
        how="left",
        distance_col="bld_m",
    )
    joined = joined[~joined.index.duplicated(keep="first")]
    far = joined["bld_m"].isna() | (joined["bld_m"] > float(_POINT_OUTSIDE_M))
    out.loc[far.index] = far.to_numpy(dtype=bool)
    return out


def _building_centroids(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Point GeoDataFrame of building centroids (projected → WGS84)."""
    if buildings.empty:
        return buildings
    work = buildings.to_crs(_PROJECTED_CRS).copy()
    work["geometry"] = work.geometry.centroid
    return work.to_crs("EPSG:4326")


def fetch_osm_building_centroids(
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
    *,
    force_refresh: bool = False,
) -> gpd.GeoDataFrame:
    """Download OSM residential/apartment buildings for ``bbox``.

    Prefers footprint polygons (GeoParquet). Falls back to the legacy
    centroid-only cache. Snap targets always use centroids derived from
    whatever geometry is available.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    footprints_path = cache_dir / _CACHE_FOOTPRINTS
    centroids_path = cache_dir / _CACHE_NAME

    if not force_refresh and footprints_path.exists():
        try:
            cached = gpd.read_parquet(footprints_path)
            if not cached.empty and cached.geometry is not None:
                if cached.crs is None:
                    cached = cached.set_crs("EPSG:4326")
                return cached
        except Exception as exc:
            logger.warning("Footprint cache unreadable (%s) — rebuilding", exc)

    # Prefer a fresh OSM footprint download over the legacy centroid-only
    # cache so road/sidewalk containment checks stay accurate.
    if not force_refresh and not footprints_path.exists() and centroids_path.exists():
        logger.info(
            "Legacy building centroid cache present without footprints — "
            "re-fetching OSM building polygons"
        )

    def _load_centroid_cache() -> gpd.GeoDataFrame:
        if not centroids_path.exists():
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        cached = pd.read_parquet(centroids_path)
        if cached.empty or not {"lon", "lat"}.issubset(cached.columns):
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
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
        fallback = _load_centroid_cache()
        if not fallback.empty:
            logger.warning("Falling back to cached building centroids")
            return fallback
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    if features.empty:
        fallback = _load_centroid_cache()
        return (
            fallback
            if not fallback.empty
            else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        )

    work = features.copy()
    if work.crs is None:
        work = work.set_crs("EPSG:4326")
    # Keep polygonal footprints when present; drop pure nodes.
    polys = work.loc[work.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if polys.empty:
        polys = work.copy()

    keep = polys[["geometry"]].copy()
    keep = keep[~keep.geometry.is_empty & keep.geometry.notna()].reset_index(drop=True)
    try:
        keep.to_parquet(footprints_path)
        logger.info(
            "Cached %s OSM building footprints → %s", len(keep), footprints_path
        )
    except Exception as exc:
        logger.warning("Could not write footprint cache: %s", exc)

    centroids = _building_centroids(keep)
    frame = pd.DataFrame({
        "lon": centroids.geometry.x.astype(float),
        "lat": centroids.geometry.y.astype(float),
    }).dropna()
    frame = frame.drop_duplicates(subset=["lon", "lat"]).reset_index(drop=True)
    frame.to_parquet(centroids_path, index=False)
    logger.info("Cached %s OSM building centroids → %s", len(frame), centroids_path)
    return (
        keep
        if not keep.empty
        else gpd.GeoDataFrame(
            frame,
            geometry=gpd.points_from_xy(frame["lon"], frame["lat"]),
            crs="EPSG:4326",
        )
    )


def _park_union_m(parks: gpd.GeoDataFrame | None) -> BaseGeometry | None:
    if parks is None or parks.empty:
        return None
    parks_m = parks.to_crs(_PROJECTED_CRS)
    union = unary_union(parks_m.geometry.values)
    return union if isinstance(union, BaseGeometry) else None


def snap_listings_to_buildings(
    listings: pd.DataFrame,
    *,
    parks: gpd.GeoDataFrame | None = None,
    buildings: gpd.GeoDataFrame | None = None,
    cache_dir: Path | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    max_snap_m: float = _DEFAULT_MAX_SNAP_M,
    force_refresh_buildings: bool = False,
) -> pd.DataFrame:
    """Move park / road / sidewalk pins onto the nearest OSM building.

    Args:
        listings: Rows with ``lat`` / ``lon``.
        parks: Optional park polygons (WGS84 or projected).
        buildings: Optional building footprints or centroids; OSM when None.
        cache_dir: Required when ``buildings`` is None.
        bbox: Required when fetching buildings.
        max_snap_m: Skip snap when nearest outside building is farther.
        force_refresh_buildings: Ignore building cache.

    Returns:
        Copy of ``listings`` with updated coords and ``location_snapped``.
    """
    out = listings.copy()
    out["location_snapped"] = False
    if out.empty:
        return out

    parks_frame = (
        parks if parks is not None else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    )
    inside_park = listing_points_inside_parks(out, parks_frame)

    bld = buildings
    if bld is None:
        if cache_dir is None or bbox is None:
            if bool(inside_park.any()):
                logger.warning(
                    "Cannot snap %s park pins: buildings/cache/bbox missing",
                    int(inside_park.sum()),
                )
            return out
        bld = fetch_osm_building_centroids(
            bbox,
            cache_dir,
            force_refresh=force_refresh_buildings,
        )
    if bld is None or bld.empty:
        logger.warning("No OSM buildings available for location snap")
        return out

    outside_bld = listing_points_outside_buildings(out, bld)
    need = inside_park | outside_bld
    n_need = int(need.sum())
    if n_need == 0:
        logger.info("No misplaced listing pins — location snap skipped")
        return out

    park_union = _park_union_m(parks_frame)
    bld_m = bld.to_crs(_PROJECTED_CRS).copy()
    bld_m["geometry"] = bld_m.geometry.centroid
    if park_union is not None:
        outside_mask = ~bld_m.geometry.within(park_union)
        candidates = bld_m.loc[outside_mask].copy()
    else:
        candidates = bld_m
    if candidates.empty:
        logger.warning("All OSM buildings lie inside parks — snap skipped")
        return out
    candidates = candidates.reset_index(drop=True)
    candidates["bld_idx"] = np.arange(len(candidates))

    bad = gpd.GeoDataFrame(
        out.loc[need, ["lat", "lon"]].copy(),
        geometry=gpd.points_from_xy(
            out.loc[need, "lon"],
            out.loc[need, "lat"],
        ),
        crs="EPSG:4326",
        index=out.loc[need].index,
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
        "Snapped %s / %s misplaced listing pins onto OSM buildings "
        "(park=%s, outside_building=%s)",
        snapped,
        n_need,
        int(inside_park.sum()),
        int(outside_bld.sum()),
    )
    return out


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
    """Snap park-interior and road/sidewalk pins onto nearest OSM buildings.

    Kept as the pipeline entrypoint name; delegates to
    :func:`snap_listings_to_buildings`.
    """
    return snap_listings_to_buildings(
        listings,
        parks=parks,
        buildings=buildings,
        cache_dir=cache_dir,
        bbox=bbox,
        max_snap_m=max_snap_m,
        force_refresh_buildings=force_refresh_buildings,
    )
