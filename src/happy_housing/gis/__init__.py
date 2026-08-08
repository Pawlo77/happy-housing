"""Spatial enrichment: OOU flags, amenity distances, air and transit proxies."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)

_DEFAULT_OOU_COORDS: list[tuple[float, float]] = [
    (20.90, 52.14),
    (21.02, 52.14),
    (21.02, 52.20),
    (20.90, 52.20),
    (20.90, 52.14),
]
"""Rough Chopin OOU rectangle used when no official polygon file is present."""


def listings_to_gdf(
    listings: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Convert listing rows with lon/lat columns into a GeoDataFrame."""
    geometry = gpd.points_from_xy(listings[lon_col], listings[lat_col])
    return gpd.GeoDataFrame(listings.copy(), geometry=geometry, crs=crs)


def load_or_create_oou_polygon(path: Path) -> gpd.GeoDataFrame:
    """Load an OOU polygon GeoJSON or write the default Chopin bootstrap polygon."""
    if path.exists():
        return gpd.read_file(path)
    polygon = Polygon(_DEFAULT_OOU_COORDS)
    frame = gpd.GeoDataFrame(
        {"zone": ["Z_bootstrap"], "name": ["Chopin OOU bootstrap"]},
        geometry=[polygon],
        crs="EPSG:4326",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(path, driver="GeoJSON")
    logger.info("Wrote bootstrap OOU polygon to %s", path)
    return frame


def flag_oou_membership(
    listings: gpd.GeoDataFrame,
    oou: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Add ``in_oou`` via spatial join (vectorized)."""
    joined = gpd.sjoin(
        listings,
        oou[["geometry"]],
        how="left",
        predicate="within",
    )
    joined["in_oou"] = joined["index_right"].notna().astype(int)
    return joined.drop(columns=["index_right"], errors="ignore")


def nearest_distance_meters(
    listings: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    projected_crs: str,
    out_col: str,
) -> gpd.GeoDataFrame:
    """Distance from each listing to nearest point geometry (meters)."""
    if points.empty:
        result = listings.copy()
        result[out_col] = float("nan")
        return result
    left = listings.to_crs(projected_crs)
    right = points.to_crs(projected_crs)
    matched = left.sjoin_nearest(right[["geometry"]], how="left", distance_col=out_col)
    matched = matched[~matched.index.duplicated(keep="first")]
    out = listings.copy()
    out[out_col] = matched.reindex(listings.index)[out_col].to_numpy()
    return out


def load_amenity_points(path: Path) -> gpd.GeoDataFrame:
    """Load amenity points from a JSON list with lon/lat/amenity fields."""
    frame = pd.read_json(path)
    return gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame["lon"], frame["lat"]),
        crs="EPSG:4326",
    )


def fetch_osm_amenities(
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
    tags: dict[str, str | list[str]] | None = None,
    fixture_path: Path | None = None,
) -> gpd.GeoDataFrame:
    """Download amenities via OSMnx; fall back to fixture if Overpass fails."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "osm_amenities.geojson"
    if cache_path.exists():
        return gpd.read_file(cache_path)

    tag_filter: dict[str, str | list[str]] = tags or {
        "amenity": ["hospital", "fire_station", "police"],
    }
    west, south, east, north = bbox
    frame = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    try:
        import osmnx as ox

        ox.settings.use_cache = True
        ox.settings.cache_folder = str(cache_dir / "osmnx")
        ox.settings.timeout = 90
        features = ox.features_from_bbox(
            bbox=(west, south, east, north),
            tags=tag_filter,
        )
        if not features.empty:
            frame = features.reset_index(drop=False)
            projected = frame.to_crs("EPSG:2180")
            frame["geometry"] = projected.geometry.centroid.to_crs("EPSG:4326")
            frame = gpd.GeoDataFrame(frame, geometry="geometry", crs="EPSG:4326")
            keep = [c for c in ("amenity", "name", "geometry") if c in frame.columns]
            frame = frame[keep]
    except Exception as exc:
        logger.warning("OSMnx amenity fetch failed: %s", exc)

    if frame.empty and fixture_path is not None and fixture_path.exists():
        logger.warning("Using amenity fixture %s", fixture_path)
        frame = load_amenity_points(fixture_path)

    if not frame.empty:
        frame.to_file(cache_path, driver="GeoJSON")
        logger.info("Cached %s amenities to %s", len(frame), cache_path)
    return frame


def fetch_osm_shops(
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
    fixture_path: Path | None = None,
) -> gpd.GeoDataFrame:
    """Download retail POIs via OSMnx; fall back to fixture if Overpass fails."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "osm_shops.geojson"
    if cache_path.exists():
        return gpd.read_file(cache_path)

    west, south, east, north = bbox
    frame = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    try:
        import osmnx as ox

        ox.settings.use_cache = True
        ox.settings.cache_folder = str(cache_dir / "osmnx")
        ox.settings.timeout = 90
        features = ox.features_from_bbox(
            bbox=(west, south, east, north),
            tags={
                "shop": [
                    "supermarket",
                    "convenience",
                    "mall",
                    "bakery",
                    "chemist",
                ]
            },
        )
        if not features.empty:
            frame = features.reset_index(drop=False)
            projected = frame.to_crs("EPSG:2180")
            frame["geometry"] = projected.geometry.centroid.to_crs("EPSG:4326")
            frame = gpd.GeoDataFrame(frame, geometry="geometry", crs="EPSG:4326")
            keep = [c for c in ("shop", "name", "geometry") if c in frame.columns]
            frame = frame[keep]
    except Exception as exc:
        logger.warning("OSMnx shop fetch failed: %s", exc)

    if frame.empty and fixture_path is not None and fixture_path.exists():
        logger.warning("Using shop fixture %s", fixture_path)
        raw = pd.read_json(fixture_path)
        frame = gpd.GeoDataFrame(
            raw,
            geometry=gpd.points_from_xy(raw["lon"], raw["lat"]),
            crs="EPSG:4326",
        )

    if not frame.empty:
        frame.to_file(cache_path, driver="GeoJSON")
        logger.info("Cached %s shops to %s", len(frame), cache_path)
    return frame


def enrich_listings(
    listings: pd.DataFrame,
    *,
    projected_crs: str,
    oou_path: Path,
    amenities: gpd.GeoDataFrame,
    gios_stations: pd.DataFrame,
    gtfs_stops: pd.DataFrame,
    commute_lon: float,
    commute_lat: float,
    gunb_points: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Attach OOU, amenity, air, transit, GUNB, and commute-distance features."""
    gdf = listings_to_gdf(listings)
    oou = load_or_create_oou_polygon(oou_path)
    gdf = flag_oou_membership(gdf, oou)

    amenity_cols = (
        ("hospital", "dist_hospital_m"),
        ("fire_station", "dist_fire_station_m"),
        ("police", "dist_police_m"),
    )
    for amenity_value, col in amenity_cols:
        if not amenities.empty and "amenity" in amenities.columns:
            subset = amenities[amenities["amenity"] == amenity_value]
        else:
            subset = amenities.iloc[0:0]
        gdf = nearest_distance_meters(gdf, subset, projected_crs, col)

    if not gios_stations.empty:
        air = gpd.GeoDataFrame(
            gios_stations,
            geometry=gpd.points_from_xy(gios_stations["lon"], gios_stations["lat"]),
            crs="EPSG:4326",
        )
        gdf = nearest_distance_meters(gdf, air, projected_crs, "dist_air_station_m")
    else:
        gdf["dist_air_station_m"] = float("nan")

    if not gtfs_stops.empty:
        stops = gpd.GeoDataFrame(
            gtfs_stops,
            geometry=gpd.points_from_xy(
                gtfs_stops["stop_lon"],
                gtfs_stops["stop_lat"],
            ),
            crs="EPSG:4326",
        )
        gdf = nearest_distance_meters(gdf, stops, projected_crs, "dist_transit_stop_m")
    else:
        gdf["dist_transit_stop_m"] = float("nan")

    if gunb_points is not None and not gunb_points.empty:
        gdf = nearest_distance_meters(gdf, gunb_points, projected_crs, "dist_gunb_m")
    else:
        gdf["dist_gunb_m"] = float("nan")

    target = gpd.GeoDataFrame(
        geometry=[Point(commute_lon, commute_lat)],
        crs="EPSG:4326",
    )
    gdf = nearest_distance_meters(gdf, target, projected_crs, "dist_daszynskiego_m")
    return gdf
