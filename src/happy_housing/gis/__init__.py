"""Spatial enrichment: noise bands, amenity / school / shop distances, transit."""

import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from happy_housing.gis.amenity_catalog import AMENITY_TYPES, dist_column
from happy_housing.gis.bike_paths import (
    BIKE_PATH_COVERAGE_RADIUS_M,
    attach_bike_path_features,
)
from happy_housing.gis.frame_util import assign_columns
from happy_housing.gis.traffic import (
    TRAFFIC_NEIGHBOURHOOD_RADIUS_M,
    attach_traffic_features,
)

logger = logging.getLogger(__name__)

AIRCRAFT_LDEN_EXPOSURE_DB: float = 55.0
"""END reporting threshold used as binary high-exposure flag (``in_oou``)."""

OSM_CACHE_SCHEMA_VERSION: int = 1
"""Bump when default OSM tag scope or cache layout changes."""

DEFAULT_OSM_AMENITY_TAGS: dict[str, bool | str | list[str]] = {
    "amenity": True,
    "station": "subway",
}
"""Overpass tags: every ``amenity=*`` plus subway stations in the bbox."""

DEFAULT_OSM_SHOP_TAGS: dict[str, bool | str | list[str]] = {
    "shop": True,
}
"""Overpass tags: every ``shop=*`` POI in the bbox."""

DEFAULT_OSM_FITNESS_TAGS: dict[str, bool | str | list[str]] = {
    "leisure": "fitness_centre",
    "amenity": "gym",
}
"""Commercial indoor gyms / clubs (``leisure=fitness_centre``, ``amenity=gym``)."""

DEFAULT_OSM_GREEN_TAGS: dict[str, bool | str | list[str]] = {
    "leisure": ["park", "garden", "nature_reserve"],
    "landuse": ["forest", "recreation_ground"],
    "natural": "wood",
}
"""Public green polygons beyond UM PLACE_SKWERY (parks, woods, forests)."""

CITY_POIS_SCHEMA_VERSION: int = 2
"""Bump when combined city-POI merge (sports + civic + OSM fitness) changes."""

_OSM_OVERPASS_TIMEOUT_S: int = 300
"""Overpass / OSMnx timeout for city-wide amenity and shop queries."""


def _osm_meta_path(cache_path: Path) -> Path:
    """Sidecar JSON path that fingerprints tag filter + schema version."""
    return cache_path.with_name(cache_path.stem + ".meta.json")


def _tags_fingerprint(tags: dict[str, bool | str | list[str]]) -> str:
    """Stable JSON fingerprint for an OSMnx tag filter."""
    return json.dumps(tags, sort_keys=True, separators=(",", ":"))


def _osm_cache_is_current(
    cache_path: Path,
    tags: dict[str, bool | str | list[str]],
) -> bool:
    """True when GeoJSON exists and meta matches current tags + schema."""
    if not cache_path.exists():
        return False
    meta_path = _osm_meta_path(cache_path)
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return int(meta.get("schema_version", -1)) == OSM_CACHE_SCHEMA_VERSION and str(
        meta.get("tags_fingerprint", "")
    ) == _tags_fingerprint(tags)


def _write_osm_cache_meta(
    cache_path: Path,
    tags: dict[str, bool | str | list[str]],
    *,
    row_count: int,
) -> None:
    """Persist cache fingerprint next to a GeoJSON artifact."""
    payload: dict[str, Any] = {
        "schema_version": OSM_CACHE_SCHEMA_VERSION,
        "tags_fingerprint": _tags_fingerprint(tags),
        "tags": tags,
        "row_count": int(row_count),
    }
    _osm_meta_path(cache_path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def listings_to_gdf(
    listings: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Convert listing rows with lon/lat columns into a GeoDataFrame."""
    geometry = gpd.points_from_xy(listings[lon_col], listings[lat_col])
    return gpd.GeoDataFrame(listings.copy(), geometry=geometry, crs=crs)


def load_noise_contours(path: Path, value_col: str) -> gpd.GeoDataFrame:
    """Load END-style noise contour polygons with a numeric band column."""
    if not path.exists():
        return gpd.GeoDataFrame(
            columns=[value_col, "geometry"], geometry=[], crs="EPSG:4326"
        )
    frame = gpd.read_file(path)
    if frame.empty or value_col not in frame.columns:
        return gpd.GeoDataFrame(
            columns=[value_col, "geometry"], geometry=[], crs="EPSG:4326"
        )
    out = frame[[value_col, "geometry"]].copy()
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out[out[value_col].notna() & out.geometry.notna() & ~out.geometry.is_empty]
    return gpd.GeoDataFrame(out, geometry="geometry", crs=frame.crs or "EPSG:4326")


def attach_contour_level(
    listings: gpd.GeoDataFrame,
    contours: gpd.GeoDataFrame,
    value_col: str,
    out_col: str,
) -> gpd.GeoDataFrame:
    """Assign max contour band containing each listing; missing → 0."""
    result = listings.copy()
    if contours.empty or value_col not in contours.columns:
        result[out_col] = 0.0
        return result
    right = contours[[value_col, "geometry"]].copy()
    if result.crs is not None and right.crs is not None and result.crs != right.crs:
        right = right.to_crs(result.crs)
    joined = gpd.sjoin(result, right, how="left", predicate="within")
    levels = (
        joined
        .groupby(joined.index, sort=False)[value_col]
        .max()
        .reindex(result.index)
        .fillna(0.0)
        .astype(float)
    )
    result[out_col] = levels.to_numpy()
    return result


def attach_aircraft_noise_levels(
    listings: gpd.GeoDataFrame,
    *,
    lden: gpd.GeoDataFrame,
    lnight: gpd.GeoDataFrame,
    exposure_db: float = AIRCRAFT_LDEN_EXPOSURE_DB,
) -> gpd.GeoDataFrame:
    """Attach ``aircraft_lden_db``, ``aircraft_lnight_db``, and exposure flag."""
    out = attach_contour_level(listings, lden, "lden_db", "aircraft_lden_db")
    out = attach_contour_level(out, lnight, "lnight_db", "aircraft_lnight_db")
    out["in_oou"] = (out["aircraft_lden_db"] >= float(exposure_db)).astype(int)
    return out


def nearest_distance_meters(
    listings: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    projected_crs: str,
    out_col: str,
    *,
    attr_cols: tuple[str, ...] | list[str] = (),
    attr_rename: dict[str, str] | None = None,
    left_projected: gpd.GeoDataFrame | None = None,
    copy: bool = True,
    target: dict[str, Any] | None = None,
) -> gpd.GeoDataFrame:
    """Distance from each listing to nearest point; optionally copy attributes.

    Pass ``left_projected`` (listings already in ``projected_crs``) and
    ``copy=False`` from enrich loops to avoid reprojecting / copying the full
    frame on every amenity type.

    When ``target`` is set, write new columns into that dict instead of
    mutating ``listings`` (caller joins once via :func:`assign_columns`).
    """
    rename = dict(attr_rename or {})
    wanted = [c for c in attr_cols if c in points.columns]
    out = listings.copy() if copy and target is None else listings
    n = len(listings)

    def _empty_values() -> dict[str, Any]:
        vals: dict[str, Any] = {out_col: np.full(n, np.nan, dtype=float)}
        for col in wanted:
            vals[rename.get(col, col)] = np.full(n, np.nan, dtype=object)
        return vals

    if points.empty:
        values = _empty_values()
        if target is not None:
            target.update(values)
            return out
        if copy:
            return assign_columns(out, values)
        for key, value in values.items():
            out[key] = value
        return out

    left = (
        left_projected if left_projected is not None else listings.to_crs(projected_crs)
    )
    right_cols = ["geometry", *wanted]
    right = points.to_crs(projected_crs)[right_cols]
    matched = left.sjoin_nearest(right, how="left", distance_col=out_col)
    matched = matched[~matched.index.duplicated(keep="first")]
    aligned = matched.reindex(listings.index)
    filled: dict[str, Any] = {out_col: aligned[out_col].to_numpy()}
    for col in wanted:
        dest = rename.get(col, col)
        filled[dest] = aligned[col].to_numpy()
    if target is not None:
        target.update(filled)
        return out
    if copy:
        return assign_columns(out, filled)
    for key, value in filled.items():
        out[key] = value
    return out


_GIOS_AQ_ATTRS: tuple[str, ...] = (
    "aq_index",
    "pm10_ug_m3",
    "pm25_ug_m3",
    "no2_ug_m3",
)
"""Station AQ columns copied onto listings from the nearest GIOŚ station."""

_GIOS_AQ_RENAME: dict[str, str] = {
    "aq_index": "air_aq_index",
    "pm10_ug_m3": "air_pm10_ug_m3",
    "pm25_ug_m3": "air_pm25_ug_m3",
    "no2_ug_m3": "air_no2_ug_m3",
}
"""Listing feature names for nearest-station air quality."""


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
    tags: dict[str, bool | str | list[str]] | None = None,
    fixture_path: Path | None = None,
    *,
    force_refresh: bool = False,
) -> gpd.GeoDataFrame:
    """Download all amenities (plus subway) via OSMnx; fixture on Overpass failure."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "osm_amenities.geojson"
    tag_filter: dict[str, bool | str | list[str]] = dict(
        tags if tags is not None else DEFAULT_OSM_AMENITY_TAGS
    )
    if not force_refresh and _osm_cache_is_current(cache_path, tag_filter):
        return gpd.read_file(cache_path)

    west, south, east, north = bbox
    frame = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    try:
        import osmnx as ox

        ox.settings.use_cache = True
        ox.settings.cache_folder = str(cache_dir / "osmnx")
        ox.settings.timeout = _OSM_OVERPASS_TIMEOUT_S
        features = ox.features_from_bbox(
            bbox=(west, south, east, north),
            tags=tag_filter,
        )
        if not features.empty:
            frame = features.reset_index(drop=False)
            projected = frame.to_crs("EPSG:2180")
            frame["geometry"] = projected.geometry.centroid.to_crs("EPSG:4326")
            frame = gpd.GeoDataFrame(frame, geometry="geometry", crs="EPSG:4326")
            if "amenity" not in frame.columns:
                frame["amenity"] = None
            if "station" in frame.columns:
                subway = frame["station"].astype(str).eq("subway")
                frame.loc[subway, "amenity"] = "subway"
            keep = [
                c
                for c in (
                    "amenity",
                    "station",
                    "railway",
                    "name",
                    "brand",
                    "operator",
                    "opening_hours",
                    "geometry",
                )
                if c in frame.columns
            ]
            frame = frame[keep]
            frame = frame[frame["amenity"].notna()]
    except Exception as exc:
        logger.warning("OSMnx amenity fetch failed: %s", exc)

    if frame.empty and fixture_path is not None and fixture_path.exists():
        logger.warning("Using amenity fixture %s", fixture_path)
        frame = load_amenity_points(fixture_path)

    if not frame.empty:
        frame.to_file(cache_path, driver="GeoJSON")
        _write_osm_cache_meta(cache_path, tag_filter, row_count=len(frame))
        logger.info("Cached %s amenities to %s", len(frame), cache_path)
    return frame


def fetch_osm_shops(
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
    fixture_path: Path | None = None,
    tags: dict[str, bool | str | list[str]] | None = None,
    *,
    force_refresh: bool = False,
) -> gpd.GeoDataFrame:
    """Download all shop POIs via OSMnx; fall back to fixture if Overpass fails."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "osm_shops.geojson"
    tag_filter: dict[str, bool | str | list[str]] = dict(
        tags if tags is not None else DEFAULT_OSM_SHOP_TAGS
    )
    if not force_refresh and _osm_cache_is_current(cache_path, tag_filter):
        return gpd.read_file(cache_path)

    west, south, east, north = bbox
    frame = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    try:
        import osmnx as ox

        ox.settings.use_cache = True
        ox.settings.cache_folder = str(cache_dir / "osmnx")
        ox.settings.timeout = _OSM_OVERPASS_TIMEOUT_S
        features = ox.features_from_bbox(
            bbox=(west, south, east, north),
            tags=tag_filter,
        )
        if not features.empty:
            frame = features.reset_index(drop=False)
            projected = frame.to_crs("EPSG:2180")
            frame["geometry"] = projected.geometry.centroid.to_crs("EPSG:4326")
            frame = gpd.GeoDataFrame(frame, geometry="geometry", crs="EPSG:4326")
            keep = [
                c
                for c in (
                    "shop",
                    "name",
                    "brand",
                    "operator",
                    "opening_hours",
                    "geometry",
                )
                if c in frame.columns
            ]
            frame = frame[keep]
            if "shop" in frame.columns:
                frame = frame[frame["shop"].notna()]
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
        _write_osm_cache_meta(cache_path, tag_filter, row_count=len(frame))
        logger.info("Cached %s shops to %s", len(frame), cache_path)
    return frame


def osm_fitness_to_city_pois(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Normalize OSM fitness features to city-POI columns."""
    empty = gpd.GeoDataFrame(
        columns=["object_type", "name", "source", "geometry"],
        geometry=[],
        crs="EPSG:4326",
    )
    if frame.empty:
        return empty
    out = frame.copy()
    if out.crs is None:
        out = out.set_crs("EPSG:4326")
    elif str(out.crs) != "EPSG:4326":
        out = out.to_crs("EPSG:4326")
    names: list[str] = []
    for _, row in out.iterrows():
        label = ""
        for key in ("name", "brand", "operator"):
            raw = row.get(key)
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                continue
            text = str(raw).strip()
            if text and text.lower() not in {"nan", "none"}:
                label = text
                break
        names.append(label or "Fitness centre")
    result = gpd.GeoDataFrame(
        {
            "object_type": ["gym"] * len(out),
            "name": names,
            "source": ["osm"] * len(out),
            "geometry": out.geometry.values,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    return result[~result.geometry.isna() & ~result.geometry.is_empty].reset_index(
        drop=True
    )


def fetch_osm_fitness_pois(
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
    tags: dict[str, bool | str | list[str]] | None = None,
    *,
    force_refresh: bool = False,
) -> gpd.GeoDataFrame:
    """Download commercial fitness centres; return city-POI-shaped points.

    Kept separate from ``fetch_osm_amenities`` because that path filters to
    rows with a non-null ``amenity`` column and drops leisure-only POIs.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "osm_fitness.geojson"
    tag_filter: dict[str, bool | str | list[str]] = dict(
        tags if tags is not None else DEFAULT_OSM_FITNESS_TAGS
    )
    if not force_refresh and _osm_cache_is_current(cache_path, tag_filter):
        cached = gpd.read_file(cache_path)
        if {"object_type", "name", "source"}.issubset(cached.columns):
            return cached
        return osm_fitness_to_city_pois(cached)

    west, south, east, north = bbox
    frame = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    try:
        import osmnx as ox

        ox.settings.use_cache = True
        ox.settings.cache_folder = str(cache_dir / "osmnx")
        ox.settings.timeout = _OSM_OVERPASS_TIMEOUT_S
        features = ox.features_from_bbox(
            bbox=(west, south, east, north),
            tags=tag_filter,
        )
        if not features.empty:
            frame = features.reset_index(drop=False)
            projected = frame.to_crs("EPSG:2180")
            frame["geometry"] = projected.geometry.centroid.to_crs("EPSG:4326")
            frame = gpd.GeoDataFrame(frame, geometry="geometry", crs="EPSG:4326")
            keep = [
                c
                for c in (
                    "leisure",
                    "amenity",
                    "name",
                    "brand",
                    "operator",
                    "geometry",
                )
                if c in frame.columns
            ]
            frame = frame[keep]
    except Exception as exc:
        logger.warning("OSMnx fitness fetch failed: %s", exc)

    pois = osm_fitness_to_city_pois(frame)
    if not pois.empty:
        pois.to_file(cache_path, driver="GeoJSON")
        _write_osm_cache_meta(cache_path, tag_filter, row_count=len(pois))
        logger.info("Cached %s OSM fitness POIs → %s", len(pois), cache_path)
    return pois


def fetch_osm_green_areas(
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
    tags: dict[str, bool | str | list[str]] | None = None,
    *,
    force_refresh: bool = False,
) -> gpd.GeoDataFrame:
    """Download OSM green polygons (parks, woods, forests); keep area geometry.

    Complements UM ``PLACE_SKWERY``, which covers official parks/squares only
    and omits large forests and many community greens.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "osm_green_areas.geojson"
    tag_filter: dict[str, bool | str | list[str]] = dict(
        tags if tags is not None else DEFAULT_OSM_GREEN_TAGS
    )
    if not force_refresh and _osm_cache_is_current(cache_path, tag_filter):
        return gpd.read_file(cache_path)

    west, south, east, north = bbox
    frame = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    try:
        import osmnx as ox

        ox.settings.use_cache = True
        ox.settings.cache_folder = str(cache_dir / "osmnx")
        ox.settings.timeout = _OSM_OVERPASS_TIMEOUT_S
        features = ox.features_from_bbox(
            bbox=(west, south, east, north),
            tags=tag_filter,
        )
        if not features.empty:
            frame = features.reset_index(drop=False)
            if frame.crs is None:
                frame = frame.set_crs("EPSG:4326")
            elif str(frame.crs) != "EPSG:4326":
                frame = frame.to_crs("EPSG:4326")
            # Polygons only — points/lines are not useful park areas.
            frame = frame[frame.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
            keep = [
                c
                for c in (
                    "name",
                    "leisure",
                    "landuse",
                    "natural",
                    "geometry",
                )
                if c in frame.columns
            ]
            frame = gpd.GeoDataFrame(frame[keep], geometry="geometry", crs="EPSG:4326")
            if "name" not in frame.columns:
                frame["name"] = None
            frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty]
    except Exception as exc:
        logger.warning("OSMnx green-area fetch failed: %s", exc)

    if not frame.empty:
        frame.to_file(cache_path, driver="GeoJSON")
        _write_osm_cache_meta(cache_path, tag_filter, row_count=len(frame))
        logger.info("Cached %s OSM green areas → %s", len(frame), cache_path)
    return frame


def _points_for_nearest(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Use centroids when features are polygons (school catchments, parks)."""
    if gdf.empty:
        return gdf
    out = gdf.copy()
    geom_types = set(out.geometry.geom_type.unique())
    if geom_types <= {"Point", "MultiPoint"}:
        return out
    projected = out.to_crs("EPSG:2180")
    out["geometry"] = projected.geometry.centroid.to_crs(out.crs or "EPSG:4326")
    return gpd.GeoDataFrame(out, geometry="geometry", crs=out.crs or "EPSG:4326")


def load_shop_points(path: Path) -> gpd.GeoDataFrame:
    """Load shop points from a JSON list with lon/lat/shop fields."""
    frame = pd.read_json(path)
    return gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame["lon"], frame["lat"]),
        crs="EPSG:4326",
    )


def enrich_listings(
    listings: pd.DataFrame,
    *,
    projected_crs: str,
    amenities: gpd.GeoDataFrame,
    gios_stations: pd.DataFrame,
    gtfs_stops: pd.DataFrame,
    commute_lon: float,
    commute_lat: float,
    gunb_points: gpd.GeoDataFrame | None = None,
    parks: gpd.GeoDataFrame | None = None,
    schools: gpd.GeoDataFrame | None = None,
    shops: gpd.GeoDataFrame | None = None,
    city_pois: gpd.GeoDataFrame | None = None,
    bike_paths: gpd.GeoDataFrame | None = None,
    bike_path_coverage_radius_m: float = BIKE_PATH_COVERAGE_RADIUS_M,
    bike_path_sample: pd.DataFrame | None = None,
    rail_areas: gpd.GeoDataFrame | None = None,
    military: gpd.GeoDataFrame | None = None,
    airport_areas: gpd.GeoDataFrame | None = None,
    traffic_sample: pd.DataFrame | None = None,
    traffic_neighbourhood_radius_m: float = TRAFFIC_NEIGHBOURHOOD_RADIUS_M,
    aircraft_lden_path: Path | None = None,
    aircraft_lnight_path: Path | None = None,
    road_lden_path: Path | None = None,
    rail_lden_path: Path | None = None,
) -> gpd.GeoDataFrame:
    """Attach noise, amenities, zones, traffic, transit, air, and GUNB."""
    gdf = listings_to_gdf(listings)
    lden = (
        load_noise_contours(aircraft_lden_path, "lden_db")
        if aircraft_lden_path is not None
        else gpd.GeoDataFrame(
            columns=["lden_db", "geometry"], geometry=[], crs="EPSG:4326"
        )
    )
    lnight = (
        load_noise_contours(aircraft_lnight_path, "lnight_db")
        if aircraft_lnight_path is not None
        else gpd.GeoDataFrame(
            columns=["lnight_db", "geometry"], geometry=[], crs="EPSG:4326"
        )
    )
    gdf = attach_aircraft_noise_levels(gdf, lden=lden, lnight=lnight)

    road = (
        load_noise_contours(road_lden_path, "lden_db")
        if road_lden_path is not None
        else gpd.GeoDataFrame(
            columns=["lden_db", "geometry"], geometry=[], crs="EPSG:4326"
        )
    )
    rail = (
        load_noise_contours(rail_lden_path, "lden_db")
        if rail_lden_path is not None
        else gpd.GeoDataFrame(
            columns=["lden_db", "geometry"], geometry=[], crs="EPSG:4326"
        )
    )
    gdf = attach_contour_level(gdf, road, "lden_db", "road_lden_db")
    gdf = attach_contour_level(gdf, rail, "lden_db", "rail_lden_db")

    # One projected frame for all nearest-distance joins (avoids Nx reproject/copy).
    left_proj = gdf.to_crs(projected_crs)
    # Collect distance / AQ columns and join once (avoids fragmented inserts).
    extras: dict[str, Any] = {}

    def _near(
        points: gpd.GeoDataFrame,
        col: str,
        *,
        attr_cols: tuple[str, ...] | list[str] = (),
        attr_rename: dict[str, str] | None = None,
    ) -> None:
        nearest_distance_meters(
            gdf,
            points,
            projected_crs,
            col,
            attr_cols=attr_cols,
            attr_rename=attr_rename,
            left_projected=left_proj,
            copy=False,
            target=extras,
        )

    amenity_cols = (
        ("hospital", "dist_hospital_m"),
        ("fire_station", "dist_fire_station_m"),
        ("police", "dist_police_m"),
        ("subway", "dist_metro_m"),
        ("charging_station", "dist_charging_station_m"),
        ("fuel", "dist_fuel_m"),
    )
    for amenity_value, col in amenity_cols:
        if not amenities.empty and "amenity" in amenities.columns:
            subset = amenities[amenities["amenity"] == amenity_value]
        else:
            subset = amenities.iloc[0:0]
        _near(subset, col)

    if parks is not None and not parks.empty:
        _near(parks, "dist_park_m")
    else:
        extras["dist_park_m"] = float("nan")

    paths = (
        bike_paths
        if bike_paths is not None
        else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    )
    gdf = attach_bike_path_features(
        gdf,
        paths,
        projected_crs=projected_crs,
        coverage_radius_m=float(bike_path_coverage_radius_m),
        sample_points=bike_path_sample,
        left_projected=left_proj,
    )

    if rail_areas is not None and not rail_areas.empty:
        _near(rail_areas, "dist_rail_area_m")
    else:
        extras["dist_rail_area_m"] = float("nan")

    if military is not None and not military.empty:
        _near(military, "dist_military_m")
    else:
        extras["dist_military_m"] = float("nan")

    if airport_areas is not None and not airport_areas.empty:
        _near(airport_areas, "dist_airport_area_m")
    else:
        extras["dist_airport_area_m"] = float("nan")

    traffic = traffic_sample if traffic_sample is not None else pd.DataFrame()
    gdf = attach_traffic_features(
        gdf,
        traffic,
        projected_crs=projected_crs,
        neighbourhood_radius_m=float(traffic_neighbourhood_radius_m),
        left_projected=left_proj,
    )

    school_src: gpd.GeoDataFrame | None = None
    if schools is not None and not schools.empty:
        school_src = _points_for_nearest(schools)
    elif not amenities.empty and "amenity" in amenities.columns:
        osm_schools = amenities[amenities["amenity"] == "school"]
        if not osm_schools.empty:
            school_src = _points_for_nearest(osm_schools)
    if school_src is not None:
        _near(school_src, "dist_school_m")
    else:
        extras["dist_school_m"] = float("nan")

    if shops is not None and not shops.empty:
        _near(_points_for_nearest(shops), "dist_shop_m")
    else:
        extras["dist_shop_m"] = float("nan")

    if (
        city_pois is not None
        and not city_pois.empty
        and "object_type" in city_pois.columns
    ):
        typed = _points_for_nearest(city_pois)
        city_keys = {
            t.key
            for t in AMENITY_TYPES
            if t.source in {"sports", "wfs"} and t.key not in {"school"}
        }
        present = set(typed["object_type"].astype(str).unique())
        for key in sorted(city_keys):
            col = dist_column(key)
            if key not in present:
                extras[col] = float("nan")
                continue
            subset = typed[typed["object_type"].astype(str) == key]
            _near(subset, col)
    else:
        for t in AMENITY_TYPES:
            if t.source in {"sports", "wfs"} and t.key != "school":
                extras[dist_column(t.key)] = float("nan")

    if not gios_stations.empty:
        air = gpd.GeoDataFrame(
            gios_stations,
            geometry=gpd.points_from_xy(gios_stations["lon"], gios_stations["lat"]),
            crs="EPSG:4326",
        )
        _near(
            air,
            "dist_air_station_m",
            attr_cols=_GIOS_AQ_ATTRS,
            attr_rename=_GIOS_AQ_RENAME,
        )
    else:
        extras["dist_air_station_m"] = float("nan")
        for col in _GIOS_AQ_RENAME.values():
            extras[col] = float("nan")

    if not gtfs_stops.empty:
        stops = gpd.GeoDataFrame(
            gtfs_stops,
            geometry=gpd.points_from_xy(
                gtfs_stops["stop_lon"],
                gtfs_stops["stop_lat"],
            ),
            crs="EPSG:4326",
        )
        _near(stops, "dist_transit_stop_m")
    else:
        extras["dist_transit_stop_m"] = float("nan")

    if gunb_points is not None and not gunb_points.empty:
        _near(gunb_points, "dist_gunb_m")
    else:
        extras["dist_gunb_m"] = float("nan")

    target = gpd.GeoDataFrame(
        geometry=[Point(commute_lon, commute_lat)],
        crs="EPSG:4326",
    )
    _near(target, "dist_daszynskiego_m")
    return assign_columns(gdf, extras)
