"""Shared data loading helpers for Streamlit pages."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from happy_housing.app.io_cache import clear_io_caches, file_mtime, read_geojson_gdf
from happy_housing.app.listing_attrs import promote_listing_flags
from happy_housing.app.metadata import RunMetadata, load_run_metadata
from happy_housing.config import ExperimentConfig, load_experiment_config
from happy_housing.ingest import build_gtfs_stop_hourly
from happy_housing.ingest.market import backfill_market_primary

if TYPE_CHECKING:
    import geopandas as gpd
    import lightgbm as lgb
    import shap

    from happy_housing.gis.gios_aq_field import AqMapOverlay
    from happy_housing.gis.warsaw_noise import NoiseMapOverlay

logger = logging.getLogger(__name__)


def config_path() -> Path:
    """Default experiment YAML path."""
    return Path("configs/experiment.yaml")


@st.cache_data(show_spinner=False)  # type: ignore[misc]
def _load_config_cached(path: str, mtime: float) -> ExperimentConfig:
    """Cached experiment config keyed by YAML path + mtime."""
    del mtime
    return load_experiment_config(Path(path))


def load_config() -> ExperimentConfig:
    """Load experiment config (mtime-keyed Streamlit cache)."""
    path = config_path()
    return _load_config_cached(  # type: ignore[no-any-return]
        str(path.resolve()), file_mtime(path)
    )


def processed_dir() -> Path:
    """Processed data directory from config."""
    return load_config().data.processed_path()


def scored_path() -> Path:
    """Scored listings parquet path."""
    return processed_dir() / "listings_scored.parquet"


def metadata_path() -> Path:
    """Run metadata JSON path."""
    return processed_dir() / "run_metadata.json"


def model_path() -> Path:
    """Persisted LightGBM booster path."""
    return processed_dir() / "lightgbm_price_per_m2.txt"


def _with_listed_month(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach ``listed_month`` (YYYY-MM) derived from ``listed_at`` when present."""
    if "listed_at" not in frame.columns:
        frame["listed_month"] = "unknown"
        return frame
    raw = frame["listed_at"].astype(str)
    parsed = pd.to_datetime(raw, errors="coerce", utc=True, format="mixed")
    # RCN GML ids embed a timestamp suffix: …_YYYY-MM-DDTHH-MM-SS
    missing = parsed.isna()
    if missing.any():
        extracted = raw.loc[missing].str.extract(r"_(\d{4}-\d{2}-\d{2})T", expand=False)
        parsed.loc[missing] = pd.to_datetime(extracted, errors="coerce", utc=True)
    frame["listed_month"] = (
        parsed.dt.tz_convert(None).dt.strftime("%Y-%m").fillna("unknown")
    )
    return frame


@st.cache_resource(show_spinner=False)  # type: ignore[misc]
def _read_scored_resource(
    path: str, mtime: float, otodom_path: str, otodom_mtime: float
) -> pd.DataFrame:
    """Cached scored frame (resource — no per-call pickle copy).

    Callers must treat the returned frame as read-only. Filter/map helpers
    already allocate new frames instead of mutating in place.
    """
    del mtime, otodom_mtime  # cache keys only
    frame = pd.read_parquet(path)
    frame = backfill_market_primary(
        frame,
        otodom_path=Path(otodom_path) if otodom_path else None,
    )
    return promote_listing_flags(_with_listed_month(frame))


def load_scored() -> pd.DataFrame:
    """Load scored listings with listed_month and market_primary derived."""
    path = scored_path()
    if not path.exists():
        logger.error("Missing scored listings at %s", path)
        raise FileNotFoundError(
            f"Missing {path}. Run pipeline from Admin or: make pipeline"
        )
    oto = processed_dir() / "otodom_listings.parquet"
    frame = _read_scored_resource(
        str(path),
        path.stat().st_mtime,
        str(oto) if oto.exists() else "",
        oto.stat().st_mtime if oto.exists() else 0.0,
    )
    logger.debug("Loaded scored listings n=%s from %s", len(frame), path.name)
    return frame


def load_metadata() -> RunMetadata | None:
    """Load last-run metadata if present."""
    return load_run_metadata(metadata_path())


@st.cache_data(show_spinner=False)  # type: ignore[misc]
def _read_amenities(
    cache_path: str, fixture_path: str, mtime: float
) -> gpd.GeoDataFrame:
    """Cached amenity load keyed by resolved paths + newest mtime."""
    del mtime
    import geopandas as gpd  # deferred: avoid cold-start tax on load_scored

    cache = Path(cache_path)
    fixture = Path(fixture_path)
    if cache.exists():
        return read_geojson_gdf(str(cache), cache.stat().st_mtime)
    if fixture.exists():
        frame = pd.read_json(fixture)
        return gpd.GeoDataFrame(
            frame,
            geometry=gpd.points_from_xy(frame["lon"], frame["lat"]),
            crs="EPSG:4326",
        )
    return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def load_amenities() -> gpd.GeoDataFrame:
    """Load OSM amenity points from cache or fixture."""
    config = load_config()
    cache = config.data.cache_path() / "osm_amenities.geojson"
    fixture = config.data.fixtures_path() / "amenities.json"
    mtimes = [p.stat().st_mtime for p in (cache, fixture) if p.exists()]
    return _read_amenities(str(cache), str(fixture), max(mtimes, default=0.0))


def load_parks() -> gpd.GeoDataFrame:
    """Load UM + OSM park polygons from cache or fixture."""
    config = load_config()
    cache = config.data.cache_path() / "warsaw_parks.geojson"
    fixture = config.data.fixtures_path() / "parks.geojson"
    for path in (cache, fixture):
        if path.exists():
            return read_geojson_gdf(str(path), path.stat().st_mtime)
    return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def load_bike_paths() -> gpd.GeoDataFrame:
    """Load OSM bicycle infrastructure lines from cache or fixture."""
    config = load_config()
    cache = config.data.cache_path() / "bike_paths.geojson"
    fixture = config.data.fixtures_path() / "bike_paths.geojson"
    for path in (cache, fixture):
        if path.exists():
            return read_geojson_gdf(str(path), path.stat().st_mtime)
    return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def load_city_pois() -> gpd.GeoDataFrame:
    """Load UM sports + civic + OSM fitness city POIs from cache or fixture."""
    config = load_config()
    cache = config.data.cache_path() / "warsaw_city_pois.geojson"
    fixture = config.data.fixtures_path() / "city_pois.geojson"
    for path in (cache, fixture):
        if path.exists():
            return read_geojson_gdf(str(path), path.stat().st_mtime)
    return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def _load_zone_fixture(filename: str) -> gpd.GeoDataFrame:
    """Load a zone-overlay GeoJSON fixture, or an empty frame."""
    path = load_config().data.fixtures_path() / filename
    if not path.exists():
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return read_geojson_gdf(str(path), path.stat().st_mtime)


def load_city_boundary() -> gpd.GeoDataFrame:
    """Warsaw administrative boundary polygon."""
    return _load_zone_fixture("warsaw_boundary.geojson")


def load_district_boundaries() -> gpd.GeoDataFrame:
    """Warsaw dzielnica (district) administrative polygons."""
    return _load_zone_fixture("warsaw_districts.geojson")


def load_military_areas() -> gpd.GeoDataFrame:
    """OSM landuse=military polygons for the map."""
    return _load_zone_fixture("military_areas.geojson")


def load_rail_areas() -> gpd.GeoDataFrame:
    """OSM landuse=railway polygons for the map."""
    return _load_zone_fixture("rail_areas.geojson")


def load_airport_areas() -> gpd.GeoDataFrame:
    """OSM aeroway=aerodrome footprints for the map."""
    return _load_zone_fixture("airport_areas.geojson")


@st.cache_resource(show_spinner=False)  # type: ignore[misc]
def _load_warsaw_noise_overlay(
    cache_root: str, mtime: float, overlay_key: str
) -> dict[str, object] | None:
    """Cached continuous UM noise BitmapLayer payload (resource — no copy)."""
    del mtime
    from happy_housing.gis.warsaw_noise import (
        load_warsaw_noise_rasters,
        prepare_noise_overlay_for_key,
    )

    cache = Path(cache_root)
    rasters = load_warsaw_noise_rasters(cache)
    if not rasters:
        return None
    bbox: tuple[float, float, float, float] | None = None
    meta_path = cache / "warsaw_noise" / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        raw = meta.get("bbox_wgs84")
        if isinstance(raw, list) and len(raw) == 4:
            bbox = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
    overlay = prepare_noise_overlay_for_key(rasters, overlay_key, bbox_wgs84=bbox)
    if overlay is None:
        return None
    return {
        "layer_key": overlay.layer_key,
        "image_data_uri": overlay.image_data_uri,
        "bounds": list(overlay.bounds),
        "width": overlay.width,
        "height": overlay.height,
    }


def load_warsaw_noise_overlay(overlay_key: str) -> NoiseMapOverlay | None:
    """Load a continuous Warsaw UM noise overlay for the selected key."""
    if not overlay_key or overlay_key == "off":
        return None
    cache = load_config().data.cache_path()
    meta = cache / "warsaw_noise" / "meta.json"
    if not meta.exists():
        return None
    payload = _load_warsaw_noise_overlay(
        str(cache.resolve()), file_mtime(meta), str(overlay_key)
    )
    if not payload:
        return None
    bounds_raw = payload["bounds"]
    if not isinstance(bounds_raw, list) or len(bounds_raw) != 4:
        return None
    from happy_housing.gis.warsaw_noise import NoiseMapOverlay

    return NoiseMapOverlay(
        layer_key=str(payload["layer_key"]),
        image_data_uri=str(payload["image_data_uri"]),
        bounds=(
            float(bounds_raw[0]),
            float(bounds_raw[1]),
            float(bounds_raw[2]),
            float(bounds_raw[3]),
        ),
        width=int(payload["width"]),
        height=int(payload["height"]),
    )


def load_warsaw_noise_heatmap(overlay_key: str) -> pd.DataFrame:
    """Deprecated densified points; prefer :func:`load_warsaw_noise_overlay`."""
    del overlay_key
    return pd.DataFrame(columns=["lon", "lat", "db", "weight"])


@st.cache_resource(show_spinner=False)  # type: ignore[misc]
def _load_gios_aq_overlay(
    cache_root: str, mtime: float, overlay_key: str
) -> dict[str, object] | None:
    """Cached continuous GIOŚ AQ BitmapLayer payload (resource — no copy)."""
    del mtime
    from happy_housing.gis.gios_aq_field import (
        load_gios_aq_field_rasters,
        prepare_aq_overlay_for_key,
    )

    cache = Path(cache_root)
    rasters = load_gios_aq_field_rasters(cache)
    if not rasters:
        return None
    bbox: tuple[float, float, float, float] | None = None
    meta_path = cache / "gios_aq_field" / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        raw = meta.get("bbox_wgs84")
        if isinstance(raw, list) and len(raw) == 4:
            bbox = (
                float(raw[0]),
                float(raw[1]),
                float(raw[2]),
                float(raw[3]),
            )
    overlay = prepare_aq_overlay_for_key(rasters, overlay_key, bbox_wgs84=bbox)
    if overlay is None:
        return None
    return {
        "layer_key": overlay.layer_key,
        "image_data_uri": overlay.image_data_uri,
        "bounds": list(overlay.bounds),
        "width": overlay.width,
        "height": overlay.height,
    }


def load_gios_aq_overlay(overlay_key: str) -> AqMapOverlay | None:
    """Load a continuous GIOŚ IDW air-quality overlay for the selected key."""
    if not overlay_key or overlay_key == "off":
        return None
    cache = load_config().data.cache_path()
    meta = cache / "gios_aq_field" / "meta.json"
    if not meta.exists():
        return None
    payload = _load_gios_aq_overlay(
        str(cache.resolve()), file_mtime(meta), str(overlay_key)
    )
    if not payload:
        return None
    bounds_raw = payload["bounds"]
    if not isinstance(bounds_raw, list) or len(bounds_raw) != 4:
        return None
    from happy_housing.gis.gios_aq_field import AqMapOverlay

    return AqMapOverlay(
        layer_key=str(payload["layer_key"]),
        image_data_uri=str(payload["image_data_uri"]),
        bounds=(
            float(bounds_raw[0]),
            float(bounds_raw[1]),
            float(bounds_raw[2]),
            float(bounds_raw[3]),
        ),
        width=int(payload["width"]),
        height=int(payload["height"]),
    )


@st.cache_resource(show_spinner=False)  # type: ignore[misc]
def _shap_bundle(
    model_file: str, mtime: float
) -> tuple[lgb.Booster, shap.TreeExplainer]:
    """Cached LightGBM booster + SHAP TreeExplainer keyed by path + mtime."""
    del mtime
    import lightgbm as lgb  # deferred: avoid cold-start tax on load_scored
    import shap

    booster = lgb.Booster(model_file=model_file)
    return booster, shap.TreeExplainer(booster)


def load_shap_bundle() -> tuple[lgb.Booster, shap.TreeExplainer] | None:
    """Return ``(booster, explainer)`` when the model file exists."""
    path = model_path()
    if not path.exists():
        return None
    # Cached fn typed Any under streamlit stubs.
    return _shap_bundle(str(path), path.stat().st_mtime)  # type: ignore[no-any-return]


def clear_data_caches() -> None:
    """Drop Streamlit + config caches after pipeline / scrape rematerializes data."""
    clear_io_caches()
    logger.info("Cleared Streamlit app data caches")


@st.cache_data(show_spinner=False)  # type: ignore[misc]
def _read_parquet_frame(path: str, mtime: float) -> pd.DataFrame:
    """Cached parquet load keyed by path + mtime."""
    del mtime
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)  # type: ignore[misc]
def _read_json_points(path: str, mtime: float) -> pd.DataFrame:
    """Cached JSON point table (shops / traffic sample)."""
    del mtime
    return pd.read_json(path)


@st.cache_data(show_spinner=False)  # type: ignore[misc]
def _shops_from_geojson(path: str, mtime: float) -> pd.DataFrame:
    """Cached shop points extracted from an OSM shops GeoJSON."""
    del mtime
    import geopandas as gpd  # deferred: avoid cold-start tax on load_scored

    gdf = gpd.read_file(path)
    return pd.DataFrame({
        "name": gdf["name"] if "name" in gdf.columns else "Shop",
        "shop": gdf["shop"] if "shop" in gdf.columns else "shop",
        "lon": gdf.geometry.x,
        "lat": gdf.geometry.y,
    })


def _load_parquet_or_empty(
    path: Path, columns: list[str] | None = None
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    return _read_parquet_frame(str(path), path.stat().st_mtime)


def load_gtfs_stops() -> pd.DataFrame:
    """Load GTFS stop points from processed parquet."""
    return _load_parquet_or_empty(
        processed_dir() / "gtfs_stops.parquet",
        ["stop_id", "stop_name", "stop_lat", "stop_lon"],
    )


def load_gtfs_route_shapes() -> pd.DataFrame:
    """Load simplified GTFS route polylines for the transit-network map."""
    path = processed_dir() / "gtfs_route_shapes.parquet"
    empty_cols = [
        "route_id",
        "route_short_name",
        "route_type",
        "mode",
        "path",
        "color_r",
        "color_g",
        "color_b",
        "color_a",
    ]
    if not path.exists():
        zip_path = load_config().data.raw_path() / "gtfs_warsaw.zip"
        if zip_path.exists():
            from happy_housing.ingest.gtfs_network import build_gtfs_route_shapes

            return build_gtfs_route_shapes(zip_path, path)
        return pd.DataFrame(columns=empty_cols)
    return _load_parquet_or_empty(path, empty_cols)


def load_gtfs_hourly() -> pd.DataFrame:
    """Load or build GTFS stop-hour departure counts."""
    path = processed_dir() / "gtfs_stop_hourly.parquet"
    if not path.exists():
        zip_path = load_config().data.raw_path() / "gtfs_warsaw.zip"
        if not zip_path.exists():
            return pd.DataFrame(columns=["stop_id", "hour", "n_departures"])
        build_gtfs_stop_hourly(zip_path, path)
    frame = _read_parquet_frame(str(path), path.stat().st_mtime)
    if "n_departures" not in frame.columns and "n" in frame.columns:
        frame = frame.rename(columns={"n": "n_departures"})
    return frame


def load_gios_stations() -> pd.DataFrame:
    """Load GIOŚ air-quality station points."""
    return _load_parquet_or_empty(
        processed_dir() / "gios_stations.parquet",
        ["station_id", "lat", "lon", "name", "city"],
    )


def load_gunb_points() -> pd.DataFrame:
    """Load GUNB permit / construction points."""
    path = processed_dir() / "gunb_mazowieckie.parquet"
    if path.exists():
        return _read_parquet_frame(str(path), path.stat().st_mtime)
    fixture = load_config().data.fixtures_path() / "gunb_points.json"
    if fixture.exists():
        return _read_json_points(str(fixture), fixture.stat().st_mtime)
    return pd.DataFrame(columns=["lon", "lat", "adres", "numer_decyzji"])


def load_shops() -> pd.DataFrame:
    """Load OSM shop POIs from cache GeoJSON or fixture JSON."""
    config = load_config()
    cache = config.data.cache_path() / "osm_shops.geojson"
    fixture = config.data.fixtures_path() / "shops.json"
    if cache.exists():
        return _shops_from_geojson(str(cache), cache.stat().st_mtime)
    if fixture.exists():
        return _read_json_points(str(fixture), fixture.stat().st_mtime)
    return pd.DataFrame(columns=["name", "shop", "lon", "lat"])


def load_traffic_sample() -> pd.DataFrame:
    """Load densified OSM road sample for the diurnal traffic-proxy heatmap."""
    config = load_config()
    cache = config.data.cache_path()
    cache_path = cache / "traffic_roads.parquet"
    fixture = config.data.fixtures_path() / "traffic_sample.json"
    if cache_path.exists():
        frame = _read_parquet_frame(str(cache_path), cache_path.stat().st_mtime)
        if not frame.empty and {"lon", "lat", "base_weight"}.issubset(frame.columns):
            return frame
    from happy_housing.gis.traffic import fetch_traffic_road_sample

    return fetch_traffic_road_sample(
        cache,
        bbox=config.bbox.as_tuple(),
        fixture_path=fixture if fixture.exists() else None,
    )
