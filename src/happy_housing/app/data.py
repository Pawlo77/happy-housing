"""Shared data loading helpers for Streamlit pages."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import lightgbm as lgb
import pandas as pd
import shap
import streamlit as st

from happy_housing.app.metadata import RunMetadata, load_run_metadata
from happy_housing.config import ExperimentConfig, load_experiment_config
from happy_housing.ingest import build_gtfs_stop_hourly

logger = logging.getLogger(__name__)


def config_path() -> Path:
    """Default experiment YAML path."""
    return Path("configs/experiment.yaml")


@lru_cache(maxsize=1)
def load_config() -> ExperimentConfig:
    """Load experiment config (process-cached)."""
    return load_experiment_config(config_path())


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


@st.cache_data(show_spinner=False)  # type: ignore[misc]
def _read_scored(path: str, mtime: float) -> pd.DataFrame:
    """Cached parquet read keyed by path + mtime."""
    del mtime  # cache key only
    frame = pd.read_parquet(path)
    return _with_listed_month(frame)


def load_scored() -> pd.DataFrame:
    """Load scored listings with listed_month derived."""
    path = scored_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run pipeline from Admin or: make pipeline"
        )
    return _read_scored(str(path), path.stat().st_mtime)


def load_metadata() -> RunMetadata | None:
    """Load last-run metadata if present."""
    return load_run_metadata(metadata_path())


@st.cache_data(show_spinner=False)  # type: ignore[misc]
def _read_amenities(
    cache_path: str, fixture_path: str, mtime: float
) -> gpd.GeoDataFrame:
    """Cached amenity load keyed by resolved paths + newest mtime."""
    del mtime
    cache = Path(cache_path)
    fixture = Path(fixture_path)
    if cache.exists():
        return gpd.read_file(cache)
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


@st.cache_data(show_spinner=False)  # type: ignore[misc]
def _read_oou(path: str, mtime: float) -> gpd.GeoDataFrame | None:
    """Cached OOU polygon keyed by path + mtime; ``mtime < 0`` means missing."""
    if mtime < 0:
        return None
    del mtime
    return gpd.read_file(path)


def load_oou() -> gpd.GeoDataFrame | None:
    """Load Chopin OOU polygon when present."""
    path = load_config().data.fixtures_path() / "oou_chopin.geojson"
    if not path.exists():
        return _read_oou(str(path), -1.0)
    return _read_oou(str(path), path.stat().st_mtime)


@st.cache_resource(show_spinner=False)  # type: ignore[misc]
def _shap_bundle(
    model_file: str, mtime: float
) -> tuple[lgb.Booster, shap.TreeExplainer]:
    """Cached LightGBM booster + SHAP TreeExplainer keyed by path + mtime."""
    del mtime
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
    load_config.cache_clear()
    _read_scored.clear()
    _read_amenities.clear()
    _read_oou.clear()
    _read_parquet_frame.clear()
    _read_json_points.clear()
    _shap_bundle.clear()
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
        gdf = gpd.read_file(cache)
        out = pd.DataFrame({
            "name": gdf["name"] if "name" in gdf.columns else "Shop",
            "shop": gdf["shop"] if "shop" in gdf.columns else "shop",
            "lon": gdf.geometry.x,
            "lat": gdf.geometry.y,
        })
        return out
    if fixture.exists():
        return _read_json_points(str(fixture), fixture.stat().st_mtime)
    return pd.DataFrame(columns=["name", "shop", "lon", "lat"])


def load_traffic_sample() -> pd.DataFrame:
    """Load corridor sample points for the diurnal traffic-proxy heatmap."""
    fixture = load_config().data.fixtures_path() / "traffic_sample.json"
    if not fixture.exists():
        return pd.DataFrame(columns=["name", "lon", "lat", "base_weight"])
    return _read_json_points(str(fixture), fixture.stat().st_mtime)
