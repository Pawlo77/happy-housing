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
    parsed = pd.to_datetime(frame["listed_at"], errors="coerce", utc=True)
    frame["listed_month"] = (
        parsed.dt.tz_localize(None).dt.strftime("%Y-%m").fillna("unknown")
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
    _shap_bundle.clear()
    logger.info("Cleared Streamlit app data caches")
