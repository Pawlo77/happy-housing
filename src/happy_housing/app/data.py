"""Shared data loading helpers for Streamlit pages."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from happy_housing.app.metadata import RunMetadata, load_run_metadata
from happy_housing.config import ExperimentConfig, load_experiment_config


def config_path() -> Path:
    """Default experiment YAML path."""
    return Path("configs/experiment.yaml")


def load_config() -> ExperimentConfig:
    """Load experiment config."""
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


def load_scored() -> pd.DataFrame:
    """Load scored listings with listed_month derived."""
    path = scored_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run pipeline from Admin or: make pipeline"
        )
    frame = pd.read_parquet(path)
    frame = frame.copy()
    if "listed_at" in frame.columns:
        parsed = pd.to_datetime(frame["listed_at"], errors="coerce", utc=True)
        frame["listed_month"] = (
            parsed.dt.tz_localize(None).dt.strftime("%Y-%m").fillna("unknown")
        )
    else:
        frame["listed_month"] = "unknown"
    return frame


def load_metadata() -> RunMetadata | None:
    """Load last-run metadata if present."""
    return load_run_metadata(metadata_path())


def load_amenities() -> gpd.GeoDataFrame:
    """Load OSM amenity points from cache or fixture."""
    config = load_config()
    cache = config.data.cache_path() / "osm_amenities.geojson"
    fixture = config.data.fixtures_path() / "amenities.json"
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


def load_oou() -> gpd.GeoDataFrame | None:
    """Load Chopin OOU polygon when present."""
    path = load_config().data.fixtures_path() / "oou_chopin.geojson"
    if not path.exists():
        return None
    return gpd.read_file(path)
