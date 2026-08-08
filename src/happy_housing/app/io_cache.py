"""Streamlit mtime-keyed I/O caches shared by app data loaders."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import pyarrow.parquet as pq
import streamlit as st

logger = logging.getLogger(__name__)


def file_mtime(path: Path) -> float:
    """Filesystem mtime, or ``-1.0`` when the path is missing."""
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return -1.0


@st.cache_data(show_spinner=False)  # type: ignore[misc]
def read_text(path: str, mtime: float) -> str:
    """Cached UTF-8 text read keyed by path + mtime."""
    del mtime
    return Path(path).read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)  # type: ignore[misc]
def read_json(path: str, mtime: float) -> Any:
    """Cached ``json.loads`` of a UTF-8 file keyed by path + mtime."""
    del mtime
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)  # type: ignore[misc]
def read_geojson_gdf(path: str, mtime: float) -> gpd.GeoDataFrame:
    """Cached GeoJSON / GeoPandas read keyed by path + mtime."""
    del mtime
    return gpd.read_file(path)


@st.cache_data(show_spinner=False)  # type: ignore[misc]
def parquet_num_rows(path: str, mtime: float) -> int | None:
    """Cached parquet footer row count keyed by path + mtime."""
    del mtime
    try:
        meta = pq.ParquetFile(path).metadata
        return int(meta.num_rows) if meta is not None else None
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Parquet row count failed for %s: %s", path, exc)
        return None


@st.cache_data(show_spinner=False)  # type: ignore[misc]
def json_record_count(path: str, mtime: float) -> int | None:
    """Cached GeoJSON / JSON array length keyed by path + mtime."""
    del mtime
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("JSON count failed for %s: %s", path, exc)
        return None
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        if payload.get("type") == "FeatureCollection":
            features = payload.get("features")
            return len(features) if isinstance(features, list) else 0
        return 1
    return None


def clear_io_caches() -> None:
    """Drop all Streamlit data/resource caches used by the app."""
    st.cache_data.clear()
    st.cache_resource.clear()
    logger.info("Cleared Streamlit cache_data + cache_resource")
