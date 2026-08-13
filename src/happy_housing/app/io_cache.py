"""Mtime-keyed I/O helpers; Streamlit-cached only inside a running app."""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import geopandas as gpd

F = TypeVar("F", bound=Callable[..., Any])
"""Callable type preserved by cache decorator."""


def _cache_data(**_kwargs: Any) -> Callable[[F], F]:
    """Use ``st.cache_data`` only when a Streamlit script runtime is active.

    CLI / pytest imports otherwise get an identity decorator and never touch
    Streamlit's cache storage (avoids ``No runtime found`` warnings).
    """

    def decorator(func: F) -> F:
        try:
            import streamlit as st
            from streamlit.runtime.scriptrunner_utils.script_run_context import (
                get_script_run_ctx,
            )
        except ImportError:
            return func
        if get_script_run_ctx(suppress_warning=True) is None:
            return func
        cached = st.cache_data(show_spinner=False)(func)
        return cached  # type: ignore[no-any-return]  # Streamlit wrapper is untyped

    return decorator


def file_mtime(path: Path) -> float:
    """Filesystem mtime, or ``-1.0`` when the path is missing."""
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return -1.0


@_cache_data()
def read_text(path: str, mtime: float) -> str:
    """Cached UTF-8 text read keyed by path + mtime."""
    del mtime
    return Path(path).read_text(encoding="utf-8")


@_cache_data()
def read_json(path: str, mtime: float) -> Any:
    """Cached ``json.loads`` of a UTF-8 file keyed by path + mtime."""
    del mtime
    return json.loads(Path(path).read_text(encoding="utf-8"))


@_cache_data()
def read_geojson_gdf(path: str, mtime: float) -> "gpd.GeoDataFrame":
    """Cached GeoJSON / GeoPandas read keyed by path + mtime."""
    del mtime
    import geopandas as gpd

    return gpd.read_file(path)


@_cache_data()
def parquet_num_rows(path: str, mtime: float) -> int | None:
    """Cached parquet footer row count keyed by path + mtime."""
    del mtime
    import pyarrow.parquet as pq

    try:
        meta = pq.ParquetFile(path).metadata
        return int(meta.num_rows) if meta is not None else None
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Parquet row count failed for %s: %s", path, exc)
        return None


@_cache_data()
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
    import streamlit as st

    st.cache_data.clear()
    st.cache_resource.clear()
    logger.info("Cleared Streamlit cache_data + cache_resource")
