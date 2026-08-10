"""Continuous Warsaw air-quality field from GIOŚ stations (IDW).

Strategic noise uses official WMS rasters. Live AQ has no equivalent hourly
city raster — only sparse stations — so we build inverse-distance-weighted
grids from current station measurements, sample them onto listings, and paint
Explore BitmapLayer overlays the same way as UM noise.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyproj
from PIL import Image

from happy_housing.gis.frame_util import assign_columns

logger = logging.getLogger(__name__)

AQ_CACHE_DIR_NAME: str = "gios_aq_field"
"""Cache subdirectory under ``data/cache`` for IDW grids + meta."""

AQ_CACHE_SCHEMA_VERSION: int = 1
"""Bump when grid layout / column set / colour legend changes."""

AQ_GRID_WIDTH: int = 500
"""Default IDW grid width (~40 m/px E-W over Warsaw bbox)."""

AQ_GRID_HEIGHT: int = 400
"""Default IDW grid height (~50 m/px N-S over Warsaw bbox)."""

AQ_IDW_POWER: float = 2.0
"""Default inverse-distance power."""

AQ_POOR_INDEX_THRESHOLD: float = 3.0
"""GIOŚ overall index ≥ this → ``in_poor_air`` (sufficient / bad / very bad)."""

_OVERLAY_ALPHA: int = 180
"""Alpha for Explore AQ BitmapLayer pixels."""

# Station measurement column → listing feature column.
AQ_VALUE_SPECS: tuple[tuple[str, str], ...] = (
    ("aq_index", "air_aq_index"),
    ("pm10_ug_m3", "air_pm10_ug_m3"),
    ("pm25_ug_m3", "air_pm25_ug_m3"),
    ("no2_ug_m3", "air_no2_ug_m3"),
)
"""``(station_col, listing_col)`` pairs written by field attach."""

AQ_FEATURE_COLUMNS: tuple[str, ...] = (
    *(listing for _, listing in AQ_VALUE_SPECS),
    "in_poor_air",
)
"""Model / enrich columns produced from the IDW air-quality field."""

AQ_MAP_OVERLAYS: tuple[tuple[str, str], ...] = (
    ("Off", "off"),
    ("AQI (IDW)", "aq_index"),
    ("PM2.5 µg/m³ (IDW)", "pm25_ug_m3"),
    ("PM10 µg/m³ (IDW)", "pm10_ug_m3"),
    ("NO₂ µg/m³ (IDW)", "no2_ug_m3"),
)
"""Explore map air-quality overlay choices: ``(label, key)``."""

# GIOŚ overall index 0-5 -> green -> purple (official-ish severity colours).
_AQI_RGB: dict[int, tuple[int, int, int]] = {
    0: (0, 180, 80),
    1: (120, 200, 60),
    2: (255, 220, 0),
    3: (255, 140, 0),
    4: (220, 50, 50),
    5: (120, 0, 90),
}
"""Discrete AQI band colours for overlay painting."""


@dataclass(slots=True)
class AqFieldRaster:
    """Georeferenced float grid for one AQ pollutant / index."""

    key: str
    """Station column key, e.g. ``aq_index`` / ``pm25_ug_m3``."""
    grid: np.ndarray
    """Float32 values shaped ``(height, width)``; NaN = no coverage."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    width: int
    height: int
    crs: str = "EPSG:2178"
    bbox_wgs84: tuple[float, float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class AqMapOverlay:
    """Georeferenced PNG overlay for Explore Map (deck.gl BitmapLayer)."""

    layer_key: str
    """Overlay key, e.g. ``aq_index``."""
    image_data_uri: str
    """``data:image/png;base64,…`` payload for pydeck."""
    bounds: tuple[float, float, float, float]
    """WGS84 ``(west, south, east, north)`` for BitmapLayer bounds."""
    width: int
    """Raster width in pixels."""
    height: int
    """Raster height in pixels."""


def _bbox_2178(
    bbox_wgs84: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """``(west, south, east, north)`` WGS84 → EPSG:2178 extent."""
    west, south, east, north = bbox_wgs84
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2178", always_xy=True)
    x0, y0 = transformer.transform(west, south)
    x1, y1 = transformer.transform(east, north)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _station_xy_2178(stations: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Project station lon/lat to EPSG:2178 arrays."""
    lon = pd.to_numeric(stations["lon"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(stations["lat"], errors="coerce").to_numpy(dtype=float)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2178", always_xy=True)
    xs, ys = transformer.transform(lon, lat)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _idw_grid(
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
    *,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: int,
    height: int,
    power: float,
) -> np.ndarray:
    """Build an IDW grid; stations with non-finite values are dropped."""
    mask = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(values)
    if not mask.any():
        return np.full((height, width), np.nan, dtype=np.float32)
    sx = xs[mask]
    sy = ys[mask]
    sv = values[mask].astype(float)
    if sx.size == 1:
        return np.full((height, width), float(sv[0]), dtype=np.float32)

    cols = (np.arange(width, dtype=float) + 0.5) / float(width)
    rows = (np.arange(height, dtype=float) + 0.5) / float(height)
    # Row 0 = north (ymax).
    gx = xmin + cols * (xmax - xmin)
    gy = ymax - rows * (ymax - ymin)
    xx, yy = np.meshgrid(gx, gy)
    flat_x = xx.ravel()
    flat_y = yy.ravel()
    # (cells, stations)
    dx = flat_x[:, None] - sx[None, :]
    dy = flat_y[:, None] - sy[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    exact = dist < 1e-6
    has_exact = exact.any(axis=1)
    # Avoid div-by-zero on exact hits; those cells are overwritten below.
    dist_safe = np.where(exact, np.inf, dist)
    w = np.where(np.isfinite(dist_safe), 1.0 / np.power(dist_safe, float(power)), 0.0)
    w_sum = w.sum(axis=1)
    safe = (~has_exact) & (w_sum > 0.0)
    out = np.full(flat_x.shape[0], np.nan, dtype=float)
    out[safe] = (w[safe] * sv[None, :]).sum(axis=1) / w_sum[safe]
    if has_exact.any():
        first_hit = np.argmax(exact, axis=1)
        out[has_exact] = sv[first_hit[has_exact]]
    return out.reshape(height, width).astype(np.float32)


def build_gios_aq_idw_grids(
    stations: pd.DataFrame,
    *,
    bbox_wgs84: tuple[float, float, float, float],
    width: int = AQ_GRID_WIDTH,
    height: int = AQ_GRID_HEIGHT,
    power: float = AQ_IDW_POWER,
) -> dict[str, AqFieldRaster]:
    """Build IDW rasters for each AQ value column present on ``stations``."""
    if stations.empty or "lon" not in stations.columns or "lat" not in stations.columns:
        return {}
    xmin, ymin, xmax, ymax = _bbox_2178(bbox_wgs84)
    xs, ys = _station_xy_2178(stations)
    out: dict[str, AqFieldRaster] = {}
    for station_col, _listing_col in AQ_VALUE_SPECS:
        if station_col not in stations.columns:
            continue
        values = pd.to_numeric(stations[station_col], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(values).any():
            continue
        grid = _idw_grid(
            xs,
            ys,
            values,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            width=int(width),
            height=int(height),
            power=float(power),
        )
        out[station_col] = AqFieldRaster(
            key=station_col,
            grid=grid,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            width=int(width),
            height=int(height),
            bbox_wgs84=(
                float(bbox_wgs84[0]),
                float(bbox_wgs84[1]),
                float(bbox_wgs84[2]),
                float(bbox_wgs84[3]),
            ),
        )
    return out


def sample_aq_field(
    raster: AqFieldRaster,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    xy: np.ndarray | None = None,
) -> np.ndarray:
    """Sample float field at lon/lat (or precomputed projected ``xy``)."""
    if xy is None:
        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", raster.crs, always_xy=True
        )
        xs, ys = transformer.transform(lon.astype(float), lat.astype(float))
        xy = np.column_stack([xs, ys])
    xs = xy[:, 0]
    ys = xy[:, 1]
    col = np.floor((xs - raster.xmin) / (raster.xmax - raster.xmin) * raster.width)
    row = np.floor((raster.ymax - ys) / (raster.ymax - raster.ymin) * raster.height)
    col = np.clip(col.astype(int), 0, raster.width - 1)
    row = np.clip(row.astype(int), 0, raster.height - 1)
    inside = (
        (xs >= raster.xmin)
        & (xs <= raster.xmax)
        & (ys >= raster.ymin)
        & (ys <= raster.ymax)
    )
    values = np.full(len(xs), np.nan, dtype=float)
    values[inside] = raster.grid[row[inside], col[inside]]
    return values


def attach_gios_aq_field_features(
    frame: pd.DataFrame,
    rasters: dict[str, AqFieldRaster],
) -> pd.DataFrame:
    """Overwrite nearest-station AQ columns with continuous IDW samples."""
    out = frame.copy()
    if not rasters or "lon" not in out.columns or "lat" not in out.columns:
        if "in_poor_air" not in out.columns:
            return assign_columns(out, {"in_poor_air": 0})
        return out
    lon = pd.to_numeric(out["lon"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(out["lat"], errors="coerce").to_numpy(dtype=float)
    first = next(iter(rasters.values()))
    transformer = pyproj.Transformer.from_crs("EPSG:4326", first.crs, always_xy=True)
    xs, ys = transformer.transform(lon, lat)
    xy = np.column_stack([xs, ys])
    cols: dict[str, object] = {}
    for station_col, listing_col in AQ_VALUE_SPECS:
        raster = rasters.get(station_col)
        if raster is None:
            if listing_col not in out.columns:
                cols[listing_col] = np.nan
            continue
        cols[listing_col] = sample_aq_field(raster, lon, lat, xy=xy)
    out = assign_columns(out, cols)
    aq = pd.to_numeric(out.get("air_aq_index"), errors="coerce")
    return assign_columns(
        out,
        {"in_poor_air": (aq >= AQ_POOR_INDEX_THRESHOLD).fillna(False).astype(int)},
    )


def _meta_path(cache_root: Path) -> Path:
    return cache_root / "meta.json"


def _stations_fingerprint(stations: pd.DataFrame) -> str:
    """Compact fingerprint so cache refreshes when measurements move."""
    cols = ["station_id", *(c for c, _ in AQ_VALUE_SPECS)]
    present = [c for c in cols if c in stations.columns]
    if not present or stations.empty:
        return "empty"
    slim = stations[present].copy()
    for col in present:
        if col != "station_id":
            slim[col] = pd.to_numeric(slim[col], errors="coerce").round(3)
    slim = slim.sort_values(present[0]).reset_index(drop=True)
    return json.dumps(
        slim.to_dict(orient="list"), sort_keys=True, separators=(",", ":")
    )


def _cache_is_current(
    cache_root: Path,
    *,
    bbox_wgs84: tuple[float, float, float, float],
    width: int,
    height: int,
    power: float,
    fingerprint: str,
) -> bool:
    meta_file = _meta_path(cache_root)
    if not meta_file.exists():
        return False
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if int(meta.get("schema_version", -1)) != AQ_CACHE_SCHEMA_VERSION:
        return False
    if list(meta.get("bbox_wgs84", [])) != list(bbox_wgs84):
        return False
    if int(meta.get("width", -1)) != int(width):
        return False
    if int(meta.get("height", -1)) != int(height):
        return False
    if abs(float(meta.get("power", -1.0)) - float(power)) > 1e-9:
        return False
    if str(meta.get("stations_fingerprint", "")) != fingerprint:
        return False
    layers = meta.get("layers") or []
    return all((cache_root / f"{key}.npy").exists() for key in layers)


def _load_cached(
    cache_root: Path,
    meta: dict[str, Any],
) -> dict[str, AqFieldRaster]:
    bbox = tuple(float(v) for v in meta["bbox_wgs84"])
    xmin, ymin, xmax, ymax = (float(v) for v in meta["bbox_2178"])
    width = int(meta["width"])
    height = int(meta["height"])
    out: dict[str, AqFieldRaster] = {}
    for key in meta.get("layers") or []:
        path = cache_root / f"{key}.npy"
        if not path.exists():
            continue
        grid = np.load(path)
        out[str(key)] = AqFieldRaster(
            key=str(key),
            grid=np.asarray(grid, dtype=np.float32),
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            width=width,
            height=height,
            bbox_wgs84=bbox,  # type: ignore[arg-type]
        )
    return out


def load_gios_aq_field_rasters(cache_dir: Path) -> dict[str, AqFieldRaster]:
    """Load cached IDW AQ rasters without rebuilding."""
    cache_root = cache_dir / AQ_CACHE_DIR_NAME
    meta_file = _meta_path(cache_root)
    if not meta_file.exists():
        return {}
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return _load_cached(cache_root, meta)


def fetch_gios_aq_field_rasters(
    cache_dir: Path,
    stations: pd.DataFrame,
    *,
    bbox_wgs84: tuple[float, float, float, float],
    width: int = AQ_GRID_WIDTH,
    height: int = AQ_GRID_HEIGHT,
    power: float = AQ_IDW_POWER,
    force_refresh: bool = False,
) -> dict[str, AqFieldRaster]:
    """Build or load cached IDW AQ rasters for the experiment bbox."""
    cache_root = cache_dir / AQ_CACHE_DIR_NAME
    cache_root.mkdir(parents=True, exist_ok=True)
    fingerprint = _stations_fingerprint(stations)
    if not force_refresh and _cache_is_current(
        cache_root,
        bbox_wgs84=bbox_wgs84,
        width=width,
        height=height,
        power=power,
        fingerprint=fingerprint,
    ):
        meta = json.loads(_meta_path(cache_root).read_text(encoding="utf-8"))
        loaded = _load_cached(cache_root, meta)
        if loaded:
            logger.info("Loaded %s cached GIOŚ AQ field layers", len(loaded))
            return loaded

    rasters = build_gios_aq_idw_grids(
        stations,
        bbox_wgs84=bbox_wgs84,
        width=width,
        height=height,
        power=power,
    )
    if not rasters:
        logger.warning("No GIOŚ AQ field rasters built (empty / missing values)")
        return {}

    bbox_2178 = _bbox_2178(bbox_wgs84)
    for key, raster in rasters.items():
        np.save(cache_root / f"{key}.npy", raster.grid)
    meta = {
        "schema_version": AQ_CACHE_SCHEMA_VERSION,
        "bbox_wgs84": list(bbox_wgs84),
        "bbox_2178": list(bbox_2178),
        "width": int(width),
        "height": int(height),
        "power": float(power),
        "crs": "EPSG:2178",
        "stations_fingerprint": fingerprint,
        "layers": sorted(rasters.keys()),
    }
    _meta_path(cache_root).write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Cached %s GIOŚ AQ field layers → %s", len(rasters), cache_root)
    return rasters


def _png_data_uri(payload: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def _aqi_grid_to_png(grid: np.ndarray) -> bytes:
    """Paint discrete AQI bands."""
    values = np.asarray(grid, dtype=float)
    h, w = values.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    finite = np.isfinite(values)
    band = np.clip(np.rint(values), 0, 5).astype(int)
    a = np.uint8(_OVERLAY_ALPHA)
    for level, rgb in _AQI_RGB.items():
        mask = finite & (band == level)
        rgba[mask, 0] = rgb[0]
        rgba[mask, 1] = rgb[1]
        rgba[mask, 2] = rgb[2]
        rgba[mask, 3] = a
    buf = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _concentration_grid_to_png(
    grid: np.ndarray,
    *,
    vmax: float,
) -> bytes:
    """Paint continuous green→yellow→red concentration field."""
    values = np.asarray(grid, dtype=float)
    h, w = values.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    finite = np.isfinite(values) & (values >= 0.0)
    if not finite.any():
        buf = BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    t = np.clip(values / float(max(vmax, 1e-6)), 0.0, 1.0)
    # Piecewise: green (0) → yellow (0.5) → red (1).
    r = np.zeros_like(t)
    g = np.zeros_like(t)
    b = np.zeros_like(t)
    low = t <= 0.5
    high = ~low
    r[low] = (t[low] / 0.5) * 255.0
    g[low] = 200.0 + (t[low] / 0.5) * 20.0
    b[low] = 40.0 * (1.0 - t[low] / 0.5)
    r[high] = 255.0
    g[high] = 220.0 * (1.0 - (t[high] - 0.5) / 0.5)
    b[high] = 0.0
    rgba[finite, 0] = np.clip(r[finite], 0, 255).astype(np.uint8)
    rgba[finite, 1] = np.clip(g[finite], 0, 255).astype(np.uint8)
    rgba[finite, 2] = np.clip(b[finite], 0, 255).astype(np.uint8)
    rgba[finite, 3] = np.uint8(_OVERLAY_ALPHA)
    buf = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


_CONCENTRATION_VMAX: dict[str, float] = {
    "pm25_ug_m3": 50.0,
    "pm10_ug_m3": 80.0,
    "no2_ug_m3": 100.0,
}
"""Soft upper colour scale for pollutant overlays (µg/m³)."""


def prepare_aq_map_overlay(
    rasters: dict[str, AqFieldRaster],
    *,
    layer_key: str,
    bbox_wgs84: tuple[float, float, float, float] | None = None,
) -> AqMapOverlay | None:
    """Build a continuous BitmapLayer overlay from one IDW grid."""
    raster = rasters.get(layer_key)
    if raster is None:
        return None
    bounds = bbox_wgs84 or raster.bbox_wgs84
    if bounds is None:
        transformer = pyproj.Transformer.from_crs(
            raster.crs, "EPSG:4326", always_xy=True
        )
        lon0, lat0 = transformer.transform(raster.xmin, raster.ymin)
        lon1, lat1 = transformer.transform(raster.xmax, raster.ymax)
        bounds = (min(lon0, lon1), min(lat0, lat1), max(lon0, lon1), max(lat0, lat1))
    if layer_key == "aq_index":
        payload = _aqi_grid_to_png(raster.grid)
    else:
        payload = _concentration_grid_to_png(
            raster.grid,
            vmax=_CONCENTRATION_VMAX.get(layer_key, 80.0),
        )
    return AqMapOverlay(
        layer_key=layer_key,
        image_data_uri=_png_data_uri(payload),
        bounds=(float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])),
        width=int(raster.width),
        height=int(raster.height),
    )


def prepare_aq_overlay_for_key(
    rasters: dict[str, AqFieldRaster],
    overlay_key: str,
    *,
    bbox_wgs84: tuple[float, float, float, float] | None = None,
) -> AqMapOverlay | None:
    """Resolve Explore overlay key → continuous BitmapLayer payload."""
    key = str(overlay_key or "").strip()
    if not key or key == "off" or not rasters:
        return None
    return prepare_aq_map_overlay(rasters, layer_key=key, bbox_wgs84=bbox_wgs84)
