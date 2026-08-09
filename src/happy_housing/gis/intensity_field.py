"""Continuous traffic / transit intensity fields for Explore BitmapLayer.

Point HeatmapLayers read as sparse blobs. These helpers bin weighted lon/lat
samples into a Warsaw grid, blur lightly, paint a georeferenced PNG, and expose
the same BitmapLayer contract as UM noise / GIOŚ AQ overlays. Traffic scales
with the diurnal curve on an absolute colour scale; transit uses hour-specific
GTFS departure weights.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import pandas as pd
import pyproj
from PIL import Image

from happy_housing.gis.traffic import TRAFFIC_DIURNAL_MAX, diurnal_traffic_weight

INTENSITY_GRID_WIDTH: int = 500
"""Default field width (~40 m/px E-W over Warsaw bbox)."""

INTENSITY_GRID_HEIGHT: int = 400
"""Default field height (~50 m/px N-S over Warsaw bbox)."""

_OVERLAY_ALPHA: int = 170
"""Alpha for Explore intensity BitmapLayer pixels."""

_BLUR_PASSES: int = 2
"""Box-blur passes after binning (smooth corridors without scipy)."""

_TRAFFIC_VMAX_FLOOR: float = 1.0
"""Minimum colour-scale ceiling for traffic overlays."""


@dataclass(frozen=True, slots=True)
class IntensityMapOverlay:
    """Georeferenced PNG overlay for Explore Map (deck.gl BitmapLayer)."""

    layer_key: str
    """Stable id, e.g. ``traffic-8`` / ``transit-8``."""
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


def _png_data_uri(payload: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def _box_blur(grid: np.ndarray, passes: int) -> np.ndarray:
    """Cheap separable 3x3 mean filter (ignores NaN / zero empty cells as 0)."""
    out = np.nan_to_num(grid.astype(np.float32), nan=0.0)
    for _ in range(max(0, int(passes))):
        padded = np.pad(out, 1, mode="edge")
        acc = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        )
        out = (acc / 9.0).astype(np.float32)
    return out


def weighted_points_density_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    weights: np.ndarray,
    *,
    bbox_wgs84: tuple[float, float, float, float],
    width: int = INTENSITY_GRID_WIDTH,
    height: int = INTENSITY_GRID_HEIGHT,
    blur_passes: int = _BLUR_PASSES,
) -> np.ndarray:
    """Bin weighted WGS84 points into an EPSG:2178 grid and blur."""
    xmin, ymin, xmax, ymax = _bbox_2178(bbox_wgs84)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2178", always_xy=True)
    xs, ys = transformer.transform(lon.astype(float), lat.astype(float))
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(w) & (w > 0.0)
    grid = np.zeros((int(height), int(width)), dtype=np.float32)
    if not mask.any():
        return grid
    xs = xs[mask]
    ys = ys[mask]
    w = w[mask]
    cols = np.floor((xs - xmin) / (xmax - xmin) * width).astype(int)
    rows = np.floor((ymax - ys) / (ymax - ymin) * height).astype(int)
    inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    if not inside.any():
        return grid
    np.add.at(grid, (rows[inside], cols[inside]), w[inside].astype(np.float32))
    return _box_blur(grid, blur_passes)


def intensity_grid_to_png(grid: np.ndarray, *, vmax: float) -> bytes:
    """Paint continuous cool→warm intensity field (transparent when empty)."""
    values = np.asarray(grid, dtype=float)
    h, w = values.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    finite = np.isfinite(values) & (values > 0.0)
    if not finite.any():
        buf = BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    t = np.clip(values / float(max(vmax, 1e-6)), 0.0, 1.0)
    # Cyan (low) → yellow → red (high), similar to deck Heatmap warm look.
    r = np.zeros_like(t)
    g = np.zeros_like(t)
    b = np.zeros_like(t)
    low = t <= 0.5
    high = ~low
    r[low] = (t[low] / 0.5) * 255.0
    g[low] = 180.0 + (t[low] / 0.5) * 40.0
    b[low] = 220.0 * (1.0 - t[low] / 0.5)
    r[high] = 255.0
    g[high] = 220.0 * (1.0 - (t[high] - 0.5) / 0.5)
    b[high] = 0.0
    rgba[finite, 0] = np.clip(r[finite], 0, 255).astype(np.uint8)
    rgba[finite, 1] = np.clip(g[finite], 0, 255).astype(np.uint8)
    rgba[finite, 2] = np.clip(b[finite], 0, 255).astype(np.uint8)
    # Fade low density so quiet areas stay see-through.
    alpha = (_OVERLAY_ALPHA * np.clip(0.25 + 0.75 * t, 0.0, 1.0)).astype(np.uint8)
    rgba[finite, 3] = alpha[finite]
    buf = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _overlay_from_grid(
    grid: np.ndarray,
    *,
    layer_key: str,
    bbox_wgs84: tuple[float, float, float, float],
    vmax: float,
) -> IntensityMapOverlay | None:
    if not np.isfinite(grid).any() or float(np.nanmax(grid)) <= 0.0:
        return None
    payload = intensity_grid_to_png(grid, vmax=float(vmax))
    return IntensityMapOverlay(
        layer_key=layer_key,
        image_data_uri=_png_data_uri(payload),
        bounds=(
            float(bbox_wgs84[0]),
            float(bbox_wgs84[1]),
            float(bbox_wgs84[2]),
            float(bbox_wgs84[3]),
        ),
        width=int(grid.shape[1]),
        height=int(grid.shape[0]),
    )


def prepare_weighted_points_overlay(
    lon: np.ndarray | pd.Series,
    lat: np.ndarray | pd.Series,
    weights: np.ndarray | pd.Series,
    *,
    bbox_wgs84: tuple[float, float, float, float],
    layer_key: str,
    width: int = INTENSITY_GRID_WIDTH,
    height: int = INTENSITY_GRID_HEIGHT,
    blur_passes: int = _BLUR_PASSES + 1,
) -> IntensityMapOverlay | None:
    """Continuous BitmapLayer field from weighted lon/lat samples."""
    xs = np.asarray(lon, dtype=float)
    ys = np.asarray(lat, dtype=float)
    ws = np.asarray(weights, dtype=float)
    grid = weighted_points_density_grid(
        xs,
        ys,
        ws,
        bbox_wgs84=bbox_wgs84,
        width=width,
        height=height,
        blur_passes=blur_passes,
    )
    finite = grid > 0.0
    if not finite.any():
        return None
    vmax = max(1.0, float(np.percentile(grid[finite], 99)))
    return _overlay_from_grid(
        grid,
        layer_key=layer_key,
        bbox_wgs84=bbox_wgs84,
        vmax=vmax,
    )


def prepare_traffic_intensity_overlay(
    sample: pd.DataFrame,
    *,
    hour: int,
    bbox_wgs84: tuple[float, float, float, float],
    width: int = INTENSITY_GRID_WIDTH,
    height: int = INTENSITY_GRID_HEIGHT,
) -> IntensityMapOverlay | None:
    """Continuous diurnal traffic-proxy field (absolute scale vs peak hour)."""
    if sample is None or getattr(sample, "empty", True):
        return None
    if "lon" not in sample.columns or "lat" not in sample.columns:
        return None
    lon = pd.to_numeric(sample["lon"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(sample["lat"], errors="coerce").to_numpy(dtype=float)
    if "base_weight" in sample.columns:
        base = pd.to_numeric(sample["base_weight"], errors="coerce").fillna(1.0)
    else:
        base = pd.Series(1.0, index=sample.index)
    base_w = base.to_numpy(dtype=float)
    base_grid = weighted_points_density_grid(
        lon,
        lat,
        base_w,
        bbox_wgs84=bbox_wgs84,
        width=width,
        height=height,
    )
    finite = base_grid > 0.0
    if not finite.any():
        return None
    # Colour ceiling = dense-corridor bin at max diurnal factor.
    p99 = float(np.percentile(base_grid[finite], 99))
    vmax = max(_TRAFFIC_VMAX_FLOOR, p99 * float(TRAFFIC_DIURNAL_MAX))
    factor = diurnal_traffic_weight(hour)
    grid = base_grid * float(factor)
    return _overlay_from_grid(
        grid,
        layer_key=f"traffic-{int(hour) % 24}",
        bbox_wgs84=bbox_wgs84,
        vmax=vmax,
    )


def prepare_transit_intensity_overlay(
    stops: pd.DataFrame,
    hourly: pd.DataFrame,
    *,
    hour: int,
    bbox_wgs84: tuple[float, float, float, float],
    width: int = INTENSITY_GRID_WIDTH,
    height: int = INTENSITY_GRID_HEIGHT,
    min_departures: int = 1,
) -> IntensityMapOverlay | None:
    """Continuous GTFS stop-hour departure field for the selected clock hour."""
    if (
        stops is None
        or hourly is None
        or getattr(stops, "empty", True)
        or getattr(hourly, "empty", True)
    ):
        return None
    hour_rows = hourly.loc[hourly["hour"].astype(int) == int(hour)].copy()
    if hour_rows.empty:
        return None
    merged = stops.merge(hour_rows, on="stop_id", how="inner")
    merged = merged.loc[
        pd.to_numeric(merged["n_departures"], errors="coerce").fillna(0)
        >= min_departures
    ]
    if merged.empty:
        return None
    if "stop_lat" in merged.columns and "stop_lon" in merged.columns:
        lat = pd.to_numeric(merged["stop_lat"], errors="coerce").to_numpy(dtype=float)
        lon = pd.to_numeric(merged["stop_lon"], errors="coerce").to_numpy(dtype=float)
    else:
        lat = pd.to_numeric(merged.get("lat"), errors="coerce").to_numpy(dtype=float)
        lon = pd.to_numeric(merged.get("lon"), errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(merged["n_departures"], errors="coerce").fillna(0.0)
    weights_arr = weights.to_numpy(dtype=float)
    grid = weighted_points_density_grid(
        lon,
        lat,
        weights_arr,
        bbox_wgs84=bbox_wgs84,
        width=width,
        height=height,
        blur_passes=_BLUR_PASSES + 1,
    )
    finite = grid > 0.0
    if not finite.any():
        return None
    vmax = max(1.0, float(np.percentile(grid[finite], 99)))
    return _overlay_from_grid(
        grid,
        layer_key=f"transit-{int(hour) % 24}",
        bbox_wgs84=bbox_wgs84,
        vmax=vmax,
    )
