"""Warsaw strategic noise maps via UM WMS (2022 LDWN / LN layers).

Samples city-wide GetMap rasters using the official GIOŚ RGB legend
(Dz.U. colour table) so every listing gets road / rail / tram / aircraft /
industry Lden + Lnight bands plus energy-sum totals.
"""

from __future__ import annotations

import base64
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import requests
from PIL import Image

logger = logging.getLogger(__name__)

WARSAW_WMS_URL: str = "https://wms.um.warszawa.pl/serwis"
"""UM Warszawa WMS root (strategic noise + many other city layers)."""

_WMS_TIMEOUT_S: float = 180.0
"""Timeout for city-wide GetMap downloads."""

_RASTER_WIDTH: int = 2000
"""GetMap pixel width covering the experiment bbox (~15 m/px E-W)."""

_RASTER_HEIGHT: int = 1600
"""GetMap pixel height covering the experiment bbox (~19 m/px N-S)."""

_CACHE_DIR_NAME: str = "warsaw_noise"
"""Subdirectory under data/cache for PNG rasters + meta.json."""

# Official strategic-map legend (Dz.U. RGB → band lower edge in dB).
# Yellow <50 is stored as 45 so models see a positive low band.
_LEGEND_RGB_TO_DB: dict[tuple[int, int, int], float] = {
    (255, 255, 0): 45.0,
    (255, 191, 15): 50.0,
    (252, 138, 30): 55.0,
    (253, 88, 5): 60.0,
    (222, 62, 62): 65.0,
    (177, 126, 217): 70.0,
    (3, 171, 226): 75.0,
    (0, 76, 230): 80.0,
}
"""GIOŚ / MKiŚ strategic noise map colour → dB lower edge."""

_LEGEND_ARRAY: np.ndarray = np.asarray(list(_LEGEND_RGB_TO_DB.keys()), dtype=np.int32)
"""RGB rows for nearest-colour matching."""

_LEGEND_DB: np.ndarray = np.asarray(list(_LEGEND_RGB_TO_DB.values()), dtype=np.float64)
"""Matching dB lower edges for ``_LEGEND_ARRAY`` rows."""


@dataclass(frozen=True, slots=True)
class NoiseLayerSpec:
    """One UM WMS noise layer wired into enrich / model / map."""

    key: str
    """Stable key, e.g. ``road_lden``."""
    wms_layer: str
    """WMS ``LAYERS`` name (2022 strategic map)."""
    out_col: str
    """Listing column written by enrich."""
    source: str
    """road / rail / tram / aircraft / industry."""
    indicator: str
    """lden or lnight."""


NOISE_LAYER_SPECS: tuple[NoiseLayerSpec, ...] = (
    NoiseLayerSpec(
        "road_lden", "HALAS_DROGOWY_LDWN_2022", "road_lden_db", "road", "lden"
    ),
    NoiseLayerSpec(
        "road_lnight", "HALAS_DROGOWY_LN_2022", "road_lnight_db", "road", "lnight"
    ),
    NoiseLayerSpec(
        "rail_lden", "HALAS_KOLEJOWY_LDWN_2022", "rail_lden_db", "rail", "lden"
    ),
    NoiseLayerSpec(
        "rail_lnight", "HALAS_KOLEJOWY_LN_2022", "rail_lnight_db", "rail", "lnight"
    ),
    NoiseLayerSpec(
        "tram_lden", "HALAS_TRAMWAJOWY_LDWN_2022", "tram_lden_db", "tram", "lden"
    ),
    NoiseLayerSpec(
        "tram_lnight", "HALAS_TRAMWAJOWY_LN_2022", "tram_lnight_db", "tram", "lnight"
    ),
    NoiseLayerSpec(
        "aircraft_lden",
        "HALAS_LOTNICZY_LDWN_2022",
        "aircraft_lden_db",
        "aircraft",
        "lden",
    ),
    NoiseLayerSpec(
        "aircraft_lnight",
        "HALAS_LOTNICZY_LN_2022",
        "aircraft_lnight_db",
        "aircraft",
        "lnight",
    ),
    NoiseLayerSpec(
        "industry_lden",
        "HALAS_PRZEMYSLOWY_LDWN_2022",
        "industry_lden_db",
        "industry",
        "lden",
    ),
    NoiseLayerSpec(
        "industry_lnight",
        "HALAS_PRZEMYSLOWY_LN_2022",
        "industry_lnight_db",
        "industry",
        "lnight",
    ),
)
"""All 2022 strategic-map source x indicator layers."""

NOISE_MAP_OVERLAYS: tuple[tuple[str, str], ...] = (
    ("Off", "off"),
    ("Total Lden", "total_lden"),
    ("Total Lnight", "total_lnight"),
    ("Road Lden", "road_lden"),
    ("Road Lnight", "road_lnight"),
    ("Rail Lden", "rail_lden"),
    ("Rail Lnight", "rail_lnight"),
    ("Tram Lden", "tram_lden"),
    ("Tram Lnight", "tram_lnight"),
    ("Aircraft Lden", "aircraft_lden"),
    ("Aircraft Lnight", "aircraft_lnight"),
    ("Industry Lden", "industry_lden"),
    ("Industry Lnight", "industry_lnight"),
)
"""Explore map noise overlay choices: ``(label, key)``."""

NOISE_FEATURE_COLUMNS: tuple[str, ...] = (
    *(s.out_col for s in NOISE_LAYER_SPECS),
    "noise_total_lden_db",
    "noise_total_lnight_db",
)
"""Model / enrich columns produced from Warsaw WMS noise."""

_LDEN_COLS: tuple[str, ...] = tuple(
    s.out_col for s in NOISE_LAYER_SPECS if s.indicator == "lden"
)
_LNIGHT_COLS: tuple[str, ...] = tuple(
    s.out_col for s in NOISE_LAYER_SPECS if s.indicator == "lnight"
)


@dataclass(slots=True)
class NoiseRaster:
    """Cached GetMap PNG + projected georeference."""

    path: Path
    layer_key: str
    out_col: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    width: int
    height: int
    crs: str = "EPSG:2178"


@dataclass(frozen=True, slots=True)
class NoiseMapOverlay:
    """Georeferenced PNG overlay for Explore Map (deck.gl BitmapLayer)."""

    layer_key: str
    """Overlay key, e.g. ``road_lden`` or ``total_lden``."""
    image_data_uri: str
    """``data:image/png;base64,…`` payload for pydeck."""
    bounds: tuple[float, float, float, float]
    """WGS84 ``(west, south, east, north)`` for BitmapLayer bounds."""
    width: int
    """Raster width in pixels."""
    height: int
    """Raster height in pixels."""


_DB_TO_RGB: dict[float, tuple[int, int, int]] = {
    db: rgb for rgb, db in _LEGEND_RGB_TO_DB.items()
}
"""dB band lower edge → official legend RGB."""

_OVERLAY_ALPHA: int = 200
"""Alpha for painted noise bands on the Explore basemap."""


def color_to_db(rgba: tuple[int, ...] | list[int]) -> float:
    """Map a WMS pixel RGBA to strategic-map band lower edge (dB)."""
    if len(rgba) >= 4 and int(rgba[3]) < 16:
        return 0.0
    rgb = (int(rgba[0]), int(rgba[1]), int(rgba[2]))
    if rgb in _LEGEND_RGB_TO_DB:
        return float(_LEGEND_RGB_TO_DB[rgb])
    # Nearest legend colour (tolerates slight WMS anti-aliasing).
    sample = np.asarray(rgb, dtype=np.int32)
    delta = _LEGEND_ARRAY.astype(np.int32) - sample
    dist = np.sum(delta * delta, axis=1)
    return float(_LEGEND_DB[int(np.argmin(dist))])


def db_to_legend_rgba(
    db: float, *, alpha: int = _OVERLAY_ALPHA
) -> tuple[int, int, int, int]:
    """Map a dB level to GIOŚ legend RGBA; quiet / missing → transparent."""
    if not np.isfinite(db) or float(db) < 45.0:
        return (0, 0, 0, 0)
    band = float(np.floor(float(db) / 5.0) * 5.0)
    band = min(80.0, max(45.0, band))
    rgb = _DB_TO_RGB.get(band)
    if rgb is None:
        # Nearest official band edge.
        edges = np.asarray(list(_DB_TO_RGB.keys()), dtype=float)
        band = float(edges[int(np.argmin(np.abs(edges - band)))])
        rgb = _DB_TO_RGB[band]
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]), int(max(0, min(255, alpha))))


def energy_sum_db(levels: list[float] | np.ndarray) -> float:
    """Energy-equivalent sum of dB bands; zeros ignored; empty → 0."""
    arr = np.asarray(levels, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size == 0:
        return 0.0
    return float(10.0 * np.log10(np.sum(10.0 ** (arr / 10.0))))


def energy_sum_db_rows(stack: np.ndarray) -> np.ndarray:
    """Row-wise energy sum for ``(n, sources)`` dB matrix; zeros ignored."""
    arr = np.asarray(stack, dtype=float)
    if arr.ndim != 2:
        raise ValueError("energy_sum_db_rows expects a 2-D array")
    positive = np.isfinite(arr) & (arr > 0.0)
    power = np.where(positive, 10.0 ** (arr / 10.0), 0.0)
    total = power.sum(axis=1)
    out = np.zeros(arr.shape[0], dtype=float)
    mask = total > 0.0
    out[mask] = 10.0 * np.log10(total[mask])
    return out


def _bbox_2178(
    bbox_wgs84: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """``(west, south, east, north)`` WGS84 → EPSG:2178 ``(xmin, ymin, xmax, ymax)``."""
    west, south, east, north = bbox_wgs84
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2178", always_xy=True)
    x0, y0 = transformer.transform(west, south)
    x1, y1 = transformer.transform(east, north)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _download_layer_png(
    spec: NoiseLayerSpec,
    *,
    bbox_2178: tuple[float, float, float, float],
    width: int,
    height: int,
    out_path: Path,
) -> None:
    """Download one GetMap PNG into ``out_path``."""
    xmin, ymin, xmax, ymax = bbox_2178
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS": spec.wms_layer,
        "STYLES": "",
        "SRS": "EPSG:2178",
        "BBOX": f"{xmin},{ymin},{xmax},{ymax}",
        "WIDTH": int(width),
        "HEIGHT": int(height),
        "FORMAT": "image/png",
        "TRANSPARENT": "TRUE",
    }
    response = requests.get(WARSAW_WMS_URL, params=params, timeout=_WMS_TIMEOUT_S)
    response.raise_for_status()
    out_path.write_bytes(response.content)


def fetch_warsaw_noise_rasters(
    cache_dir: Path,
    *,
    bbox: tuple[float, float, float, float],
    force_refresh: bool = False,
    width: int = _RASTER_WIDTH,
    height: int = _RASTER_HEIGHT,
    max_workers: int = 4,
) -> dict[str, NoiseRaster]:
    """Download / cache all 2022 strategic noise GetMap rasters.

    Args:
        cache_dir: Experiment cache root (creates ``warsaw_noise/``).
        bbox: ``(west, south, east, north)`` WGS84.
        force_refresh: Re-download even when PNG + meta exist.
        width: Raster width in pixels.
        height: Raster height in pixels.
        max_workers: Parallel GetMap downloads.

    Returns:
        Mapping of layer key → georeferenced raster.
    """
    root = cache_dir / _CACHE_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    bbox_2178 = _bbox_2178(bbox)
    xmin, ymin, xmax, ymax = bbox_2178
    meta_path = root / "meta.json"
    meta = {
        "bbox_wgs84": list(bbox),
        "bbox_2178": list(bbox_2178),
        "width": int(width),
        "height": int(height),
        "crs": "EPSG:2178",
        "layers": {s.key: s.wms_layer for s in NOISE_LAYER_SPECS},
    }
    # Re-download when cached GetMap size does not match the requested resolution.
    size_mismatch = False
    if meta_path.exists() and not force_refresh:
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            size_mismatch = int(old.get("width", 0)) != int(width) or int(
                old.get("height", 0)
            ) != int(height)
        except (json.JSONDecodeError, TypeError, ValueError):
            size_mismatch = True
    refresh = bool(force_refresh or size_mismatch)

    def _one(spec: NoiseLayerSpec) -> NoiseRaster:
        png = root / f"{spec.key}.png"
        if refresh or not png.exists() or png.stat().st_size < 100:
            logger.info("Downloading WMS noise layer %s", spec.wms_layer)
            _download_layer_png(
                spec,
                bbox_2178=bbox_2178,
                width=width,
                height=height,
                out_path=png,
            )
        return NoiseRaster(
            path=png,
            layer_key=spec.key,
            out_col=spec.out_col,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            width=int(width),
            height=int(height),
        )

    rasters: dict[str, NoiseRaster] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        futures = {pool.submit(_one, spec): spec.key for spec in NOISE_LAYER_SPECS}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                rasters[key] = fut.result()
            except Exception as exc:
                logger.warning("Noise layer %s download failed: %s", key, exc)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(
        "Cached %s/%s Warsaw noise rasters → %s",
        len(rasters),
        len(NOISE_LAYER_SPECS),
        root,
    )
    return rasters


def load_warsaw_noise_rasters(cache_dir: Path) -> dict[str, NoiseRaster]:
    """Load georeferenced rasters from cache without downloading."""
    root = cache_dir / _CACHE_DIR_NAME
    meta_path = root / "meta.json"
    if not meta_path.exists():
        return {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    xmin, ymin, xmax, ymax = (float(v) for v in meta["bbox_2178"])
    width = int(meta["width"])
    height = int(meta["height"])
    out: dict[str, NoiseRaster] = {}
    for spec in NOISE_LAYER_SPECS:
        png = root / f"{spec.key}.png"
        if not png.exists():
            continue
        out[spec.key] = NoiseRaster(
            path=png,
            layer_key=spec.key,
            out_col=spec.out_col,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            width=width,
            height=height,
        )
    return out


def _decode_db_array(raster: NoiseRaster) -> np.ndarray:
    """Decode PNG to float32 dB grid shaped ``(height, width)``."""
    image = Image.open(raster.path).convert("RGBA")
    pixels = np.asarray(image, dtype=np.uint8)
    h, w = pixels.shape[0], pixels.shape[1]
    flat = pixels.reshape(-1, 4)
    alpha = flat[:, 3]
    rgb = flat[:, :3].astype(np.int16)
    # Exact legend hits.
    out = np.zeros(flat.shape[0], dtype=np.float32)
    transparent = alpha < 16
    for rgb_key, db in _LEGEND_RGB_TO_DB.items():
        mask = (
            (~transparent)
            & (rgb[:, 0] == rgb_key[0])
            & (rgb[:, 1] == rgb_key[1])
            & (rgb[:, 2] == rgb_key[2])
        )
        out[mask] = float(db)
    # Nearest for remaining opaque anti-aliased pixels.
    remain = (~transparent) & (out == 0.0)
    if remain.any():
        leftover = rgb[remain]
        delta = leftover[:, None, :] - _LEGEND_ARRAY[None, :, :]
        dist = np.sum(delta * delta, axis=2)
        out[remain] = _LEGEND_DB[np.argmin(dist, axis=1)].astype(np.float32)
    return out.reshape(h, w)


def raster_sample_db(
    raster: NoiseRaster,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    xy: np.ndarray | None = None,
) -> np.ndarray:
    """Sample dB band at lon/lat (or precomputed projected ``xy``)."""
    grid = _decode_db_array(raster)
    if xy is None:
        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", raster.crs, always_xy=True
        )
        xs, ys = transformer.transform(lon.astype(float), lat.astype(float))
        xy = np.column_stack([xs, ys])
    xs = xy[:, 0]
    ys = xy[:, 1]
    # Image row 0 is north (ymax).
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
    values = np.zeros(len(xs), dtype=np.float64)
    values[inside] = grid[row[inside], col[inside]]
    return values


def attach_warsaw_noise_features(
    frame: pd.DataFrame,
    rasters: dict[str, NoiseRaster],
) -> pd.DataFrame:
    """Attach all noise source bands + energy-sum totals to listings."""
    out = frame.copy()
    lon = pd.to_numeric(out["lon"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(out["lat"], errors="coerce").to_numpy(dtype=float)
    xy: np.ndarray | None = None
    if rasters:
        first = next(iter(rasters.values()))
        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", first.crs, always_xy=True
        )
        xs, ys = transformer.transform(lon, lat)
        xy = np.column_stack([xs, ys])
    for spec in NOISE_LAYER_SPECS:
        raster = rasters.get(spec.key)
        if raster is None:
            out[spec.out_col] = 0.0
            continue
        out[spec.out_col] = raster_sample_db(raster, lon, lat, xy=xy)

    lden_stack = np.column_stack([
        pd.to_numeric(out[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        for c in _LDEN_COLS
    ])
    lnight_stack = np.column_stack([
        pd.to_numeric(out[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        for c in _LNIGHT_COLS
    ])
    out["noise_total_lden_db"] = energy_sum_db_rows(lden_stack)
    out["noise_total_lnight_db"] = energy_sum_db_rows(lnight_stack)
    # High-noise flag for Explore filters (total Lden ≥ 55 dB).
    out["in_high_noise"] = (out["noise_total_lden_db"] >= 55.0).astype(int)
    # Keep legacy aircraft OOU flag aligned with UM aircraft Lden.
    out["in_oou"] = (out["aircraft_lden_db"] >= 55.0).astype(int)
    return out


def prepare_noise_map_points(
    rasters: dict[str, NoiseRaster],
    *,
    layer_key: str,
    stride: int = 4,
    max_points: int = 12_000,
) -> pd.DataFrame:
    """Densify a noise raster to lon/lat/weight points (legacy HeatmapLayer)."""
    raster = rasters.get(layer_key)
    if raster is None:
        return pd.DataFrame(columns=["lon", "lat", "db", "weight"])
    grid = _decode_db_array(raster)
    return _grid_to_map_points(raster, grid, stride=stride, max_points=max_points)


def prepare_total_noise_map_points(
    rasters: dict[str, NoiseRaster],
    *,
    indicator: str = "lden",
    stride: int = 4,
    max_points: int = 12_000,
) -> pd.DataFrame:
    """Energy-sum source rasters into legacy heatmap points."""
    specs = [s for s in NOISE_LAYER_SPECS if s.indicator == indicator]
    grids: list[np.ndarray] = []
    ref: NoiseRaster | None = None
    for spec in specs:
        raster = rasters.get(spec.key)
        if raster is None:
            continue
        grids.append(_decode_db_array(raster))
        ref = raster
    if ref is None or not grids:
        return pd.DataFrame(columns=["lon", "lat", "db", "weight"])
    stack = np.stack(grids, axis=0)
    # Flatten pixels → energy sum → reshape.
    flat = stack.reshape(stack.shape[0], -1).T
    total = energy_sum_db_rows(flat).reshape(ref.height, ref.width).astype(np.float32)
    return _grid_to_map_points(ref, total, stride=stride, max_points=max_points)


def _png_bytes_to_data_uri(payload: bytes) -> str:
    """Encode PNG bytes as a ``data:image/png;base64,…`` URI."""
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def _apply_overlay_alpha(png_path: Path, *, alpha: int = _OVERLAY_ALPHA) -> bytes:
    """Re-encode a WMS PNG with uniform alpha on opaque legend pixels."""
    image = Image.open(png_path).convert("RGBA")
    pixels = np.asarray(image, dtype=np.uint8).copy()
    opaque = pixels[:, :, 3] >= 16
    pixels[opaque, 3] = np.uint8(max(0, min(255, alpha)))
    out = Image.fromarray(pixels, mode="RGBA")
    buf = BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _db_grid_to_png_bytes(grid: np.ndarray, *, alpha: int = _OVERLAY_ALPHA) -> bytes:
    """Paint a dB grid with the official legend colours → PNG bytes."""
    values = np.asarray(grid, dtype=float)
    h, w = values.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    quiet = ~np.isfinite(values) | (values < 45.0)
    band = np.floor(values / 5.0) * 5.0
    band = np.clip(band, 45.0, 80.0)
    a = np.uint8(max(0, min(255, alpha)))
    for edge, rgb in _DB_TO_RGB.items():
        mask = (~quiet) & (np.abs(band - float(edge)) < 0.1)
        rgba[mask, 0] = rgb[0]
        rgba[mask, 1] = rgb[1]
        rgba[mask, 2] = rgb[2]
        rgba[mask, 3] = a
    image = Image.fromarray(rgba, mode="RGBA")
    buf = BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _resolve_overlay_bounds(
    rasters: dict[str, NoiseRaster],
    bbox_wgs84: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    """Prefer explicit WGS84 bbox; else invert the first raster's EPSG:2178 extent."""
    if bbox_wgs84 is not None:
        west, south, east, north = bbox_wgs84
        return (float(west), float(south), float(east), float(north))
    if not rasters:
        return None
    ref = next(iter(rasters.values()))
    transformer = pyproj.Transformer.from_crs(ref.crs, "EPSG:4326", always_xy=True)
    lon0, lat0 = transformer.transform(ref.xmin, ref.ymin)
    lon1, lat1 = transformer.transform(ref.xmax, ref.ymax)
    return (min(lon0, lon1), min(lat0, lat1), max(lon0, lon1), max(lat0, lat1))


def prepare_noise_map_overlay(
    rasters: dict[str, NoiseRaster],
    *,
    layer_key: str,
    bbox_wgs84: tuple[float, float, float, float] | None = None,
) -> NoiseMapOverlay | None:
    """Build a continuous BitmapLayer overlay from one cached WMS PNG."""
    raster = rasters.get(layer_key)
    if raster is None or not raster.path.exists():
        return None
    bounds = _resolve_overlay_bounds(rasters, bbox_wgs84)
    if bounds is None:
        return None
    payload = _apply_overlay_alpha(raster.path)
    return NoiseMapOverlay(
        layer_key=layer_key,
        image_data_uri=_png_bytes_to_data_uri(payload),
        bounds=bounds,
        width=int(raster.width),
        height=int(raster.height),
    )


def prepare_total_noise_map_overlay(
    rasters: dict[str, NoiseRaster],
    *,
    indicator: str = "lden",
    bbox_wgs84: tuple[float, float, float, float] | None = None,
) -> NoiseMapOverlay | None:
    """Energy-sum all source rasters into one continuous legend-coloured PNG."""
    specs = [s for s in NOISE_LAYER_SPECS if s.indicator == indicator]
    grids: list[np.ndarray] = []
    ref: NoiseRaster | None = None
    for spec in specs:
        raster = rasters.get(spec.key)
        if raster is None:
            continue
        grids.append(_decode_db_array(raster))
        ref = raster
    if ref is None or not grids:
        return None
    bounds = _resolve_overlay_bounds(rasters, bbox_wgs84)
    if bounds is None:
        return None
    stack = np.stack(grids, axis=0)
    flat = stack.reshape(stack.shape[0], -1).T
    total = energy_sum_db_rows(flat).reshape(ref.height, ref.width).astype(np.float32)
    payload = _db_grid_to_png_bytes(total)
    return NoiseMapOverlay(
        layer_key=f"total_{indicator}",
        image_data_uri=_png_bytes_to_data_uri(payload),
        bounds=bounds,
        width=int(ref.width),
        height=int(ref.height),
    )


def prepare_noise_overlay_for_key(
    rasters: dict[str, NoiseRaster],
    overlay_key: str,
    *,
    bbox_wgs84: tuple[float, float, float, float] | None = None,
) -> NoiseMapOverlay | None:
    """Resolve Explore overlay key → continuous BitmapLayer payload."""
    key = str(overlay_key or "").strip()
    if not key or key == "off" or not rasters:
        return None
    if key == "total_lden":
        return prepare_total_noise_map_overlay(
            rasters, indicator="lden", bbox_wgs84=bbox_wgs84
        )
    if key == "total_lnight":
        return prepare_total_noise_map_overlay(
            rasters, indicator="lnight", bbox_wgs84=bbox_wgs84
        )
    return prepare_noise_map_overlay(rasters, layer_key=key, bbox_wgs84=bbox_wgs84)


def _grid_to_map_points(
    raster: NoiseRaster,
    grid: np.ndarray,
    *,
    stride: int,
    max_points: int,
) -> pd.DataFrame:
    """Sample non-zero grid cells to lon/lat heatmap points."""
    rows = np.arange(0, raster.height, max(1, int(stride)))
    cols = np.arange(0, raster.width, max(1, int(stride)))
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    db = grid[rr, cc].ravel()
    mask = db > 0.0
    if not mask.any():
        return pd.DataFrame(columns=["lon", "lat", "db", "weight"])
    rr = rr.ravel()[mask]
    cc = cc.ravel()[mask]
    db = db[mask]
    xs = raster.xmin + (cc + 0.5) / raster.width * (raster.xmax - raster.xmin)
    ys = raster.ymax - (rr + 0.5) / raster.height * (raster.ymax - raster.ymin)
    transformer = pyproj.Transformer.from_crs(raster.crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xs, ys)
    frame = pd.DataFrame({
        "lon": lon,
        "lat": lat,
        "db": db.astype(float),
        "weight": np.clip((db.astype(float) - 40.0) / 40.0, 0.05, 1.0),
    })
    if len(frame) > max_points:
        frame = frame.sample(n=int(max_points), random_state=42).reset_index(drop=True)
    return frame
