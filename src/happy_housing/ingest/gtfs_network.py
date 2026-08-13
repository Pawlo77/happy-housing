"""Build simplified GTFS route polylines for Explore map PathLayers."""

import logging
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# GTFS route_type → map mode key.
_ROUTE_TYPE_MODE: dict[int, str] = {
    0: "tram",
    1: "metro",
    2: "rail",
    3: "bus",
    5: "tram",
    11: "bus",  # trolleybus
    12: "rail",
}
"""GTFS route_type integers mapped to display modes."""

MODE_RGBA: dict[str, list[int]] = {
    "metro": [196, 30, 58, 230],
    "tram": [182, 0, 0, 220],
    "bus": [0, 101, 178, 200],
    "rail": [22, 163, 74, 220],
    "other": [100, 116, 139, 180],
}
"""RGBA fills for transit modes (Warsaw-ish palette)."""

DEFAULT_MAX_POINTS_PER_SHAPE = 72
"""Evenly downsample long GTFS shapes for Streamlit message size."""


def route_type_to_mode(route_type: object) -> str:
    """Map a GTFS ``route_type`` code to a display mode key."""
    try:
        code = int(str(route_type))
    except (TypeError, ValueError):
        return "other"
    return _ROUTE_TYPE_MODE.get(code, "other")


def simplify_shape_points(
    points: list[tuple[float, float]],
    max_points: int = DEFAULT_MAX_POINTS_PER_SHAPE,
) -> list[list[float]]:
    """Keep first/last points and evenly sample the middle."""
    if max_points < 2 or len(points) <= max_points:
        return [[float(lon), float(lat)] for lon, lat in points]
    n = len(points)
    indexes = [0]
    for i in range(1, max_points - 1):
        indexes.append(round(i * (n - 1) / (max_points - 1)))
    indexes.append(n - 1)
    # Deduplicate consecutive picks from rounding.
    out: list[list[float]] = []
    prev = -1
    for idx in indexes:
        if idx == prev:
            continue
        lon, lat = points[idx]
        out.append([float(lon), float(lat)])
        prev = idx
    return out


def _read_gtfs_csv(
    archive: zipfile.ZipFile, filename: str, **kwargs: Any
) -> pd.DataFrame:
    member = next(name for name in archive.namelist() if Path(name).name == filename)
    with archive.open(member) as handle:
        return pd.read_csv(handle, **kwargs)


def build_gtfs_route_shapes(
    gtfs_zip: Path,
    output_parquet: Path,
    *,
    max_points_per_shape: int = DEFAULT_MAX_POINTS_PER_SHAPE,
) -> pd.DataFrame:
    """Export one simplified polyline per route (most-used shape_id).

    Columns: route_id, route_short_name, route_type, mode, path, color_*.
    """
    with zipfile.ZipFile(gtfs_zip) as archive:
        names = {Path(n).name for n in archive.namelist()}
        for need in ("routes.txt", "trips.txt", "shapes.txt"):
            if need not in names:
                msg = f"GTFS zip missing {need}"
                raise FileNotFoundError(msg)
        routes = _read_gtfs_csv(archive, "routes.txt")
        keep_route_cols = [
            c
            for c in (
                "route_id",
                "route_short_name",
                "route_long_name",
                "route_type",
                "route_color",
            )
            if c in routes.columns
        ]
        routes = routes.loc[:, keep_route_cols]
        trips = _read_gtfs_csv(archive, "trips.txt", usecols=["route_id", "shape_id"])
        shapes = _read_gtfs_csv(
            archive,
            "shapes.txt",
            usecols=["shape_id", "shape_pt_sequence", "shape_pt_lat", "shape_pt_lon"],
        )

    trips = trips.dropna(subset=["route_id", "shape_id"])
    shape_pick = (
        trips
        .groupby(["route_id", "shape_id"], sort=False)
        .size()
        .reset_index(name="n")
        .sort_values(["route_id", "n"], ascending=[True, False])
        .drop_duplicates(subset=["route_id"], keep="first")
    )
    routes = routes.merge(
        shape_pick[["route_id", "shape_id"]], on="route_id", how="inner"
    )
    shapes = shapes.copy()
    shapes["shape_pt_lat"] = pd.to_numeric(shapes["shape_pt_lat"], errors="coerce")
    shapes["shape_pt_lon"] = pd.to_numeric(shapes["shape_pt_lon"], errors="coerce")
    shapes["shape_pt_sequence"] = pd.to_numeric(
        shapes["shape_pt_sequence"], errors="coerce"
    )
    shapes = shapes.dropna(subset=["shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"])
    shapes = shapes.sort_values(["shape_id", "shape_pt_sequence"])

    wanted = set(routes["shape_id"].astype(str))
    shapes = shapes.loc[shapes["shape_id"].astype(str).isin(wanted)]

    path_by_shape: dict[str, list[list[float]]] = {}
    for shape_id, group in shapes.groupby("shape_id", sort=False):
        pts = list(
            zip(
                group["shape_pt_lon"].astype(float).tolist(),
                group["shape_pt_lat"].astype(float).tolist(),
                strict=True,
            )
        )
        path_by_shape[str(shape_id)] = simplify_shape_points(
            pts, max_points=max_points_per_shape
        )

    records: list[dict[str, Any]] = []
    for row in routes.itertuples(index=False):
        shape_id = str(row.shape_id)
        path = path_by_shape.get(shape_id)
        if not path or len(path) < 2:
            continue
        mode = route_type_to_mode(row.route_type)
        rgba = MODE_RGBA.get(mode, MODE_RGBA["other"])
        short = str(getattr(row, "route_short_name", "") or row.route_id)
        records.append({
            "route_id": str(row.route_id),
            "route_short_name": short,
            "route_type": int(row.route_type),
            "mode": mode,
            "path": path,
            "color_r": int(rgba[0]),
            "color_g": int(rgba[1]),
            "color_b": int(rgba[2]),
            "color_a": int(rgba[3]),
            "tooltip_name": f"{mode.title()} {short}",
            "tooltip_detail": f"GTFS route · {mode}",
        })

    frame = pd.DataFrame.from_records(records)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_parquet, index=False)
    logger.info(
        "Wrote %s GTFS route shapes (%s modes) → %s",
        len(frame),
        sorted(frame["mode"].unique()) if not frame.empty else [],
        output_parquet,
    )
    return frame
