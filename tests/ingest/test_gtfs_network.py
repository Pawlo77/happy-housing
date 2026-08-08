"""Tests for GTFS route-shape network export (lines by mode)."""

from __future__ import annotations

import zipfile
from pathlib import Path

from happy_housing.ingest.gtfs_network import (
    MODE_RGBA,
    build_gtfs_route_shapes,
    route_type_to_mode,
    simplify_shape_points,
)


def _write_mini_gtfs(path: Path) -> None:
    routes = "route_id,route_short_name,route_long_name,route_type,route_color\n"
    routes += "M1,M1,Metro Kabaty,1,C41E3A\n"
    routes += "T1,1,Tram Annopol,0,B60000\n"
    routes += "B100,100,Bus loop,3,0065B2\n"
    trips = "route_id,trip_id,shape_id\n"
    trips += "M1,t1,shape_m\n"
    trips += "M1,t2,shape_m\n"
    trips += "T1,t3,shape_t\n"
    trips += "B100,t4,shape_b\n"
    shapes = "shape_id,shape_pt_sequence,shape_pt_lat,shape_pt_lon\n"
    for i in range(10):
        shapes += f"shape_m,{i},52.2{i},21.0{i}\n"
    for i in range(6):
        shapes += f"shape_t,{i},52.3{i},21.1{i}\n"
    for i in range(12):
        shapes += f"shape_b,{i},52.1{i},20.9{i}\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("routes.txt", routes)
        archive.writestr("trips.txt", trips)
        archive.writestr("shapes.txt", shapes)


def test_route_type_to_mode() -> None:
    assert route_type_to_mode(0) == "tram"
    assert route_type_to_mode(1) == "metro"
    assert route_type_to_mode(2) == "rail"
    assert route_type_to_mode(3) == "bus"
    assert route_type_to_mode(99) == "other"
    assert "metro" in MODE_RGBA


def test_simplify_shape_points_caps_length() -> None:
    pts = [(21.0 + i * 0.001, 52.2 + i * 0.001) for i in range(100)]
    out = simplify_shape_points(pts, max_points=10)
    assert len(out) == 10
    assert out[0] == list(pts[0])
    assert out[-1] == list(pts[-1])


def test_build_gtfs_route_shapes_one_path_per_route(tmp_path: Path) -> None:
    zip_path = tmp_path / "mini.zip"
    _write_mini_gtfs(zip_path)
    out = tmp_path / "gtfs_route_shapes.parquet"
    frame = build_gtfs_route_shapes(zip_path, out, max_points_per_shape=5)
    assert out.exists()
    assert set(frame["mode"]) == {"metro", "tram", "bus"}
    assert len(frame) == 3
    metro = frame.loc[frame["mode"] == "metro"].iloc[0]
    assert metro["route_short_name"] == "M1"
    path = metro["path"]
    assert isinstance(path, list)
    assert len(path) <= 5
    assert path[0][0] == path[0][0]  # lon first
    assert float(metro["color_r"]) == MODE_RGBA["metro"][0]
