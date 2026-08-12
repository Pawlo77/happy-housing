"""Tests for transit path / stop network map prep."""

from __future__ import annotations

import pandas as pd

from happy_housing.app.map_layers import (
    prepare_transit_paths,
    prepare_transit_stops_network,
)
from happy_housing.app.maps import transit_network_layers


def test_prepare_transit_paths_coerces_numpy_paths_to_plain_lists() -> None:
    """Parquet may store path as ndarray-of-ndarrays; pydeck needs list[list]."""
    import numpy as np

    frame = pd.DataFrame({
        "mode": ["metro"],
        "path": [np.array([np.array([21.0, 52.2]), np.array([21.1, 52.3])])],
        "route_short_name": ["M1"],
        "color_r": [196],
        "color_g": [30],
        "color_b": [58],
        "color_a": [230],
        "tooltip_name": ["Metro M1"],
        "tooltip_detail": ["GTFS"],
    })
    out = prepare_transit_paths(frame, modes={"metro"})
    assert len(out) == 1
    path = out.iloc[0]["path"]
    assert isinstance(path, list)
    assert path == [[21.0, 52.2], [21.1, 52.3]]
    assert all(isinstance(pt, list) for pt in path)

    layers = transit_network_layers(out, pd.DataFrame(), modes={"metro"})
    path_layers = [layer for layer in layers if layer.type == "PathLayer"]
    assert path_layers
    import json

    import pydeck as pdk

    payload = json.loads(pdk.Deck(layers=path_layers).to_json())
    row_path = payload["layers"][0]["data"][0]["path"]
    assert isinstance(row_path, list)
    assert row_path[0] == [21.0, 52.2]


def test_prepare_transit_paths_filters_modes() -> None:
    frame = pd.DataFrame({
        "mode": ["metro", "bus", "tram"],
        "path": [[[21.0, 52.2], [21.1, 52.3]]] * 3,
        "route_short_name": ["M1", "100", "1"],
        "color_r": [1, 2, 3],
        "color_g": [1, 2, 3],
        "color_b": [1, 2, 3],
        "color_a": [200, 200, 200],
        "tooltip_name": ["a", "b", "c"],
        "tooltip_detail": ["x", "y", "z"],
        "extra": [9, 9, 9],
    })
    out = prepare_transit_paths(frame, modes={"metro", "tram"})
    assert set(out["mode"]) == {"metro", "tram"}
    assert "extra" not in out.columns


def test_transit_network_layers_emit_path_and_stops() -> None:
    shapes = pd.DataFrame({
        "mode": ["metro"],
        "path": [[[21.0, 52.2], [21.05, 52.25], [21.1, 52.3]]],
        "route_short_name": ["M1"],
        "color_r": [196],
        "color_g": [30],
        "color_b": [58],
        "color_a": [230],
        "tooltip_name": ["Metro M1"],
        "tooltip_detail": ["GTFS"],
    })
    stops = pd.DataFrame({
        "stop_id": ["a"],
        "stop_name": ["Centrum"],
        "stop_lat": [52.23],
        "stop_lon": [21.01],
    })
    layers = transit_network_layers(shapes, stops, modes={"metro"})
    types = {layer.type for layer in layers}
    assert "PathLayer" in types
    assert "IconLayer" in types


def test_prepare_transit_stops_network_caps_default() -> None:
    """Default cap keeps pydeck payloads small; max_stops=0 keeps all."""
    from happy_housing.app.map_layers import MAX_TRANSIT_STOP_MAP_POINTS

    n = 7135
    stops = pd.DataFrame({
        "stop_id": [f"s{i}" for i in range(n)],
        "stop_name": [f"Stop {i}" for i in range(n)],
        "stop_lat": [52.10 + (i % 200) * 0.001 for i in range(n)],
        "stop_lon": [20.85 + (i // 200) * 0.001 for i in range(n)],
    })
    out = prepare_transit_stops_network(stops)
    assert len(out) == MAX_TRANSIT_STOP_MAP_POINTS
    assert len(prepare_transit_stops_network(stops, max_stops=0)) == n
    assert set(out.columns) >= {"lat", "lon", "tooltip_name"}
