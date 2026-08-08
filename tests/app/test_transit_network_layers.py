"""Tests for transit path / stop network map prep."""

from __future__ import annotations

import pandas as pd

from happy_housing.app.map_layers import (
    prepare_transit_paths,
)
from happy_housing.app.maps import transit_network_layers


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
