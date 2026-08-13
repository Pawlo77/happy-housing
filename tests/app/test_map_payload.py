"""Tests for slim / capped map payloads (Streamlit message size)."""

import json

import pandas as pd

from happy_housing.app.map_layers import (
    limit_map_points,
    prepare_gunb_for_map,
    slim_map_frame,
)
from happy_housing.app.maps import gunb_layer


def test_slim_map_frame_keeps_only_hover_and_position() -> None:
    frame = pd.DataFrame({
        "lon": [21.0],
        "lat": [52.2],
        "tooltip_name": ["x"],
        "tooltip_detail": ["y"],
        "adres": ["huge " * 100],
        "nazwisko_projektanta": ["secret"],
        "icon_size": [20.0],
    })
    slim = slim_map_frame(frame)
    assert set(slim.columns) <= {
        "lon",
        "lat",
        "tooltip_name",
        "tooltip_detail",
        "icon_size",
        "radius",
        "color_r",
        "color_g",
        "color_b",
        "color_a",
    }
    assert "adres" not in slim.columns


def test_limit_map_points_caps_rows() -> None:
    frame = pd.DataFrame({"lon": range(100), "lat": range(100)})
    out = limit_map_points(frame, max_points=10, seed=0)
    assert len(out) == 10


def test_prepare_gunb_for_map_caps_and_prefers_warsaw() -> None:
    rows = []
    for i in range(50):
        rows.append({
            "lon": 21.0 + i * 0.001,
            "lat": 52.2,
            "adres": f"addr-{i}",
            "miasto": "Warszawa",
            "numer_decyzji": f"D-{i}",
        })
    for i in range(50):
        rows.append({
            "lon": 19.0 + i * 0.001,
            "lat": 52.4,
            "adres": f"lodz-{i}",
            "miasto": "Łódź",
            "numer_decyzji": f"L-{i}",
        })
    out = prepare_gunb_for_map(pd.DataFrame(rows), max_points=20)
    assert len(out) == 20
    assert out["tooltip_name"].str.startswith("addr-").all()


def test_gunb_layer_payload_stays_small_with_dense_input() -> None:
    """~1e5 attribute-heavy rows must not serialize megabytes of icons/attrs."""
    n = 20_000
    frame = pd.DataFrame({
        "lon": [21.0 + (i % 200) * 0.001 for i in range(n)],
        "lat": [52.2 + (i // 200) * 0.001 for i in range(n)],
        "adres": [f"ul. Testowa {i} " + ("x" * 80) for i in range(n)],
        "numer_decyzji": [f"DEC-{i}" for i in range(n)],
        "miasto": ["Warszawa"] * n,
        "nazwisko_projektanta": ["Kowalski"] * n,
        "projektant_numer_uprawnien": ["UPR"] * n,
    })
    layer = gunb_layer(frame)
    assert layer is not None
    assert layer.type == "IconLayer"
    data = layer.data
    assert len(data) <= 3000
    raw = json.dumps(data, default=str)
    assert len(raw) < 2_000_000
    assert "nazwisko_projektanta" not in raw
    row0 = data[0]
    assert isinstance(row0, dict)
    assert row0.get("icon") == "marker"
    assert "data:image/svg+xml" not in str(row0.get("icon"))
