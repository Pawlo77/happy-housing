"""Tests for continuous traffic / transit intensity BitmapLayer overlays."""

from __future__ import annotations

import pandas as pd

from happy_housing.gis.intensity_field import (
    prepare_traffic_intensity_overlay,
    prepare_transit_intensity_overlay,
)

_BBOX = (20.85, 52.10, 21.28, 52.37)


def test_traffic_overlay_is_bitmap_payload_and_dims_with_hour() -> None:
    sample = pd.DataFrame({
        "lon": [21.0, 21.01, 21.02, 20.95],
        "lat": [52.23, 52.24, 52.22, 52.25],
        "base_weight": [10.0, 8.0, 2.0, 5.0],
    })
    night = prepare_traffic_intensity_overlay(sample, hour=3, bbox_wgs84=_BBOX)
    peak = prepare_traffic_intensity_overlay(sample, hour=8, bbox_wgs84=_BBOX)
    assert night is not None
    assert peak is not None
    assert night.image_data_uri.startswith("data:image/png;base64,")
    assert peak.image_data_uri.startswith("data:image/png;base64,")
    assert night.bounds == _BBOX
    assert night.width > 0 and night.height > 0
    # Absolute diurnal scale → peak PNG differs from quiet night.
    assert night.image_data_uri != peak.image_data_uri
    assert night.layer_key == "traffic-3"
    assert peak.layer_key == "traffic-8"


def test_transit_overlay_uses_hour_departures() -> None:
    stops = pd.DataFrame({
        "stop_id": ["a", "b", "c"],
        "stop_name": ["A", "B", "C"],
        "stop_lat": [52.23, 52.24, 52.25],
        "stop_lon": [21.0, 21.01, 21.02],
    })
    hourly = pd.DataFrame({
        "stop_id": ["a", "b", "a", "c"],
        "hour": [8, 8, 3, 3],
        "n_departures": [40, 12, 2, 30],
    })
    rush = prepare_transit_intensity_overlay(stops, hourly, hour=8, bbox_wgs84=_BBOX)
    quiet = prepare_transit_intensity_overlay(stops, hourly, hour=3, bbox_wgs84=_BBOX)
    assert rush is not None
    assert quiet is not None
    assert rush.layer_key == "transit-8"
    assert quiet.layer_key == "transit-3"
    assert rush.image_data_uri != quiet.image_data_uri


def test_empty_inputs_yield_none_overlay() -> None:
    assert (
        prepare_traffic_intensity_overlay(pd.DataFrame(), hour=8, bbox_wgs84=_BBOX)
        is None
    )
    assert (
        prepare_transit_intensity_overlay(
            pd.DataFrame(), pd.DataFrame(), hour=8, bbox_wgs84=_BBOX
        )
        is None
    )
