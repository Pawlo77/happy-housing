"""Tests for Admin commute-target live search + compare map helpers."""

from __future__ import annotations

from happy_housing.app.commute_target import (
    CommuteTarget,
    commute_compare_points,
    commute_compare_view,
)
from happy_housing.app.geocode import live_search_ready
from happy_housing.app.maps import commute_compare_layers


def test_live_search_ready_requires_min_chars() -> None:
    assert live_search_ready("") is False
    assert live_search_ready("  a ") is False
    assert live_search_ready("pl") is True
    assert live_search_ready("plac") is True


def test_commute_compare_points_saved_and_draft() -> None:
    saved = CommuteTarget(lon=20.983, lat=52.230, label="Rondo Daszyńskiego")
    draft = CommuteTarget(lon=21.002, lat=52.243, label="Plac Bankowy")
    frame = commute_compare_points(saved=saved, draft=draft)
    assert list(frame["role"]) == ["saved", "draft"]
    assert float(frame.loc[frame["role"] == "saved", "lon"].iloc[0]) == 20.983
    assert float(frame.loc[frame["role"] == "draft", "lat"].iloc[0]) == 52.243
    assert "Saved:" in str(frame.loc[frame["role"] == "saved", "tooltip_name"].iloc[0])
    assert "New:" in str(frame.loc[frame["role"] == "draft", "tooltip_name"].iloc[0])


def test_commute_compare_points_same_coords_still_two_rows() -> None:
    pin = CommuteTarget(lon=20.983, lat=52.230, label="Same")
    frame = commute_compare_points(saved=pin, draft=pin)
    assert len(frame) == 2


def test_commute_compare_view_centers_between_pins() -> None:
    saved = CommuteTarget(lon=20.98, lat=52.23, label="A")
    draft = CommuteTarget(lon=21.02, lat=52.25, label="B")
    lat, lon, zoom = commute_compare_view(saved=saved, draft=draft)
    assert abs(lat - 52.24) < 1e-6
    assert abs(lon - 21.0) < 1e-6
    assert 10.0 <= zoom <= 14.0


def test_commute_compare_layers_builds_two_icons() -> None:
    saved = CommuteTarget(lon=20.983, lat=52.230, label="Saved")
    draft = CommuteTarget(lon=21.002, lat=52.243, label="Draft")
    layers = commute_compare_layers(saved=saved, draft=draft)
    assert len(layers) == 2
    assert {layer.id for layer in layers} == {
        "commute-saved",
        "commute-draft",
    }
