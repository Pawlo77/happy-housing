"""Tests for Admin commute-target live search + compare map helpers."""

from unittest.mock import MagicMock, patch

from happy_housing.app.commute_target import (
    CommuteTarget,
    commute_compare_points,
    commute_compare_view,
)
from happy_housing.app.geocode import (
    PlaceSuggestion,
    format_place_option_labels,
    live_search_ready,
    resolve_place_pick,
    search_places,
)
from happy_housing.app.maps import commute_compare_layers


def test_live_search_ready_requires_min_chars() -> None:
    assert live_search_ready("") is False
    assert live_search_ready("  a ") is False
    assert live_search_ready("pl") is True
    assert live_search_ready("plac") is True


def test_resolve_place_pick_uses_index_not_duplicate_label() -> None:
    hits = (
        {"label": "Same Place", "lon": 20.1, "lat": 52.1, "osm_id": 1},
        {"label": "Same Place", "lon": 20.2, "lat": 52.2, "osm_id": 2},
    )
    labels = format_place_option_labels(hits)
    assert labels[0] != labels[1]
    first = resolve_place_pick(labels[0], hits, labels)
    second = resolve_place_pick(labels[1], hits, labels)
    assert first is not None and second is not None
    assert float(first["lon"]) == 20.1  # type: ignore[index]
    assert float(second["lon"]) == 20.2  # type: ignore[index]


def test_resolve_place_pick_with_place_suggestion() -> None:
    hits = [
        PlaceSuggestion("A", 20.0, 52.0, "node", 1),
        PlaceSuggestion("B", 21.0, 53.0, "way", 2),
    ]
    labels = format_place_option_labels(hits)
    picked = resolve_place_pick(labels[1], hits, labels)
    assert isinstance(picked, PlaceSuggestion)
    assert picked.lon == 21.0


def test_nominatim_throttle_sleeps_between_calls() -> None:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = []
    import happy_housing.app.geocode as geo

    with (
        patch("happy_housing.app.geocode.requests.get", return_value=response),
        patch("happy_housing.app.geocode.time.sleep") as sleep,
    ):
        geo._last_request_monotonic = 0.0
        with patch(
            "happy_housing.app.geocode.time.monotonic",
            side_effect=[10.0, 10.0],
        ):
            search_places("aa")
        geo._last_request_monotonic = 10.0
        with patch(
            "happy_housing.app.geocode.time.monotonic",
            side_effect=[10.2, 11.2],
        ):
            search_places("bb")
        assert sleep.called
        assert float(sleep.call_args.args[0]) >= 0.5


def test_commute_compare_view_zoom_buckets() -> None:
    same = CommuteTarget(lon=20.98, lat=52.23, label="A")
    _, _, z_same = commute_compare_view(saved=same, draft=same)
    assert z_same == 13.0
    far = CommuteTarget(lon=21.2, lat=52.4, label="B")
    _, _, z_far = commute_compare_view(saved=same, draft=far)
    assert z_far == 10.0


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
