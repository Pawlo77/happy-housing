"""Tests for OSM Nominatim place suggestions."""

from unittest.mock import MagicMock, patch

import pytest

from happy_housing.app.geocode import PlaceSuggestion, search_places


def test_search_places_parses_nominatim_payload() -> None:
    payload = [
        {
            "display_name": "Rondo Daszyńskiego, Wola, Warszawa, Poland",
            "lon": "20.983",
            "lat": "52.230",
            "osm_type": "node",
            "osm_id": 1,
        },
        {
            "display_name": "Plac Bankowy, Warszawa",
            "lon": "21.002",
            "lat": "52.243",
            "osm_type": "way",
            "osm_id": 2,
        },
    ]
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = payload
    with patch("happy_housing.app.geocode.requests.get", return_value=response) as get:
        hits = search_places("daszyńskiego", limit=5)
    assert len(hits) == 2
    assert hits[0] == PlaceSuggestion(
        label="Rondo Daszyńskiego, Wola, Warszawa, Poland",
        lon=20.983,
        lat=52.230,
        osm_type="node",
        osm_id=1,
    )
    assert get.call_args.kwargs["params"]["q"] == "daszyńskiego"
    assert get.call_args.kwargs["params"]["countrycodes"] == "pl"
    assert "HappyHousing" in get.call_args.kwargs["headers"]["User-Agent"]


def test_search_places_blank_query_returns_empty() -> None:
    assert search_places("  ") == []


def test_search_places_http_error_raises() -> None:
    response = MagicMock()
    response.raise_for_status.side_effect = RuntimeError("boom")
    with (
        patch("happy_housing.app.geocode.requests.get", return_value=response),
        pytest.raises(RuntimeError),
    ):
        search_places("warsaw")
