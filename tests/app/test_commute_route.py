"""Tests for door-to-door commute route plans (OTP legs + OSRM geometry)."""

from unittest.mock import MagicMock, patch

from happy_housing.app.commute_route import (
    format_commute_route_markdown,
    parse_otp_itinerary,
    path_records_for_plan,
)
from happy_housing.gis.routing import (
    osrm_route_geojson,
    otp_plan_best_itinerary,
    otp_plan_duration_seconds,
)


def test_otp_duration_is_seconds_not_milliseconds() -> None:
    """OTP2 itinerary.duration is already seconds."""
    payload = {
        "plan": {
            "itineraries": [
                {"duration": 600},
                {"duration": 900},
            ]
        }
    }
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = payload
    with patch("happy_housing.gis.routing.requests.get", return_value=mock_resp):
        seconds = otp_plan_duration_seconds(
            (21.0, 52.23),
            (20.983, 52.23),
            base_url="http://otp.test",
        )
    assert seconds == 600.0


def test_parse_otp_itinerary_builds_legs_and_paths() -> None:
    # Tiny polyline: single point near Warsaw (approx).
    itinerary = {
        "duration": 1200,
        "walkTime": 300,
        "transitTime": 900,
        "legs": [
            {
                "mode": "WALK",
                "duration": 180,
                "distance": 220.0,
                "from": {"name": "Origin", "lat": 52.23, "lon": 21.0},
                "to": {"name": "Bus stop A", "lat": 52.231, "lon": 21.001},
                "legGeometry": {"points": "_p~iF~ps|U"},
            },
            {
                "mode": "BUS",
                "routeShortName": "175",
                "headsign": "Centrum",
                "duration": 900,
                "distance": 4500.0,
                "from": {"name": "Bus stop A", "lat": 52.231, "lon": 21.001},
                "to": {"name": "Bus stop B", "lat": 52.24, "lon": 20.99},
                "legGeometry": {"points": "_p~iF~ps|U_ulLnnqC"},
            },
            {
                "mode": "WALK",
                "duration": 120,
                "distance": 150.0,
                "from": {"name": "Bus stop B", "lat": 52.24, "lon": 20.99},
                "to": {"name": "Destination", "lat": 52.241, "lon": 20.989},
                "legGeometry": {"points": "_ulLnnqC"},
            },
        ],
    }
    plan = parse_otp_itinerary(itinerary, mode="transit")
    assert plan is not None
    assert plan.mode == "transit"
    assert plan.duration_s == 1200.0
    assert len(plan.legs) == 3
    assert plan.legs[1].route_short_name == "175"
    assert plan.legs[1].from_name == "Bus stop A"
    assert plan.legs[1].to_name == "Bus stop B"
    assert len(plan.legs[0].path) >= 1
    md = format_commute_route_markdown(plan)
    assert "175" in md
    assert "Bus stop A" in md
    records = path_records_for_plan(plan)
    assert len(records) == 3
    assert "path" in records[0]


def test_otp_plan_itinerary_passes_hour_as_time() -> None:
    payload = {
        "plan": {
            "itineraries": [
                {
                    "duration": 600,
                    "legs": [
                        {
                            "mode": "WALK",
                            "duration": 600,
                            "distance": 800.0,
                            "from": {"name": "A", "lat": 52.23, "lon": 21.0},
                            "to": {"name": "B", "lat": 52.24, "lon": 20.98},
                            "legGeometry": {"points": "_p~iF~ps|U"},
                        }
                    ],
                }
            ]
        }
    }
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = payload
    with patch("happy_housing.gis.routing.requests.get", return_value=mock_resp) as get:
        itinerary = otp_plan_best_itinerary(
            (21.0, 52.23),
            (20.98, 52.24),
            base_url="http://otp.test",
            hour=8,
        )
    assert itinerary is not None
    params = get.call_args.kwargs.get("params") or {}
    assert params.get("time") == "08:00:00"
    assert "date" in params


def test_osrm_route_geometry_parses_geojson() -> None:
    payload = {
        "code": "Ok",
        "routes": [
            {
                "distance": 3200.0,
                "duration": 480.0,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[21.0, 52.23], [20.99, 52.235], [20.98, 52.24]],
                },
            }
        ],
    }
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = payload
    with patch("happy_housing.gis.routing.requests.get", return_value=mock_resp) as get:
        out = osrm_route_geojson(
            (21.0, 52.23),
            (20.98, 52.24),
            base_url="http://osrm.test",
            profile="driving",
        )
    assert out is not None
    assert float(out["duration"]) == 480.0
    assert len(out["coordinates"]) == 3  # type: ignore[arg-type]
    url = str(get.call_args.args[0])
    assert "/route/v1/driving/" in url


def test_osrm_foot_route_rescales_car_graph_duration() -> None:
    """Car-only OSRM accepts /foot/ but returns driving times — rescale to walk."""
    payload = {
        "code": "Ok",
        "routes": [
            {
                "distance": 2182.0,
                "duration": 240.0,  # ~33 km/h — car, not walk
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[21.0, 52.23], [20.98, 52.24]],
                },
            }
        ],
    }
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = payload
    with patch("happy_housing.gis.routing.requests.get", return_value=mock_resp):
        out = osrm_route_geojson(
            (21.0, 52.23),
            (20.98, 52.24),
            base_url="http://osrm.test",
            profile="foot",
        )
    assert out is not None
    assert float(out["distance"]) == 2182.0
    # 2.182 km at 5 km/h → 26.184 min → ~1571 s
    assert abs(float(out["duration"]) - 1571.04) < 1.0


def test_fetch_offer_commute_route_walk_uses_foot_url() -> None:
    from happy_housing.app.commute_route import fetch_offer_commute_route

    with patch(
        "happy_housing.app.commute_route.osrm_route_geojson",
        return_value={
            "duration": 600.0,
            "distance": 1000.0,
            "coordinates": [[21.0, 52.23], [20.98, 52.24]],
        },
    ) as route_mock:
        plan = fetch_offer_commute_route(
            origin_lon=21.0,
            origin_lat=52.23,
            dest_lon=20.98,
            dest_lat=52.24,
            mode="walk",
            hour=8,
            osrm_url="http://car.test",
            otp_url="http://otp.test",
            osrm_foot_url="http://foot.test",
            osrm_bike_url="http://bike.test",
        )
    assert plan is not None
    assert route_mock.call_args.kwargs["base_url"] == "http://foot.test"
    assert route_mock.call_args.kwargs["profile"] == "foot"


def test_fetch_offer_commute_route_walk_uses_walk_pace() -> None:
    from happy_housing.app.commute_route import fetch_offer_commute_route

    with patch(
        "happy_housing.app.commute_route.osrm_route_geojson",
        return_value={
            "duration": 1571.0,
            "distance": 2182.0,
            "coordinates": [[21.0, 52.23], [20.98, 52.24]],
        },
    ):
        plan = fetch_offer_commute_route(
            origin_lon=21.0,
            origin_lat=52.23,
            dest_lon=20.98,
            dest_lat=52.24,
            mode="walk",
            hour=8,
            osrm_url="http://osrm.test",
            otp_url="http://otp.test",
        )
    assert plan is not None
    assert plan.mode == "walk"
    assert abs(plan.duration_s / 60.0 - 26.0) < 1.0
    assert "26 min" in plan.summary
    md = format_commute_route_markdown(plan)
    assert "26 min" in md
    assert "2,182 m" in md


def test_fetch_offer_commute_route_car_builds_plan() -> None:
    from happy_housing.app.commute_route import fetch_offer_commute_route

    with patch(
        "happy_housing.app.commute_route.osrm_route_geojson",
        return_value={
            "duration": 480.0,
            "distance": 3200.0,
            "coordinates": [[21.0, 52.23], [20.98, 52.24]],
        },
    ):
        plan = fetch_offer_commute_route(
            origin_lon=21.0,
            origin_lat=52.23,
            dest_lon=20.98,
            dest_lat=52.24,
            mode="car",
            hour=8,
            osrm_url="http://osrm.test",
            otp_url="http://otp.test",
        )
    assert plan is not None
    assert plan.mode == "car"
    assert len(plan.legs) == 1
    assert plan.legs[0].mode == "CAR"
