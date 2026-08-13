"""Tests for Otodom amenity flag promotion (basement, parking, attic)."""

from happy_housing.ingest.otodom import (
    flags_from_extras_and_tags,
    flatten_otodom_ad,
)


def test_flags_from_extras_split_garage_and_outdoor_parking() -> None:
    flags = flags_from_extras_and_tags(
        extras={"garage", "balcony"},
        tags=set(),
        floor_label=None,
        floor_no=2,
    )
    assert flags["has_garage"] == 1
    assert flags["has_outdoor_parking"] == 0
    assert flags["has_balcony"] == 1
    assert flags["has_basement"] == 0
    assert flags["is_attic"] == 0

    parking = flags_from_extras_and_tags(
        extras={"parking"},
        tags={"PARKING_SPOT"},
        floor_label="GARRET",
        floor_no=99,
    )
    assert parking["has_garage"] == 0
    assert parking["has_outdoor_parking"] == 1
    assert parking["is_attic"] == 1
    assert parking["has_basement"] == 0

    basement = flags_from_extras_and_tags(
        extras={"basement"},
        tags={"BASEMENT"},
        floor_label="GROUND",
        floor_no=0,
    )
    assert basement["has_basement"] == 1


def test_flatten_otodom_ad_promotes_basement_and_parking() -> None:
    ad = {
        "advertId": "1",
        "characteristics": [
            {"key": "floor_no", "value": "3"},
            {"key": "building_ownership", "value": "full_ownership"},
        ],
        "additionalInformation": [
            {"label": "extras_types", "values": ["basement", "parking"]},
        ],
        "location": {"coordinates": {"latitude": 52.23, "longitude": 21.0}},
        "description": "Nice flat",
    }
    row = flatten_otodom_ad(ad)
    assert row["has_basement"] == 1
    assert row["has_outdoor_parking"] == 1
    assert row["has_garage"] == 0
    assert row["is_attic"] == 0
