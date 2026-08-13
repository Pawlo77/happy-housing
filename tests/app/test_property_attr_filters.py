"""Tests for property-attribute filter application."""

import pandas as pd

from happy_housing.app.filters import FilterState, apply_filters
from happy_housing.app.listing_attrs import (
    apply_property_attr_filters,
    promote_listing_flags,
)
from happy_housing.app.stats_filters import StatsFilterState, apply_stats_filters


def test_apply_property_attr_filters_garage_and_ownership() -> None:
    frame = pd.DataFrame({
        "has_garage": [1, 0, 1],
        "ownership": ["full", "usufruct", "full"],
        "object_type": ["flat", "flat", "house"],
    })
    out = apply_property_attr_filters(
        frame, has_garage="yes", ownership="full", object_type="flat"
    )
    assert len(out) == 1
    assert int(out.iloc[0]["has_garage"]) == 1


def test_promote_listing_flags_from_tags() -> None:
    frame = pd.DataFrame({
        "tags_json": ['["BASEMENT", "PARKING_SPOT"]', "[]"],
        "floor_label": ["GARRET", "GROUND"],
        "floor_no": [99, 0],
        "estate": ["FLAT", "HOUSE"],
    })
    out = promote_listing_flags(frame)
    assert int(out.iloc[0]["has_basement"]) == 1
    assert int(out.iloc[0]["has_outdoor_parking"]) == 1
    assert int(out.iloc[0]["is_attic"]) == 1
    assert out.iloc[0]["object_type"] == "flat"
    assert out.iloc[1]["object_type"] == "house"


def test_explore_and_stats_filters_include_property_attrs() -> None:
    frame = pd.DataFrame({
        "district": ["wola", "wola"],
        "source": ["otodom", "otodom"],
        "price_per_m2": [10_000.0, 12_000.0],
        "has_basement": [1, 0],
        "plumbing_replaced": [1, 0],
    })
    explore = apply_filters(
        frame,
        FilterState(has_basement="yes", plumbing_replaced="yes"),
    )
    assert len(explore) == 1
    stats = apply_stats_filters(
        frame,
        StatsFilterState(has_basement="yes", plumbing_replaced="yes"),
    )
    assert len(stats) == 1
