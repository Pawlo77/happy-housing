"""Tests for named Explore map filter presets."""

from __future__ import annotations

from pathlib import Path

from happy_housing.app.filter_presets import (
    FilterPreset,
    apply_filter_preset_to_session,
    clear_filter_presets,
    delete_filter_preset,
    filter_presets_path,
    load_filter_presets,
    save_filter_preset,
    snapshot_filter_preset_from_session,
    suggest_filter_preset_name,
    upsert_filter_preset,
)
from happy_housing.app.filters import FilterState


def test_filter_state_round_trip() -> None:
    state = FilterState(
        district="mokotow",
        oou="outside",
        max_price_per_m2=18000.0,
        max_age_months=6,
        has_garage="yes",
        floor_min=1,
        floor_max=5,
        year_built_min=1990,
    )
    restored = FilterState.from_dict(state.to_dict())
    assert restored.district == "mokotow"
    assert restored.oou == "outside"
    assert restored.max_price_per_m2 == 18000.0
    assert restored.max_age_months == 6
    assert restored.has_garage == "yes"
    assert restored.floor_min == 1
    assert restored.floor_max == 5
    assert restored.year_built_min == 1990
    assert restored.source == "(all)"


def test_filter_state_from_dict_ignores_unknown_and_coerces() -> None:
    restored = FilterState.from_dict({
        "district": "wola",
        "max_price_per_m2": "15000",
        "max_age_months": "3",
        "floor_min": "",
        "bogus": True,
    })
    assert restored.district == "wola"
    assert restored.max_price_per_m2 == 15000.0
    assert restored.max_age_months == 3
    assert restored.floor_min is None


def test_suggest_filter_preset_name_meaningful() -> None:
    name = suggest_filter_preset_name(
        FilterState(
            district="mokotow",
            oou="outside",
            frontage="quiet",
            has_garage="yes",
            max_price_pln=950_000.0,
            max_price_per_m2=17000.0,
            max_age_months=4,
        ),
        starred_only=True,
        ppm_ceiling=25000.0,
        price_ceiling=2_000_000.0,
    )
    lower = name.lower()
    assert "mokotow" in lower
    assert "outside" in lower or "oou" in lower or "aircraft" in lower
    assert "quiet" in lower
    assert "garage" in lower
    assert "star" in lower
    assert "17" in name or "zł" in lower
    assert "950" in name or "pln" in lower


def test_suggest_filter_preset_name_defaults() -> None:
    name = suggest_filter_preset_name(
        FilterState(),
        starred_only=False,
        ppm_ceiling=20000.0,
        price_ceiling=2_000_000.0,
    )
    assert name == "All offers"


def test_save_load_delete_presets(tmp_path: Path) -> None:
    path = filter_presets_path(tmp_path)
    assert load_filter_presets(path) == []

    preset = FilterPreset(
        name="Quiet Wola",
        starred_only=False,
        filters=FilterState(district="wola", frontage="quiet"),
    )
    upsert_filter_preset(path, preset, max_presets=10)
    loaded = load_filter_presets(path)
    assert len(loaded) == 1
    assert loaded[0].name == "Quiet Wola"
    assert loaded[0].filters.district == "wola"
    assert loaded[0].filters.frontage == "quiet"

    upsert_filter_preset(
        path,
        FilterPreset(
            name="Quiet Wola",
            starred_only=True,
            filters=FilterState(district="wola", frontage="quiet", has_balcony="yes"),
        ),
        max_presets=10,
    )
    loaded = load_filter_presets(path)
    assert len(loaded) == 1
    assert loaded[0].starred_only is True
    assert loaded[0].filters.has_balcony == "yes"

    save_filter_preset(
        path,
        FilterPreset(name="Budget", filters=FilterState(max_price_per_m2=12000.0)),
        max_presets=10,
    )
    assert [p.name for p in load_filter_presets(path)] == ["Quiet Wola", "Budget"]

    assert delete_filter_preset(path, "Quiet Wola") is True
    assert [p.name for p in load_filter_presets(path)] == ["Budget"]
    assert delete_filter_preset(path, "missing") is False

    clear_filter_presets(path)
    assert load_filter_presets(path) == []


def test_upsert_respects_max_presets(tmp_path: Path) -> None:
    path = filter_presets_path(tmp_path)
    for i in range(3):
        upsert_filter_preset(
            path,
            FilterPreset(name=f"P{i}", filters=FilterState(district=f"d{i}")),
            max_presets=2,
        )
    names = [p.name for p in load_filter_presets(path)]
    assert names == ["P1", "P2"]


def test_apply_filter_preset_to_session() -> None:
    session: dict[str, object] = {"explore_district": "(all)"}
    preset = FilterPreset(
        name="x",
        starred_only=True,
        filters=FilterState(
            district="wola",
            oou="outside",
            max_noise_total_lden_db=55.0,
            max_price_pln=900_000.0,
            max_price_per_m2=15000.0,
            max_age_months=6,
            has_garage="yes",
            floor_min=2,
            floor_max=None,
            year_built_min=None,
        ),
    )
    apply_filter_preset_to_session(
        session,
        preset,
        age_max=24,
        ppm_max=30000,
        price_max=2_500_000,
    )
    assert session["explore_district"] == "wola"
    assert session["explore_oou"] == "outside"
    assert session["explore_max_noise_total"] == 55.0
    assert session["explore_max_price_pln"] == 900_000
    assert session["explore_max_ppm"] == 15000
    assert session["explore_max_age_months"] == 6
    assert session["explore_starred_only"] is True
    assert session["explore_garage"] == "yes"
    assert session["explore_floor_min"] == 2
    assert session["explore_floor_max"] == 99
    assert session["explore_year_built_min"] == 0


def test_snapshot_filter_preset_from_session_round_trip() -> None:
    session: dict[str, object] = {}
    original = FilterPreset(
        name="Quiet Wola",
        starred_only=True,
        filters=FilterState(
            district="wola",
            frontage="quiet",
            max_price_pln=1_100_000.0,
            max_price_per_m2=16000.0,
            max_age_months=3,
            has_balcony="yes",
            floor_min=1,
            floor_max=4,
            year_built_min=2000,
        ),
    )
    apply_filter_preset_to_session(
        session,
        original,
        age_max=36,
        ppm_max=40000,
        price_max=3_000_000,
    )
    snap = snapshot_filter_preset_from_session(
        session,
        age_max=36,
        ppm_max=40000,
        price_max=3_000_000,
        name="Quiet Wola",
    )
    assert snap.name == "Quiet Wola"
    assert snap.starred_only is True
    assert snap.filters.district == "wola"
    assert snap.filters.frontage == "quiet"
    assert snap.filters.max_price_pln == 1_100_000.0
    assert snap.filters.max_price_per_m2 == 16000.0
    assert snap.filters.max_age_months == 3
    assert snap.filters.has_balcony == "yes"
    assert snap.filters.floor_min == 1
    assert snap.filters.floor_max == 4
    assert snap.filters.year_built_min == 2000
