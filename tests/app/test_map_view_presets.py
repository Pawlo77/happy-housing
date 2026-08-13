"""Tests for named Explore map-view (layer) presets."""

from pathlib import Path

from happy_housing.app.map_layers import LayerFlags
from happy_housing.app.map_view_presets import (
    MapViewPreset,
    apply_map_view_preset_to_session,
    clear_map_view_presets,
    delete_map_view_preset,
    load_map_view_presets,
    map_view_presets_path,
    snapshot_map_view_preset_from_session,
    suggest_map_view_preset_name,
    upsert_map_view_preset,
)


def test_layer_flags_round_trip() -> None:
    flags = LayerFlags(
        offers=True,
        transit_show_metro=True,
        transit_show_bus=False,
        shop_show_grocery=True,
        shop_show_fashion=False,
        noise_overlay="total_lden",
        aq_overlay="aq_index",
        city_poi_types=("gym", "library"),
    )
    restored = LayerFlags.from_dict(flags.to_dict())
    assert restored.transit_show_metro is True
    assert restored.transit_show_bus is False
    assert restored.shop_show_grocery is True
    assert restored.shop_show_fashion is False
    assert restored.noise_overlay == "total_lden"
    assert restored.aq_overlay == "aq_index"
    assert restored.city_poi_types == ("gym", "library")


def test_layer_flags_from_dict_legacy_master_off_clears_children() -> None:
    restored = LayerFlags.from_dict({
        "shops": False,
        "shop_show_grocery": True,
        "transit_network": False,
        "transit_show_metro": True,
    })
    assert restored.shop_show_grocery is False
    assert restored.transit_show_metro is False


def test_suggest_map_view_preset_name() -> None:
    name = suggest_map_view_preset_name(
        LayerFlags(
            offers=True,
            transit_show_metro=True,
            shop_show_grocery=True,
            parks=True,
            noise_overlay="total_lden",
        ),
        hour=19,
        basemap="Satellite",
    )
    lower = name.lower()
    assert "metro" in lower
    assert "grocer" in lower
    assert "park" in lower
    assert "satellite" in lower
    assert "19" in name


def test_suggest_map_view_preset_name_defaults() -> None:
    name = suggest_map_view_preset_name(LayerFlags(), hour=8, basemap="Streets")
    assert name == "Default map"


def test_save_load_delete_map_view_presets(tmp_path: Path) -> None:
    path = map_view_presets_path(tmp_path)
    assert load_map_view_presets(path) == []
    upsert_map_view_preset(
        path,
        MapViewPreset(
            name="Night transit",
            layers=LayerFlags(transit_show_metro=True, transit_intensity=True),
            hour=22,
            basemap="Satellite",
        ),
        max_presets=10,
    )
    loaded = load_map_view_presets(path)
    assert len(loaded) == 1
    assert loaded[0].name == "Night transit"
    assert loaded[0].hour == 22
    assert loaded[0].basemap == "Satellite"
    assert loaded[0].layers.transit_show_metro is True

    upsert_map_view_preset(
        path,
        MapViewPreset(
            name="Night transit",
            layers=LayerFlags(transit_show_metro=True, traffic=True),
            hour=21,
            basemap="Streets",
        ),
        max_presets=10,
    )
    loaded = load_map_view_presets(path)
    assert len(loaded) == 1
    assert loaded[0].hour == 21
    assert loaded[0].layers.traffic is True

    upsert_map_view_preset(
        path,
        MapViewPreset(name="Parks", layers=LayerFlags(parks=True), hour=12),
        max_presets=10,
    )
    assert [p.name for p in load_map_view_presets(path)] == ["Night transit", "Parks"]
    assert delete_map_view_preset(path, "Night transit") is True
    assert [p.name for p in load_map_view_presets(path)] == ["Parks"]
    clear_map_view_presets(path)
    assert load_map_view_presets(path) == []


def test_upsert_map_view_respects_max(tmp_path: Path) -> None:
    path = map_view_presets_path(tmp_path)
    for i in range(3):
        upsert_map_view_preset(
            path,
            MapViewPreset(name=f"V{i}", layers=LayerFlags(parks=bool(i)), hour=i),
            max_presets=2,
        )
    assert [p.name for p in load_map_view_presets(path)] == ["V1", "V2"]


def test_apply_and_snapshot_map_view_session_round_trip() -> None:
    session: dict[str, object] = {}
    original = MapViewPreset(
        name="Quiet evening",
        layers=LayerFlags(
            offers=True,
            transit_show_metro=True,
            transit_show_bus=False,
            parks=True,
            shop_show_grocery=True,
            shop_show_other=False,
            noise_overlay="total_lden",
            aq_overlay="off",
            city_poi_types=("gym",),
            military=False,
        ),
        hour=20,
        basemap="Satellite",
    )
    apply_map_view_preset_to_session(session, original)
    snap = snapshot_map_view_preset_from_session(session, name="Quiet evening")
    assert snap.name == "Quiet evening"
    assert snap.hour == 20
    assert snap.basemap == "Satellite"
    assert snap.layers.transit_show_metro is True
    assert snap.layers.transit_show_bus is False
    assert snap.layers.shop_show_grocery is True
    assert snap.layers.shop_show_other is False
    assert snap.layers.parks is True
    assert snap.layers.noise_overlay == "total_lden"
    assert snap.layers.city_poi_types == ("gym",)
    assert snap.layers.military is False
