"""Named Explore map-view presets (layers + hour + basemap)."""

import json
import logging
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Self

from happy_housing.app.io_cache import read_json
from happy_housing.app.map_layers import LayerFlags
from happy_housing.gis.amenity_catalog import (
    AMENITY_TYPES,
    city_map_objects_by_group,
)
from happy_housing.gis.gios_aq_field import AQ_MAP_OVERLAYS
from happy_housing.gis.warsaw_noise import NOISE_MAP_OVERLAYS

logger = logging.getLogger(__name__)

_BOOL_SESSION_KEYS: dict[str, str] = {
    "offers": "explore_ly_offers",
    "city_boundary": "explore_ly_city_boundary",
    "district_boundaries": "explore_ly_district_boundaries",
    "commute_target": "explore_ly_commute",
    "transit_show_metro": "explore_ly_transit_metro",
    "transit_show_tram": "explore_ly_transit_tram",
    "transit_show_bus": "explore_ly_transit_bus",
    "transit_show_rail": "explore_ly_transit_rail",
    "transit_intensity": "explore_ly_transit_intensity",
    "traffic": "explore_ly_traffic",
    "bike_paths": "explore_ly_bike_paths",
    "charging_stations": "explore_ly_charging",
    "gas_stations": "explore_ly_gas",
    "shop_show_grocery": "explore_ly_shop_grocery",
    "shop_show_health_beauty": "explore_ly_shop_health_beauty",
    "shop_show_fashion": "explore_ly_shop_fashion",
    "shop_show_home": "explore_ly_shop_home",
    "shop_show_mobility": "explore_ly_shop_mobility",
    "shop_show_other": "explore_ly_shop_other",
    "hospitals": "explore_ly_hospitals",
    "fire_stations": "explore_ly_fire",
    "police": "explore_ly_police",
    "military": "explore_ly_military",
    "rail_areas": "explore_ly_rail_areas",
    "airport": "explore_ly_airport",
    "parks": "explore_ly_parks",
    "air": "explore_ly_air",
    "gunb": "explore_ly_gunb",
}
"""``LayerFlags`` bool field → Streamlit widget key."""

_LAYER_NAME_BITS: tuple[tuple[str, str], ...] = (
    ("offers", "offers"),
    ("transit_show_metro", "metro"),
    ("transit_show_tram", "tram"),
    ("transit_show_bus", "bus"),
    ("transit_show_rail", "rail lines"),
    ("transit_intensity", "transit heat"),
    ("traffic", "traffic"),
    ("bike_paths", "bike"),
    ("shop_show_grocery", "groceries"),
    ("shop_show_health_beauty", "health shops"),
    ("shop_show_fashion", "fashion"),
    ("shop_show_home", "home shops"),
    ("shop_show_mobility", "car/bike shops"),
    ("shop_show_other", "other shops"),
    ("hospitals", "hospitals"),
    ("fire_stations", "fire"),
    ("police", "police"),
    ("charging_stations", "EV"),
    ("gas_stations", "gas"),
    ("parks", "parks"),
    ("air", "AQI stations"),
    ("gunb", "GUNB"),
    ("city_boundary", "boundary"),
    ("district_boundaries", "districts"),
    ("commute_target", "commute"),
    ("military", "military"),
    ("rail_areas", "rail land"),
    ("airport", "airport"),
)
"""Enabled-layer fragments for auto names (skip default-on env zones unless off)."""

_DEFAULT_ON_SKIP_IN_NAME: frozenset[str] = frozenset({"offers"})
"""Layers that are True by default — only mention when False (as ``no X``)."""


@dataclass(slots=True)
class MapViewPreset:
    """One named Explore map-view snapshot."""

    name: str
    """User-facing preset title."""
    layers: LayerFlags
    """Map overlay flags."""
    hour: int = 8
    """Time-of-day slider (0-23) for transit/traffic."""
    basemap: str = "Streets"
    """``Streets`` or ``Satellite``."""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable payload."""
        return {
            "name": self.name,
            "hour": int(self.hour),
            "basemap": self.basemap,
            "layers": self.layers.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | object) -> Self | None:
        """Parse one preset; invalid/empty name → ``None``."""
        if not isinstance(payload, Mapping):
            return None
        name = str(payload.get("name") or "").strip()
        if not name:
            return None
        raw_layers = payload.get("layers")
        if not isinstance(raw_layers, Mapping):
            raw_layers = {
                k: v
                for k, v in payload.items()
                if k not in {"name", "hour", "basemap", "layers"}
            }
        hour_raw = payload.get("hour", 8)
        try:
            hour = int(hour_raw)
        except (TypeError, ValueError):
            hour = 8
        hour = min(23, max(0, hour))
        basemap = str(payload.get("basemap") or "Streets").strip() or "Streets"
        if basemap not in {"Streets", "Satellite"}:
            basemap = "Streets"
        return cls(
            name=name,
            layers=LayerFlags.from_dict(raw_layers),
            hour=hour,
            basemap=basemap,
        )


def map_view_presets_path(processed: Path) -> Path:
    """JSON path for Explore map-view presets."""
    return processed / "explore_map_view_presets.json"


def load_map_view_presets(path: Path) -> list[MapViewPreset]:
    """Load named presets; missing/invalid → empty list."""
    if not path.exists():
        return []
    try:
        payload = read_json(str(path), path.stat().st_mtime)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        logger.warning("Failed to load map-view presets %s: %s", path, exc)
        return []
    return _parse_presets_payload(payload)


def save_map_view_presets(path: Path, presets: list[MapViewPreset]) -> None:
    """Persist presets JSON (stable order)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"presets": [p.to_dict() for p in presets]}
    path.write_text(
        json.dumps(body, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def clear_map_view_presets(path: Path) -> None:
    """Delete all saved map-view presets."""
    save_map_view_presets(path, [])


def delete_map_view_preset(path: Path, name: str) -> bool:
    """Remove preset by exact name. Return whether anything was deleted."""
    needle = str(name).strip()
    presets = load_map_view_presets(path)
    kept = [p for p in presets if p.name != needle]
    if len(kept) == len(presets):
        return False
    save_map_view_presets(path, kept)
    return True


def upsert_map_view_preset(
    path: Path,
    preset: MapViewPreset,
    *,
    max_presets: int = 30,
) -> list[MapViewPreset]:
    """Insert or replace by name; drop oldest when over ``max_presets``."""
    name = str(preset.name).strip()
    if not name:
        raise ValueError("Map-view preset name must be non-empty")
    cleaned = MapViewPreset(
        name=name,
        layers=preset.layers,
        hour=int(preset.hour),
        basemap=str(preset.basemap),
    )
    presets = [p for p in load_map_view_presets(path) if p.name != name]
    presets.append(cleaned)
    cap = max(1, int(max_presets))
    if len(presets) > cap:
        presets = presets[-cap:]
    save_map_view_presets(path, presets)
    return presets


def suggest_map_view_preset_name(
    layers: LayerFlags,
    *,
    hour: int = 8,
    basemap: str = "Streets",
) -> str:
    """Build a short human-readable name from active map layers."""
    parts: list[str] = []
    if str(basemap) == "Satellite":
        parts.append("Satellite")
    for field_name, label in _LAYER_NAME_BITS:
        enabled = bool(getattr(layers, field_name))
        if field_name in _DEFAULT_ON_SKIP_IN_NAME:
            if not enabled:
                parts.append(f"no {label}")
            continue
        if enabled:
            parts.append(label)
    if layers.noise_overlay and layers.noise_overlay != "off":
        parts.append(f"noise:{layers.noise_overlay}")
    if layers.aq_overlay and layers.aq_overlay != "off":
        parts.append(f"aq:{layers.aq_overlay}")
    if layers.city_poi_types:
        parts.append(f"{len(layers.city_poi_types)} POIs")
    if int(hour) != 8:
        parts.append(f"{int(hour):02d}:00")
    if not parts:
        return "Default map"
    name = " · ".join(parts)
    if len(name) > 80:
        name = name[:77].rstrip(" ·") + "…"
    return name


def apply_map_view_preset_to_session(
    session: MutableMapping[str, Any],
    preset: MapViewPreset,
) -> None:
    """Write Streamlit Map-layers widget keys from ``preset``."""
    layers = preset.layers
    session["explore_basemap"] = (
        "Satellite" if preset.basemap == "Satellite" else "Streets"
    )
    session["explore_hour"] = int(min(23, max(0, int(preset.hour))))
    for field_name, key in _BOOL_SESSION_KEYS.items():
        session[key] = bool(getattr(layers, field_name))
    session["explore_ly_noise_overlay"] = _overlay_label(
        NOISE_MAP_OVERLAYS, layers.noise_overlay
    )
    session["explore_ly_aq_overlay"] = _overlay_label(
        AQ_MAP_OVERLAYS, layers.aq_overlay
    )
    selected = set(layers.city_poi_types)
    session["explore_ly_city_pois_sports"] = _poi_labels_for_group("Sports", selected)
    session["explore_ly_city_pois_civic"] = _poi_labels_for_group("Civic", selected)


def snapshot_map_view_preset_from_session(
    session: Mapping[str, Any],
    name: str = "",
) -> MapViewPreset:
    """Build a preset from Map-layers widget session keys."""
    defaults = LayerFlags()
    data: dict[str, object] = {}
    for field in fields(LayerFlags):
        data[field.name] = getattr(defaults, field.name)
    for field_name, key in _BOOL_SESSION_KEYS.items():
        if key in session:
            data[field_name] = bool(session[key])
    noise_label = str(session.get("explore_ly_noise_overlay") or "Off")
    aq_label = str(session.get("explore_ly_aq_overlay") or "Off")
    data["noise_overlay"] = _overlay_key(NOISE_MAP_OVERLAYS, noise_label)
    data["aq_overlay"] = _overlay_key(AQ_MAP_OVERLAYS, aq_label)
    sport_keys = _poi_keys_from_labels(
        "Sports", session.get("explore_ly_city_pois_sports")
    )
    civic_keys = _poi_keys_from_labels(
        "Civic", session.get("explore_ly_city_pois_civic")
    )
    data["city_poi_types"] = sport_keys + civic_keys
    hour_raw = session.get("explore_hour", 8)
    try:
        hour = int(hour_raw)
    except (TypeError, ValueError):
        hour = 8
    basemap = str(session.get("explore_basemap") or "Streets").strip() or "Streets"
    if basemap not in {"Streets", "Satellite"}:
        basemap = "Streets"
    return MapViewPreset(
        name=str(name or "").strip(),
        layers=LayerFlags.from_dict(data),
        hour=min(23, max(0, hour)),
        basemap=basemap,
    )


def _parse_presets_payload(payload: object) -> list[MapViewPreset]:
    if isinstance(payload, list):
        raw_list = payload
    elif isinstance(payload, Mapping):
        raw_list = payload.get("presets", [])
    else:
        return []
    if not isinstance(raw_list, list):
        return []
    out: list[MapViewPreset] = []
    seen: set[str] = set()
    for item in raw_list:
        preset = MapViewPreset.from_dict(item)
        if preset is None or preset.name in seen:
            continue
        seen.add(preset.name)
        out.append(preset)
    return out


def _overlay_label(options: tuple[tuple[str, str], ...], key: str) -> str:
    needle = str(key or "off")
    for label, value in options:
        if value == needle:
            return label
    return options[0][0] if options else "Off"


def _overlay_key(options: tuple[tuple[str, str], ...], label: str) -> str:
    needle = str(label or "")
    for option_label, value in options:
        if option_label == needle:
            return value
    return "off"


def _poi_labels_for_group(group: str, selected_keys: set[str]) -> list[str]:
    keys = set(city_map_objects_by_group(group))
    label_by_key = {t.key: t.label for t in AMENITY_TYPES if t.key in keys}
    return [label_by_key[k] for k in selected_keys if k in label_by_key]


def _poi_keys_from_labels(group: str, labels: object) -> tuple[str, ...]:
    if not isinstance(labels, (list, tuple)):
        return ()
    keys = set(city_map_objects_by_group(group))
    label_to_key = {t.label: t.key for t in AMENITY_TYPES if t.key in keys}
    out: list[str] = []
    for label in labels:
        key = label_to_key.get(str(label))
        if key is not None:
            out.append(key)
    return tuple(out)
