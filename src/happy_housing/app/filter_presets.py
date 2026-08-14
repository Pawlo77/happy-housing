"""Named Explore map filter presets (save / load / delete)."""

import json
import logging
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from happy_housing.app.filters import FilterState, coerce_listing_kind
from happy_housing.app.io_cache import read_json

logger = logging.getLogger(__name__)

_DEFAULT_ALL: str = "(all)"
"""Sidebar selectbox sentinel for no filter."""

_YES_NO_FIELDS: tuple[str, ...] = (
    "has_garage",
    "has_outdoor_parking",
    "has_basement",
    "has_balcony",
    "has_garden",
    "has_terrace",
    "has_lift",
    "is_attic",
    "needs_renovation",
    "plumbing_replaced",
    "wiring_replaced",
)
"""Tri-state amenity fields on ``FilterState``."""

_ZONE_LABELS: dict[str, str] = {
    "oou": "aircraft",
    "high_noise": "noise",
    "poor_air": "air",
}
"""Short labels for zone inside/outside filters in auto names."""

_AMENITY_LABELS: dict[str, str] = {
    "has_garage": "garage",
    "has_outdoor_parking": "outdoor parking",
    "has_basement": "basement",
    "has_balcony": "balcony",
    "has_garden": "garden",
    "has_terrace": "terrace",
    "has_lift": "lift",
    "is_attic": "attic",
    "needs_renovation": "needs reno",
    "plumbing_replaced": "plumbing",
    "wiring_replaced": "wiring",
}
"""Human labels for amenity auto-name fragments."""


@dataclass(slots=True)
class FilterPreset:
    """One named Explore filter snapshot (sidebar + starred-only)."""

    name: str
    """User-facing preset title."""
    filters: FilterState
    """Sidebar ``FilterState`` payload."""
    starred_only: bool = False
    """Whether Map/Board were limited to starred offers."""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable payload."""
        return {
            "name": self.name,
            "starred_only": bool(self.starred_only),
            "filters": self.filters.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | object) -> Self | None:
        """Parse one preset; invalid/empty name → ``None``."""
        if not isinstance(payload, Mapping):
            return None
        name = str(payload.get("name") or "").strip()
        if not name:
            return None
        raw_filters = payload.get("filters")
        if not isinstance(raw_filters, Mapping):
            raw_filters = {
                k: v
                for k, v in payload.items()
                if k not in {"name", "starred_only", "filters"}
            }
        return cls(
            name=name,
            starred_only=bool(payload.get("starred_only", False)),
            filters=FilterState.from_dict(raw_filters),
        )


def filter_presets_path(processed: Path) -> Path:
    """JSON path for Explore filter presets."""
    return processed / "explore_filter_presets.json"


def load_filter_presets(path: Path) -> list[FilterPreset]:
    """Load named presets; missing/invalid → empty list."""
    if not path.exists():
        return []
    try:
        payload = read_json(str(path), path.stat().st_mtime)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        logger.warning("Failed to load filter presets %s: %s", path, exc)
        return []
    return _parse_presets_payload(payload)


def save_filter_presets(path: Path, presets: list[FilterPreset]) -> None:
    """Persist presets JSON (stable order)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"presets": [p.to_dict() for p in presets]}
    path.write_text(
        json.dumps(body, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def clear_filter_presets(path: Path) -> None:
    """Delete all saved filter presets."""
    save_filter_presets(path, [])


def delete_filter_preset(path: Path, name: str) -> bool:
    """Remove preset by exact name. Return whether anything was deleted."""
    needle = str(name).strip()
    presets = load_filter_presets(path)
    kept = [p for p in presets if p.name != needle]
    if len(kept) == len(presets):
        return False
    save_filter_presets(path, kept)
    return True


def upsert_filter_preset(
    path: Path,
    preset: FilterPreset,
    *,
    max_presets: int = 30,
) -> list[FilterPreset]:
    """Insert or replace by name; drop oldest when over ``max_presets``."""
    name = str(preset.name).strip()
    if not name:
        raise ValueError("Filter preset name must be non-empty")
    cleaned = FilterPreset(
        name=name,
        filters=preset.filters,
        starred_only=bool(preset.starred_only),
    )
    presets = [p for p in load_filter_presets(path) if p.name != name]
    presets.append(cleaned)
    cap = max(1, int(max_presets))
    if len(presets) > cap:
        presets = presets[-cap:]
    save_filter_presets(path, presets)
    return presets


def save_filter_preset(
    path: Path,
    preset: FilterPreset,
    *,
    max_presets: int = 30,
) -> list[FilterPreset]:
    """Alias for ``upsert_filter_preset`` (append/replace by name)."""
    return upsert_filter_preset(path, preset, max_presets=max_presets)


def suggest_filter_preset_name(
    state: FilterState,
    *,
    starred_only: bool = False,
    ppm_ceiling: float | None = None,
    price_ceiling: float | None = None,
) -> str:
    """Build a short human-readable name from active filter facets."""
    parts: list[str] = []
    if starred_only:
        parts.append("starred")
    if state.district != _DEFAULT_ALL:
        parts.append(_title_token(state.district))
    if state.source != _DEFAULT_ALL:
        parts.append(_title_token(state.source))
    if state.listing_kind != _DEFAULT_ALL:
        parts.append(state.listing_kind)
    if state.object_type != _DEFAULT_ALL:
        parts.append(state.object_type)
    if state.ownership != _DEFAULT_ALL:
        parts.append(f"own:{state.ownership}")
    for field_name, label in _ZONE_LABELS.items():
        value = str(getattr(state, field_name))
        if value != _DEFAULT_ALL:
            parts.append(f"{value} {label}")
    if state.frontage != _DEFAULT_ALL:
        parts.append(f"{state.frontage} side")
    for field_name in _YES_NO_FIELDS:
        value = str(getattr(state, field_name))
        if value == _DEFAULT_ALL:
            continue
        label = _AMENITY_LABELS.get(field_name, field_name)
        if value == "yes":
            parts.append(label)
        elif value == "no":
            parts.append(f"no {label}")
        else:
            parts.append(f"{label}:{value}")
    if state.max_noise_total_lden_db is not None:
        parts.append(f"≤{int(state.max_noise_total_lden_db)} dB")
    if state.max_traffic_weight_nearby is not None:
        parts.append(f"traffic≤{int(state.max_traffic_weight_nearby)}")
    if state.max_major_traffic_weight_nearby is not None:
        parts.append(f"major≤{int(state.max_major_traffic_weight_nearby)}")
    if state.max_price_pln is not None and (
        price_ceiling is None or float(state.max_price_pln) < float(price_ceiling) - 0.5
    ):
        pln = float(state.max_price_pln)
        if pln >= 1_000_000:
            parts.append(f"≤{pln / 1_000_000:.1f}".rstrip("0").rstrip(".") + "M PLN")
        else:
            parts.append(f"≤{round(pln / 1000)}k PLN")
    if state.max_price_per_m2 is not None and (
        ppm_ceiling is None or float(state.max_price_per_m2) < float(ppm_ceiling) - 0.5
    ):
        parts.append(f"≤{round(float(state.max_price_per_m2) / 1000)}k zł/m²")
    if state.max_age_months is not None:
        parts.append(f"≤{int(state.max_age_months)} mo")
    if state.floor_min is not None or state.floor_max is not None:
        lo = state.floor_min if state.floor_min is not None else "…"
        hi = state.floor_max if state.floor_max is not None else "…"
        parts.append(f"fl {lo}-{hi}")
    if state.year_built_min is not None:
        parts.append(f"built≥{int(state.year_built_min)}")
    if state.year_renovated_min is not None:
        parts.append(f"reno≥{int(state.year_renovated_min)}")
    if not parts:
        return "All offers"
    name = " · ".join(parts)
    if len(name) > 80:
        name = name[:77].rstrip(" ·") + "…"
    return name


def apply_filter_preset_to_session(
    session: MutableMapping[str, Any],
    preset: FilterPreset,
    *,
    age_max: int,
    ppm_max: int,
    price_max: int,
) -> None:
    """Write Streamlit widget keys so the next Filters render matches ``preset``."""
    state = preset.filters
    session["explore_district"] = state.district
    session["explore_source"] = state.source
    session["explore_listing_kind"] = coerce_listing_kind(state.listing_kind)
    session["explore_oou"] = state.oou
    session["explore_high_noise"] = state.high_noise
    session["explore_poor_air"] = state.poor_air
    session["explore_max_noise_total"] = float(state.max_noise_total_lden_db or 0.0)
    session["explore_max_traffic_weight"] = float(
        state.max_traffic_weight_nearby or 0.0
    )
    session["explore_max_major_traffic_weight"] = float(
        state.max_major_traffic_weight_nearby or 0.0
    )
    session["explore_frontage"] = state.frontage
    price = state.max_price_pln
    if price is None:
        session["explore_max_price_pln"] = int(price_max)
    else:
        session["explore_max_price_pln"] = round(float(price))
    ppm = state.max_price_per_m2
    if ppm is None:
        session["explore_max_ppm"] = int(ppm_max)
    else:
        session["explore_max_ppm"] = round(float(ppm))
    if age_max > 0:
        if state.max_age_months is None:
            session["explore_max_age_months"] = int(age_max)
        else:
            session["explore_max_age_months"] = int(
                min(max(0, int(state.max_age_months)), int(age_max))
            )
    session["explore_starred_only"] = bool(preset.starred_only)
    session["explore_object_type"] = state.object_type
    session["explore_ownership"] = state.ownership
    session["explore_floor_min"] = (
        -1 if state.floor_min is None else int(state.floor_min)
    )
    session["explore_floor_max"] = (
        99 if state.floor_max is None else int(state.floor_max)
    )
    session["explore_year_built_min"] = (
        0 if state.year_built_min is None else int(state.year_built_min)
    )
    session["explore_year_renovated_min"] = (
        0 if state.year_renovated_min is None else int(state.year_renovated_min)
    )
    session["explore_garage"] = state.has_garage
    session["explore_outdoor_parking"] = state.has_outdoor_parking
    session["explore_basement"] = state.has_basement
    session["explore_balcony"] = state.has_balcony
    session["explore_garden"] = state.has_garden
    session["explore_terrace"] = state.has_terrace
    session["explore_lift"] = state.has_lift
    session["explore_attic"] = state.is_attic
    session["explore_renovation"] = state.needs_renovation
    session["explore_plumbing"] = state.plumbing_replaced
    session["explore_wiring"] = state.wiring_replaced


def snapshot_filter_preset_from_session(
    session: Mapping[str, Any],
    *,
    age_max: int,
    ppm_max: int,
    price_max: int,
    name: str = "",
) -> FilterPreset:
    """Build a preset from Explore widget session keys (defaults when missing)."""

    def _str(key: str, default: str = _DEFAULT_ALL) -> str:
        raw = session.get(key, default)
        text = str(raw if raw is not None else default).strip()
        return text or default

    def _float_cap(key: str) -> float | None:
        raw = session.get(key, 0.0)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    price_raw = session.get("explore_max_price_pln", price_max)
    try:
        price_value = float(price_raw)
    except (TypeError, ValueError):
        price_value = float(price_max)

    ppm_raw = session.get("explore_max_ppm", ppm_max)
    try:
        ppm_value = float(ppm_raw)
    except (TypeError, ValueError):
        ppm_value = float(ppm_max)

    max_age: int | None = None
    if age_max > 0 and "explore_max_age_months" in session:
        try:
            picked = int(session["explore_max_age_months"])
        except (TypeError, ValueError):
            picked = age_max
        if picked < age_max:
            max_age = picked

    floor_min_raw = session.get("explore_floor_min", -1)
    floor_max_raw = session.get("explore_floor_max", 99)
    year_built_raw = session.get("explore_year_built_min", 0)
    year_reno_raw = session.get("explore_year_renovated_min", 0)
    try:
        floor_min_i = int(floor_min_raw)
    except (TypeError, ValueError):
        floor_min_i = -1
    try:
        floor_max_i = int(floor_max_raw)
    except (TypeError, ValueError):
        floor_max_i = 99
    try:
        year_built_i = int(year_built_raw)
    except (TypeError, ValueError):
        year_built_i = 0
    try:
        year_reno_i = int(year_reno_raw)
    except (TypeError, ValueError):
        year_reno_i = 0

    state = FilterState(
        district=_str("explore_district"),
        source=_str("explore_source"),
        listing_kind=coerce_listing_kind(_str("explore_listing_kind")),
        oou=_str("explore_oou"),
        high_noise=_str("explore_high_noise"),
        poor_air=_str("explore_poor_air"),
        max_noise_total_lden_db=_float_cap("explore_max_noise_total"),
        max_traffic_weight_nearby=_float_cap("explore_max_traffic_weight"),
        max_major_traffic_weight_nearby=_float_cap("explore_max_major_traffic_weight"),
        frontage=_str("explore_frontage"),
        max_price_pln=price_value,
        max_price_per_m2=ppm_value,
        max_age_months=max_age,
        object_type=_str("explore_object_type"),
        ownership=_str("explore_ownership"),
        has_garage=_str("explore_garage"),
        has_outdoor_parking=_str("explore_outdoor_parking"),
        has_basement=_str("explore_basement"),
        has_balcony=_str("explore_balcony"),
        has_garden=_str("explore_garden"),
        has_terrace=_str("explore_terrace"),
        has_lift=_str("explore_lift"),
        is_attic=_str("explore_attic"),
        needs_renovation=_str("explore_renovation"),
        plumbing_replaced=_str("explore_plumbing"),
        wiring_replaced=_str("explore_wiring"),
        floor_min=None if floor_min_i < 0 else floor_min_i,
        floor_max=None if floor_max_i >= 99 else floor_max_i,
        year_built_min=None if year_built_i <= 0 else year_built_i,
        year_renovated_min=None if year_reno_i <= 0 else year_reno_i,
    )
    return FilterPreset(
        name=str(name or "").strip(),
        filters=state,
        starred_only=bool(session.get("explore_starred_only", False)),
    )


def _parse_presets_payload(payload: object) -> list[FilterPreset]:
    if isinstance(payload, list):
        raw_list = payload
    elif isinstance(payload, Mapping):
        raw_list = payload.get("presets", [])
    else:
        return []
    if not isinstance(raw_list, list):
        return []
    out: list[FilterPreset] = []
    seen: set[str] = set()
    for item in raw_list:
        preset = FilterPreset.from_dict(item)
        if preset is None or preset.name in seen:
            continue
        seen.add(preset.name)
        out.append(preset)
    return out


def _title_token(value: str) -> str:
    text = str(value).strip()
    if not text:
        return text
    if text.islower() or "_" in text:
        return text.replace("_", " ").title()
    return text
