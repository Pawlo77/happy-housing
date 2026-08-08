"""Explore user preferences: structured palette + Map/Board hard filters."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Self

import pandas as pd

from happy_housing.app.io_cache import read_json

logger = logging.getLogger(__name__)

OOU_ANY: str = "(any)"
"""No aircraft-noise hard filter."""

OOU_OUTSIDE_ONLY: str = "outside_only"
"""Hard-exclude Lden ≥55 dB listings."""

OOU_PREFER_OUTSIDE: str = "prefer_outside"
"""Soft preference only (scorer); no hard filter."""

OOU_INSIDE_OK: str = "inside_ok"
"""Explicitly allow inside noise contours."""

_OOU_VALUES: frozenset[str] = frozenset({
    OOU_ANY,
    OOU_OUTSIDE_ONLY,
    OOU_PREFER_OUTSIDE,
    OOU_INSIDE_OK,
})


@dataclass(slots=True)
class UserPreferences:
    """Structured buyer preferences for Explore Map/Board/News."""

    enabled: bool = False
    """When True, hard-filter Map/Board and drive recommendations."""
    districts: list[str] = field(default_factory=list)
    """Allowed districts; empty = any."""
    sources: list[str] = field(default_factory=list)
    """Allowed listing sources; empty = any."""
    oou: str = OOU_ANY
    """Aircraft Lden: (any) / prefer_outside / outside_only / inside_ok."""
    min_price_pln: float | None = None
    """Lower bound on asking price."""
    max_price_pln: float | None = None
    """Upper bound on asking price."""
    min_price_per_m2: float | None = None
    """Lower bound on listing zł/m²."""
    max_price_per_m2: float | None = None
    """Upper bound on listing zł/m²."""
    min_area_m2: float | None = None
    """Minimum floor area."""
    max_area_m2: float | None = None
    """Maximum floor area."""
    min_rooms: int | None = None
    """Minimum room count."""
    max_rooms: int | None = None
    """Maximum room count."""
    min_year_built: int | None = None
    """Oldest acceptable build year."""
    max_commute_min: float | None = None
    """Max minutes to Daszyńskiego."""
    max_dist_transit_m: float | None = None
    """Max metres to nearest transit stop."""
    max_dist_hospital_m: float | None = None
    """Max metres to nearest hospital."""
    max_dist_fire_m: float | None = None
    """Max metres to nearest fire station."""
    max_dist_police_m: float | None = None
    """Max metres to nearest police station."""
    max_dist_air_m: float | None = None
    """Max metres to nearest air-quality station."""
    max_dist_gunb_m: float | None = None
    """Max metres to nearest GUNB site."""
    max_dist_park_m: float | None = None
    """Max metres to nearest UM Warszawa park / square."""
    max_dist_bike_path_m: float | None = None
    """Max metres to nearest OSM bike path."""
    max_noise_total_lden_db: float | None = None
    """Max energy-sum total Lden (dB) from UM strategic noise maps."""
    max_delta_price_per_m2: float | None = None
    """Max listing - model delta (e.g. 0 = fair or better)."""
    require_underpriced: bool = False
    """When True, keep only negative ``delta_price_per_m2``."""
    nl_text: str = ""
    """Last natural-language preference note (audit / LLM input)."""

    def __post_init__(self) -> None:
        if self.oou not in _OOU_VALUES:
            self.oou = OOU_ANY
        self.districts = [str(d).strip() for d in self.districts if str(d).strip()]
        self.sources = [str(s).strip() for s in self.sources if str(s).strip()]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable payload."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | object) -> Self:
        """Build preferences from a mapping; unknown keys ignored."""
        if not isinstance(payload, Mapping):
            return cls()
        known = {f.name for f in fields(cls)}
        data: dict[str, Any] = {}
        for key, value in payload.items():
            name = str(key)
            if name not in known:
                continue
            data[name] = value
        # Coerce list fields.
        for list_key in ("districts", "sources"):
            if list_key in data and not isinstance(data[list_key], list):
                text = str(data[list_key] or "")
                data[list_key] = [p.strip() for p in text.split(",") if p.strip()]
        for float_key in (
            "min_price_pln",
            "max_price_pln",
            "min_price_per_m2",
            "max_price_per_m2",
            "min_area_m2",
            "max_area_m2",
            "max_commute_min",
            "max_dist_transit_m",
            "max_dist_hospital_m",
            "max_dist_fire_m",
            "max_dist_police_m",
            "max_dist_air_m",
            "max_dist_gunb_m",
            "max_dist_park_m",
            "max_dist_bike_path_m",
            "max_noise_total_lden_db",
            "max_delta_price_per_m2",
        ):
            if float_key in data:
                data[float_key] = _opt_float(data[float_key])
        for int_key in ("min_rooms", "max_rooms", "min_year_built"):
            if int_key in data:
                data[int_key] = _opt_int(data[int_key])
        if "enabled" in data:
            data["enabled"] = bool(data["enabled"])
        if "require_underpriced" in data:
            data["require_underpriced"] = bool(data["require_underpriced"])
        if "nl_text" in data:
            data["nl_text"] = str(data["nl_text"] or "")
        if "oou" in data:
            data["oou"] = str(data["oou"] or OOU_ANY)
        return cls(**data)


def preferences_path(processed: Path) -> Path:
    """JSON path for Explore user preferences."""
    return processed / "user_preferences.json"


def load_preferences(path: Path) -> UserPreferences:
    """Load preferences; missing/invalid → defaults."""
    if not path.exists():
        return UserPreferences()
    try:
        payload = read_json(str(path), path.stat().st_mtime)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        logger.warning("Failed to load preferences %s: %s", path, exc)
        return UserPreferences()
    return UserPreferences.from_dict(payload)


def save_preferences(path: Path, prefs: UserPreferences) -> None:
    """Persist preferences JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(prefs.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _opt_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _opt_int(value: object) -> int | None:
    number = _opt_float(value)
    if number is None:
        return None
    return round(number)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float("nan"), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def apply_preferences(frame: pd.DataFrame, prefs: UserPreferences) -> pd.DataFrame:
    """Hard-filter rows to match set preference fields (no-op when disabled)."""
    if frame.empty or not prefs.enabled:
        return frame.copy()
    mask = pd.Series(True, index=frame.index, dtype=bool)
    if prefs.districts and "district" in frame.columns:
        wanted = {d.casefold() for d in prefs.districts}
        mask &= frame["district"].astype(str).str.casefold().isin(wanted)
    if prefs.sources and "source" in frame.columns:
        wanted_src = {s.casefold() for s in prefs.sources}
        mask &= frame["source"].astype(str).str.casefold().isin(wanted_src)
    if prefs.oou == OOU_OUTSIDE_ONLY and "in_oou" in frame.columns:
        mask &= _numeric(frame, "in_oou").eq(0)
    if prefs.min_price_pln is not None:
        mask &= _numeric(frame, "price_pln").ge(float(prefs.min_price_pln))
    if prefs.max_price_pln is not None:
        mask &= _numeric(frame, "price_pln").le(float(prefs.max_price_pln))
    if prefs.min_price_per_m2 is not None:
        mask &= _numeric(frame, "price_per_m2").ge(float(prefs.min_price_per_m2))
    if prefs.max_price_per_m2 is not None:
        mask &= _numeric(frame, "price_per_m2").le(float(prefs.max_price_per_m2))
    if prefs.min_area_m2 is not None:
        mask &= _numeric(frame, "area_m2").ge(float(prefs.min_area_m2))
    if prefs.max_area_m2 is not None:
        mask &= _numeric(frame, "area_m2").le(float(prefs.max_area_m2))
    if prefs.min_rooms is not None:
        mask &= _numeric(frame, "rooms").ge(float(prefs.min_rooms))
    if prefs.max_rooms is not None:
        mask &= _numeric(frame, "rooms").le(float(prefs.max_rooms))
    if prefs.min_year_built is not None:
        mask &= _numeric(frame, "year_built").ge(float(prefs.min_year_built))
    if prefs.max_commute_min is not None:
        mask &= _numeric(frame, "commute_min").le(float(prefs.max_commute_min))
    if prefs.max_dist_transit_m is not None:
        mask &= _numeric(frame, "dist_transit_stop_m").le(
            float(prefs.max_dist_transit_m)
        )
    if prefs.max_dist_hospital_m is not None:
        mask &= _numeric(frame, "dist_hospital_m").le(float(prefs.max_dist_hospital_m))
    if prefs.max_dist_fire_m is not None:
        mask &= _numeric(frame, "dist_fire_station_m").le(float(prefs.max_dist_fire_m))
    if prefs.max_dist_police_m is not None:
        mask &= _numeric(frame, "dist_police_m").le(float(prefs.max_dist_police_m))
    if prefs.max_dist_air_m is not None:
        mask &= _numeric(frame, "dist_air_station_m").le(float(prefs.max_dist_air_m))
    if prefs.max_dist_gunb_m is not None:
        mask &= _numeric(frame, "dist_gunb_m").le(float(prefs.max_dist_gunb_m))
    if prefs.max_dist_park_m is not None:
        mask &= _numeric(frame, "dist_park_m").le(float(prefs.max_dist_park_m))
    if prefs.max_dist_bike_path_m is not None:
        mask &= _numeric(frame, "dist_bike_path_m").le(
            float(prefs.max_dist_bike_path_m)
        )
    if prefs.max_noise_total_lden_db is not None:
        mask &= _numeric(frame, "noise_total_lden_db").le(
            float(prefs.max_noise_total_lden_db)
        )
    if prefs.max_delta_price_per_m2 is not None:
        mask &= _numeric(frame, "delta_price_per_m2").le(
            float(prefs.max_delta_price_per_m2)
        )
    if prefs.require_underpriced:
        mask &= _numeric(frame, "delta_price_per_m2").lt(0.0)
    return frame.loc[mask].reset_index(drop=True)
