"""Pure display helpers for the Explore Offer subtabs."""

from __future__ import annotations

from typing import Any

import pandas as pd

_DELTA_NEAR_ZL = 500.0
"""Abs delta (zł/m²) treated as fair / yellow band."""


def round_meters(value: object) -> int | None:
    """Round a distance to whole metres; return ``None`` when missing."""
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return round(number)


def format_meters(value: object) -> str:
    """Human-readable whole-metre distance."""
    metres = round_meters(value)
    if metres is None:
        return "—"
    return f"{metres:,} m"


def delta_band(delta: float) -> str:
    """Classify listing-model delta as ``green``, ``yellow``, or ``red``."""
    if delta < -_DELTA_NEAR_ZL:
        return "green"
    if delta > _DELTA_NEAR_ZL:
        return "red"
    return "yellow"


def delta_hex(delta: float) -> str:
    """Hex colour for a listing-model delta band."""
    return {
        "green": "#15803D",
        "yellow": "#A16207",
        "red": "#B91C1C",
    }[delta_band(delta)]


def livability_rows(row: pd.Series) -> pd.DataFrame:
    """Livability signals with distances rounded to whole metres."""
    specs: tuple[tuple[str, str, str], ...] = (
        ("aircraft_lden_db", "Aircraft Lden (dB)", "db"),
        ("aircraft_lnight_db", "Aircraft Lnight (dB)", "db"),
        ("in_oou", "Lden ≥ 55 dB", "flag"),
        ("commute_min", "Commute to Daszyńskiego", "min"),
        ("dist_transit_stop_m", "Nearest transit stop", "m"),
        ("dist_metro_m", "Nearest metro", "m"),
        ("dist_hospital_m", "Nearest hospital", "m"),
        ("dist_fire_station_m", "Nearest fire station", "m"),
        ("dist_police_m", "Nearest police", "m"),
        ("dist_air_station_m", "Nearest air station", "m"),
        ("dist_gunb_m", "Nearest GUNB permit", "m"),
        ("dist_daszynskiego_m", "Distance to Daszyńskiego", "m"),
    )
    out: list[dict[str, Any]] = []
    for col, label, kind in specs:
        raw = row.get(col)
        if kind == "flag":
            display = "Yes" if int(raw or 0) == 1 else "No"
        elif kind == "min":
            metres_proxy = round_meters(raw)
            display = "—" if metres_proxy is None else f"{metres_proxy} min"
        elif kind == "db":
            metres_proxy = round_meters(raw)
            if metres_proxy is None:
                display = "—"
            elif metres_proxy <= 0:
                display = "below band / 0"
            else:
                display = f"≥ {metres_proxy} dB"
        else:
            display = format_meters(raw)
        out.append({"signal": label, "value": display})
    return pd.DataFrame(out)


def offer_fact_rows(row: pd.Series) -> pd.DataFrame:
    """Core offer facts for the Overview subtab."""
    facts: list[tuple[str, object]] = [
        ("Listing ID", row.get("listing_id")),
        ("Title", row.get("title")),
        ("District", row.get("district")),
        ("Source", row.get("source")),
        ("Listed at", row.get("listed_at")),
        ("Market", row.get("market")),
        ("Rooms", row.get("rooms")),
        ("Floor", row.get("floor_no")),
        ("Building floors", row.get("building_floors_num")),
        ("Area (m²)", round_meters(row.get("area_m2"))),
        ("Year built", row.get("year_built")),
        ("Building type", row.get("building_type")),
        ("Ownership", row.get("building_ownership")),
        ("Heating", row.get("heating")),
        ("Rent (PLN/mo)", row.get("rent")),
        ("Balcony", "Yes" if int(row.get("has_balcony") or 0) == 1 else "No"),
        ("Garage / parking", "Yes" if int(row.get("has_garage") or 0) == 1 else "No"),
        ("Lift", "Yes" if int(row.get("has_lift") or 0) == 1 else "No"),
        ("Private owner", "Yes" if int(row.get("is_private_owner") or 0) else "No"),
        ("Price (PLN)", row.get("price_pln")),
        ("Latitude", row.get("lat")),
        ("Longitude", row.get("lon")),
    ]
    return pd.DataFrame(facts, columns=["field", "value"])
