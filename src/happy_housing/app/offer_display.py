"""Pure display helpers for the Explore Offer subtabs."""

import json
import re
from html.parser import HTMLParser
from typing import Any

import pandas as pd

from happy_housing.app.commute_modes import (
    COMMUTE_MODE_COLUMN,
    COMMUTE_MODE_LABELS,
    coerce_commute_display_mode,
)
from happy_housing.ingest.fx import format_original_price

_DELTA_NEAR_ZL = 500.0
"""Abs delta (zł/m²) treated as fair / yellow band."""

_ALLOWED_HTML_TAGS: frozenset[str] = frozenset({
    "p",
    "br",
    "hr",
    "ul",
    "ol",
    "li",
    "strong",
    "b",
    "em",
    "i",
    "u",
    "a",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "blockquote",
    "div",
    "span",
})
"""Tags kept when rendering Otodom listing descriptions."""

_VOID_HTML_TAGS: frozenset[str] = frozenset({"br", "hr"})
"""Self-closing tags in the allowlist."""

_SKIP_HTML_TAGS: frozenset[str] = frozenset({
    "script",
    "style",
    "iframe",
    "object",
    "embed",
    "svg",
    "math",
})
"""Tags whose content is dropped entirely."""


class _ListingHtmlSanitizer(HTMLParser):
    """Allowlist sanitizer for Otodom HTML descriptions (no CSS)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        name = tag.lower()
        if name in _SKIP_HTML_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth or name not in _ALLOWED_HTML_TAGS:
            return
        if name == "a":
            href = ""
            for key, value in attrs:
                if key.lower() == "href" and value:
                    href = value.strip()
                    break
            if href.startswith(("http://", "https://", "/")):
                safe = href.replace('"', "&quot;")
                self._parts.append(f'<a href="{safe}" rel="noopener noreferrer">')
            else:
                self._parts.append("<a>")
            return
        if name in _VOID_HTML_TAGS:
            self._parts.append(f"<{name}>")
            return
        self._parts.append(f"<{name}>")

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if name in _SKIP_HTML_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth or name not in _ALLOWED_HTML_TAGS:
            return
        if name in _VOID_HTML_TAGS:
            return
        self._parts.append(f"</{name}>")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self._parts.append(
            data.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )

    def handle_entityref(self, name: str) -> None:
        if self._skip_depth:
            return
        self._parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self._skip_depth:
            return
        self._parts.append(f"&#{name};")

    def result(self) -> str:
        """Return sanitized HTML."""
        return "".join(self._parts)


def sanitize_listing_html(raw: str) -> str:
    """Allowlist-sanitize listing HTML for safe Streamlit rendering.

    Keeps basic text markup (paragraphs, lists, bold/italic, links). Drops
    scripts, styles, images, and event-handler attributes.
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    if "<" not in text:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    parser = _ListingHtmlSanitizer()
    try:
        parser.feed(text)
        parser.close()
    except Exception:
        return re.sub(r"<[^>]+>", "", text)
    return parser.result().strip()


def _urls_from_json_cell(value: object) -> list[str]:
    """Parse a JSON list of URL strings from a dataframe cell."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [str(item).strip() for item in payload if str(item).strip()]


def image_urls_from_row(row: pd.Series | dict[str, Any]) -> list[str]:
    """Listing photo URLs (multi-image JSON, else single ``image_url``)."""
    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    urls = _urls_from_json_cell(getter("image_urls_json"))
    if urls:
        return urls
    single = getter("image_url")
    if isinstance(single, str) and single.strip():
        return [single.strip()]
    return []


def floor_plan_urls_from_row(row: pd.Series | dict[str, Any]) -> list[str]:
    """Floor-plan / layout image URLs from ``floor_plan_urls_json``."""
    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    return _urls_from_json_cell(getter("floor_plan_urls_json"))


def round_meters(value: object) -> int | None:
    """Round a distance to whole metres; return ``None`` when missing."""
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]  # pandas/object scalar
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


def flag_yes_no(value: object) -> str:
    """Render a 0/1-ish flag; NaN / missing → ``No`` (never raise on NaN)."""
    number = pd.to_numeric(value, errors="coerce")
    try:
        as_float = float(number)
    except (TypeError, ValueError):
        return "No"
    if as_float != as_float:
        return "No"
    return "Yes" if int(as_float) == 1 else "No"


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


def livability_rows(
    row: pd.Series,
    *,
    commute_label: str = "commute target",
    commute_mode: object | None = None,
) -> pd.DataFrame:
    """Livability signals with distances rounded to whole metres."""
    short = commute_label if len(commute_label) <= 40 else commute_label[:37] + "..."
    mode = coerce_commute_display_mode(commute_mode)
    primary_col = COMMUTE_MODE_COLUMN[mode]
    primary_label = f"{COMMUTE_MODE_LABELS[mode]} to {short}"
    other_specs = tuple(
        (col, f"{COMMUTE_MODE_LABELS[m]} to {short}", "min")
        for m, col in COMMUTE_MODE_COLUMN.items()
        if m != mode
    )
    specs: tuple[tuple[str, str, str], ...] = (
        ("noise_total_lden_db", "Total Lden (dB, all sources)", "db"),
        ("noise_total_lnight_db", "Total Lnight (dB, all sources)", "db"),
        ("in_high_noise", "Total Lden ≥ 55 dB", "flag"),
        ("road_lden_db", "Road Lden (dB)", "db"),
        ("road_lnight_db", "Road Lnight (dB)", "db"),
        ("rail_lden_db", "Rail Lden (dB)", "db"),
        ("rail_lnight_db", "Rail Lnight (dB)", "db"),
        ("tram_lden_db", "Tram Lden (dB)", "db"),
        ("tram_lnight_db", "Tram Lnight (dB)", "db"),
        ("aircraft_lden_db", "Aircraft Lden (dB)", "db"),
        ("aircraft_lnight_db", "Aircraft Lnight (dB)", "db"),
        ("industry_lden_db", "Industry Lden (dB)", "db"),
        ("industry_lnight_db", "Industry Lnight (dB)", "db"),
        ("in_oou", "Aircraft Lden ≥ 55 dB", "flag"),
        (primary_col, primary_label, "min"),
        *other_specs,
        ("dist_transit_stop_m", "Nearest transit stop", "m"),
        ("dist_metro_m", "Nearest metro", "m"),
        ("dist_hospital_m", "Nearest hospital", "m"),
        ("dist_fire_station_m", "Nearest fire station", "m"),
        ("dist_police_m", "Nearest police", "m"),
        ("dist_charging_station_m", "Nearest EV charger", "m"),
        ("dist_fuel_m", "Nearest gas station", "m"),
        ("dist_air_station_m", "Nearest air station", "m"),
        ("air_aq_index", "Air AQI (IDW)", "aq"),
        ("air_pm10_ug_m3", "PM10 µg/m³ (IDW)", "ug"),
        ("air_pm25_ug_m3", "PM2.5 µg/m³ (IDW)", "ug"),
        ("air_no2_ug_m3", "NO₂ µg/m³ (IDW)", "ug"),
        ("in_poor_air", "AQI ≥ 3 (IDW field)", "flag"),
        ("dist_gunb_m", "Nearest GUNB permit", "m"),
        ("dist_park_m", "Nearest park / square", "m"),
        ("dist_bike_path_m", "Nearest bike path", "m"),
        ("bike_path_coverage_m", "Bike path coverage nearby", "m"),
        ("dist_rail_area_m", "Nearest rail yard / corridor", "m"),
        ("dist_military_m", "Nearest military area", "m"),
        ("dist_airport_area_m", "Nearest airport footprint", "m"),
        ("dist_traffic_road_m", "Nearest drive road (sample)", "m"),
        ("dist_major_road_m", "Nearest major road (sample)", "m"),
        ("traffic_weight_nearby", "Traffic weight nearby", "aq"),
        ("major_traffic_weight_nearby", "Major-road traffic weight nearby", "aq"),
        ("nearest_highway", "Nearest highway class", "text"),
        ("street_facing_hint", "Street-facing (description hint)", "flag"),
        ("quiet_side_hint", "Quiet / courtyard side (description hint)", "flag"),
        ("dist_daszynskiego_m", f"Distance to {short}", "m"),
    )
    out: list[dict[str, Any]] = []
    for col, label, kind in specs:
        raw = row.get(col)
        if kind == "flag":
            display = flag_yes_no(raw)
        elif kind == "text":
            display = str(raw).strip() if raw is not None and str(raw).strip() else "—"
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
        elif kind == "aq":
            number = pd.to_numeric(raw, errors="coerce")
            try:
                as_float = float(number)
            except (TypeError, ValueError):
                display = "—"
            else:
                display = "—" if as_float != as_float else f"{as_float:.0f}"
        elif kind == "ug":
            number = pd.to_numeric(raw, errors="coerce")
            try:
                as_float = float(number)
            except (TypeError, ValueError):
                display = "—"
            else:
                display = "—" if as_float != as_float else f"{as_float:.1f} µg/m³"
        else:
            display = format_meters(raw)
        out.append({"signal": label, "value": display})
    return pd.DataFrame(out)


def format_fact_value(value: object) -> str:
    """Render a fact cell as text so ``st.dataframe`` Arrow serialization succeeds."""
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if number != number:
            return "—"
        if number.is_integer():
            return f"{int(number)}"
        return f"{number:g}"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return "—"
    return text


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
        ("Year renovated", row.get("year_renovated")),
        ("Object type", row.get("object_type")),
        ("Building type", row.get("building_type")),
        (
            "Ownership",
            (
                row.get("ownership")
                if pd.notna(row.get("ownership"))
                and str(row.get("ownership") or "").strip()
                else row.get("building_ownership")
            ),
        ),
        ("Needs renovation", flag_yes_no(row.get("needs_renovation"))),
        ("Heating", row.get("heating")),
        ("Rent (PLN/mo)", row.get("rent")),
        ("Balcony", flag_yes_no(row.get("has_balcony"))),
        ("Terrace", flag_yes_no(row.get("has_terrace"))),
        ("Garden", flag_yes_no(row.get("has_garden"))),
        ("Garage", flag_yes_no(row.get("has_garage"))),
        ("Outdoor parking", flag_yes_no(row.get("has_outdoor_parking"))),
        ("Basement", flag_yes_no(row.get("has_basement"))),
        ("Attic", flag_yes_no(row.get("is_attic"))),
        ("Lift", flag_yes_no(row.get("has_lift"))),
        ("Plumbing replaced", flag_yes_no(row.get("plumbing_replaced"))),
        ("Wiring replaced", flag_yes_no(row.get("wiring_replaced"))),
        ("Private owner", flag_yes_no(row.get("is_private_owner"))),
        (
            "Price (original)",
            format_original_price(row.get("price_original"), row.get("price_currency")),
        ),
        ("Price (PLN)", row.get("price_pln")),
        ("Latitude", row.get("lat")),
        ("Longitude", row.get("lon")),
    ]
    return pd.DataFrame(
        [(field, format_fact_value(value)) for field, value in facts],
        columns=["field", "value"],
    )
