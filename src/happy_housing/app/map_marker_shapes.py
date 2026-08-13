"""Emoji SVG markers for Explore map POI layers.

ScatterplotLayer is circles-only. IconLayer + inline SVG keeps glyphs offline
and selection/hover-friendly without custom page CSS.
"""

from typing import Any
from urllib.parse import quote

from happy_housing.gis.amenity_catalog import amenity_keys

MarkerIcon = dict[str, Any]
"""deck.gl IconLayer ``get_icon`` payload (url / width / height / anchorY)."""

_FALLBACK_EMOJI_PALETTE: tuple[str, ...] = (
    "⭐",
    "🔹",
    "🔸",
    "💠",
    "✳️",
    "❇️",
    "🔶",
    "🔷",
    "▪️",
    "▫️",
    "◾",
    "◽",
    "⬛",
    "⬜",
    "🔘",
    "⚪",
)
"""Rotating glyphs for unknown amenity catalog keys."""

_SEMANTIC_EMOJIS: dict[str, str] = {
    # Core amenities / layers
    "hospital": "🏥",
    "fire_station": "🚒",
    "police": "👮",
    "police_um": "👮",
    "shop": "🛍️",
    "shops": "🛍️",
    "grocery": "🛒",
    "health_beauty": "💄",
    "fashion": "👗",
    "home": "🛋️",
    "mobility": "🚲",
    "other": "🏪",
    "air": "💨",
    "gunb": "🏗️",
    "transit": "🚌",
    "commute": "📍",
    "focus": "📍",
    "address": "📍",
    "charging_station": "🔌",
    "fuel": "⛽",
    "default": "📍",
    # Sports
    "gym": "🏋️",
    "outdoor_gym": "🏋️",
    "street_workout": "🤸",
    "indoor_pool": "🏊",
    "outdoor_pool": "🏊",
    "fitness": "💪",
    "sauna": "🧖",
    "sports_hall": "🏟️",
    "gymnasium": "🏟️",
    "football_pitch": "⚽",
    "multipurpose_pitch": "🏟️",
    "basketball": "🏀",
    "volleyball": "🏐",
    "beach_volleyball": "🏐",
    "handball": "🤾",
    "beach_handball": "🤾",
    "field_hockey": "🏑",
    "baseball": "⚾",
    "tennis": "🎾",
    "tennis_wall": "🎾",
    "squash": "🏸",
    "ricochet": "🏸",
    "running_track": "🏃",
    "stadium": "🏟️",
    "athletics_field": "🏃",
    "skatepark": "🛹",
    "pumptrack": "🛹",
    "bowling": "🎳",
    "climbing_wall": "🧗",
    "table_tennis": "🏓",
    "minigolf": "⛳",
    "shooting_range": "🎯",
    "boules": "⚪",
    "archery": "🏹",
    "ski_slope": "⛷️",
    "luge_track": "🛷",
    "roller_ski": "🛼",
    "indoor_ice_rink": "⛸️",
    "outdoor_ice_rink": "⛸️",
    "ice_track": "⛸️",
    # Civic
    "park": "🌳",
    "school": "🏫",
    "library": "📚",
    "pharmacy": "💊",
    "bike_share": "🚲",
    "park_and_ride": "🅿️",
    "hotel": "🏨",
    "theater": "🎭",
    "cemetery": "🪦",
    "dormitory": "🛏️",
    "metro": "🚇",
    "metro_entrance": "🚇",
    "senior_club": "🧓",
}
"""Hand-picked emojis for familiar layer / amenity / shop kinds."""


def _catalog_emoji_map() -> dict[str, str]:
    """Stable emoji per amenity catalog key, overridden by semantics."""
    out: dict[str, str] = {}
    n = len(_FALLBACK_EMOJI_PALETTE)
    for i, key in enumerate(amenity_keys()):
        out[key] = _FALLBACK_EMOJI_PALETTE[i % n]
    out.update(_SEMANTIC_EMOJIS)
    return out


MARKER_EMOJI_BY_KIND: dict[str, str] = _catalog_emoji_map()
"""Emoji glyph per map layer / amenity / shop kind."""

_EMOJI_LEGEND: tuple[tuple[str, str], ...] = (
    ("Offers", "○"),
    ("Hospital", "🏥"),
    ("Fire", "🚒"),
    ("Police", "👮"),
    ("Grocery", "🛒"),
    ("Gym", "🏋️"),
    ("Pool", "🏊"),
    ("Air", "💨"),
    ("GUNB", "🏗️"),
    ("Transit", "🚌"),
    ("Commute / focus", "📍"),
)
"""Human legend rows for Map caption."""


def emoji_for_kind(kind: str) -> str:
    """Return emoji glyph for a layer / amenity / shop key."""
    key = str(kind or "").strip().lower()
    if key in MARKER_EMOJI_BY_KIND:
        return MARKER_EMOJI_BY_KIND[key]
    if key:
        idx = sum(ord(c) for c in key) % len(_FALLBACK_EMOJI_PALETTE)
        return _FALLBACK_EMOJI_PALETTE[idx]
    return MARKER_EMOJI_BY_KIND["default"]


def rgba_to_css(rgba: list[int] | tuple[int, ...]) -> str:
    """Convert 3-4 channel 0-255 RGBA to CSS ``rgba()``."""
    r, g, b = (int(rgba[0]), int(rgba[1]), int(rgba[2]))
    alpha = float(rgba[3]) / 255.0 if len(rgba) > 3 else 1.0
    return f"rgba({r},{g},{b},{alpha:.3f})"


def render_emoji_marker_svg(emoji: str, fill_css: str) -> str:
    """64x64 SVG: tinted disc + centered emoji for IconLayer atlases."""
    stroke = "rgba(255,255,255,0.95)"
    common = (
        'xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64"'
    )
    # Soft colour disc keeps category tint; emoji carries the type signal.
    disc = (
        f'<circle cx="32" cy="32" r="28" fill="{fill_css}" '
        f'stroke="{stroke}" stroke-width="3"/>'
    )
    # Escape XML-significant chars if any slip into glyphs (emoji-safe normally).
    safe = emoji.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = (
        f'<text x="32" y="34" font-size="30" text-anchor="middle" '
        f'dominant-baseline="central">{safe}</text>'
    )
    return f"<svg {common}>{disc}{text}</svg>"


def svg_data_url(svg: str) -> str:
    """Encode SVG markup as a ``data:image/svg+xml`` URL."""
    return "data:image/svg+xml;charset=utf-8," + quote(svg, safe="")


def marker_icon(
    kind: str,
    fill_rgba: list[int] | tuple[int, ...],
) -> MarkerIcon:
    """IconLayer payload for ``kind`` tinted with ``fill_rgba``."""
    emoji = emoji_for_kind(kind)
    url = svg_data_url(render_emoji_marker_svg(emoji, rgba_to_css(fill_rgba)))
    return {
        "url": url,
        "width": 128,
        "height": 128,
        "anchorY": 64,
    }


def map_shape_legend_caption() -> str:
    """One-line Map legend for emoji marker reminder."""
    parts = [f"{glyph} {label}" for label, glyph in _EMOJI_LEGEND]
    return "Markers · " + " · ".join(parts)
