"""Distinct SVG marker silhouettes for Explore map POI layers.

ScatterplotLayer is circles-only. IconLayer + inline SVG keeps shapes offline
and selection/hover-friendly without custom page CSS.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

MarkerIcon = dict[str, Any]
"""deck.gl IconLayer ``get_icon`` payload (url / width / height / anchorY)."""

MARKER_SHAPE_BY_KIND: dict[str, str] = {
    "hospital": "cross",
    "fire_station": "triangle",
    "police": "diamond",
    "shop": "square",
    "shops": "square",
    "air": "hexagon",
    "gunb": "star",
    "transit": "circle",
    "commute": "pin",
    "focus": "pin",
    "default": "circle",
}
"""Logical shape name per map layer / amenity kind."""

_SHAPE_LEGEND: tuple[tuple[str, str, str], ...] = (
    ("Offers", "○", "circle (Δ colour)"),
    ("Hospital", "✚", "cross"),
    ("Fire", "▲", "triangle"),
    ("Police", "◆", "diamond"),
    ("Shop", "■", "square"),
    ("Air", "⬡", "hexagon"),
    ("GUNB", "★", "star"),
    ("Transit", "●", "circle"),
    ("Commute / focus", "📍", "pin"),
)
"""Human legend rows for Map caption."""


def shape_for_kind(kind: str) -> str:
    """Return silhouette name for a layer / amenity key."""
    key = str(kind or "").strip().lower()
    return MARKER_SHAPE_BY_KIND.get(key, MARKER_SHAPE_BY_KIND["default"])


def rgba_to_css(rgba: list[int] | tuple[int, ...]) -> str:
    """Convert 3-4 channel 0-255 RGBA to CSS ``rgba()``."""
    r, g, b = (int(rgba[0]), int(rgba[1]), int(rgba[2]))
    alpha = float(rgba[3]) / 255.0 if len(rgba) > 3 else 1.0
    return f"rgba({r},{g},{b},{alpha:.3f})"


def render_shape_svg(shape: str, fill_css: str) -> str:
    """Compact 64x64 SVG path for ``shape`` filled with ``fill_css``."""
    stroke = "rgba(255,255,255,0.92)"
    common = (
        'xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64"'
    )
    if shape == "square":
        body = (
            f'<rect x="10" y="10" width="44" height="44" rx="4" '
            f'fill="{fill_css}" stroke="{stroke}" stroke-width="4"/>'
        )
    elif shape == "triangle":
        body = (
            f'<polygon points="32,8 56,54 8,54" fill="{fill_css}" '
            f'stroke="{stroke}" stroke-width="4" stroke-linejoin="round"/>'
        )
    elif shape == "diamond":
        body = (
            f'<polygon points="32,6 58,32 32,58 6,32" fill="{fill_css}" '
            f'stroke="{stroke}" stroke-width="4" stroke-linejoin="round"/>'
        )
    elif shape == "cross":
        body = (
            f'<rect x="26" y="8" width="12" height="48" rx="2" fill="{fill_css}" '
            f'stroke="{stroke}" stroke-width="3"/>'
            f'<rect x="8" y="26" width="48" height="12" rx="2" fill="{fill_css}" '
            f'stroke="{stroke}" stroke-width="3"/>'
        )
    elif shape == "hexagon":
        body = (
            f'<polygon points="32,6 54,19 54,45 32,58 10,45 10,19" '
            f'fill="{fill_css}" stroke="{stroke}" stroke-width="4" '
            f'stroke-linejoin="round"/>'
        )
    elif shape == "star":
        body = (
            f'<polygon points="32,6 38,24 56,24 42,36 48,54 32,44 16,54 22,36 '
            f'8,24 26,24" fill="{fill_css}" stroke="{stroke}" stroke-width="3" '
            f'stroke-linejoin="round"/>'
        )
    elif shape == "pin":
        body = (
            f'<path d="M32 4C21 4 12 13 12 24c0 14 20 36 20 36s20-22 20-36'
            f'C52 13 43 4 32 4z" fill="{fill_css}" stroke="{stroke}" '
            f'stroke-width="3"/>'
            f'<circle cx="32" cy="24" r="7" fill="{stroke}"/>'
        )
    else:
        body = (
            f'<circle cx="32" cy="32" r="22" fill="{fill_css}" '
            f'stroke="{stroke}" stroke-width="4"/>'
        )
    return f"<svg {common}>{body}</svg>"


def svg_data_url(svg: str) -> str:
    """Encode SVG markup as a ``data:image/svg+xml`` URL."""
    return "data:image/svg+xml;charset=utf-8," + quote(svg, safe="")


def marker_icon(
    kind: str,
    fill_rgba: list[int] | tuple[int, ...],
) -> MarkerIcon:
    """IconLayer payload for ``kind`` tinted with ``fill_rgba``."""
    shape = shape_for_kind(kind)
    url = svg_data_url(render_shape_svg(shape, rgba_to_css(fill_rgba)))
    anchor_y = 128 if shape == "pin" else 64
    return {
        "url": url,
        "width": 128,
        "height": 128,
        "anchorY": anchor_y,
    }


def map_shape_legend_caption() -> str:
    """One-line Map legend for enabled-style silhouette reminder."""
    parts = [f"{glyph} {label}" for label, glyph, _hint in _SHAPE_LEGEND]
    return "Shapes · " + " · ".join(parts)
