"""Door-to-door commute route plans for the Explore Offer Map."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from happy_housing.app.commute_modes import (
    COMMUTE_DISPLAY_MODES,
    COMMUTE_MODE_LABELS,
    coerce_commute_display_mode,
)
from happy_housing.gis.polyline import decode_google_polyline
from happy_housing.gis.routing import osrm_route_geojson, otp_plan_best_itinerary

_LEG_RGBA: dict[str, tuple[int, int, int, int]] = {
    "WALK": (100, 116, 139, 220),
    "BICYCLE": (34, 197, 94, 230),
    "BUS": (37, 99, 235, 230),
    "TRAM": (220, 38, 38, 230),
    "SUBWAY": (147, 51, 234, 230),
    "RAIL": (15, 23, 42, 230),
    "CAR": (15, 107, 92, 230),
}
"""RGBA stroke colours per OTP/OSRM leg mode."""

_DEFAULT_LEG_RGBA: tuple[int, int, int, int] = (51, 65, 85, 220)
"""Fallback path colour."""


@dataclass(frozen=True, slots=True)
class CommuteRouteLeg:
    """One walk / ride / drive segment of a commute plan."""

    mode: str
    """OTP/OSRM mode token (``WALK``, ``BUS``, ``CAR``, …)."""
    from_name: str
    """Start place / stop name."""
    to_name: str
    """End place / stop name."""
    duration_s: float
    """Leg duration in seconds."""
    distance_m: float | None
    """Leg distance in metres when known."""
    route_short_name: str
    """Transit route short name (e.g. ``175``); empty for walk/car."""
    headsign: str
    """Transit headsign when present."""
    path: list[list[float]] = field(default_factory=list)
    """``[[lon, lat], …]`` polyline for the map."""


@dataclass(frozen=True, slots=True)
class CommuteRoutePlan:
    """Best door-to-door commute for one mode."""

    mode: str
    """Display mode: ``transit`` / ``car`` / ``bike`` / ``walk``."""
    duration_s: float
    """Total duration in seconds."""
    distance_m: float | None
    """Total distance in metres when known."""
    legs: tuple[CommuteRouteLeg, ...]
    """Ordered itinerary legs."""
    summary: str
    """One-line human summary."""


def parse_otp_itinerary(
    itinerary: dict[str, Any],
    *,
    mode: str = "transit",
) -> CommuteRoutePlan | None:
    """Build a ``CommuteRoutePlan`` from one OTP itinerary dict."""
    if not isinstance(itinerary, dict):
        return None
    raw_legs = itinerary.get("legs") or []
    if not isinstance(raw_legs, list) or not raw_legs:
        return None
    legs: list[CommuteRouteLeg] = []
    for raw in raw_legs:
        if not isinstance(raw, dict):
            continue
        from_raw = raw.get("from")
        to_raw = raw.get("to")
        geom_raw = raw.get("legGeometry")
        from_place: dict[str, Any] = from_raw if isinstance(from_raw, dict) else {}
        to_place: dict[str, Any] = to_raw if isinstance(to_raw, dict) else {}
        geom: dict[str, Any] = geom_raw if isinstance(geom_raw, dict) else {}
        points = decode_google_polyline(str(geom.get("points") or ""))
        if len(points) < 2:
            try:
                points = [
                    [float(from_place["lon"]), float(from_place["lat"])],
                    [float(to_place["lon"]), float(to_place["lat"])],
                ]
            except (KeyError, TypeError, ValueError):
                points = []
        duration = float(raw.get("duration") or 0.0)
        distance_raw = raw.get("distance")
        try:
            distance_m = float(distance_raw) if distance_raw is not None else None
        except (TypeError, ValueError):
            distance_m = None
        legs.append(
            CommuteRouteLeg(
                mode=str(raw.get("mode") or "WALK").upper(),
                from_name=str(from_place.get("name") or "Start").strip() or "Start",
                to_name=str(to_place.get("name") or "End").strip() or "End",
                duration_s=duration,
                distance_m=distance_m,
                route_short_name=str(raw.get("routeShortName") or "").strip(),
                headsign=str(raw.get("headsign") or "").strip(),
                path=points,
            )
        )
    if not legs:
        return None
    total = float(itinerary.get("duration") or sum(leg.duration_s for leg in legs))
    dist_vals = [leg.distance_m for leg in legs if leg.distance_m is not None]
    total_dist = float(sum(dist_vals)) if dist_vals else None
    label = COMMUTE_MODE_LABELS.get(coerce_commute_display_mode(mode), str(mode))
    return CommuteRoutePlan(
        mode=coerce_commute_display_mode(mode),
        duration_s=total,
        distance_m=total_dist,
        legs=tuple(legs),
        summary=f"{label} · {total / 60.0:.0f} min · {len(legs)} leg(s)",
    )


def format_commute_route_markdown(plan: CommuteRoutePlan) -> str:
    """Markdown itinerary for the Offer Map panel."""
    lines = [
        f"**{plan.summary}**",
        "",
    ]
    for idx, leg in enumerate(plan.legs, start=1):
        mins = max(1, round(leg.duration_s / 60.0)) if leg.duration_s else 0
        dist = (
            f" · {leg.distance_m:,.0f} m"
            if leg.distance_m is not None and leg.distance_m == leg.distance_m
            else ""
        )
        if leg.mode in {"WALK", "BICYCLE", "CAR"}:
            verb = {
                "WALK": "Walk",
                "BICYCLE": "Bike",
                "CAR": "Drive",
            }.get(leg.mode, leg.mode.title())
            lines.append(
                f"{idx}. **{verb}** {leg.from_name} → {leg.to_name} ({mins} min{dist})"
            )
        else:
            route = leg.route_short_name or leg.mode
            head = f" toward {leg.headsign}" if leg.headsign else ""
            lines.append(
                f"{idx}. **{leg.mode.title()} {route}**{head}: "
                f"board at {leg.from_name}, alight at {leg.to_name} "
                f"({mins} min{dist})"
            )
    return "\n".join(lines)


def path_records_for_plan(plan: CommuteRoutePlan) -> list[dict[str, Any]]:
    """Rows for a pydeck PathLayer (one row per leg with geometry)."""
    rows: list[dict[str, Any]] = []
    for idx, leg in enumerate(plan.legs):
        if len(leg.path) < 2:
            continue
        rgba = _LEG_RGBA.get(leg.mode, _DEFAULT_LEG_RGBA)
        label = leg.route_short_name or leg.mode.title()
        detail = f"{leg.from_name} → {leg.to_name} · {leg.duration_s / 60.0:.0f} min"
        rows.append({
            "path": leg.path,
            "color_r": rgba[0],
            "color_g": rgba[1],
            "color_b": rgba[2],
            "color_a": rgba[3],
            "tooltip_name": f"{label}",
            "tooltip_detail": detail,
            "leg_index": idx,
        })
    return rows


def fetch_offer_commute_route(
    *,
    origin_lon: float,
    origin_lat: float,
    dest_lon: float,
    dest_lat: float,
    mode: object | None,
    hour: int,
    osrm_url: str,
    otp_url: str,
) -> CommuteRoutePlan | None:
    """Fetch the best route for ``mode`` at ``hour`` (0-23 local)."""
    display = coerce_commute_display_mode(mode)
    origin = (float(origin_lon), float(origin_lat))
    dest = (float(dest_lon), float(dest_lat))
    if display == "transit":
        itinerary = otp_plan_best_itinerary(
            origin,
            dest,
            base_url=otp_url,
            hour=int(hour),
        )
        if itinerary is None:
            return None
        return parse_otp_itinerary(dict(itinerary), mode="transit")
    profile = {"car": "driving", "bike": "bike", "walk": "foot"}[display]
    payload = osrm_route_geojson(
        origin,
        dest,
        base_url=osrm_url,
        profile=profile,
    )
    if payload is None:
        return None
    coords = payload.get("coordinates")
    if not isinstance(coords, list) or len(coords) < 2:
        return None
    path = [[float(p[0]), float(p[1])] for p in coords if isinstance(p, list)]
    dur_raw = payload.get("duration", 0.0)
    dist_raw = payload.get("distance", 0.0)
    duration_s = float(dur_raw) if isinstance(dur_raw, (int, float)) else 0.0
    distance_m = float(dist_raw) if isinstance(dist_raw, (int, float)) else 0.0
    mode_token = {"car": "CAR", "bike": "BICYCLE", "walk": "WALK"}[display]
    label = COMMUTE_MODE_LABELS[display]
    leg = CommuteRouteLeg(
        mode=mode_token,
        from_name="Offer",
        to_name="Commute target",
        duration_s=duration_s,
        distance_m=distance_m,
        route_short_name="",
        headsign="",
        path=path,
    )
    return CommuteRoutePlan(
        mode=display,
        duration_s=duration_s,
        distance_m=distance_m,
        legs=(leg,),
        summary=f"{label} · {duration_s / 60.0:.0f} min · {distance_m / 1000.0:.1f} km",
    )


def commute_route_mode_options() -> list[tuple[str, str]]:
    """``(label, mode_key)`` pairs for Offer Map selectboxes."""
    return [(COMMUTE_MODE_LABELS[m], m) for m in COMMUTE_DISPLAY_MODES]
