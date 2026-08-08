"""City / military / rail / airport polygon overlays for Explore maps."""

from __future__ import annotations

from typing import Any

import geopandas as gpd

ZONE_FILL: dict[str, list[int]] = {
    "city_boundary": [55, 65, 81, 18],
    "military": [107, 114, 128, 110],
    "rail_areas": [71, 85, 105, 95],
    "airport": [100, 116, 139, 100],
}
"""RGBA fills — muted grays; airport slightly cooler than military."""

ZONE_LINE: dict[str, list[int]] = {
    "city_boundary": [31, 41, 55, 230],
    "military": [75, 85, 99, 210],
    "rail_areas": [51, 65, 85, 200],
    "airport": [71, 85, 105, 210],
}
"""RGBA outlines; city boundary is the strongest stroke."""


def prepare_zone_records(
    frame: gpd.GeoDataFrame | None,
    *,
    fill_rgba: list[int],
    line_rgba: list[int],
    detail: str,
    default_name: str = "Zone",
) -> list[dict[str, Any]]:
    """Flatten polygons/multipolygons into pydeck PolygonLayer records."""
    if frame is None or frame.empty:
        return []
    work = frame.copy()
    if work.crs is not None and str(work.crs) != "EPSG:4326":
        work = work.to_crs("EPSG:4326")
    records: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        name = default_name
        if "name" in work.columns and row.get("name") is not None:
            text = str(row.get("name")).strip()
            if text and text.lower() not in {"nan", "none"}:
                name = text
        polys = (
            [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
        )
        for poly in polys:
            if poly.geom_type != "Polygon":
                continue
            coords = [[float(x), float(y)] for x, y in poly.exterior.coords]
            records.append({
                "coordinates": [coords],
                "fill_color": list(fill_rgba),
                "line_color": list(line_rgba),
                "tooltip_name": name,
                "tooltip_detail": detail,
            })
    return records
