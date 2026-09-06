"""Pure helpers for Explore map layer payloads and visibility flags."""

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from typing import Self

import geopandas as gpd
import numpy as np
import pandas as pd

from happy_housing.app.commute_modes import (
    commute_series,
    commute_tooltip_label,
)
from happy_housing.gis.traffic import diurnal_traffic_weight
from happy_housing.ingest.listing_common import LISTING_ACTIVE_COLUMN, listing_is_active

_DELTA_COLOR_SCALE_ZL: float = 1_500.0
"""Δ zł/m² soft-saturation scale for offer pin colours (tanh)."""

_DELTA_UNDER_RGB: tuple[int, int, int] = (20, 160, 85)
"""Teal-green — strongly underpriced."""

_DELTA_MID_RGB: tuple[int, int, int] = (190, 150, 40)
"""Amber — near-fair Δ ≈ 0."""

_DELTA_OVER_RGB: tuple[int, int, int] = (215, 35, 30)
"""Danger red — strongly overpriced."""

_DELTA_COLOR_ALPHA: int = 200
"""Offer pin alpha."""

_LAYER_BOOL_FIELDS: frozenset[str] = frozenset({
    "offers",
    "hospitals",
    "fire_stations",
    "police",
    "charging_stations",
    "gas_stations",
    "shop_show_grocery",
    "shop_show_health_beauty",
    "shop_show_fashion",
    "shop_show_home",
    "shop_show_mobility",
    "shop_show_other",
    "transit_intensity",
    "transit_show_metro",
    "transit_show_tram",
    "transit_show_bus",
    "transit_show_rail",
    "traffic",
    "bike_paths",
    "air",
    "gunb",
    "parks",
    "city_boundary",
    "district_boundaries",
    "military",
    "rail_areas",
    "airport",
    "commute_target",
})
"""Boolean fields on ``LayerFlags``."""

_SHOP_SHOW_FIELDS: tuple[str, ...] = (
    "shop_show_grocery",
    "shop_show_health_beauty",
    "shop_show_fashion",
    "shop_show_home",
    "shop_show_mobility",
    "shop_show_other",
)
"""Shop subcategory toggles (no master Shops flag)."""

_TRANSIT_SHOW_FIELDS: tuple[str, ...] = (
    "transit_show_metro",
    "transit_show_tram",
    "transit_show_bus",
    "transit_show_rail",
)
"""Transit mode toggles (no master Transit lines flag)."""


def listing_delta_rgba(
    delta_zl_m2: float,
    scale_zl: float = _DELTA_COLOR_SCALE_ZL,
) -> tuple[int, int, int, int]:
    """Map listing-model delta zl/m2 to continuous RGBA (green → amber → red)."""
    scale = max(100.0, float(scale_zl))
    t = 0.5 * (float(np.tanh(float(delta_zl_m2) / scale)) + 1.0)
    if t <= 0.5:
        u = t * 2.0
        lo, hi = _DELTA_UNDER_RGB, _DELTA_MID_RGB
    else:
        u = (t - 0.5) * 2.0
        lo, hi = _DELTA_MID_RGB, _DELTA_OVER_RGB
    rgb = tuple(round(lo[i] + u * (hi[i] - lo[i])) for i in range(3))
    return (rgb[0], rgb[1], rgb[2], _DELTA_COLOR_ALPHA)


def listing_delta_rgba_columns(
    delta: pd.Series,
    scale_zl: float = _DELTA_COLOR_SCALE_ZL,
) -> pd.DataFrame:
    """Vectorized RGBA columns for offer ScatterplotLayer fills."""
    values = pd.to_numeric(delta, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    scale = max(100.0, float(scale_zl))
    t = 0.5 * (np.tanh(values / scale) + 1.0)
    left = t <= 0.5
    u = np.where(left, t * 2.0, (t - 0.5) * 2.0)
    lo_arr = np.empty((len(values), 3), dtype=float)
    hi_arr = np.empty((len(values), 3), dtype=float)
    lo_arr[left] = _DELTA_UNDER_RGB
    lo_arr[~left] = _DELTA_MID_RGB
    hi_arr[left] = _DELTA_MID_RGB
    hi_arr[~left] = _DELTA_OVER_RGB
    rgb = np.rint(lo_arr + u[:, None] * (hi_arr - lo_arr)).astype(int)
    return pd.DataFrame(
        {
            "color_r": rgb[:, 0],
            "color_g": rgb[:, 1],
            "color_b": rgb[:, 2],
            "color_a": _DELTA_COLOR_ALPHA,
        },
        index=delta.index,
    )


@dataclass(slots=True)
class LayerFlags:
    """Which Explore map overlays are enabled."""

    offers: bool = True
    """Scored listing pins."""
    noise_overlay: str = "off"
    """UM WMS noise overlay key (``off`` / ``total_lden`` / ``road_lden`` / …)."""
    aq_overlay: str = "off"
    """GIOŚ IDW AQ overlay key (``off`` / ``aq_index`` / ``pm25_ug_m3`` / …)."""
    hospitals: bool = False
    """OSM hospitals."""
    fire_stations: bool = False
    """OSM fire stations."""
    police: bool = False
    """OSM police stations."""
    charging_stations: bool = False
    """OSM EV charging stations."""
    gas_stations: bool = False
    """OSM fuel / petrol stations."""
    shop_show_grocery: bool = False
    """OSM grocery / food shop markers."""
    shop_show_health_beauty: bool = False
    """OSM chemist / beauty / salon shop markers."""
    shop_show_fashion: bool = False
    """OSM clothes / shoes / jewelry markers."""
    shop_show_home: bool = False
    """OSM furniture / electronics / DIY markers."""
    shop_show_mobility: bool = False
    """OSM car / bike shop markers."""
    shop_show_other: bool = False
    """Remaining OSM shop=* markers."""
    transit_intensity: bool = False
    """GTFS stop-hour departure heatmap (popularity)."""
    transit_show_metro: bool = False
    """GTFS metro route polylines + stops."""
    transit_show_tram: bool = False
    """GTFS tram route polylines + stops."""
    transit_show_bus: bool = False
    """GTFS bus route polylines + stops."""
    transit_show_rail: bool = False
    """GTFS rail / SKM-style route polylines + stops."""
    traffic: bool = False
    """Road-activity heatmap proxy for the selected hour."""
    bike_paths: bool = False
    """OSM bicycle infrastructure polylines."""
    air: bool = False
    """GIOŚ air-quality stations."""
    gunb: bool = False
    """GUNB construction / permit points."""
    parks: bool = False
    """UM PLACE_SKWERY + OSM green polygons."""
    city_poi_types: tuple[str, ...] = ()
    """Selected city / sports / civic object_type keys for map markers."""
    city_boundary: bool = False
    """Warsaw administrative boundary outline."""
    district_boundaries: bool = False
    """Warsaw dzielnica (district) administrative polygons."""
    military: bool = False
    """OSM landuse=military polygons (olive)."""
    rail_areas: bool = False
    """OSM landuse=railway yard / corridor polygons (steel blue)."""
    airport: bool = False
    """Aerodrome / airport footprints (violet; not Lden noise)."""
    commute_target: bool = False
    """Admin-configured commute destination pin."""

    def any_shops(self) -> bool:
        """True when any shop subcategory overlay is enabled."""
        return any(bool(getattr(self, name)) for name in _SHOP_SHOW_FIELDS)

    def any_transit_network(self) -> bool:
        """True when any transit mode overlay is enabled."""
        return any(bool(getattr(self, name)) for name in _TRANSIT_SHOW_FIELDS)

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable payload for map-view presets."""
        payload = asdict(self)
        payload["city_poi_types"] = list(self.city_poi_types)
        return payload

    @classmethod
    def from_dict(cls, payload: object) -> Self:
        """Build flags from a mapping; unknown keys ignored."""
        if not isinstance(payload, Mapping):
            return cls()
        known = {f.name for f in fields(cls)}
        data: dict[str, object] = {}
        for key, value in payload.items():
            name = str(key)
            if name not in known:
                continue
            data[name] = value
        # Legacy master toggles: parent off forced children off in the old UI.
        if "shops" in payload and not bool(payload.get("shops")):
            for name in _SHOP_SHOW_FIELDS:
                data[name] = False
        if "transit_network" in payload and not bool(payload.get("transit_network")):
            for name in _TRANSIT_SHOW_FIELDS:
                data[name] = False
        if "city_poi_types" in data:
            raw = data["city_poi_types"]
            if isinstance(raw, (list, tuple)):
                data["city_poi_types"] = tuple(str(x) for x in raw if str(x).strip())
            elif raw is None or raw == "":
                data["city_poi_types"] = ()
            else:
                data["city_poi_types"] = (str(raw),)
        for str_key in ("noise_overlay", "aq_overlay"):
            if str_key in data:
                text = str(data[str_key] if data[str_key] is not None else "off")
                data[str_key] = text.strip() or "off"
        for bool_key in _LAYER_BOOL_FIELDS:
            if bool_key in data:
                data[bool_key] = bool(data[bool_key])
        return cls(**data)  # type: ignore[arg-type]  # Dataclass fields validated above


def explore_datasets_needed(
    *,
    section: str,
    flags: LayerFlags,
    offer_subtab: str = "Overview",
) -> frozenset[str]:
    """Which heavy Explore datasets to load for the active tab + layer flags.

    Board / Chat / Prefs skip GIS entirely. News only needs star-news sync.
    Map / Spatial load overlays gated by ``flags``. Offer loads GIS only when
    the Offer → Map subtab is open.
    """
    name = str(section or "").strip()
    if name == "News":
        return frozenset({"star_news"})
    if name == "Offer" and str(offer_subtab or "Overview") != "Map":
        return frozenset()
    if name not in {"Map", "Offer", "Spatial"}:
        return frozenset()

    needed: set[str] = set()
    # Spatial only needs traffic + optional zone polygons — not Map amenity GIS.
    if name == "Spatial":
        needed.add("traffic")
        if flags.city_boundary:
            needed.add("city_boundary")
        if flags.district_boundaries:
            needed.add("district_boundaries")
        if flags.military:
            needed.add("military")
        if flags.rail_areas:
            needed.add("rail_areas")
        if flags.airport:
            needed.add("airport")
        return frozenset(needed)

    if flags.noise_overlay and flags.noise_overlay != "off":
        needed.add("warsaw_noise")
    if flags.aq_overlay and flags.aq_overlay != "off":
        needed.add("gios_aq")
    if flags.city_boundary:
        needed.add("city_boundary")
    if flags.district_boundaries:
        needed.add("district_boundaries")
    if flags.military:
        needed.add("military")
    if flags.rail_areas:
        needed.add("rail_areas")
    if flags.airport:
        needed.add("airport")
    if flags.parks:
        needed.add("parks")
    if flags.city_poi_types:
        needed.add("city_pois")
    if flags.hospitals or flags.fire_stations or flags.police:
        needed.add("amenities")
    if flags.charging_stations or flags.gas_stations:
        needed.add("amenities")
    if flags.any_shops():
        needed.add("shops")
    if flags.transit_intensity:
        needed.update({"stops", "hourly"})
    if flags.any_transit_network():
        needed.update({"stops", "route_shapes"})
    if flags.traffic:
        needed.add("traffic")
    if flags.bike_paths:
        needed.add("bike_paths")
    if flags.air:
        needed.add("air")
    if flags.gunb:
        needed.add("gunb")
    return frozenset(needed)


def keep_explore_map_mounted(section: str) -> bool:
    """Keep Explore Map pydeck mounted only on the Map tab.

    Offer used to remount Map for camera persistence; that forced full GIS +
    deck serialize on every Offer interaction. Prefer snappy Offer UX; pan/zoom
    may reset when returning to Map (stable ``selection_key`` still helps
    within Map reruns).
    """
    return str(section or "").strip() == "Map"


def explore_needs_recommended_pins(
    section: str,
    offer_subtab: str = "Overview",
) -> bool:
    """True when gold-ring recommended pins are drawn this render."""
    name = str(section or "").strip()
    if name == "Map":
        return True
    return name == "Offer" and str(offer_subtab or "Overview") == "Map"


def recommended_pins_memo_key(
    *,
    scored_mtime: float,
    scored_rows: int,
    prefs_payload: Mapping[str, object],
    starred: frozenset[str],
    top_n: int,
    demotion_alpha: float,
    w_bargain: float,
    w_commute: float,
    w_transit: float,
    w_overprice: float,
    otodom_pool_default: bool,
    amenity_weights: Mapping[str, float] | None,
    commute_mode: object,
    delta_scale_zl: float = 2000.0,
    dist_cap_m: float = 2000.0,
    opened_penalty: float = 0.5,
    skipped_penalty: float = 3.0,
) -> tuple[object, ...]:
    """Session memo key for map recommended-pin ids (skip full rescoring)."""
    weights = tuple(
        sorted((str(k), float(v)) for k, v in dict(amenity_weights or {}).items())
    )
    prefs_items = tuple(
        sorted((str(k), repr(v)) for k, v in dict(prefs_payload).items())
    )
    return (
        float(scored_mtime),
        int(scored_rows),
        prefs_items,
        tuple(sorted(str(x) for x in starred)),
        int(top_n),
        round(float(demotion_alpha), 6),
        round(float(w_bargain), 6),
        round(float(w_commute), 6),
        round(float(w_transit), 6),
        round(float(w_overprice), 6),
        bool(otodom_pool_default),
        weights,
        str(commute_mode or ""),
        round(float(delta_scale_zl), 1),
        round(float(dist_cap_m), 1),
        round(float(opened_penalty), 6),
        round(float(skipped_penalty), 6),
    )


def explore_datasets_needed_preserving_map(
    *,
    section: str,
    flags: LayerFlags,
    offer_subtab: str = "Overview",
) -> frozenset[str]:
    """Alias of ``explore_datasets_needed`` (Map no longer stays mounted on Offer)."""
    return explore_datasets_needed(
        section=section,
        flags=flags,
        offer_subtab=offer_subtab,
    )


MAX_GUNB_MAP_POINTS = 3000
"""Hard cap for GUNB IconLayer rows sent to the browser."""

MAX_SHOP_MAP_POINTS = 5000
"""Hard cap for OSM shop markers on the map."""

MAX_TRANSIT_STOP_MAP_POINTS = 2500
"""Cap transit-network stop markers (full ZTM ~7k bloated pydeck JSON)."""

SHOP_CATEGORY_LABELS: dict[str, str] = {
    "grocery": "Groceries",
    "health_beauty": "Health & beauty",
    "fashion": "Fashion",
    "home": "Home & electronics",
    "mobility": "Car & bike",
    "other": "Other shops",
}
"""Stable shop subcategory id → Explore checkbox label."""

_SHOP_CATEGORY_TAGS: dict[str, frozenset[str]] = {
    "grocery": frozenset({
        "supermarket",
        "convenience",
        "greengrocer",
        "bakery",
        "butcher",
        "pastry",
        "deli",
        "alcohol",
        "wine",
        "beverages",
        "cheese",
        "seafood",
        "farm",
        "kiosk",
        "newsagent",
        "tobacco",
        "tea",
        "coffee",
        "confectionery",
        "chocolate",
        "ice_cream",
    }),
    "health_beauty": frozenset({
        "chemist",
        "beauty",
        "hairdresser",
        "optician",
        "cosmetics",
        "perfume",
        "massage",
        "laundry",
        "dry_cleaning",
        "hearing_aids",
        "medical_supply",
        "herbalist",
    }),
    "fashion": frozenset({
        "clothes",
        "shoes",
        "jewelry",
        "jewellery",
        "tailor",
        "bag",
        "fashion_accessories",
        "watches",
        "leather",
        "fabric",
    }),
    "home": frozenset({
        "furniture",
        "electronics",
        "mobile_phone",
        "computer",
        "appliance",
        "doityourself",
        "hardware",
        "houseware",
        "interior_decoration",
        "carpet",
        "curtain",
        "lighting",
        "kitchen",
        "bathroom_furnishing",
    }),
    "mobility": frozenset({
        "car",
        "car_repair",
        "car_parts",
        "bicycle",
        "motorcycle",
        "tyres",
        "tires",
    }),
}
"""OSM ``shop=*`` tags grouped into Explore subcategories (remainder → other)."""

_TAG_TO_CATEGORY: dict[str, str] = {
    tag: category for category, tags in _SHOP_CATEGORY_TAGS.items() for tag in tags
}
"""Flat ``shop`` tag → category lookup."""


def shop_category_for_tag(tag: object) -> str:
    """Map an OSM ``shop=*`` value to a stable Explore subcategory id."""
    raw = str(tag or "").strip().lower()
    if not raw or raw in {"nan", "none"}:
        return "other"
    return _TAG_TO_CATEGORY.get(raw, "other")


def filter_shops_by_categories(
    frame: pd.DataFrame,
    categories: set[str] | frozenset[str],
) -> pd.DataFrame:
    """Keep shop rows whose ``shop`` tag maps into ``categories``."""
    if frame is None or getattr(frame, "empty", True) or not categories:
        return pd.DataFrame()
    out = frame.copy()
    if "shop" not in out.columns:
        # Untyped shops only show under ``other``.
        return out if "other" in categories else pd.DataFrame()
    cats = out["shop"].map(shop_category_for_tag)
    return out.loc[cats.isin(set(categories))].reset_index(drop=True)


MAX_CITY_POI_MAP_POINTS = 4000
"""Hard cap for UM city / sports POI markers on the map."""

_MAP_PAYLOAD_COLUMNS: tuple[str, ...] = (
    "lon",
    "lat",
    "listing_id",
    "tooltip_name",
    "tooltip_detail",
    "icon_size",
    "radius",
    "color_r",
    "color_g",
    "color_b",
    "color_a",
    "line_r",
    "line_g",
    "line_b",
    "line_a",
    "line_width",
    "is_recommended",
    "weight",
    "n_departures",
    "stop_id",
    "stop_name",
)
"""Columns safe to serialize into pydeck / Streamlit messages."""


def slim_map_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop attribute-heavy columns before sending points to the browser."""
    if frame.empty:
        return frame.reset_index(drop=True)
    cols = [c for c in _MAP_PAYLOAD_COLUMNS if c in frame.columns]
    return frame.loc[:, cols].reset_index(drop=True)


def limit_map_points(
    frame: pd.DataFrame,
    max_points: int,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample down dense POI tables so Streamlit stays under message limits."""
    if max_points <= 0 or frame.empty or len(frame) <= max_points:
        return frame.reset_index(drop=True)
    return frame.sample(n=int(max_points), random_state=seed).reset_index(drop=True)


def prepare_gunb_for_map(
    frame: pd.DataFrame,
    max_points: int = MAX_GUNB_MAP_POINTS,
) -> pd.DataFrame:
    """Warsaw-preferring, capped, tooltip-only GUNB points for the map."""
    if frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    work["lat"] = pd.to_numeric(work.get("lat"), errors="coerce")
    work["lon"] = pd.to_numeric(work.get("lon"), errors="coerce")
    work = work.dropna(subset=["lat", "lon"])
    if work.empty:
        return work
    if "miasto" in work.columns:
        city = work["miasto"].astype(str)
        warsaw = work.loc[city.str.contains("Warszawa", case=False, na=False)]
        if not warsaw.empty:
            work = warsaw
    else:
        # Rough Warsaw bbox when city labels are missing.
        work = work.loc[
            work["lon"].between(20.85, 21.30) & work["lat"].between(52.10, 52.40)
        ]
    if work.empty:
        return work.reset_index(drop=True)
    work = limit_map_points(work, max_points)
    if "adres" in work.columns:
        name = work["adres"].fillna("GUNB").astype(str)
    elif "numer_decyzji" in work.columns:
        name = work["numer_decyzji"].fillna("GUNB").astype(str)
    else:
        name = pd.Series("GUNB", index=work.index)
    work["tooltip_name"] = name.str.slice(0, 80)
    work["tooltip_detail"] = "GUNB · construction / permit proxy"
    return slim_map_frame(work)


_LISTING_MAP_SOURCE_COLUMNS: tuple[str, ...] = (
    "listing_id",
    "lat",
    "lon",
    "delta_price_per_m2",
    "price_per_m2",
    "area_m2",
    "rooms",
    "district",
    "commute_min",
    "transit_commute_min",
    "bike_commute_min",
    "walk_commute_min",
    LISTING_ACTIVE_COLUMN,
)
"""Columns read when building offer map pins (skip description / media)."""


def prepare_listings_for_map(
    frame: pd.DataFrame,
    *,
    recommended_ids: frozenset[str] | set[str] | None = None,
    commute_mode: object | None = None,
    delta_scale_zl: float = _DELTA_COLOR_SCALE_ZL,
) -> pd.DataFrame:
    """Listing pins with delta colours and rich hover fields.

    Recommended ids get larger radii and a clear tooltip badge so they
    stand out among standard offer pins. ``commute_mode`` selects which
    commute column appears on hover (default: public transport).
    Closed asks (``listing_active=0``) stay on the map when the caller
    includes them — smaller grey pins with a Closed tooltip badge.
    """
    if frame.empty:
        return pd.DataFrame()
    keep = [c for c in _LISTING_MAP_SOURCE_COLUMNS if c in frame.columns]
    if "lat" not in keep or "lon" not in keep or "listing_id" not in keep:
        return pd.DataFrame()
    out = frame.loc[:, keep].copy()
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["lat", "lon"])
    if out.empty:
        return pd.DataFrame()
    out["listing_id"] = out["listing_id"].astype(str)
    recommended = {str(x) for x in (recommended_ids or ())}
    out["is_recommended"] = out["listing_id"].isin(recommended)
    delta = pd.to_numeric(out.get("delta_price_per_m2"), errors="coerce").fillna(0.0)
    colors = listing_delta_rgba_columns(delta, scale_zl=delta_scale_zl)
    out["color_r"] = colors["color_r"].to_numpy()
    out["color_g"] = colors["color_g"].to_numpy()
    out["color_b"] = colors["color_b"].to_numpy()
    out["color_a"] = colors["color_a"].to_numpy()
    area = pd.to_numeric(out.get("area_m2"), errors="coerce")
    base_radius = (area.fillna(50.0) * 0.7).clip(22, 55)
    out["radius"] = base_radius.where(
        ~out["is_recommended"], (base_radius * 2.2).clip(48, 120)
    )
    if LISTING_ACTIVE_COLUMN in out.columns:
        is_closed = ~out[LISTING_ACTIVE_COLUMN].map(listing_is_active)
    else:
        is_closed = pd.Series(False, index=out.index)
    out["radius"] = out["radius"].where(~is_closed, (out["radius"] * 0.55).clip(12, 32))
    closed_arr = is_closed.to_numpy()
    out["color_r"] = np.where(closed_arr, 120, out["color_r"])
    out["color_g"] = np.where(closed_arr, 120, out["color_g"])
    out["color_b"] = np.where(closed_arr, 128, out["color_b"])
    out["color_a"] = np.where(closed_arr, 110, out["color_a"])
    # Gold stroke for recommended; soft white for the rest.
    out["line_r"] = np.where(out["is_recommended"], 255, 255)
    out["line_g"] = np.where(out["is_recommended"], 196, 255)
    out["line_b"] = np.where(out["is_recommended"], 40, 255)
    out["line_a"] = np.where(out["is_recommended"], 230, 160)
    out["line_width"] = np.where(out["is_recommended"], 3, 1)
    out["tooltip_name"] = out["listing_id"]
    ppm = pd.to_numeric(out.get("price_per_m2"), errors="coerce")
    rooms = (
        pd.to_numeric(out["rooms"], errors="coerce")
        if "rooms" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    commute = commute_series(out, commute_mode)
    commute_word = commute_tooltip_label(commute_mode)
    district = (
        out["district"].astype(str)
        if "district" in out.columns
        else pd.Series("", index=out.index)
    )

    def _fmt_int(series: pd.Series, suffix: str = "") -> pd.Series:
        return series.map(lambda v: f"{v:,.0f}{suffix}" if pd.notna(v) else "—")

    def _fmt_signed(series: pd.Series, suffix: str = "") -> pd.Series:
        return series.map(lambda v: f"{v:+,.0f}{suffix}" if pd.notna(v) else "—")

    badge = out["is_recommended"].map(
        lambda flag: "★ Recommended · " if bool(flag) else ""
    )
    closed_badge = is_closed.map(lambda flag: "Closed · " if bool(flag) else "")
    out["tooltip_detail"] = (
        badge
        + closed_badge
        + district
        + " · "
        + _fmt_int(ppm, " zł/m²")
        + " · Δ "
        + _fmt_signed(delta)
        + "<br/>"
        + _fmt_int(area, " m²")
        + " · "
        + rooms.map(lambda v: f"{v:g} rm" if pd.notna(v) else "—")
        + " · "
        + f"{commute_word} "
        + commute.map(lambda v: f"{v:.0f} min" if pd.notna(v) else "—")
    )
    return slim_map_frame(out)


def prepare_point_layer(
    frame: pd.DataFrame,
    *,
    lat_col: str,
    lon_col: str,
    name_col: str,
    detail: str,
    layer_label: str,
) -> pd.DataFrame:
    """Normalize a point table for scatter + hover."""
    if frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out["lat"] = pd.to_numeric(out[lat_col], errors="coerce")
    out["lon"] = pd.to_numeric(out[lon_col], errors="coerce")
    out = out.dropna(subset=["lat", "lon"])
    if out.empty:
        return out
    out["tooltip_name"] = out[name_col].astype(str).str.slice(0, 80)
    out["tooltip_detail"] = f"{layer_label} · {detail}"
    out["layer_label"] = layer_label
    return slim_map_frame(out)


def prepare_city_pois_for_map(
    frame: gpd.GeoDataFrame | pd.DataFrame,
    *,
    object_types: frozenset[str] | set[str] | tuple[str, ...],
    max_points: int = MAX_CITY_POI_MAP_POINTS,
) -> pd.DataFrame:
    """Filter city POIs by ``object_type`` and slim for IconLayer."""
    if frame is None or getattr(frame, "empty", True):
        return pd.DataFrame()
    if not object_types:
        return pd.DataFrame()
    out = frame.copy()
    if "object_type" not in out.columns:
        return pd.DataFrame()
    wanted = {str(t) for t in object_types}
    out = out.loc[out["object_type"].astype(str).isin(wanted)]
    if out.empty:
        return pd.DataFrame()
    if "geometry" in out.columns:
        out["lon"] = out.geometry.x
        out["lat"] = out.geometry.y
    out["lat"] = pd.to_numeric(out.get("lat"), errors="coerce")
    out["lon"] = pd.to_numeric(out.get("lon"), errors="coerce")
    out = out.dropna(subset=["lat", "lon"])
    out = limit_map_points(out, max_points)
    if out.empty:
        return out
    name = (
        out["name"].fillna("POI").astype(str)
        if "name" in out.columns
        else pd.Series("POI", index=out.index)
    )
    kind = out["object_type"].astype(str)
    out["tooltip_name"] = name.str.slice(0, 80)
    out["tooltip_detail"] = "City map · " + kind.str.replace("_", " ", regex=False)
    out["layer_label"] = kind
    return slim_map_frame(out)


def prepare_shops_for_map(
    frame: pd.DataFrame,
    max_points: int = MAX_SHOP_MAP_POINTS,
) -> pd.DataFrame:
    """Shop POIs with category + OSM tag in the hover detail."""
    if frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if "lat" not in out.columns and "geometry" in out.columns:
        out["lon"] = out.geometry.x
        out["lat"] = out.geometry.y
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["lat", "lon"])
    out = limit_map_points(out, max_points)
    if "shop" in out.columns:
        shop = out["shop"].astype(str)
    else:
        shop = pd.Series("shop", index=out.index)
    if "name" in out.columns:
        name = out["name"].fillna("Shop").astype(str)
    else:
        name = pd.Series("Shop", index=out.index)
    category = shop.map(shop_category_for_tag)
    label = category.map(lambda c: SHOP_CATEGORY_LABELS.get(c, "Other shops"))
    out["shop_category"] = category
    out["tooltip_name"] = name.str.slice(0, 80)
    out["tooltip_detail"] = label.astype(str) + " · " + shop
    out["layer_label"] = label
    return slim_map_frame(out)


def prepare_transit_for_hour(
    stops: pd.DataFrame,
    hourly: pd.DataFrame,
    *,
    hour: int,
    min_departures: int = 3,
    max_stops: int = 500,
) -> pd.DataFrame:
    """GTFS stops active at ``hour``, sized by departure count."""
    if stops.empty or hourly.empty:
        return pd.DataFrame()
    hour_rows = hourly.loc[hourly["hour"].astype(int) == int(hour)].copy()
    if hour_rows.empty:
        return pd.DataFrame()
    merged = stops.merge(hour_rows, on="stop_id", how="inner")
    merged = merged.loc[merged["n_departures"] >= min_departures]
    if merged.empty:
        return merged
    merged = merged.sort_values("n_departures", ascending=False).head(max_stops)
    merged["lat"] = pd.to_numeric(merged["stop_lat"], errors="coerce")
    merged["lon"] = pd.to_numeric(merged["stop_lon"], errors="coerce")
    merged = merged.dropna(subset=["lat", "lon"])
    merged["radius"] = (18.0 + merged["n_departures"].astype(float) * 0.7).clip(25, 70)
    merged["tooltip_name"] = merged["stop_name"].astype(str)
    merged["tooltip_detail"] = (
        f"Transit · {int(hour):02d}:00 · "
        + merged["n_departures"].astype(int).astype(str)
        + " departures"
    )
    merged["layer_label"] = "Transit"
    merged["weight"] = merged["n_departures"].astype(float)
    return slim_map_frame(merged.reset_index(drop=True))


def coerce_path_coordinates(path: object) -> list[list[float]]:
    """Normalize a polyline to plain ``[[lon, lat], …]`` for pydeck PathLayer.

    Parquet round-trips often yield ``numpy.ndarray`` of coordinate pairs.
    ``json.dumps(..., default=str)`` then emits a non-JSON string and deck.gl
    draws nothing.
    """
    if path is None or isinstance(path, (str, bytes)):
        return []
    if not isinstance(path, Iterable):
        return []
    out: list[list[float]] = []
    for point in path:
        if isinstance(point, (str, bytes)):
            continue
        try:
            lon_lat = np.asarray(point, dtype=float).reshape(-1)
            if lon_lat.size < 2:
                continue
            out.append([float(lon_lat[0]), float(lon_lat[1])])
        except (TypeError, ValueError):
            continue
    return out


def prepare_transit_paths(
    frame: pd.DataFrame,
    modes: set[str] | frozenset[str] | None = None,
) -> pd.DataFrame:
    """Filter route polylines to enabled modes for PathLayer."""
    if frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    if modes is not None:
        work = work.loc[work["mode"].astype(str).isin(modes)]
    if work.empty:
        return work.reset_index(drop=True)
    if "path" in work.columns:
        work = work.copy()
        work["path"] = [coerce_path_coordinates(p) for p in work["path"]]
        work = work.loc[work["path"].map(len) >= 2]
    if work.empty:
        return work.reset_index(drop=True)
    keep = [
        c
        for c in (
            "path",
            "mode",
            "route_short_name",
            "color_r",
            "color_g",
            "color_b",
            "color_a",
            "tooltip_name",
            "tooltip_detail",
        )
        if c in work.columns
    ]
    return work.loc[:, keep].reset_index(drop=True)


def prepare_transit_stops_network(
    stops: pd.DataFrame,
    max_stops: int = MAX_TRANSIT_STOP_MAP_POINTS,
) -> pd.DataFrame:
    """Slim stop markers for the transit-network overlay.

    Default caps at ``MAX_TRANSIT_STOP_MAP_POINTS`` so pydeck messages stay
    small. Pass ``max_stops=0`` to keep every geocoded stop.
    """
    if stops.empty:
        return pd.DataFrame()
    work = stops.copy()
    work["lat"] = pd.to_numeric(work.get("stop_lat", work.get("lat")), errors="coerce")
    work["lon"] = pd.to_numeric(work.get("stop_lon", work.get("lon")), errors="coerce")
    work = work.dropna(subset=["lat", "lon"])
    work = limit_map_points(work, max_stops)
    name = (
        work["stop_name"].astype(str)
        if "stop_name" in work.columns
        else pd.Series("Stop", index=work.index)
    )
    work["tooltip_name"] = name.str.slice(0, 80)
    work["tooltip_detail"] = "GTFS stop"
    return slim_map_frame(work)


def traffic_weights_for_hour(sample: pd.DataFrame, hour: int) -> pd.DataFrame:
    """Scale road-sample base weights by the diurnal traffic curve.

    Returns only ``lon`` / ``lat`` / ``weight`` — HeatmapLayer is non-pickable,
    so tooltip columns only inflate the pydeck JSON payload.
    """
    if sample.empty:
        return pd.DataFrame(columns=["lon", "lat", "weight"])
    out = sample.copy()
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["base_weight"] = pd.to_numeric(out["base_weight"], errors="coerce").fillna(1.0)
    out = out.dropna(subset=["lat", "lon"])
    factor = diurnal_traffic_weight(hour)
    out["weight"] = out["base_weight"] * factor
    return out[["lon", "lat", "weight"]].reset_index(drop=True)
