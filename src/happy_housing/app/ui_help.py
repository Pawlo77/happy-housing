"""Hover help copy for Explore / Stats / Prefs controls.

Explains what each control shows and how it relates to price attractiveness
or living conditions. Wired via Streamlit ``help=`` (sidebar ``?`` tooltip).
"""

from __future__ import annotations

from happy_housing.app.map_layers import MAX_GUNB_MAP_POINTS

# Map overlay toggles — keys match LayerFlags field names.
LAYER_HELP: dict[str, str] = {
    "offers": (
        "Scored listing pins coloured by Δ zł/m² (listing - model). "
        "Green = model says underpriced; red = overpriced. "
        "Hover pins for details; click a pin, then Open in Offer. "
        "Valuation only — does not change living-condition overlays."
    ),
    "oou": (
        "EEA END Chopin aircraft Lden / Lnight contour bands (≥55 dB). "
        "Higher aircraft noise hurts sleep/outdoor comfort and often lowers "
        "willingness-to-pay; the model also uses Lden / in_oou as features."
    ),
    "transit_intensity": (
        "Real GTFS stop-hour departure heatmap + busy stops at the sidebar "
        "clock. Shows where service is dense (popularity), not the physical "
        "line map."
    ),
    "transit_network": (
        "GTFS route polylines plus stop pins from shapes.txt. "
        "Line colours: metro red · tram dark red · bus blue · rail green. "
        "Toggle modes below. Strong transit access raises livability; "
        "dist_transit_stop_m enters the model."
    ),
    "transit_show_metro": (
        "Draw metro routes (red). Warsaw M1/M2 from GTFS route_type=1."
    ),
    "transit_show_tram": ("Draw tram routes (dark red). GTFS route_type=0."),
    "transit_show_bus": (
        "Draw bus routes (blue). Densest layer — simplify/sample for map size."
    ),
    "transit_show_rail": ("Draw rail / SKM-style routes (green). GTFS route_type=2."),
    "traffic": (
        "Diurnal road-activity heatmap proxy at the sidebar hour "
        "(not live sensors). Peak hours ≈ more congestion/noise — worse "
        "living comfort near busy corridors; context only, not a model feature."
    ),
    "shops": (
        "OpenStreetMap retail / shop POIs (full shop=* set). "
        "Nearby daily shopping raises convenience and neighbourhood amenity; "
        "shops are map context, not a trained distance feature yet."
    ),
    "hospitals": (
        "OSM hospitals. Closer emergency / specialist care improves safety "
        "perception and livability; dist_hospital_m is a model proximity feature "
        "and Prefs can cap max distance."
    ),
    "fire_stations": (
        "OSM fire stations. Proximity is a public-safety amenity signal; "
        "dist_fire_station_m feeds the model and Prefs distance caps."
    ),
    "police": (
        "OSM police stations. Nearby stations can feel safer (mixed amenity); "
        "dist_police_m is a model feature and Prefs can require a max distance."
    ),
    "air": (
        "GIOŚ monitoring stations with nearest-station AQI and PM10/PM2.5/NO2. "
        "Model uses dist_air_station_m plus air_aq_index / air_pm*_ug_m3 / "
        "air_no2_ug_m3 — distance alone is not enough."
    ),
    "gunb": (
        "GUNB / GUGiK building-permit points. Map shows a Warsaw sample "
        f"(max {MAX_GUNB_MAP_POINTS:,} points) so Streamlit stays under "
        "message limits — full Mazowieckie dump is too large for the browser. "
        "Nearby permits hint at construction disruption or future supply; "
        "dist_gunb_m is a model feature."
    ),
    "parks": (
        "UM Warszawa PLACE_SKWERY parks & squares (same layer as testmapa). "
        "Closer green space raises day-to-day livability and often supports "
        "price; dist_park_m is a model feature and Prefs can cap max distance."
    ),
    "city_boundary": (
        "Warsaw city administrative outline (OSM). Context only — shows where "
        "offers sit relative to the city edge."
    ),
    "military": (
        "OSM landuse=military areas in olive. Restricted land — usually not "
        "buildable housing; context for nearby offers."
    ),
    "rail_areas": (
        "OSM landuse=railway yards / corridor land (steel blue). Distinct from "
        "GTFS rail line overlay. Nearby yards can mean noise/vibration."
    ),
    "airport": (
        "OSM aerodrome footprints in violet. Footprint of Chopin / "
        "Babice / military airfields — not the amber Lden noise contours."
    ),
    "commute_target": (
        "Commute destination pin (Admin → General; default Rondo Daszyńskiego). "
        "Reference for commute minutes / dist_daszynskiego_m — shorter trips "
        "raise job-access livability and strongly affect price attractiveness "
        "in the model."
    ),
}
"""Sidebar Map layer checkbox help texts."""

BASEMAP_HELP: str = (
    "Streets = Carto light basemap for labels/context. "
    "Satellite = Esri imagery for roof/yard/green cover — living-condition "
    "context only; does not change model scores. "
    "POI layers use distinct silhouettes — hover map pins for details."
)
"""Basemap segmented-control help."""

FILTER_HELP: dict[str, str] = {
    "district": (
        "Limit Map/Board to one Warsaw district. District mix drives local "
        "price levels and amenity access."
    ),
    "source": (
        "Otodom asks vs RCN closed sales. Mix of sources changes what "
        "'price' means (asking vs transaction)."
    ),
    "oou": (
        "Filter by aircraft Lden ≥55 dB (EEA END). Inside = noisier living "
        "conditions and usually weaker price attractiveness."
    ),
    "max_ppm": (
        "Cap asking zł/m². Hard budget filter on price attractiveness; "
        "does not change livability overlays."
    ),
    "max_age_months": (
        "Keep offers no older than this many months (from listed_at / "
        "listed_month). Fresher asks ≈ current market; stale asks may be "
        "mispriced or less liquid. Max = no age filter."
    ),
}
"""Explore sidebar filter help texts."""

STATS_FILTER_HELP: dict[str, str] = {
    "districts": (
        "Multi-select districts for Stats charts. Empty = all. "
        "District mix drives local price and amenity patterns."
    ),
    "sources": ("Otodom asks vs RCN sales in Stats. Empty = all sources."),
    "months": ("Restrict Stats to listed months. Empty = all months in the frame."),
    "valuation": (
        "Underpriced = listing - model < 0 (bargain heat); overpriced = "
        "asking looks rich vs model. Price-attractiveness cut only."
    ),
    "oou": (
        "Aircraft Lden ≥55 dB filter. Inside bands ≈ worse noise livability "
        "and often softer willingness-to-pay."
    ),
    "price": "Listing zł/m² range — price attractiveness band for Stats.",
    "area": "Floor-area range — unit size affects both price and living space.",
    "rooms": "Room-count range — layout livability and price tier.",
}
"""Stats sidebar filter help texts."""

PREF_HELP: dict[str, str] = {
    "enabled": (
        "When on, Map and Board keep only offers that pass these hard "
        "constraints (price, size, noise, proximity)."
    ),
    "districts": "Preferred districts — living location and local price levels.",
    "sources": "Allowed listing sources (asks vs sales).",
    "oou": (
        "Aircraft noise preference. Avoiding Lden ≥55 dB protects sleep/"
        "outdoor comfort and often aligns with stronger residual value."
    ),
    "min_price": "Floor on total asking price (budget / quality floor).",
    "max_price": "Ceiling on total asking price (hard budget).",
    "min_ppm": "Floor on zł/m² — skip ultra-cheap outliers if desired.",
    "max_ppm": "Ceiling on zł/m² — core price-attractiveness budget.",
    "min_area": "Minimum floor area — living-space floor.",
    "max_area": "Maximum floor area — avoid oversized units if desired.",
    "min_rooms": "Minimum rooms — household fit / livability.",
    "max_rooms": "Maximum rooms — cap oversized layouts if desired.",
    "min_year": "Minimum year built — fabric / energy / comfort proxy.",
    "max_commute": (
        "Max modelled commute minutes to Daszyńskiego. Shorter = better "
        "job access and usually stronger price support."
    ),
    "max_transit": (
        "Max metres to nearest GTFS stop. Closer transit = better daily "
        "mobility; also a model feature."
    ),
    "max_delta": (
        "Max listing-model zł/m². Negative/zero prefers bargains "
        "(price attractiveness)."
    ),
    "require_under": (
        "Keep only negative delta (model says underpriced). Strict "
        "price-attractiveness gate."
    ),
    "max_hospital": (
        "Max metres to nearest hospital. Closer care = safety/livability; "
        "model uses dist_hospital_m."
    ),
    "max_fire": ("Max metres to fire station. Public-safety amenity; model feature."),
    "max_police": (
        "Max metres to police station. Safety amenity signal; model feature."
    ),
    "max_air": ("Max metres to GIOŚ station (monitor location, not pollution level)."),
    "max_gunb": (
        "Max metres to GUNB permit. Avoid dense construction if you want "
        "quieter living; model uses dist_gunb_m."
    ),
    "max_park": (
        "Max metres to nearest UM park/square (PLACE_SKWERY). Closer greenery "
        "improves livability; model uses dist_park_m."
    ),
}
"""Explore Prefs form field help texts."""

SPATIAL_METRIC_HELP: dict[str, str] = {
    "price_ppm": (
        "Heat of asking zł/m². Warm = expensive pocket — price level, not "
        "bargain quality. Cross-check Bargain heat for attractiveness."
    ),
    "listing_density": (
        "Where open asks pile up. High density = more choice / competition, "
        "not proven sales liquidity."
    ),
    "sold_density": (
        "RCN closed deals. Warm = proven transaction activity — stronger "
        "signal of real market depth than asks alone."
    ),
    "avg_area": (
        "Larger mean floor plates. Bigger units change living space and "
        "usually total price; zł/m² can still be moderate."
    ),
    "popularity": (
        "Weight toward fresher listed_month. Fresh inventory ≈ current "
        "demand interest; stale stock may linger for a reason."
    ),
    "commute": (
        "Shorter modelled trips to Daszyńskiego score warmer. Job-access "
        "livability and a major price driver in the model."
    ),
    "bargain": (
        "Underpriced residual intensity (listing - model). Warm = stronger "
        "price attractiveness vs fair value — core bargain hunt map."
    ),
    "traffic": (
        "OSM diurnal road proxy at the sidebar hour. Warm = busier corridors "
        "— noise/congestion trade-off for living comfort."
    ),
}
"""Explore Spatial metric selectbox help (keyed by SpatialMetric.key)."""


def layer_help(flag_name: str) -> str:
    """Return Map-layer help for a ``LayerFlags`` field name."""
    return LAYER_HELP[flag_name]
