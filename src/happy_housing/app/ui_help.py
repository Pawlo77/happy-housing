"""Hover help copy for Explore / Stats / Prefs controls.

Explains what each control shows and how it relates to price attractiveness
or living conditions. Wired via Streamlit ``help=`` (sidebar ``?`` tooltip).
"""

from __future__ import annotations

from happy_housing.app.map_layers import MAX_GUNB_MAP_POINTS

# Map overlay toggles — keys match LayerFlags field names.
LAYER_HELP: dict[str, str] = {
    "offers": (
        "Scored listing pins coloured by Δ zł/m² (listing - model) on a "
        "continuous green → amber → red ramp (not binary). "
        "Deeper green = more underpriced vs model; deeper red = more overpriced. "
        "Hover pins for details; click a pin, then Open in Offer. "
        "Valuation only — does not change living-condition overlays."
    ),
    "noise_overlay": (
        "UM Warszawa 2022 strategic noise GetMap as a continuous city-wide "
        "raster overlay (same bands as mapa.um.warszawa.pl): total energy-sum "
        "or road / rail / tram / aircraft / industry x Lden or Lnight. "
        "Roads/corridors stay loud; quiet parks stay transparent. "
        "Model features include each source + noise_total_lden_db."
    ),
    "aq_overlay": (
        "Continuous air-quality field interpolated (IDW) from live GIOŚ "
        "Warsaw stations — AQI, PM2.5, PM10, or NO₂. Not an official AQ map; "
        "sparse sensors smoothed across the city. Listing features use the "
        "same field (air_aq_index / air_pm*_ug_m3 / air_no2_ug_m3 + "
        "in_poor_air). Station pins remain available separately."
    ),
    "transit_intensity": (
        "Continuous city-wide GTFS departure field for the sidebar clock hour "
        "(BitmapLayer, same style as noise / AQ). Busy stop clusters glow "
        "warmer; slide the hour to see service rise and fall. Shows popularity, "
        "not the physical line map (use Transit lines & stops for that)."
    ),
    "transit_network": (
        "Turn on to draw GTFS route polylines + stops. Then pick modes below: "
        "metro red · tram dark red · bus blue · rail green. "
        "Strong transit access raises livability; dist_transit_stop_m enters "
        "the model."
    ),
    "transit_show_metro": (
        "Draw metro routes (red). Requires Transit lines & stops. "
        "Warsaw M1/M2 from GTFS route_type=1."
    ),
    "transit_show_tram": (
        "Draw tram routes (dark red). Requires Transit lines & stops. "
        "GTFS route_type=0."
    ),
    "transit_show_bus": (
        "Draw bus routes (blue). Requires Transit lines & stops. "
        "Densest layer — simplify/sample for map size."
    ),
    "transit_show_rail": (
        "Draw rail / SKM-style routes (green). Requires Transit lines & stops. "
        "GTFS route_type=2."
    ),
    "traffic": (
        "Continuous diurnal street-activity field along densified OSM drive "
        "roads (BitmapLayer). Warm corridors ≈ busier; scales with the "
        "sidebar hour — not live vehicle counts. Model uses dist_traffic_road_m, "
        "traffic_weight_nearby, major-road variants, and nearest_highway."
    ),
    "bike_paths": (
        "OSM bicycle infrastructure (cycleways / designated bike routes) as "
        "cyan polylines. Model uses dist_bike_path_m (nearest) and "
        "bike_path_coverage_m (metres of path within the Admin coverage "
        "radius). Prefs can cap max distance to the nearest path."
    ),
    "shops": (
        "OpenStreetMap retail / shop POIs (full shop=* set). "
        "Turn on, then pick subtypes below (groceries, fashion, …). "
        "Nearby daily shopping raises convenience; shops are map context, "
        "not a trained distance feature yet."
    ),
    "shop_show_grocery": (
        "Supermarkets, convenience, bakeries, greengrocers, alcohol, etc. "
        "Requires Shops."
    ),
    "shop_show_health_beauty": (
        "Chemists, beauty, hairdressers, opticians, laundry. Requires Shops."
    ),
    "shop_show_fashion": ("Clothes, shoes, jewelry, bags, tailor. Requires Shops."),
    "shop_show_home": (
        "Furniture, electronics, phones, DIY / hardware, housewares. Requires Shops."
    ),
    "shop_show_mobility": ("Car dealers / repair, bike shops, tyres. Requires Shops."),
    "shop_show_other": (
        "Remaining shop=* tags (florists, vacant, specialty, …). Requires Shops."
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
    "charging_stations": (
        "OSM EV charging stations (amenity=charging_station). "
        "dist_charging_station_m enters the model; use Admin weights to prefer "
        "or avoid proximity."
    ),
    "gas_stations": (
        "OSM fuel / petrol stations (amenity=fuel). Convenience for drivers; "
        "dist_fuel_m is a model feature. Not the same as EV charging."
    ),
    "air": (
        "GIOŚ monitoring stations (sensors for the IDW field). Prefer the "
        "Air quality field overlay for city-wide AQI/PM; model uses "
        "dist_air_station_m plus interpolated air_aq_index / air_pm*_ug_m3 / "
        "air_no2_ug_m3 and in_poor_air."
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
    "city_poi_types": (
        "UM sports objects (mapa.um.warszawa.pl/obiekty-sportowe) plus civic "
        "WFS POIs (libraries, pharmacies, bike-share, …) and OSM commercial "
        "fitness clubs under Gym. OSM hospitals / police / fire / shops / EV / "
        "gas stations are separate checkboxes (not duplicated here). "
        "Multiselect filters which types draw; default gyms, fitness studios, "
        "swimming pools. Admin -> Recommendations sets +/- weights that nudge "
        "the bargain recommender by proximity."
    ),
    "city_boundary": (
        "Warsaw city administrative outline (OSM). Context only — shows where "
        "offers sit relative to the city edge."
    ),
    "military": (
        "OSM landuse=military areas in olive. Restricted land — usually not "
        "buildable housing; dist_military_m is a model feature."
    ),
    "rail_areas": (
        "OSM landuse=railway yards / corridor land (steel blue). Distinct from "
        "GTFS rail line overlay and rail_lden_db noise bands. Nearby yards can "
        "mean noise/vibration; dist_rail_area_m feeds the model."
    ),
    "airport": (
        "OSM aerodrome footprints in violet. Footprint of Chopin / "
        "Babice / military airfields — not the amber Lden noise contours. "
        "dist_airport_area_m is a model proximity feature."
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
    "POI layers use emoji markers (👮 police, 🛒 groceries, …) — "
    "hover pins for details."
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
        "Filter by aircraft Lden ≥55 dB (UM strategic map / legacy END). "
        "Inside = noisier aircraft exposure."
    ),
    "high_noise": (
        "Filter by total energy-sum Lden ≥55 dB across road, rail, tram, "
        "aircraft, and industry. Stronger livability cut than aircraft-only."
    ),
    "poor_air": (
        "Filter by interpolated GIOŚ AQI ≥3 (sufficient / bad / very bad). "
        "Uses the continuous IDW field, not only the nearest station pin."
    ),
    "max_noise_total": (
        "Cap energy-sum total Lden (dB). 0 = no cap. Quieter homes usually "
        "command stronger willingness-to-pay; model uses noise_total_lden_db."
    ),
    "max_traffic_weight": (
        "Cap OSM traffic-proxy weight sum near the listing (0 = off). "
        "Higher = denser / heavier roads nearby (class x sample proxy)."
    ),
    "max_major_traffic_weight": (
        "Cap major-road-only traffic weight nearby (0 = off). "
        "Filters arterial / primary corridors more than quiet residential."
    ),
    "frontage": (
        "Filter by Otodom description hints: street = phrases like "
        "'od ulicy'; quiet = 'od podwórza' / courtyard. Unknown when "
        "neither phrase appears — not geometric facade orientation."
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
    "high_noise": (
        "Total energy-sum Lden ≥55 dB (road+rail+tram+aircraft+industry). "
        "Stronger city-wide noise cut than aircraft-only."
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
    "max_bike_path": (
        "Max metres to nearest OSM bike path / cycleway. Model also uses "
        "bike_path_coverage_m (path length nearby)."
    ),
    "max_noise_total": (
        "Max energy-sum total Lden (dB) from UM 2022 maps (road+rail+tram+"
        "aircraft+industry). 0 = no cap. Hard filter when Prefs enabled."
    ),
    "max_traffic_weight": (
        "Max OSM traffic-proxy weight sum nearby. 0 = no cap. "
        "Hard filter when Prefs enabled."
    ),
    "max_major_traffic_weight": (
        "Max major-road traffic-proxy weight nearby. 0 = no cap."
    ),
    "frontage": (
        "Prefer street-facing or quiet/courtyard side based on Otodom "
        "description phrases (not surveyed facade orientation)."
    ),
}
"""Explore Prefs form field help texts."""

SPATIAL_METRIC_HELP: dict[str, str] = {
    "price_ppm": (
        "Continuous zł/m² field (Model predictions = fair price surface; "
        "Historical = interpolated asks). Dark pins = historical listing "
        "prices. Warm = expensive pocket — price level, not bargain quality."
    ),
    "listing_density": (
        "Continuous offer-density field with listing pins. High density = "
        "more choice / competition, not proven sales liquidity."
    ),
    "sold_density": (
        "Continuous RCN deal field with transaction pins. Warm = proven "
        "activity — stronger signal of real market depth than asks alone."
    ),
    "avg_area": (
        "Continuous mean-area field with listing pins. Bigger units change "
        "living space and usually total price; zł/m² can still be moderate."
    ),
    "popularity": (
        "Continuous freshness field with listing pins. Fresh inventory ≈ "
        "current demand interest; stale stock may linger for a reason."
    ),
    "commute": (
        "Continuous commute-ease field with listing pins. Shorter modelled "
        "trips to Daszyńskiego score warmer — a major price driver."
    ),
    "bargain": (
        "Model surface = underpriced residual (listing - model). Historical "
        "surface = interpolated asks (replaces model). Pins mark offers."
    ),
    "traffic": (
        "OSM densified drive-road BitmapLayer x hour curve. Warm = busier "
        "streets — noise/congestion trade-off for living comfort."
    ),
}
"""Explore Spatial metric selectbox help (keyed by SpatialMetric.key)."""


def layer_help(flag_name: str) -> str:
    """Return Map-layer help for a ``LayerFlags`` field name."""
    return LAYER_HELP[flag_name]
