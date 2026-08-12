"""Human-readable descriptions for model feature names (chart hover)."""

from __future__ import annotations

import pandas as pd

from happy_housing.gis.amenity_catalog import amenity_type

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "area_m2": "Living area in square metres",
    "rooms": "Number of rooms",
    "year_built": "Building construction year",
    "floor_no": "Floor number of the unit",
    "building_floors_num": "Total floors in the building",
    "market_primary": "Primary market (developer) listing flag",
    "has_balcony": "Listing has a balcony",
    "has_garage": "Listing has a garage / parking",
    "has_lift": "Building has a lift",
    "has_terrace": "Listing has a terrace",
    "has_garden": "Listing has a garden",
    "is_private_owner": "Private-owner listing (not agency)",
    "rent": "Monthly rent / admin fee (zł) when present",
    "building_type": "Building type (block, tenement, …)",
    "heating": "Heating type",
    "construction_status": "Construction / finish status",
    "building_material": "Primary building material",
    "district": "Warsaw district",
    "aircraft_lden_db": "Aircraft day-evening-night average noise (dB)",
    "aircraft_lnight_db": "Aircraft night-time average noise (dB)",
    "in_oou": "Inside UM aircraft restricted / OOU zone",
    "road_lden_db": "Road day-evening-night average noise (dB)",
    "road_lnight_db": "Road night-time average noise (dB)",
    "rail_lden_db": "Rail day-evening-night average noise (dB)",
    "rail_lnight_db": "Rail night-time average noise (dB)",
    "tram_lden_db": "Tram day-evening-night average noise (dB)",
    "tram_lnight_db": "Tram night-time average noise (dB)",
    "industry_lden_db": "Industry day-evening-night average noise (dB)",
    "industry_lnight_db": "Industry night-time average noise (dB)",
    "noise_total_lden_db": ("Total day-evening-night average noise (dB, energy sum)"),
    "noise_total_lnight_db": ("Total night-time average noise (dB, energy sum)"),
    "in_high_noise": "Total day-evening-night average noise ≥ 55 dB flag",
    "dist_air_station_m": "Distance to nearest GIOŚ air station (m)",
    "air_aq_index": "Air quality index from IDW field (0-5)",
    "air_pm10_ug_m3": "PM10 from IDW field (ug/m3)",
    "air_pm25_ug_m3": "PM2.5 from IDW field (ug/m3)",
    "air_no2_ug_m3": "NO2 from IDW field (ug/m3)",
    "in_poor_air": "IDW AQI >= 3 (poor air) flag",
    "dist_transit_stop_m": "Distance to nearest GTFS transit stop (m)",
    "dist_daszynskiego_m": "Straight-line distance to Admin commute target (m)",
    "commute_min": "Model car commute minutes to commute target",
    "transit_commute_min": "Transit commute minutes to commute target",
    "bike_commute_min": "Bike commute minutes to commute target",
    "walk_commute_min": "Walk commute minutes to commute target",
    "log_commute_min": "log1p(car commute minutes)",
    "log_transit_commute_min": "log1p(transit commute minutes)",
    "log_bike_commute_min": "log1p(bike commute minutes)",
    "log_walk_commute_min": "log1p(walk commute minutes)",
    "dist_gunb_m": "Distance to nearest GUNB permit point (m)",
    "dist_bike_path_m": "Distance to nearest bike path (m)",
    "bike_path_coverage_m": "Metres of bike path within coverage radius",
    "dist_rail_area_m": "Distance to nearest rail land / yard (m)",
    "dist_military_m": "Distance to nearest military area (m)",
    "dist_airport_area_m": "Distance to nearest airport footprint (m)",
    "dist_traffic_road_m": "Distance to nearest drive-road sample (m)",
    "traffic_weight_nearby": "Local traffic-proxy weight sum nearby",
    "dist_major_road_m": "Distance to nearest major road sample (m)",
    "major_traffic_weight_nearby": ("Major-road traffic-proxy weight sum nearby"),
    "nearest_highway": "Nearest OSM highway class (drive sample)",
    "street_facing_hint": (
        "Listing text suggests street-facing windows (Otodom description)"
    ),
    "quiet_side_hint": (
        "Listing text suggests courtyard / quiet side (Otodom description)"
    ),
    "crime_rate_per_1000": "District crimes per 1,000 residents",
    "crime_criminal_rate_per_1000": "District criminal offences per 1,000 residents",
    "crime_detectability_pct": "District crime detectability (%)",
    "nbhd_rcn_median_ppm": "Local Rejestr Cen median sale zł/m²",
    "nbhd_rcn_mean_ppm": "Local Rejestr Cen mean sale zł/m²",
    "nbhd_rcn_idw_ppm": "Local Rejestr Cen inverse-distance-weighted sale zł/m²",
    "nbhd_rcn_n": "Count of local Rejestr Cen comparable sales",
    "nbhd_rcn_nearest_m": "Distance to nearest Rejestr Cen sale (m)",
    "photo_quality_mean": "Mean CLIP interior quality score",
    "photo_quality_max": "Max CLIP interior quality score",
    "photo_quality_std": "Std of CLIP interior quality scores",
    "n_photos_scored": "Number of listing photos scored",
    "share_kitchen": "Share of photos classified as kitchen",
    "share_bathroom": "Share of photos classified as bathroom",
    "share_living": "Share of photos classified as living room",
    "share_bedroom": "Share of photos classified as bedroom",
    "share_exterior": "Share of photos classified as exterior",
    "luxury_share": "Share of photos matching luxury interior prompts",
    "has_floor_plan": "Floor-plan image present",
    "n_floor_plans": "Count of floor-plan images",
    "floor_plan_conf": "Floor-plan detection confidence",
    "dist_hospital_m": "Nearest hospital (m)",
    "dist_fire_station_m": "Nearest fire station (m)",
    "dist_police_m": "Nearest police station (m)",
    "dist_metro_m": "Nearest metro station (m)",
    "dist_park_m": "Nearest park / square (m)",
    "dist_shop_m": "Nearest shop (m)",
    "dist_school_m": "Nearest school (m)",
    "dist_charging_station_m": "Nearest EV charging station (m)",
    "dist_fuel_m": "Nearest gas station (m)",
}
"""Stable feature key → short hover description."""


def feature_description(name: object) -> str:
    """Return a short human description for a model feature column name."""
    key = str(name or "").strip()
    if not key:
        return ""
    known = FEATURE_DESCRIPTIONS.get(key)
    if known is not None:
        return known
    if key.startswith("dist_") and key.endswith("_m"):
        amenity_key = key[len("dist_") : -len("_m")]
        entry = amenity_type(amenity_key)
        if entry is not None:
            return f"Nearest {entry.label} (m)"
        nice = amenity_key.replace("_", " ")
        return f"Nearest {nice} (m)"
    return key.replace("_", " ")


def feature_catalog_frame(columns: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Build a Feature / Description table for Home and similar UIs."""
    names = [str(c).strip() for c in columns if str(c).strip()]
    return pd.DataFrame({
        "feature": names,
        "description": [feature_description(name) for name in names],
    })
