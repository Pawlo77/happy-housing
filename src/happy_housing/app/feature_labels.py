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
    "aircraft_lden_db": "Aircraft noise Lden (dB)",
    "aircraft_lnight_db": "Aircraft noise Lnight (dB)",
    "in_oou": "Inside UM aircraft restricted / OOU zone",
    "road_lden_db": "Road noise Lden (dB)",
    "road_lnight_db": "Road noise Lnight (dB)",
    "rail_lden_db": "Rail noise Lden (dB)",
    "rail_lnight_db": "Rail noise Lnight (dB)",
    "tram_lden_db": "Tram noise Lden (dB)",
    "tram_lnight_db": "Tram noise Lnight (dB)",
    "industry_lden_db": "Industry noise Lden (dB)",
    "industry_lnight_db": "Industry noise Lnight (dB)",
    "noise_total_lden_db": "Total noise Lden (dB, energy sum)",
    "noise_total_lnight_db": "Total noise Lnight (dB, energy sum)",
    "in_high_noise": "Total Lden ≥ 55 dB flag",
    "dist_air_station_m": "Distance to nearest GIOŚ air station (m)",
    "air_aq_index": "Air quality index from IDW field (0-5)",
    "air_pm10_ug_m3": "PM10 from IDW field (ug/m3)",
    "air_pm25_ug_m3": "PM2.5 from IDW field (ug/m3)",
    "air_no2_ug_m3": "NO2 from IDW field (ug/m3)",
    "in_poor_air": "IDW AQI >= 3 (poor air) flag",
    "dist_transit_stop_m": "Distance to nearest GTFS transit stop (m)",
    "dist_daszynskiego_m": "Straight-line distance to Daszynskiego (m)",
    "commute_min": "Model car commute minutes to commute target",
    "transit_commute_min": "Transit commute minutes to commute target",
    "dist_gunb_m": "Distance to nearest GUNB permit point (m)",
    "dist_bike_path_m": "Distance to nearest bike path (m)",
    "bike_path_coverage_m": "Metres of bike path within coverage radius",
    "dist_rail_area_m": "Distance to nearest rail land / yard (m)",
    "dist_military_m": "Distance to nearest military area (m)",
    "dist_airport_area_m": "Distance to nearest airport footprint (m)",
    "dist_traffic_road_m": "Distance to nearest drive-road sample (m)",
    "traffic_weight_nearby": "Local traffic-proxy weight sum nearby",
    "crime_rate_per_1000": "District crimes per 1,000 residents",
    "crime_criminal_rate_per_1000": "District criminal offences per 1,000 residents",
    "crime_detectability_pct": "District crime detectability (%)",
    "nbhd_rcn_median_ppm": "Local RCN median sale zl/m2",
    "nbhd_rcn_mean_ppm": "Local RCN mean sale zl/m2",
    "nbhd_rcn_idw_ppm": "Local RCN IDW sale zl/m2",
    "nbhd_rcn_n": "Count of local RCN comparable sales",
    "nbhd_rcn_nearest_m": "Distance to nearest RCN sale (m)",
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
