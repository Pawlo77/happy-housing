"""Canonical city / OSM amenity types for map filters and recommender weights.

Sports types follow `mapa.um.warszawa.pl/obiekty-sportowe/` (OSRID groups).
Civic types come from UM WFS; EV charging is OSM-only (not on the city portal).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AmenityType:
    """One filterable / weightable amenity category."""

    key: str
    """Stable id used in ``dist_<key>_m``, settings, and map filters."""
    label: str
    """Admin / Explore display label."""
    group: str
    """UI grouping (Sports, Civic, OSM, …)."""
    source: str
    """Primary data source: ``sports`` | ``wfs`` | ``osm`` | ``um_parks``."""
    default_map_visible: bool = False
    """Explore map filter default."""
    default_weight: float = 0.0
    """Recommender weight: + near preferred, - near avoided, 0 neutral."""


# OSRID → catalog key (portal sports objects).
SPORT_OSRID_TO_KEY: dict[str, str] = {
    "1": "running_track",
    "2": "running_track",
    "4": "baseball",
    "5": "field_hockey",
    "6": "field_hockey",
    "7": "basketball",
    "8": "basketball",
    "9": "handball",
    "10": "handball",
    "11": "beach_handball",
    "12": "beach_handball",
    "13": "volleyball",
    "14": "volleyball",
    "16": "beach_volleyball",
    "18": "beach_volleyball",
    "19": "football_pitch",
    "20": "football_pitch",
    "21": "multipurpose_pitch",
    "22": "boules",
    "23": "sports_hall",
    "24": "sports_hall",
    "26": "ricochet",
    "27": "squash",
    "28": "tennis",
    "29": "tennis",
    "30": "bowling",
    "34": "ski_slope",
    "35": "indoor_pool",
    "36": "indoor_pool",
    "37": "indoor_pool",
    "38": "outdoor_pool",
    "39": "outdoor_pool",
    "40": "outdoor_pool",
    "42": "pumptrack",
    "44": "gymnasium",
    "45": "fitness",
    "46": "sauna",
    "47": "gym",
    "48": "outdoor_gym",
    "49": "skatepark",
    "52": "stadium",
    "53": "stadium",
    "54": "stadium",
    "56": "stadium",
    "59": "street_workout",
    "60": "shooting_range",
    "61": "climbing_wall",
    "65": "archery",
    "69": "luge_track",
    "71": "roller_ski",
    "73": "athletics_field",
    "74": "athletics_field",
    "76": "athletics_field",
    "77": "athletics_field",
    "79": "athletics_field",
    "80": "tennis_wall",
    "81": "minigolf",
    "82": "table_tennis",
    "ODKRYTE": "outdoor_ice_rink",
    "ZAKRYTE": "indoor_ice_rink",
    "TOR": "ice_track",
}
"""MapViewer sports ``OSRID`` values → amenity catalog keys."""

AMENITY_TYPES: tuple[AmenityType, ...] = (
    # Sports (portal)
    AmenityType("gym", "Gym (siłownia)", "Sports", "sports", True, 0.0),
    AmenityType("outdoor_gym", "Outdoor gym", "Sports", "sports", True, 0.0),
    AmenityType("street_workout", "Street workout", "Sports", "sports", False, 0.0),
    AmenityType("indoor_pool", "Indoor swimming pool", "Sports", "sports", True, 0.0),
    AmenityType("outdoor_pool", "Outdoor swimming pool", "Sports", "sports", True, 0.0),
    AmenityType("fitness", "Fitness studio", "Sports", "sports", False, 0.0),
    AmenityType("sauna", "Sauna", "Sports", "sports", False, 0.0),
    AmenityType("sports_hall", "Sports hall", "Sports", "sports", False, 0.0),
    AmenityType("gymnasium", "Gymnasium", "Sports", "sports", False, 0.0),
    AmenityType("football_pitch", "Football pitch", "Sports", "sports", False, 0.0),
    AmenityType(
        "multipurpose_pitch",
        "Multipurpose pitch",
        "Sports",
        "sports",
        False,
        0.0,
    ),
    AmenityType("basketball", "Basketball court", "Sports", "sports", False, 0.0),
    AmenityType("volleyball", "Volleyball court", "Sports", "sports", False, 0.0),
    AmenityType(
        "beach_volleyball",
        "Beach volleyball",
        "Sports",
        "sports",
        False,
        0.0,
    ),
    AmenityType("handball", "Handball court", "Sports", "sports", False, 0.0),
    AmenityType("beach_handball", "Beach handball", "Sports", "sports", False, 0.0),
    AmenityType("field_hockey", "Field hockey", "Sports", "sports", False, 0.0),
    AmenityType("baseball", "Baseball / softball", "Sports", "sports", False, 0.0),
    AmenityType("tennis", "Tennis court", "Sports", "sports", False, 0.0),
    AmenityType("tennis_wall", "Tennis wall", "Sports", "sports", False, 0.0),
    AmenityType("squash", "Squash", "Sports", "sports", False, 0.0),
    AmenityType("ricochet", "Ricochet", "Sports", "sports", False, 0.0),
    AmenityType("running_track", "Running track", "Sports", "sports", False, 0.0),
    AmenityType("stadium", "Stadium", "Sports", "sports", False, 0.0),
    AmenityType("athletics_field", "Athletics field", "Sports", "sports", False, 0.0),
    AmenityType("skatepark", "Skatepark", "Sports", "sports", False, 0.0),
    AmenityType("pumptrack", "Pumptrack", "Sports", "sports", False, 0.0),
    AmenityType("bowling", "Bowling", "Sports", "sports", False, 0.0),
    AmenityType("climbing_wall", "Climbing wall", "Sports", "sports", False, 0.0),
    AmenityType("table_tennis", "Outdoor table tennis", "Sports", "sports", False, 0.0),
    AmenityType("minigolf", "Minigolf", "Sports", "sports", False, 0.0),
    AmenityType("shooting_range", "Shooting range", "Sports", "sports", False, 0.0),
    AmenityType("boules", "Boules / pétanque", "Sports", "sports", False, 0.0),
    AmenityType("archery", "Archery range", "Sports", "sports", False, 0.0),
    AmenityType("ski_slope", "Ski slope", "Sports", "sports", False, 0.0),
    AmenityType("luge_track", "Luge track", "Sports", "sports", False, 0.0),
    AmenityType("roller_ski", "Roller-ski trail", "Sports", "sports", False, 0.0),
    AmenityType("indoor_ice_rink", "Indoor ice rink", "Sports", "sports", False, 0.0),
    AmenityType("outdoor_ice_rink", "Outdoor ice rink", "Sports", "sports", False, 0.0),
    AmenityType("ice_track", "Ice skating track", "Sports", "sports", False, 0.0),
    # Civic UM WFS / parks
    AmenityType("park", "Park / square", "Civic", "um_parks", False, 0.0),
    AmenityType("school", "School", "Civic", "wfs", False, 0.0),
    AmenityType("library", "Library", "Civic", "wfs", False, 0.0),
    AmenityType("pharmacy", "Pharmacy", "Civic", "wfs", False, 0.0),
    AmenityType("bike_share", "Bike-share station", "Civic", "wfs", False, 0.0),
    AmenityType("park_and_ride", "Park & Ride", "Civic", "wfs", False, 0.0),
    AmenityType("hotel", "Hotel", "Civic", "wfs", False, 0.0),
    AmenityType("theater", "Theatre", "Civic", "wfs", False, 0.0),
    AmenityType("police_um", "Police (city map)", "Civic", "wfs", False, 0.0),
    AmenityType("cemetery", "Cemetery", "Civic", "wfs", False, 0.0),
    AmenityType("dormitory", "Dormitory", "Civic", "wfs", False, 0.0),
    AmenityType("metro_entrance", "Metro entrance (city)", "Civic", "wfs", False, 0.0),
    AmenityType("senior_club", "Senior club", "Civic", "wfs", False, 0.0),
    # OSM (already enriched or EV)
    AmenityType("hospital", "Hospital (OSM)", "OSM", "osm", False, 0.0),
    AmenityType("fire_station", "Fire station (OSM)", "OSM", "osm", False, 0.0),
    AmenityType("police", "Police (OSM)", "OSM", "osm", False, 0.0),
    AmenityType("metro", "Metro station (OSM)", "OSM", "osm", False, 0.0),
    AmenityType("shop", "Shop (OSM)", "OSM", "osm", False, 0.0),
    AmenityType(
        "charging_station",
        "EV charging (OSM)",
        "OSM",
        "osm",
        False,
        0.0,
    ),
)
"""All amenity categories exposed in Admin + Explore filters."""

_AMENITY_BY_KEY: dict[str, AmenityType] = {t.key: t for t in AMENITY_TYPES}
"""Lookup by catalog key."""

DEFAULT_MAP_OBJECT_TYPES: tuple[str, ...] = tuple(
    t.key for t in AMENITY_TYPES if t.default_map_visible
)
"""Explore map defaults: gyms + swimming pools."""

RECOMMENDER_DIST_CAP_M: float = 2000.0
"""Distance at which proximity score reaches 0 for amenity weights."""

# Keys whose dist_* columns already exist under these exact names.
LEGACY_DIST_KEYS: frozenset[str] = frozenset({
    "park",
    "school",
    "hospital",
    "fire_station",
    "police",
    "metro",
    "shop",
})
"""Amenity keys that map to pre-existing ``dist_*_m`` enrich columns."""


def amenity_type(key: str) -> AmenityType | None:
    """Return catalog entry for ``key``, or None."""
    return _AMENITY_BY_KEY.get(key)


def amenity_keys() -> tuple[str, ...]:
    """All catalog keys in definition order."""
    return tuple(t.key for t in AMENITY_TYPES)


def dist_column(key: str) -> str:
    """Listing column for nearest distance to amenity ``key``."""
    if key == "fire_station":
        return "dist_fire_station_m"
    if key == "metro":
        return "dist_metro_m"
    return f"dist_{key}_m"


def sport_key_for_osrid(osrid: object) -> str | None:
    """Map a sports OSRID (int/float/str) to a catalog key."""
    if osrid is None:
        return None
    raw = str(osrid).strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    return SPORT_OSRID_TO_KEY.get(raw)


def default_amenity_weights() -> dict[str, float]:
    """Zero-weight map for every catalog key (Admin defaults)."""
    return {t.key: float(t.default_weight) for t in AMENITY_TYPES}


def coerce_amenity_weights(raw: object) -> dict[str, float]:
    """Merge persisted weights with catalog defaults; clip to [-1, 1]."""
    out = default_amenity_weights()
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        sk = str(key)
        if sk not in out:
            continue
        try:
            weight = float(value)
        except (TypeError, ValueError):
            continue
        out[sk] = max(-1.0, min(1.0, weight))
    return out
