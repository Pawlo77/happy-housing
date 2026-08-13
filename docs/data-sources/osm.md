# OpenStreetMap (amenities, shops, fitness, green)

| | |
|--|--|
| **Catalog key** | `osm` (amenities/shops); green also feeds `warsaw_parks` |
| **Module** | `src/happy_housing/gis/__init__.py` |
| **URL** | https://www.openstreetmap.org/ via OSMnx / Overpass |
| **Artifacts** | `data/cache/osm_amenities.geojson`, `osm_shops.geojson`, `osm_fitness.geojson`, `osm_green_areas.geojson` (+ `.meta.json`) |

## Why we download it

Livability distances (hospital, fire, police, subway, shops, …), commercial
gyms for city POIs, and **green polygons** that UM PLACE_SKWERY omits.

## What we extract

| Fetch | Tags | Geometry | Columns |
|-------|------|----------|---------|
| Amenities | `amenity=*`, `station=subway` | Centroid points | amenity, name, brand, … |
| Shops | `shop=*` | Centroid points | shop, name, … |
| Fitness | `leisure=fitness_centre`, `amenity=gym` | Points → city POI | object_type=gym |
| Green | park/garden/nature_reserve, forest/recreation_ground, wood | **Polygons kept** | name, leisure/landuse/natural |

## How download / increment works

- `features_from_bbox` with experiment bbox; 300s Overpass timeout.
- Cache + tag fingerprint meta; `force_gis_refresh` busts cache.
- Fixture JSON fallback for amenities/shops when Overpass fails.
- `use_osm=false` → fixtures only.

## Volume

Thousands of amenity/shop points in the Warsaw bbox; hundreds–thousands of green
polygons depending on tagging.

## Completeness

**Intentional subset** — bbox-limited community map, not a cadastre. Amenity
path drops leisure-only rows (fitness/green use dedicated fetches). See
[warsaw-parks.md](warsaw-parks.md) for green merge.
