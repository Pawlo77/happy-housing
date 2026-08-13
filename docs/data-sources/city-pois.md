# City POIs (civic WFS + sports MapViewer + OSM fitness)

| | |
|--|--|
| **Catalog key** | `city_pois` |
| **Module** | `src/happy_housing/ingest/warsaw_map.py` (`fetch_warsaw_city_pois`) |
| **URLs** | WFS `wfs.um.warszawa.pl`; MapViewer `testmapa.um.warszawa.pl/mapviewer/dataserver` |
| **Artifact** | `data/cache/warsaw_city_pois.geojson` (+ schema meta) |

## Why we download it

Explore city-POI layers and recommender proximity nudges (libraries, pharmacies,
sports halls, gyms, metro entrances, …) beyond generic OSM amenities.

## What we extract

Unified columns: `object_type`, `name`, `source` (`wfs` | `sports` | `osm`), geometry.

| Bucket | Layers / query |
|--------|----------------|
| Civic WFS | libraries, pharmacies, bike-share, P+R, hotels, theatres, police, cemeteries, dorms, metro entrances, senior clubs |
| Sports | JDBC themes `OBIEKTY_SPORTOWE_MAPA_V`, ice rinks; exclude inactive / some OSRIDs |
| OSM fitness | commercial gyms merged into `object_type=gym` |

## How download / increment works

Parallel civic WFS + MapViewer + OSM fitness; combine; schema-versioned cache.
`maxFeatures=10000` per WFS layer with truncation warning. BBox used for OSM
fitness.

## Volume

Civic layers typically hundreds–low thousands each (under 10k in 2026-08 audit).
Sports: active city inventory from MapViewer SQL.

## Completeness

**Mostly complete** for listed civic typeNames (under cap). Sports **intentional
subset** (active objects, OSRID exclusions). OSM gyms fill commercial fitness
gaps in the city inventory.
