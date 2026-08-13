# Schools (UM education WFS)

| | |
|--|--|
| **Catalog key** | `warsaw_schools` |
| **Module** | `src/happy_housing/ingest/warsaw_map.py` (`fetch_warsaw_schools`) |
| **URL** | `https://wfs.um.warszawa.pl/serwis` |
| **Artifact** | `data/cache/warsaw_schools.geojson` |

## Why we download it

`dist_school_m` for the price model and Explore livability context.

## What we extract

First non-empty **point/building** education layer among discovered + candidate
typeNames (SZKOLY / OSWIATA / PRZEDSZKOLA / …), with a normalized `name`
column. **Catchment** layers matching `OBWOD*` are skipped (they are school
district polygons, not school locations).

Enrichment fallback: OSM `amenity=school` from the amenities layer when the
schools frame is empty (`enrich_listings`).

## How download / increment works

1. GetCapabilities → education typeNames (catchments filtered).
2. Try candidates until one GetFeature returns rows; cache GeoJSON.
3. Fixture `schools.geojson` if WFS fails.
4. Cache skip unless `force_gis_refresh`.

## Volume

Depends on which typeName exists on the live server (often hundreds of POIs).
Catchment layer `OBWODY_SZKOL_PODSTAWOWYCH` is intentionally unused.

## Completeness

**Fixed for design.** Previously the first match could be catchments → wrong
nearest-school distances. Now POI-oriented layers only + OSM fallback.
