# Parks & green space (UM + OSM)

| | |
|--|--|
| **Catalog key** | `warsaw_parks` |
| **Module** | `src/happy_housing/ingest/warsaw_map.py` (`fetch_warsaw_parks`), `gis.fetch_osm_green_areas` |
| **URLs** | WFS `https://wfs.um.warszawa.pl/serwis` typeName `ns92528565:PLACE_SKWERY`; OSM Overpass |
| **Artifact** | `data/cache/warsaw_parks.geojson` + `warsaw_parks.meta.json` |

## Why we download it

Nearest green distance (`dist_park_m`), Explore park polygons, and
snap-listings-out-of-parks. Official skwers alone under-represent forests and
community greens.

## What we extract

| Source | Content | Columns |
|--------|---------|---------|
| UM PLACE_SKWERY | Official parks / squares / skwers | `name` (from NAZWA_*), `park_source=um_place_skwery`, polygon |
| OSM green | park, garden, nature_reserve, forest, recreation_ground, wood | `name`, `park_source=osm`, polygon |

Merged into one GeoJSON (EPSG:4326). Schema version **2**.

## How download / increment works

1. WFS GetFeature (`maxFeatures=10000`) for PLACE_SKWERY → intermediate UM cache.
2. If pipeline passes experiment `bbox`, fetch OSM green polygons.
3. `merge_park_frames` → write combined cache + meta (`n_um`, `n_osm`, schema).
4. Skip network when meta schema matches unless `force_gis_refresh`.
5. Fixture `data/fixtures/parks.geojson` if both sides empty.

## Volume (live 2026-08)

- UM PLACE_SKWERY: **427** features (≪ 10k — not truncated).
- OSM green: bbox-dependent (typically hundreds–thousands of polygons).

## Completeness

**Fixed.** Historical gap was **layer scope**, not pagination: PLACE_SKWERY ≠ all
green space. Pipeline now merges OSM green. Truncation warning fires if any WFS
page returns exactly `maxFeatures`. Details: [completeness.md](completeness.md).
