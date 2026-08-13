# Traffic & bicycle infrastructure (OSM)

| | |
|--|--|
| **Catalog keys** | `traffic`, `bike_paths` |
| **Modules** | `gis/traffic.py`, `gis/bike_paths.py` |
| **URL** | OpenStreetMap via OSMnx |
| **Artifacts** | `traffic_roads.parquet`; `bike_paths.geojson`, `bike_path_sample.parquet` |

## Why we download it

Diurnal street-activity **proxy** and bike-path coverage for Explore Map /
Spatial features and listing distances — not measured AADT.

## What we extract

- Traffic: densified road sample points (`lon`, `lat`, `base_weight`, `highway`,
  …) → listing neighbourhood weights / major-road distance.
- Bike: cycleway / designated bicycle ways → `dist_bike_path_m`, coverage metres.

## How download / increment works

OSMnx within experiment bbox; file cache; `force_gis_refresh` to rebuild.
Traffic graph excludes `service`; sample capped (~28k points). Bike map draws a
longest-segment sample for UI size.

## Volume

Tens of thousands of traffic sample points; thousands of bike ways (map-capped).

## Completeness

**Intentional OSM proxy.** Missing informal paths / untagged lanes; traffic is
heuristic diurnal weights, not sensors.
