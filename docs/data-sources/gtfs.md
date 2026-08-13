# ZTM Warsaw GTFS

| | |
|--|--|
| **Catalog key** | `gtfs` |
| **Module** | `src/happy_housing/ingest/__init__.py`, `gtfs_network.py` |
| **URL** | `ingest_urls.gtfs_zip` (default mkuran.pl Warsaw feed) |
| **Artifacts** | `gtfs_stops.parquet`, `gtfs_stop_hourly.parquet`, `gtfs_route_shapes.parquet` |

## Why we download it

Transit proximity (`dist_transit_*`), Explore stop/route layers, commute context.

## What we extract

- Full `stops.txt` columns needed for stop_id / lat / lon (+ names).
- Hourly departure counts per stop.
- Simplified route shapes (mode, colors, downsampled path).

## How download / increment works

Zip via `download_file` (`.url` sidecar). Skip re-export when stop/hourly/shape
parquets are newer than the zip unless forced.

## Volume

Full Warsaw GTFS: thousands of stops; large `stop_times` aggregated to hourly.

## Completeness

**Complete** for the published Warsaw feed bundle (not a national multi-operator
merge).
