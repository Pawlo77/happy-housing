# GIOŚ air quality

| | |
|--|--|
| **Catalog key** | `gios` |
| **Modules** | `ingest/__init__.py` (stations), `ingest/gios_aq.py`, `gis/gios_aq_field.py` |
| **URL** | https://powietrze.gios.gov.pl/ (API v1) |
| **Artifacts** | `gios_stations.parquet`; AQ columns on stations; optional IDW rasters |

## Why we download it

Nearest-station air features and Explore BitmapLayer AQ field (alongside noise).

## What we extract

- Stations (national): id, lat/lon, name, city, address/admin ids.
- AQ for **Warsaw** stations: index + PM10 / PM2.5 / NO₂ µg/m³ (and index bands).
- Listing features: nearest-station distances + sampled field values when enabled.

## How download / increment works

Paginated station list (500/page). AQ HTTP per Warsaw station; fixture fallback.
Reuse parquet unless force / missing AQ columns. IDW field uses experiment bbox.

## Volume

Hundreds of national stations; handful of Warsaw stations with live AQ.

## Completeness

Stations **complete** (API). AQ **intentional Warsaw subset**. Edge listings may
nearest-join a non-Warsaw station without AQ values.
