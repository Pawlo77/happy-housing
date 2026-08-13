# Noise (EEA aircraft + UM strategic WMS)

| | |
|--|--|
| **Catalog keys** | `oou` (EEA fixtures), `um_noise` (UM WMS 2022) |
| **Modules** | `gis/__init__.py` (contour attach), `gis/warsaw_noise.py` |
| **URLs** | EEA END Discomap (fixtures); UM mapa WMS GetMap |
| **Artifacts** | `data/fixtures/chopin_aircraft_*.geojson`; `data/cache/warsaw_noise/*.png` |

## Why we download it

Aircraft / road / rail / tram / industry exposure for `in_oou`, noise dB
features, and Explore overlays. UM 2022 strategic maps are the live city
source; EEA fixtures cover Chopin aircraft contours offline.

## What we extract

- EEA: Lden / Lnight contour polygons → listing `aircraft_*_db`, `in_oou`.
- UM WMS: 10 rasters (sources × Lden/Lnight) → per-source dB, totals, flags.
  Legend RGB → 45–80 dB bands; point sample nearest pixel.

## How download / increment works

- EEA: **no live download** — ship fixtures (END 2012/update2015 Chopin).
- UM: parallel GetMap PNG cache for experiment bbox @ ~2000×1600; refresh on
  force or size mismatch. When rasters succeed they override same-named
  fixture columns.

## Volume

Fixture polygons for Chopin envelope; 10 PNG rasters covering the bbox.

## Completeness

**Intentional.** UM path complete for published 2022 WMS layers at configured
resolution. EEA road/rail fixtures are placeholders unless replaced. Raster
sampling has edge error vs true contour polygons.
