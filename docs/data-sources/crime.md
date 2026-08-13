# GUS crime (Panorama dzielnic)

| | |
|--|--|
| **Catalog key** | `crime` |
| **Module** | `src/happy_housing/ingest/crime.py` |
| **URL** | GUS US Warszawa Panorama dzielnic Excel annex (`.7z`) |
| **Artifact** | `data/processed/crime_by_district.parquet` |

## Why we download it

District-level crime rates / detectability for Stats and listing joins by
dzielnica — not point-level incidents.

## What we extract

Per district: totals, criminal subset, population, rates per 1,000,
detectability %, year, `district_key`.

## How download / increment works

Download `.7z` → extract → parse sheets → overwrite parquet. Fixture JSON on
failure.

## Volume

18 Warsaw dzielnice × one annex year (URL-driven; 2024 in current wiring).

## Completeness

**Complete** for the stated district annex. **Intentional aggregate** (no
geocoded incidents).
