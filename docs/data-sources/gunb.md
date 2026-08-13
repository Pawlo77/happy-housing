# GUNB building permits

| | |
|--|--|
| **Catalog key** | `gunb` |
| **Module** | `src/happy_housing/ingest/gunb.py` |
| **URL** | GUGiK WFS + Mazowieckie bulk zip |
| **Artifact** | `data/processed/gunb_mazowieckie.parquet` (+ per-layer WFS caches) |

## Why we download it

Nearby construction / permit activity proxy (`dist_gunb_m`) for disruption and
future supply signal.

## What we extract

Geolocated points: `lon`, `lat`, `adres`, `numer_decyzji`, `miasto`, plus WFS
`ms:*` attributes. Bulk CSV zip is archival (often without coordinates).

## How download / increment works

1. Prefer GUGiK WFS layers (default **recent years** 2024–2026; full history
   optional).
2. Page GetFeature (5000); client-filter Warsaw metro (city / TERC / bbox).
3. Per-layer parquet cache under raw/; merge + dedupe.

## Volume

Thousands of Warsaw-metro points for recent years; full history much larger.
Map UI samples points (message-size cap).

## Completeness

**Intentional recent-year subset** by default. National WFS + client filter is
slow and can fail; retries/cache mitigate. Enable full history in Admin when
needed.
