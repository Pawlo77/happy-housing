# UM Warszawa RCN (notarial transactions)

| | |
|--|--|
| **Catalog key** | `rcn` |
| **Module** | `src/happy_housing/ingest/rcn.py` |
| **URL** | Published GML zip (see `ingest_urls.rcn_zip` in experiment / Admin) |
| **Artifact** | `data/processed/rcn_transactions.parquet` |

## Why we download it

Deed / notarial prices for fair-value context, neighborhood features, and
optional deed-based analysis — not the default ask-label training target.

## What we extract

Joined `RCN_Transakcja` → `RCN_Nieruchomosc` → `RCN_Dzialka`:

- `lon`, `lat` (EPSG:2178 → WGS84; district centroid fallback)
- `price_pln`, `area_m2`, `price_per_m2`, `district`, `listed_at`
- Market / transaction / parcel metadata (`rodzaj_rynku`, `adres_dzialki`, …)

## How download / increment works

1. `download_file` with `.url` sidecar — re-download when URL changes.
2. Stream-parse GML; fingerprint zip + `rcn_max_rows` in step cache.
3. Soft Warsaw box filter on coordinates; drop price ≤ 0 / unlinked rows.

## Volume

Hundreds of thousands of raw GML features; kept Warsaw-geolocated rows are far
fewer. Snapshot zip is version-pinned in config.

## Completeness

**Intentional subset** of the published dump: resolvable coordinates (or
district fallback), positive price, Warsaw-ish bbox. Training remap may further
clip extreme area / zł/m².
