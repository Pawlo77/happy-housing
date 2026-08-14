# Nieruchomości Online (listing asks)

| | |
|--|--|
| **Catalog key** | `nol` |
| **Module** | `src/happy_housing/ingest/nol.py` |
| **URL** | https://www.nieruchomosci-online.pl/ (Warsaw flat **sale** search) |
| **Artifact** | `data/processed/nol_listings.parquet` |

## Why we download it

Second asking-price universe for Warsaw flats (between Otodom and OLX in
dedup priority). Same buyer schema as Otodom/OLX: Explore, map, recommender,
and default LightGBM labels.

## Fetch path

No official API and no PyPI client. Search and detail HTML are fetched with
existing `resilient_get` + `bind_http_proxy`. Parsers use stdlib JSON:

- Search `application/ld+json` CollectionPage → nested Offers
- Search `tilesData:` JSON for lat/lon, floor, year
- Detail JSON-LD `@type: Apartment` for amenities, geo, images

IDs: URL `…/{digits}.html` → `listing_id=NOL-{id}`.

## What we extract

- Identity / geo: `listing_id`, `url`, `lat`, `lon`, `district`, `listed_at`
- Price: `price_pln`, `price_per_m2`, FX fields when non-PLN
- Size / layout: `area_m2`, `rooms`, `floor_no`, `year_built`, building attrs
- Flags: balcony, garage, lift, parking, basement, …
- Text / media: `description`, photo URLs, `characteristics_json`

Empty-price developer cards are skipped. Rows missing lat/lon stay in parquet
for detail backfill; pipeline ingest keeps geocoded + closed rows.

## How download / increment works

1. Walk search pages (`Od najnowszych` is the site default; `&p=N`).
2. Incremental mode (default): load existing parquet; stop after two consecutive
   fully-known newest pages; merge without wiping detail fields.
3. Caps: `nol_max_pages` / `nol_max_listings` (null = unlimited).
4. Detail JSON-LD runs as a background Ops job (`nol_fetch_details: false`).
5. 404/410 → `listing_active=0` (Stats keeps the ask; map hides it).
6. On scrape failure: reuse cached parquet and re-apply FX.

## Admin

- **Sources** — caps, incremental, include-rent (off), inline details (off).
- **Ops** — “NOL scrape” job; scrape-in-pipeline toggle; shared detail backfill.

## Completeness

**Intentional subset.** Warsaw sale flats first; rentals opt-in. Not houses or
plots. Unchanged older listings are not re-walked every run. Dedup prefers
Otodom over NOL over OLX over fixtures.
