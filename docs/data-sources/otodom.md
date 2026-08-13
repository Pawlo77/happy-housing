# Otodom (listing asks)

| | |
|--|--|
| **Catalog key** | `otodom` |
| **Module** | `src/happy_housing/ingest/otodom.py` |
| **URL** | https://www.otodom.pl/ (Warsaw flat **sale** search) |
| **Artifact** | `data/processed/otodom_listings.parquet` |

## Why we download it

Primary **asking-price** universe for Warsaw flats: labels for LightGBM, Explore
board/map, recommender, Offer AI, and Stats. Transaction deeds (RCN) are
context — asks drive the default price model.

## What we extract

From search cards + detail `__NEXT_DATA__`:

- Identity / geo: `listing_id`, `url`, `lat`, `lon`, `district`, `listed_at`
- Price: `price_pln`, `price_per_m2`, FX fields when non-PLN
- Size / layout: `area_m2`, `rooms`, `floor`, `year_built`, building attrs
- Flags: balcony, garage, lift, parking, …
- Text / media: `description`, photo URLs / media JSON, `characteristics_json`

Rows missing lat/lon/price/area are dropped.

## How download / increment works

1. Walk search pages (`by=LATEST`, `direction=DESC` — newest first).
2. Incremental mode (default): load existing parquet; stop after fully-known
   newest pages and/or `listed_at` watermark; detail-fetch only new or
   price-changed IDs; merge.
3. Caps: `otodom_max_pages` / `otodom_max_listings` (null = unlimited).
4. On scrape failure: reuse cached parquet and re-apply FX.

## Volume (typical)

Tens of thousands of active Warsaw sale listings (varies with market). Full
refresh is heavy (HTML + detail concurrency); incremental is the steady state.

## Completeness

**Intentional subset.** Not rentals, not all Mazowsze suburbs outside Otodom’s
Warsaw search, and unchanged older listings are not re-scraped every run.
For a full re-walk, disable incremental / raise caps in Admin → Sources / Run.
