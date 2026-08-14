# OLX

Warsaw flat listings from the public OLX offers API via
[`olx-api-wrapper`](https://pypi.org/project/olx-api-wrapper/)
(`olx.public.OlxPublic`), not the Partner OAuth CRUD API.

## Purpose

Third asking-price source alongside Otodom. Default = **sale** flats
(`category_id=14`). Optional rent (`category_id=15`) is stored with
`listing_kind=rent` and excluded from LightGBM ask-price labels.

## Download behaviour

1. Load Warsaw district ids from
   `https://www.olx.pl/api/v1/geo-encoder/cities/17871/districts`
   (hardcoded fallback if the geo-encoder fails).
2. For each district, call `get_offers` with
   `sort_by=created_at:desc`, `city_id=17871`, `region_id=2`,
   `district_id=<id>`.
3. City-wide offset pagination overlaps heavily — **district tiling** is
   required to get past the ~1000 `total_elements` window.
4. Incremental mode merges into `data/processed/olx_listings.parquet` and
   early-stops a district after consecutive fully-known newest pages.

IDs: `OLX-{offer_id}`. Parquet: `olx_listings.parquet`.

## Admin

- **Sources** — caps, incremental, include-rent.
- **Ops** — “Run OLX scrape” background job; scrape-in-pipeline toggle.

## Completeness

Intentional subset: sale-first, district-tiled, rate-limited. Not a full
historical dump. Overlap with Otodom is expected; the pipeline collapses
cross-portal duplicates (shared geo fingerprint or cover photo) before enrich
and scoring, preferring Otodom over OLX.

## Feature parity

OLX does **not** use a separate detail-backfill job. Each offer comes from one
public API response (coords, description, photos, and structured ``params``).
Normalize maps those params plus description tokens into the same buyer-listing
schema as Otodom (amenity flags, heating, build year, rent, …). Otodom defaults
to search cards + background detail backfill for the same columns.
