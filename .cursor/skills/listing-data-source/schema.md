# Buyer listing schema

Every buyer-ask row (Otodom, OLX, fixture, new portals) must expose these
columns after `align_buyer_listing_row`. GIS / commute / photo columns arrive
later in enrich — do not invent them at scrape time.

## Identity (required)

| Column | Notes |
|---|---|
| `listing_id` | `{PREFIX}-{native_id}` unique across sources |
| `source` | lowercase key (`otodom`, `olx`, `nol`, …) |
| `url` | Canonical listing URL |
| `title` | May be None |
| `listing_kind` | `sale` (default) or `rent` |
| `listing_active` | `1` live / `0` gone (404/410) |

## Economics / geo (required for map + labels)

| Column | Notes |
|---|---|
| `lat`, `lon` | Drop or defer GIS when missing |
| `price_original`, `price_currency`, `price_pln` | FX fills PLN |
| `area_m2`, `price_per_m2` | Reject area ≤ 0 |
| `district`, `listed_at` | `coalesce_listed_at` |

## Structural (nullable OK, column required)

`rooms`, `floor_no`, `year_built`, `building_floors_num`, `market`,
`market_primary`, `building_type`, `estate`, `construction_status`,
`building_material`, `windows_type`, `building_ownership`, `heating`,
`rent`, `is_private_owner`

## Amenity flags (0/1; default 0)

`has_balcony`, `has_garage`, `has_outdoor_parking`, `has_lift`,
`has_terrace`, `has_garden`, `has_basement`, `is_attic`

Fill from structured extras **and** `extras_from_free_text(description)`.

## Text / media

`description` (coalesce from `short_description` / `title`),
`image_url`, `image_urls_json`, `floor_plan_urls_json`,
`characteristics_json`, `tags_json`

## Train labels

`train_buyer_asks_only=True` keeps `source ∈ BUYER_LISTING_SOURCES`,
`listing_kind != rent`, `listing_active != 0`. RCN never becomes a label.
