---
name: listing-data-source
description: >-
  Add a buyer listing portal (Otodom/OLX-class asks) to happy-housing with the
  same feature schema, incremental download, and shared NLP/GIS/train path.
  Use when adding Nieruchomości Online, Gratka, Morizon, Adresowo, Domiporta,
  or any new listing data source; when the user asks for a new scraper,
  portal ingest, or cross-source feature parity.
---

# Listing data source

Follow `happy-housing-development` TDD. This skill covers **buyer ask portals
only** (Explore / map / train labels). RCN / GIOŚ / GTFS stay GIS or comps.

## Fetch gate (do this first)

Stop if no allowed fetch path. **Do not** ship a new HTTP client, retry loop,
User-Agent stack, or portal SDK.

Priority:

1. **Official public API + existing PyPI client** (OLX = `olx-api-wrapper`).
2. **PyPI package** that already paginates and returns structured offers.
3. **Site-emitted structured payload** via existing
   `happy_housing.ingest.http_client.resilient_get` + a parser package
   (`extruct` for JSON-LD; stdlib `json` for `__NEXT_DATA__`). Same session
   helpers as Otodom (`bind_http_proxy`).
4. **Paid SaaS** (Apify, PropAPIS) only if user explicitly accepts a secret
   key + extra runtime dep. Default **off**.

Reject: BeautifulSoup-only card scraping, Playwright, new `requests.get`
wrappers, copy-paste retry/sleep from another module.

Search PyPI + publisher docs before writing fetch code. Record the chosen
path in `docs/data-sources/<source>.md`.

## Same features

Normalize **every** offer through `align_buyer_listing_row` /
`prepare_listings_for_model`. Column contract:
[schema.md](schema.md).

Sparse fields stay `None` / 0 — **columns must exist**. Do not invent a
source-specific schema.

Search-only vs detail:

| Pattern | When | Example |
|---|---|---|
| One-shot complete | API/package returns coords + attrs | OLX |
| Search + detail backfill | Search cards lack coords/attrs | Otodom, likely NOL |

If search lacks lat/lon: persist cards, register in
`ingest/detail_backfill.py`, skip GIS until geocoded. Closed 404/410 →
`listing_active=0` (keep for Stats, hide map).

## Incremental (required)

Default `*_incremental: true`. Full rebuild only when user unchecks Admin.

Required scrape loop:

1. Newest-first sort when the source supports it.
2. Load existing `{source}_listings.parquet`.
3. Skip / merge via `merge_listing_frames` (fresh wins on `listing_id`).
4. Early-stop after N consecutive pages of already-known ids
   (`stop_after_known_pages`, Otodom/OLX = 2).
5. Optional `listed_at` watermark.
6. Caps: `None` = unlimited; `resolve_ingest_cap` / `reached_cap`.
7. Apply FX: `apply_fx_to_listings_frame`.

Never delete known rows on a failed scrape — keep parquet, log, continue
pipeline.

## Wire-in checklist

Copy Otodom/OLX; do not fork a second pipeline.

- [ ] `src/happy_housing/ingest/<source>.py` — `normalize_*` + `scrape_*_warsaw`
- [ ] Tests: fake client / fake page; incremental early-stop; schema columns
- [ ] `BUYER_LISTING_SOURCES` + `BUYER_SOURCE_KEEP_PRIORITY`
      (`listing_common.py`). Default keep order: Otodom > richer geo source >
      OLX > fixture. Same-source rows never merge.
- [ ] `listing_id` prefix unique (`OTO-`, `OLX-`, `NOL-`, …)
- [ ] `IngestLimitsConfig` + `configs/experiment.yaml` + `from_dict`
- [ ] `AppSettings` + Admin **Sources** + **Ops** scrape button + pipeline
      checkbox (`pipeline_ingest.IngestFlags`)
- [ ] `pipeline.py` ingest part: geocoded (+ closed) rows, `source=` tag
- [ ] Dedup already runs after concat — no extra call
- [ ] Detail backfill registry if search-sparse
- [ ] `DEFAULT_SOURCE_CATALOG` in `app/metadata.py`
- [ ] `docs/data-sources/<source>.md` + row in `docs/data-sources/README.md`
- [ ] `make <source>` target
- [ ] Admin Settings for every user-facing toggle (AGENTS.md)

Shared post-ingest path (already source-agnostic — do not fork):

`prepare_listings_for_model` → `promote_listing_flags` (tags + description
NLP) → optional `enrich_frame_listing_attrs` (LM) → GIS / commute / photo
→ train with `train_buyer_asks_only` (Otodom+OLX+fixture+new sale asks;
RCN stays comps).

Rentals: `listing_kind=rent` OK to store; train already drops rent.

## Performance

Stream pages; bound workers; do not materialize HTML for all listings if
search cards suffice. Detail backfill = batched Ops job, not inline
pipeline default.

## Portal notes

Search JSON-LD + `tilesData` via `resilient_get` (no extra HTTP stack).
Detail JSON-LD backfill matches Otodom. Do not add Apify/PropAPIS.
