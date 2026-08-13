# Completeness audit (2026-08)

Validation that each pipeline source downloads what we claim, with emphasis on
gaps like the historical “missing Warsaw parks” issue.

## Method

1. Trace `pipeline.py` + `configs/experiment.yaml` for every fetch.
2. Read ingest modules for caps, filters, early-stop, and cache rules.
3. Live probe UM WFS (`https://wfs.um.warszawa.pl/serwis`) for
   `PLACE_SKWERY` and education typeNames.
4. Confirm MapGuide `startIndex` does **not** page (same first feature IDs at
   offsets 0 / 100 / 400).

## Parks regression (root cause + fix)

| Claim | Finding |
|-------|---------|
| “We download all Warsaw parks” via PLACE_SKWERY | **False historically.** Layer is official parks/squares only (**427** features live, well under `maxFeatures=10000` — not a truncation bug). |
| Missing forests / large greens (e.g. Las Kabacki) | Expected: not in PLACE_SKWERY; no `LASY` WFS typeName. |
| Truncation at 10k | Not the parks bug (427 ≪ 10k). Still a risk for huge layers (e.g. trees). |

**Fix shipped:** `fetch_warsaw_parks` merges UM PLACE_SKWERY with OSM green
polygons (`leisure=park|garden|nature_reserve`, `landuse=forest|recreation_ground`,
`natural=wood`) inside the experiment bbox. Cache schema `PARKS_CACHE_SCHEMA=2`
invalidates old UM-only GeoJSON. See [warsaw-parks.md](warsaw-parks.md).

## Schools

| Claim | Finding |
|-------|---------|
| First education WFS layer = school POIs | **Was wrong** — discovery preferred `OBWODY_SZKOL_PODSTAWOWYCH` (catchment polygons). Centroids ≠ school buildings. |

**Fix shipped:** catchments filtered out of discovery + candidates; OSM
`amenity=school` remains enrich fallback when WFS empty. See
[warsaw-schools.md](warsaw-schools.md).

## WFS pagination

- Client sends `maxFeatures=10000`.
- UM MapGuide **ignores** `startIndex` for paging → cannot walk past 10k with
  offsets.
- Mitigation: warn when `len == maxFeatures`; parks/civic layers checked live
  are under the cap; trees (`ZIELEN_DRZEWA`) exceed it and are **not** ingested.

## Per-source verdicts

| Source | Verdict | Notes |
|--------|---------|-------|
| Otodom | Intentional subset | Warsaw sale flats; incremental early-stop |
| RCN | Intentional subset | Geolocated deeds in loose Warsaw box |
| OSM amenities/shops | Intentional subset | Experiment bbox; OSM tagging gaps |
| Parks | Complete for design | UM official + OSM green merge |
| Schools | Complete for design | POI layers only; OSM fallback |
| City POIs | Mostly complete | Civic &lt;10k; sports SQL exclusions intentional |
| GTFS | Complete | Full Warsaw feed zip |
| GIOŚ stations | Complete | National API pages |
| GIOŚ AQ | Intentional | Warsaw stations enriched |
| GUNB | Intentional / risk | Recent years default; national WFS scan |
| Crime | Complete (scope) | 2024 district annex |
| EEA aircraft | Intentional | Fixture END 2012/2015 Chopin |
| UM noise WMS | Complete (scope) | 2022 strategic layers × bbox raster |
| Traffic / bike | Intentional | OSM proxy, sample caps |
| FX | Complete | NBP mid for currencies seen |

## Remaining gaps (not bugs unless we expand scope)

1. **Otodom rentals / suburbs** — out of search URL scope.
2. **GUNB full history** — Admin / `gunb_wfs_full_history` only.
3. **EEA road/rail** — fixture placeholders unless replaced.
4. **OSM completeness** — community tagging, not a cadastre.
5. **WFS layers &gt;10k** — would need bbox tiling (not implemented); none of the
   currently ingested civic layers hit the cap in the 2026-08 probe.
