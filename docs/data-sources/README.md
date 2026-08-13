# Data sources

Technical reference for every remote and fixture data source in
**happy-housing**. Each page covers purpose, download / incremental behaviour,
extracted fields, volume, cache layout, and a **completeness** verdict from a
2026-08 live audit against the publisher.

Pipeline wiring: [`src/happy_housing/pipeline.py`](../../src/happy_housing/pipeline.py).
BBox / caps: [`configs/experiment.yaml`](../../configs/experiment.yaml).
Home cards: [`DEFAULT_SOURCE_CATALOG`](../../src/happy_housing/app/metadata.py).

## Documents

| Doc | Key | Publisher | Completeness |
|-----|-----|-----------|--------------|
| [Otodom](otodom.md) | `otodom` | otodom.pl | Intentional subset (sale flats); incremental |
| [OLX](olx.md) | `olx` | olx.pl public API | Intentional subset (sale; rent optional); district-tiled |
| [RCN](rcn.md) | `rcn` | UM Warszawa | Intentional subset (geolocated deeds) |
| [OpenStreetMap](osm.md) | `osm` | OSM / Overpass | Intentional bbox subset |
| [Parks & green](warsaw-parks.md) | `warsaw_parks` | UM WFS + OSM | **Fixed** — UM skwers + OSM green |
| [Schools](warsaw-schools.md) | `warsaw_schools` | UM WFS (+ OSM fallback) | **Fixed** — catchments skipped |
| [City POIs](city-pois.md) | `city_pois` | UM WFS + MapViewer + OSM | Civic under 10k; sports intentional |
| [GTFS](gtfs.md) | `gtfs` | mkuran.pl / ZTM | Complete for feed |
| [GIOŚ](gios.md) | `gios` | powietrze.gios.gov.pl | Stations national; AQ Warsaw |
| [GUNB](gunb.md) | `gunb` | GUGiK / GUNB | Recent years by default |
| [Crime](crime.md) | `crime` | GUS US Warszawa | Complete for 2024 district annex |
| [Noise](noise.md) | `oou` / `um_noise` | EEA + UM WMS | Intentional (strategic maps) |
| [Traffic & bike](traffic-bike.md) | `traffic` / `bike_paths` | OSM | Intentional proxy / OSM tags |
| [FX](fx.md) | `fx` | NBP | Complete for currencies seen |

## Completeness summary

See [completeness.md](completeness.md) for the audit method, the parks
regression, WFS `maxFeatures` behaviour, and remaining known gaps.

## Design principles

1. **Warsaw-first** — experiment bbox crops OSM / traffic / bike / noise rasters.
2. **Cache + force** — GIS layers skip when cache exists unless Admin **Force GIS refresh**.
3. **Fixtures offline** — network failure → `data/fixtures/` where wired.
4. **Honest scope** — incremental Otodom and recent-year GUNB are deliberate, not bugs.
5. **Warn on truncation** — WFS returns hitting `maxFeatures=10000` log a warning
   (UM `startIndex` paging is unreliable on this MapGuide server).
