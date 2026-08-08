# happy-housing

**See what a Warsaw flat is really worth — and why.**

`happy-housing` turns open market and city data into a local decision aid for buyers, renters, and investors: fair asking-price estimates, location risk/opportunity signals, and an interactive map of how prices and livability shift over time.

Asking prices alone mislead. Two similar flats can differ by metro access, aircraft noise, nearby construction, air quality, or commute time to the centre. This project joins those signals with transaction history and listings so you can judge whether an offer is expensive, fair, or a bargain — with an explanation, not a black box.

## Who it is for

- **Buyers & renters** — compare listing price to a model “fair” price and inspect what drives the gap (metro vs sirens vs pollution).
- **Investors** — spot districts where planned infrastructure (e.g. new metro) may lift value, or where multi-year construction may hurt comfort.
- **Analysts & researchers** — reproducible Warsaw housing pipeline on free public data, no paid map or valuation APIs.

## What you get

| Capability | Business value |
|------------|----------------|
| Live / recent listings + official transaction prices | Market asking vs what people actually paid nearby |
| Spatial livability layers | Noise (emergency services, airport OOU), air quality, construction, planned transit |
| Commute & transit context | Realistic access to the city core and ZTM frequency, not just “distance to stop” |
| Fair-price model | Estimated zł/m² from location and property features |
| Explainable AI (SHAP) | Waterfall of “+metro / −fire station / −OOU” in zł terms |
| Time-aware map | Replay how prices and acoustic / investment pressure evolve with infrastructure |

## Motivation

Warsaw’s housing market moves fast, and public portals show asking prices without the spatial context that drives comfort and long-term value. Commercial valuators and paid APIs are expensive or opaque. Meanwhile the city and state publish rich free data — Rejestr Cen (RCN), ZTM schedules, GIOŚ air quality, GUNB permits, OpenStreetMap — that almost never land in one buyer-facing tool.

`happy-housing` exists to close that gap: **one local, zero-paid-API stack** that answers “Is this flat overpriced for this street?” and “What will change here in the next few years?”

## Methodology (high level)

1. **Ingest** — Collect listings and official transaction / transit / environmental feeds into local files.
2. **Enrich** — Attach vectorized GIS features (distances, zone flags, air stations, permits, routing) to every property.
3. **Store** — Columnar GeoParquet + in-process DuckDB for fast filters (no external database server).
4. **Model** — Train a tabular regressor on zł/m²; explain each prediction with SHAP.
5. **Present** — Streamlit app: offer card (listing vs model), SHAP waterfall, and interactive heatmap with time playback.

Principles: **free/open sources only**, **spatial vectorization** (no naive coordinate loops), **local compute** for cost and privacy.

Roadmap and technical phases: [docs/plan.md](docs/plan.md).

## Status

Early scaffold (Phase 0). Pipeline phases (ingest → GIS → storage → ML → GUI) are planned and landing incrementally. Expect fixtures and offline-friendly tests before live scrapers and full city extracts.

## Quick start

```bash
make install
make check
```

Edit `configs/experiment.yaml` for run metadata and MLflow settings.

```bash
make mlflow          # optional local tracking UI
make help            # all targets
```

```python
from pathlib import Path
from happy_housing.config import load_experiment_config

config = load_experiment_config(Path("configs/experiment.yaml"))
```

## Repository layout

| Path | Purpose |
|------|---------|
| `src/happy_housing/` | Ingest, GIS, storage, models, app |
| `configs/` | Experiment YAML |
| `data/fixtures/` | Small committed samples (raw dumps stay local / gitignored) |
| `docs/plan.md` | Phased plan |
| `tests/` | Pytest |

## License

Apache License 2.0. See [LICENSE](LICENSE).

Copyright 2026 Paweł Pozorski.
