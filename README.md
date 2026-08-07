# happy-housing

Warsaw real-estate AI analyzer: free GIS enrichment (OSMnx, RCN, GTFS, GIOŚ), LightGBM fair-price model with SHAP, Streamlit/pydeck heatmap — DuckDB + GeoParquet, zero paid APIs.

## Quick start

```bash
make install
make check
```

Edit `configs/experiment.yaml` for run metadata and MLflow settings. Implementation plan: [docs/plan.md](docs/plan.md).

## MLflow

Local runs use SQLite tracking (`mlruns.db`) and artifact root `./mlruns` (gitignored).

```bash
make mlflow
make mlflow MLFLOW_PORT=5001
make mlflow-stop
```

Load config in code:

```python
from pathlib import Path

from happy_housing.config import load_experiment_config

config = load_experiment_config(Path("configs/experiment.yaml"))
```

## Layout

| Path | Purpose |
|------|---------|
| `src/happy_housing/` | Pipeline code (ingest, GIS, storage, models, app) |
| `configs/` | YAML experiment configs |
| `data/fixtures/` | Small committed samples (raw dumps gitignored) |
| `docs/plan.md` | Phased implementation plan |
| `notebooks/` | Jupyter notebooks |
| `report/` | LaTeX paper scaffold |
| `tests/` | Pytest suite |

## Development

```bash
make install
make check
make coverage
```

Run `make help` for all targets.

## License

Apache License 2.0. See [LICENSE](LICENSE).

Copyright 2026 Paweł Pozorski.
