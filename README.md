> **Template repo** — replace `__PROJECT_*__` tokens and rename `src/__PROJECT_MODULE__/` before use. See [TEMPLATE.md](TEMPLATE.md).

# __PROJECT_SLUG__

__PROJECT_DESCRIPTION__

## Quick start

```bash
make install
make check
```

Edit `configs/experiment.yaml` for run metadata and MLflow settings.

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

from __PROJECT_MODULE__.config import load_experiment_config

config = load_experiment_config(Path("configs/experiment.yaml"))
```

## Layout

| Path | Purpose |
|------|---------|
| `src/__PROJECT_MODULE__/` | Experiment code (config, orchestration, training) |
| `configs/` | YAML experiment configs |
| `notebooks/` | Jupyter notebooks |
| `report/` | Interspeech LaTeX paper scaffold |
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

Copyright __COPYRIGHT_YEAR__ __AUTHOR_NAME__.
