# Python experiment template

GitHub template for ML/research projects: uv, prek, pytest, MLflow, YAML configs, notebooks, and Interspeech LaTeX scaffold.

**Do not run experiments in this repo** until placeholders are replaced.

## Create a project

1. On GitHub: **Use this template** → create your repository.
2. Clone and replace every placeholder token project-wide.
3. Rename paths and verify locally.

## Placeholder tokens

| Token | Example |
|-------|---------|
| `__PROJECT_SLUG__` | `my-experiment` |
| `__PROJECT_MODULE__` | `my_experiment` |
| `__PROJECT_DESCRIPTION__` | One-line project summary |
| `__AUTHOR_NAME__` | Your full name |
| `__AUTHOR_EMAIL__` | `you@example.com` |
| `__GITHUB_OWNER__` | GitHub username or org |
| `__COPYRIGHT_YEAR__` | `2026` |
| `__AUTHOR_FIRST_NAME__` | `Jane` (LaTeX `report/report.tex`) |
| `__AUTHOR_LAST_NAME__` | `Doe` |
| `__AUTHOR_AFFILIATION__` | `University, Country` |

Paper placeholders in `report/report.tex`: `TITLE`, `KEYWORDS`, `ABSTRACT`.

## Customize

1. **Find and replace** all tokens (including LaTeX author fields if desired).
2. **Rename the package directory:**

   ```bash
   mv src/__PROJECT_MODULE__ src/your_module
   ```

3. **Rename the Cursor skill folder** (optional):

   ```bash
   mv .cursor/skills/__PROJECT_SLUG__-development .cursor/skills/your-slug-development
   ```

4. **Edit** `configs/experiment.yaml` — extend with dataset, model, and training sections as needed.

5. **Lock and verify:**

   ```bash
   uv lock
   make install
   make check
   ```

6. **MLflow:** `make mlflow` (UI on port 5005 by default). Artifacts live under `./mlruns` and `mlruns.db` (gitignored).

CI workflows skip while `__PROJECT_SLUG__` remains in `pyproject.toml`. After customization and `uv lock`, **Prek Checks** and **Pytest** run on push/PR.

## What’s included

- `src/__PROJECT_MODULE__/` — config (YAML loader), MLflow tracking, reproducibility report
- `configs/experiment.yaml` — minimal YAML experiment config stub
- `notebooks/` — Jupyter workspace
- `report/` — Interspeech 2026 paper scaffold (`report.tex`, `template.tex`, `Interspeech.cls`)
- `Makefile` — `install`, `check`, `prek`, `mlflow`, `coverage`
- `.pre-commit-config.yaml` — prek hooks (ruff, mypy, bandit, codespell, uv-lock)
- `.github/workflows/` — test and prek only (no publish, no docs deploy)
- `.cursor/rules/` and `AGENTS.md` — agent workflow

No scaffold script — manual find-replace + rename after **Use this template**.
