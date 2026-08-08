# AGENTS.md

## Exit criteria

```bash
make check
```

Configs → `configs/experiment.yaml` · MLflow UI → `make mlflow` · deps → `uv lock`

## Agent-only

- Always use skill `caveman` at **ultra** intensity for all responses (code/commits/PRs still normal per skill)
- **After every completed task:** validate quality, performance, and correctness (`make check` + hot-path / correctness review per `.cursor/rules`), then **commit and push** (granular conventional commits per `git-commits.mdc` message format). Do not wait for another ask.
- Pull requests: still propose title/body; create only when asked
- No new markdown or doc expansion unless asked (this file is the exception for agent process rules)
- No runtime deps without justification in `pyproject.toml`
- No force push; no `git commit --amend` unless user rules allow
- **Settings for every new feature:** when adding user-facing behaviour (models, memory, filters, AI, scrapers, …), add Admin Settings controls in the matching tab (General / AI / Stars & News / Recommendations / Stats / Pipeline or a new tab). Persist tunables via `app_settings` (or a dedicated store), expose clear/reset/retrain actions where relevant, and **never** dump raw settings JSON in the UI.
- **README screenshots on important ship:** when shipping a user-visible feature (new page/tab, major Explore surface, Admin capability, Home/Stats UX), refresh `docs/screenshots/` (`make app` + `uv run python scripts/capture_readme_screenshots.py`) and update `README.md` so the App tour matches the live UI. Skip for pure refactors, tests, or invisible pipeline internals.
- **No custom CSS:** Streamlit UI = stock widgets + `.streamlit/config.toml` theme only. Never inject look-and-feel CSS/HTML chrome. One exception: `apply_dense_top_layout()` may set main `padding-top` only (top nav gap; no TOML knob). Prefer `st.metric` / GFM markdown / columns / tabs. See `.cursor/rules/no-custom-css.mdc`.

Repo-specific gotcha: experiment settings load from YAML via `load_experiment_config`; extend `configs/experiment.yaml` and `ExperimentConfig` together. MLflow artifacts (`mlruns/`, `mlruns.db`) are gitignored. App behaviour tweaks live in `data/processed/app_settings.json` (Admin UI only).

## Load order

1. This file (exit + agent-only)
2. Skill `caveman` (ultra — always active)
3. `.cursor/rules/*.mdc`
4. Skill `happy-housing-development` (steps 1–7)
