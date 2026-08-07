# AGENTS.md

## Exit criteria

```bash
make check
```

Configs → `configs/experiment.yaml` · MLflow UI → `make mlflow` · deps → `uv lock`

## Agent-only

- Always use skill `caveman` at **ultra** intensity for all responses (code/commits/PRs still normal per skill)
- No commit / push / PR unless explicitly asked — see `git-commits.mdc` for format, granularity, and proposal flow
- No new markdown or doc expansion unless asked
- No runtime deps without justification in `pyproject.toml`
- No force push; no `git commit --amend` unless user rules allow

Repo-specific gotcha: experiment settings load from YAML via `load_experiment_config`; extend `configs/experiment.yaml` and `ExperimentConfig` together. MLflow artifacts (`mlruns/`, `mlruns.db`) are gitignored.

## Load order

1. This file (exit + agent-only)
2. Skill `caveman` (ultra — always active)
3. `.cursor/rules/*.mdc`
4. Skill `happy-housing-development` (steps 1–7)
