---
name: __PROJECT_SLUG__-development
description: >-
  Ordered implementation workflow for __PROJECT_SLUG__. Use for any code, test,
  config, hook, or CI change.
---

# __PROJECT_SLUG__ development

Follow rules in `.cursor/rules/*.mdc`. Agent exit criteria in `AGENTS.md`.

## 1. Read

- Task scope only
- Existing code in `src/__PROJECT_MODULE__/`, `configs/`, and `tests/` for conventions
- Porting from another repo → minimum slice, not bulk copy

## 2. Red

- Failing test first (`test_*.py`, mirror `src/` paths)
- Bug → regression test before fix
- Test behavior, not trivia

## 3. Green

- Smallest typed implementation that passes
- Reuse before new abstractions; extend `ExperimentConfig` when adding YAML fields

## 4. Refactor

- Simplify while green; no unrelated cleanup

## 5. Validate

After each task, **test and analyze your own work** before hand-off:

- Re-read the diff for scope creep, style drift, and rule violations (bare `*`, docstrings, imports)
- Run `make check`; update `configs/experiment.yaml` and `ExperimentConfig` together when schema changes
- Confirm tests cover real behavior and meaningful edge cases — not only happy paths or trivia
- Cross-check experiment boundaries, tracking helpers, and config loader consistency
- **Performance review** — analyze hot paths for time/memory scaling; prefer streaming, batching, and lowest-resource correct implementations (see `performance.mdc`)
- Fix every issue found during validation; do not hand off known defects

```bash
make check
```

Config changes → update `configs/experiment.yaml` + tests · deps → `uv lock`

## 6. Hand off

Stop unless user asks: commit, push, PR, new markdown, doc expansion.

New Makefile target → `name: ## help` + `.PHONY`

## Pitfalls

`pre-commit` → prek · `pip install` → pyproject + `uv lock` · implement before test → back to step 2 · YAML and `ExperimentConfig` out of sync → update both
