ifneq ("$(wildcard .env)","")
	include .env
	export
endif

export PYTHONPATH=.
export OMP_NUM_THREADS=1

.DEFAULT_GOAL := help

.PHONY: help install update clean prek prek-all check tests coverage mlflow mlflow-stop datasets

MLFLOW_HOST ?= 127.0.0.1
MLFLOW_PORT ?= 5005
MLFLOW_WORKERS ?= 1

help: ## Show available targets
	@printf "Available targets:\n"
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install: ## Set up the development environment
	uv python install 3.12
	uv sync --all-groups
	uv run prek install

update: ## Update dependencies to their latest versions
	uv lock --upgrade
	uv sync --all-groups

clean: ## Remove virtual environment and cache files
	rm -rf .venv/
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d \( -name ".cache" -o -name "*__pycache__" -o -name ".*_cache" \) -exec rm -rf {} + 2>/dev/null || true

prek: ## Run git hook checks on changed files
	uv run prek run --fail-fast

prek-all: ## Run git hook checks on all files
	uv run prek run --all-files

check: ## Run tests and all quality hooks
	$(MAKE) tests
	$(MAKE) prek-all

tests: ## Run the test suite
	uv run pytest tests/

coverage: ## Run tests with coverage report
	uv run pytest --cov=./src/happy_housing --cov-report=term-missing tests/

datasets: ## Download or build datasets (project-specific)
	@echo "Not implemented — add dataset targets for your experiment."

mlflow: ## Launch MLflow UI for local runs
	@PORT="$(MLFLOW_PORT)"; \
	if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$$PORT" -sTCP:LISTEN >/dev/null 2>&1; then \
		echo "Port $$PORT is already in use. Run 'make mlflow-stop' or use another port: make mlflow MLFLOW_PORT=5001"; \
		lsof -nP -iTCP:"$$PORT" -sTCP:LISTEN; \
		exit 1; \
	fi; \
	uv run mlflow ui --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns --host "$(MLFLOW_HOST)" --port "$$PORT" --workers "$(MLFLOW_WORKERS)"

mlflow-stop: ## Stop local MLflow UI processes
	@PORT="$(MLFLOW_PORT)"; \
	PIDS="$$( ( \
		pgrep -f 'mlflow.server.fastapi_app' || true; \
		pgrep -f 'python -m mlflow' || true; \
		pgrep -f 'mlflow ui' || true; \
		pgrep -f 'mlflow server' || true \
	) | sort -u )"; \
	if [ -z "$$PIDS" ] && command -v lsof >/dev/null 2>&1; then \
		PORT_PIDS="$$(lsof -tiTCP:"$$PORT" -sTCP:LISTEN 2>/dev/null || true)"; \
		if [ -n "$$PORT_PIDS" ]; then \
			for PID in $$PORT_PIDS; do \
				ARGS="$$(ps -p $$PID -o args= 2>/dev/null || true)"; \
				case "$$ARGS" in \
					*mlflow*|*fastapi_app:app*) PIDS="$$PIDS $$PID" ;; \
				esac; \
			done; \
			PIDS="$$(printf '%s\n' $$PIDS | awk 'NF' | sort -u | tr '\n' ' ')"; \
		fi; \
	fi; \
	if [ -n "$$PIDS" ]; then \
		echo "Stopping MLflow UI processes: $$PIDS"; \
		kill $$PIDS; \
	else \
		echo "No MLflow UI process found."; \
		if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$$PORT" -sTCP:LISTEN >/dev/null 2>&1; then \
			echo "Note: port $$PORT is occupied by a non-MLflow process:"; \
			lsof -nP -iTCP:"$$PORT" -sTCP:LISTEN; \
		fi; \
	fi
