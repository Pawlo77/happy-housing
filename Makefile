ifneq ("$(wildcard .env)","")
	include .env
	export
endif

export PYTHONPATH=.
export OMP_NUM_THREADS=1

.DEFAULT_GOAL := help

.PHONY: help install update clean prek prek-all check tests coverage mlflow mlflow-stop datasets otodom pipeline app

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

datasets: ## Download GTFS, GIOŚ, RCN, GUNB into data/
	uv run python -c "from pathlib import Path; from happy_housing.config import load_experiment_config; from happy_housing.ingest import download_gtfs_stops, fetch_gios_stations, download_file; from happy_housing.ingest.rcn import extract_rcn_transactions; from happy_housing.ingest.gunb import download_gunb_mazowieckie; c=load_experiment_config(Path('configs/experiment.yaml')); r=c.data.raw_path(); p=c.data.processed_path(); f=c.data.fixtures_path(); r.mkdir(parents=True, exist_ok=True); p.mkdir(parents=True, exist_ok=True); download_gtfs_stops(c.ingest_urls.gtfs_zip, r, p/'gtfs_stops.parquet'); fetch_gios_stations(c.ingest_urls.gios_stations, p/'gios_stations.parquet'); z=download_file(c.ingest_urls.rcn_zip, r/'rcn_warsaw.zip', timeout_s=1800); extract_rcn_transactions(z, p/'rcn_transactions.parquet', max_rows=40000); download_gunb_mazowieckie(r, p/'gunb_mazowieckie.parquet', fixture_path=f/'gunb_points.json')"

otodom: ## Scrape Otodom Warsaw flats (rate-limited)
	uv run python -c "from pathlib import Path; from happy_housing.ingest.otodom import scrape_otodom_warsaw; scrape_otodom_warsaw(Path('data/processed/otodom_listings.parquet'), max_pages=50, max_listings=1200)"

pipeline: ## Full ingest → GIS → score → MLflow train
	uv run python -m happy_housing.pipeline

app: ## Launch Streamlit GUI (requires pipeline outputs)
	uv run streamlit run src/happy_housing/app/streamlit_app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true

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
