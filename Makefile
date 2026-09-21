# SentinelVision AI — common tasks
.DEFAULT_GOAL := help
PY := ./.venv/bin/python
SHELL := /bin/bash

.PHONY: help setup backend frontend build test test-fast lint typecheck fmt \
        mojo models diagnostics diagnostics-json benchmark clean reset-db

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Full development setup (add WITH_MOJO=1 for the Mojo kernels)
	@./scripts/setup.sh $(if $(WITH_MOJO),--with-mojo,)

backend: ## Run the API with reload
	$(PY) -m uvicorn backend.main:app --reload --host 0.0.0.0 --port $${APP_PORT:-8008}

frontend: ## Run the Vite dev server
	cd frontend && npm run dev

build: ## Production build of the frontend
	cd frontend && npm run build

test: ## Run the full backend test suite
	$(PY) -m pytest

test-fast: ## Skip the slower pipeline tests
	$(PY) -m pytest -k "not TestPipeline and not TestFilePlayback"

lint: ## Lint backend and tests
	$(PY) -m ruff check backend/ tests/ scripts/

fmt: ## Auto-fix lint findings
	$(PY) -m ruff check --fix backend/ tests/ scripts/

typecheck: ## Type-check backend and frontend
	$(PY) -m mypy backend/ || true
	cd frontend && npx tsc --noEmit

mojo: ## Build the Mojo acceleration kernels
	@./backend/inference/mojo/build.sh

models: ## Download and export the person + PPE detection models
	$(PY) scripts/fetch_models.py --ppe

hardhat-metrics: ## Measure the hard-hat validator (cap false-positive rate)
	$(PY) scripts/hardhat_metrics.py --sweep

diagnostics: ## Print what is actually running (models, backends, PPE, Mojo)
	@$(PY) scripts/diagnostics.py

diagnostics-json: ## Same, as raw JSON
	@$(PY) scripts/diagnostics.py --json

benchmark: ## Measure per-stage latency, FPS and memory on real frames
	@$(PY) scripts/benchmark.py

reset-db: ## Delete the development database (destructive)
	@read -p "Delete data/surveillance.db and all recorded events? [y/N] " ok; \
	 [[ $$ok == y ]] && rm -f data/surveillance.db* && echo "database removed" || echo "cancelled"

clean: ## Remove caches and build output
	find . -type d -name __pycache__ -not -path "./.venv*" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache tests/_artifacts frontend/dist
