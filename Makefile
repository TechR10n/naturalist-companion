SHELL := /bin/bash

UV ?= uv
PYTHON_VERSION ?= 3.12

.PHONY: help setup setup-base setup-dev setup-gcp setup-ollama setup-dbrx lock test smoke smoke-rag smoke-vertex check web clean

help:
	@echo "Targets:"
	@echo "  make setup        # uv sync for base + dev extras"
	@echo "  make setup-base   # uv sync base dependencies only"
	@echo "  make setup-gcp    # uv sync base + dev + gcp extras"
	@echo "  make setup-ollama # uv sync base + dev + ollama extras"
	@echo "  make setup-dbrx   # uv sync base + dev + dbrx extras"
	@echo "  make lock         # refresh uv.lock for Python $(PYTHON_VERSION)"
	@echo "  make test         # run unittest suite"
	@echo "  make smoke        # run offline route-guide smoke check"
	@echo "  make smoke-rag    # run local RAG smoke check (fallback data)"
	@echo "  make smoke-vertex # run Vertex import smoke check"
	@echo "  make check        # run test + smoke"
	@echo "  make web          # run Flask app in debug mode"
	@echo "  make clean        # remove local outputs/cache artifacts"

setup: setup-dev

setup-base:
	$(UV) sync --python $(PYTHON_VERSION)

setup-dev:
	$(UV) sync --python $(PYTHON_VERSION) --extra dev

setup-gcp:
	$(UV) sync --python $(PYTHON_VERSION) --extra dev --extra gcp

setup-ollama:
	$(UV) sync --python $(PYTHON_VERSION) --extra dev --extra ollama

setup-dbrx:
	$(UV) sync --python $(PYTHON_VERSION) --extra dev --extra dbrx

lock:
	$(UV) lock --python $(PYTHON_VERSION)

test:
	$(UV) run python -m unittest discover -s tests -p 'test_*.py'

smoke:
	$(UV) run python scripts/smoke_route_guide.py --no-write

smoke-rag:
	$(UV) run --extra ollama python scripts/smoke_local_rag.py --fallback-data

smoke-vertex:
	$(UV) run --extra gcp python scripts/smoke_vertex_ai.py

check: test smoke

web:
	$(UV) run python -m naturalist_companion --debug

clean:
	rm -rf out/guide
	rm -rf out/stategraph out/stategraph_eval out/stategraph_release_gate out/stategraph_store
	find . -path './.venv' -prune -o -name '__pycache__' -type d -exec rm -rf {} +
	find . -path './.venv' -prune -o -name '.pytest_cache' -type d -exec rm -rf {} +
	find . -path './.venv' -prune -o -name '.mypy_cache' -type d -exec rm -rf {} +
	find . -path './.venv' -prune -o -name '.ruff_cache' -type d -exec rm -rf {} +
	find . -path './.venv' -prune -o -name '.DS_Store' -type f -delete
