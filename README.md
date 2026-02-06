# Agentic Naturalist Companion (`naturalist-companion`)

Agentic Naturalist is a companion team of data + storytelling: intellectual sherpas and logisticians for field learning and route-based exploration.

Python package (import) name: `naturalist_companion`.

Project layout (code):
- `src/naturalist_companion/`: all runtime code (web app + offline route-guide graph) plus `templates/` and `static/`.
- `scripts/`: runnable scenarios / smoke tests.
- `tests/`: unit + smoke tests.

## Product direction

We are building toward a real, App Store-ready product that combines:
- Route-based, citation-grounded guides
- Camera-based identification with calibrated uncertainty
- A local-first dev workflow with cloud paths for quality

Start here: `docs/PRD.md`

Wireframes for early baseline flows: `docs/wireframes.md`

## Quickstart (recommended: uv + Makefile)

1) Set up the local environment (recommended Python **3.12**):
- `make setup`
- This creates/syncs `.venv` via `uv` with base + dev dependencies.

Provider profiles:
- Vertex/GCP: `make setup-gcp`
- Ollama/local: `make setup-ollama`
- Databricks/DBRX: `make setup-dbrx`

2) Run checks:
- `make test`
- `make smoke`
- `make check` (runs both)

PyCharm shortcut:
- Use shared run configurations:
  - `Task: setup`
  - `Task: test`
  - `Task: smoke`
  - `Task: clean`
- These call `scripts/dev_tasks.py`, which delegates to the same `make` targets used in CLI/CI.

3) Run the web app:
- `make web`
- Open `http://127.0.0.1:8000`
- Route guide API: `POST /api/guide`
- Camera path: `POST /api/vision` accepts multipart `images` or JSON `images[].data_base64`.

Optional: refresh lockfile:
- `make lock`

Legacy bootstrap path is still available:
- `./scripts/bootstrap_vertex.sh`

## Offline Route Guide Graph (minimal data)

This repo includes a tiny **offline** route-guide graph (no API calls) to validate graph wiring and smoke-test all nodes.

- Run the route guide smoke: `uv run python scripts/smoke_route_guide.py`
- Run without writing files: `make smoke`
- Run unit tests: `make test`

## StateGraph production path (local)

This path keeps StateGraph orchestration fixed while moving retrieval to persistent, partitioned stores.

1) Build/update persistent retrieval partitions:
- Deterministic local data: `make stategraph-refresh`
- Live Wikipedia data: `make stategraph-refresh-real`

2) Run eval harness against persistent indexes:
- `make stategraph-eval`
- `make stategraph-cache-observe` (shows response/retrieval cache events across repeated runs)

3) Run real-data release gate (promotion blocker):
- `make stategraph-gate`
- Gate fails if pass-rate/citation/quality thresholds are not met.
- If top-k is raised above cap, gate requires net-gain vs strict-cap baseline.
- Promotion command: `make stategraph-promote` (writes promotion record only when gate passes).

4) Optional macOS scheduler (launchd):
- `make stategraph-launchd-install`
- Add `--load` if you want immediate `launchctl bootstrap`.

Artifacts to inspect:
- Refresh reports: `out/stategraph_store/refresh_reports/`
- Eval reports: `out/stategraph_eval/`
- Release gate report: `release_gate.json` under the gate run artifact root.

## ANC notebooks (optional)

If you want the ANC notebook baselines:

Notebook policy:
- Keep notebook grounding and factual citations Wikipedia-only.
- For notebook picture experiments, use a small number of uploaded images sourced from Wikipedia/Wikimedia files.
- Use Flask/app for non-Wikipedia, multimodal, and broader provider experiments.
- After notebook baselines are complete, expand to other public data sources (for example USGS) in app/flask paths.

1) Configure:
- `cp .env.example .env` and fill in `GOOGLE_CLOUD_PROJECT` (and optionally `GOOGLE_CLOUD_LOCATION`)

2) Auth once:
- `gcloud auth application-default login`

3) Run:
- `jupyter lab` → open `notebooks/anc_gcp.ipynb` (or `notebooks/anc_dbrx.ipynb`, `notebooks/anc_ollama.ipynb`)

Optional notebook smoke execution:
- `.venv/bin/jupyter nbconvert --to notebook --execute notebooks/anc_ollama.ipynb --output anc_ollama.executed.ipynb`

## Docs (current)

- `docs/README.md`
- `docs/PRD.md`
- `docs/wireframes.md`
- `docs/anc_spec_gcp.md`
- `docs/anc_spec_dbrx.md`
- `docs/anc_spec.ollama.md`
- `docs/architecture_camera_vision.md`
- `docs/diagrams/README.md`
