# Agentic Naturalist Companion (`naturalist-companion`)

Agentic Naturalist is a companion team of data + storytelling: intellectual sherpas and logisticians for field learning and route-based exploration.

Short Python package name: `anc` (Agentic Naturalist Companion).

Project layout (code):
- `src/anc/`: main package (web app + CLI entrypoints).
- `src/anc/agentic_wikipedia/`: offline MVP graph + toy data utilities.
- `scripts/`: runnable scenarios / smoke tests.
- `tests/`: unit + smoke tests.

## Quickstart (Flask, offline)

1) Set up Python + deps (recommended Python **3.12**):
- `./scripts/bootstrap_vertex.sh`
- `source .venv/bin/activate`
- `python -m pip install -e .`

2) Run the web app:
- `python -m anc --debug`
- Open `http://127.0.0.1:8000` (the page can generate a tiny offline guide via `POST /api/mvp`).

## Offline LangGraph MVP (minimal data)

This repo includes a tiny **offline** LangGraph MVP (no API calls) to validate graph wiring and smoke-test all nodes.

- Run the MVP: `python scripts/smoke_langgraph_mvp.py`
- Run without writing files: `python scripts/smoke_langgraph_mvp.py --no-write`
- Run the smoke test: `python -m unittest discover -s tests -p 'test_*.py'`

## Vertex AI notebook (optional)

If you want the original “Agentic Wikipedia” Vertex AI wiring (Wikipedia slice → embeddings → local vector store → grounded Q&A):

1) Configure:
- `cp .env.example .env` and fill in `GOOGLE_CLOUD_PROJECT` (and optionally `GOOGLE_CLOUD_LOCATION`)

2) Auth once:
- `gcloud auth application-default login`

3) Run:
- `jupyter lab` → open `notebooks/agentic_wikipedia_gcp.ipynb`

## Docs (current)

- `docs/README.md`
- `docs/agentic_wikipedia_gcp_spec.md`
- `docs/agentic_wikipedia_dbrx_spec.md`
- `docs/diagrams/README.md`
