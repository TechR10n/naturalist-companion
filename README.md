# Agentic Naturalist Companion (`naturalist-companion`)

Agentic Naturalist is a companion team of data + storytelling: intellectual sherpas and logisticians for field learning and route-based exploration.

Python package (import) name: `naturalist_companion`.

Project layout (code):
- `src/`: all runtime code (web app + offline MVP graph) plus `templates/` and `static/`.
- `scripts/`: runnable scenarios / smoke tests.
- `tests/`: unit + smoke tests.

## Product direction

We are building toward a real, App Store-ready product that combines:
- Route-based, citation-grounded guides
- Camera-based identification with calibrated uncertainty
- A local-first dev workflow with cloud paths for quality

Start here: `docs/PRD.md`

Wireframes for early MVP flows: `docs/wireframes.md`

## Quickstart (Flask, offline)

1) Set up Python + deps (recommended Python **3.12**):
- `./scripts/bootstrap_vertex.sh`
- `source .venv/bin/activate`
- `python -m pip install -e .`

Dependency profiles (choose one notebook/provider target):
- GCP/Vertex: `python -m pip install -r requirements-gcp-dev.txt`
- Ollama/local: `python -m pip install -r requirements-ollama-dev.txt`
- Databricks/DBRX: `python -m pip install -r requirements-dbrx-dev.txt`
- Each profile includes the same local RAG stack (`sentence-transformers` + `chromadb` + `langchain-chroma` + `langchain-huggingface`).

2) Run the web app:
- `python -m naturalist_companion --debug`
- Open `http://127.0.0.1:8000` (the page can generate a tiny offline guide via `POST /api/mvp`).
- Camera path: `POST /api/vision` accepts image uploads (multipart `images`) or base64 JSON (`images[].data_base64`).

## Offline LangGraph MVP (minimal data)

This repo includes a tiny **offline** LangGraph MVP (no API calls) to validate graph wiring and smoke-test all nodes.

- Run the MVP: `python scripts/smoke_langgraph_mvp.py`
- Run without writing files: `python scripts/smoke_langgraph_mvp.py --no-write`
- Run the smoke test: `python -m unittest discover -s tests -p 'test_*.py'`

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
