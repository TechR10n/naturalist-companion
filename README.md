# Agentic Naturalist Companion (`naturalist-companion`)

Agentic Naturalist is a companion team of data + storytelling: intellectual sherpas and logisticians for field learning and route-based exploration.

Short Python package name: `anc` (Agentic Naturalist Companion).

## Quickstart (Flask, offline)

1) Set up Python + deps (recommended Python **3.12**):
- `./scripts/bootstrap_vertex.sh`
- `source .venv/bin/activate`

2) Run the web app:
- `python -m anc --debug`
- Open `http://127.0.0.1:8000` (the page can generate a tiny offline guide via `POST /api/mvp`).

## Narrow starter use-case

Start with **one corridor** and a small question set, so you can tell (quickly) whether retrieval is putting the right pages in front of the model.

- Recommended starter: `docs/use_case_appalachia_i81.md`

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

## Camera + geology (future direction)

If you want to turn this into a “roadside guide you can point at a roadcut,” here are some architecture notes for an iPhone camera → Vertex AI multimodal workflow:

- Architecture + diagrams: `docs/architecture_camera_vision.md`
- PlantUML diagrams (demo-ready): `docs/diagrams/README.md`
- Prompt + output contract: `docs/vision_prompt_schema.md`
- Product/use-case sketch: `docs/use_case_camera_roadcuts.md`

## Naturalist flâneur (future direction)

If you want to treat this as an investor-style demo for an iOS “AI Naturalist” (vegetation + ecology + geology), start here:

- Investor demo framing + script: `docs/investor_demo_naturalist_flaneur.md`
- Use-case sketch: `docs/use_case_naturalist_flaneur.md`
- v0 (Wikipedia-only) MVP spec: `docs/roadside_naturalist_v0_mvp_spec.md`
- Vision prompt + output contract: `docs/vision_prompt_schema_naturalist.md`

## More details

- Full local setup: `docs/setup_vertex_ai.md`
- Interview demo script: `docs/interview_demo.md`
