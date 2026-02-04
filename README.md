# Agentic Wikipedia (Vertex AI POC)

Think of this repo as a small roadside exhibit: we pull a handful of Wikipedia pages on a narrow topic, press them into embeddings on Vertex AI, stash them in a local vector store, and ask questions that can be traced back to the pages we picked up along the way.

## Quickstart (macOS, local Jupyter)

### Mile 0: Pack the toolkit

1) Set up Python + deps (recommended Python **3.12**):
- `cp .env.example .env` and fill in `GOOGLE_CLOUD_PROJECT` (and optionally `GOOGLE_CLOUD_LOCATION`)
- `./scripts/bootstrap_vertex.sh`
  - Optional (FAISS): `python -m pip install -r requirements-faiss.txt`

2) Get your GCP “day pass” (one-time):
- `gcloud auth application-default login`

3) Roll out:
- `source .venv/bin/activate`
- `jupyter lab` → open `notebooks/agentic_wikipedia_gcp.ipynb`

## Narrow starter use-case

Start with **one corridor** and a small question set, so you can tell (quickly) whether retrieval is putting the right pages in front of the model.

- Recommended starter: `docs/use_case_appalachia_i81.md`

## Camera + geology (future direction)

If you want to turn this into a “roadside guide you can point at a roadcut,” here are some architecture notes for an iPhone camera → Vertex AI multimodal workflow:

- Architecture + diagrams: `docs/architecture_camera_vision.md`
- Prompt + output contract: `docs/vision_prompt_schema.md`
- Product/use-case sketch: `docs/use_case_camera_roadcuts.md`

## More details

- Full local setup: `docs/setup_vertex_ai.md`
