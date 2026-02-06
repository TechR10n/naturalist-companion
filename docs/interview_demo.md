# Interview Demo Script (Product + Tech)

This script is a quick way to narrate the product direction while showing a working pipeline.

## 1) One-minute product pitch

"Agentic Naturalist is a pocket field guide. You can ask a question about your route or point your camera at a roadcut. The system offers a small set of likely answers, explains what it sees, and cites Wikipedia so the user can verify. It is designed to be honest about uncertainty and to ask for the next best observation."

## 2) Live pipeline demo (local)

- `python scripts/smoke_route_guide.py`
- Show output: `out/guide/guide.md` and `out/guide/guide.json`

Key points to highlight
- Retrieval happens before generation.
- All facts point back to citations.

## 3) Live pipeline demo (GCP)

- `cp .env.example .env` and set `GOOGLE_CLOUD_PROJECT`.
- `gcloud auth application-default login`
- `jupyter lab` -> open `notebooks/anc_gcp.ipynb`

Key points to highlight
- Vertex AI embeddings + Gemini.
- Local vector store for iteration speed.
- Wikipedia-only grounding.

## 4) Camera vision direction (conceptual)

- Show `docs/architecture_camera_vision.md`.
- Explain how camera hypotheses are grounded via the same RAG pipeline.

## 5) Future path (Databricks)

- Show `docs/diagrams/06_anc_databricks_production_app.puml`.
- Emphasize parity with the baseline notebook.
