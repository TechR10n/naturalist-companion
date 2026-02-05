# ANC Spec - GCP Baseline

## Status

Current proof-of-concept baseline. The canonical implementation details now live in `/Users/ryan/Developer/naturalist-companion/notebooks/anc_gcp.ipynb`.

## Scope

- Keep notebook behavior minimal and reproducible.
- Demonstrate LangGraph orchestration with simple deterministic outputs.
- Preserve citation-first intent (Wikipedia-only for factual claims in full mode).
- Allow a small number of uploaded pictures in notebook runs when files are sourced from Wikipedia/Wikimedia pages.

## Architecture Intent

- Data source: Wikipedia content + metadata (`title`, `summary`, `source`)
- Orchestration: LangGraph
- Cloud target: Vertex AI embeddings + generation (represented as stubs in the minimal notebook)
- Output: concise answer + citation references

## Run/Config Notes

Typical environment variables for full GCP mode:
- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION`
- `VERTEX_LLM_MODEL`
- `VERTEX_EMBEDDING_MODEL`
- `WIKIPEDIA_QUERY`
- `WIKIPEDIA_MAX_DOCS`
- `WIKIPEDIA_TOP_K`

## Development Policy

1. Validate notebook baseline in PyCharm and online notebook environments.
2. Stop adding major features in notebooks after baseline proof.
3. Continue product functionality in the localhost Flask app.
4. After all notebook baselines are complete, expand to additional public sources (for example USGS) in app/flask paths.
5. Start iOS planning after Flask behavior is stable.
