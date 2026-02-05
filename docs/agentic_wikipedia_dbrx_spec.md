# Agentic Wikipedia — Databricks Spec (Future)

## Status

**Future / roadmap.** This spec defines the Databricks migration target while keeping the current GCP notebook path functional.

## Minimum baseline (must remain)

The following must continue to run as the baseline reference:

- `notebooks/agentic_wikipedia_gcp.ipynb`

This notebook is the minimum viable path for Wikipedia-grounded Q&A and acts as the parity harness for Databricks work.

## Target architecture (Databricks)

- Workspace notebooks for development and debugging.
- Jobs / Workflows for scheduled ingestion and refresh.
- Model Serving for LLM + embeddings endpoints.
- Vector Search backed by Delta tables in Unity Catalog.
- MLflow for experiments and evaluation runs.
- Optional API layer for productization.

See diagram: `docs/diagrams/03_future_databricks_architecture.puml`.

## Phased plan

### Phase 0 — Parity harness (now)

- Keep `agentic_wikipedia_gcp.ipynb` as the canonical baseline.
- Normalize chunking, metadata, and citation rules so outputs are comparable across stacks.

### Phase 1 — Databricks notebook port

- Port the workflow to a Databricks notebook using `databricks-langchain`, `ChatDatabricks`, and `DatabricksEmbeddings`.
- Use **FAISS or Chroma** first to keep the port simple.
- Validate output parity with the baseline notebook (same questions, same slice).

### Phase 2 — Vector Search + Model Serving

- Move embeddings and LLM calls to **Model Serving**.
- Persist vectors in **Databricks Vector Search**.
- Schedule Wikipedia refresh via **Jobs / Workflows**.
- Log evaluations and retrieval metrics in **MLflow**.

### Phase 3 — Optional API layer

- Add a thin API layer for app use-cases.
- Keep the API response schema identical to the baseline outputs.

## Guardrails (non-negotiable)

- Wikipedia-only sourcing for factual claims.
- Citations must be `*.wikipedia.org/wiki/...`.
- If sources are insufficient, explicitly say so.

## Deliverables

- A Databricks notebook path that matches baseline outputs.
- A Vector Search-backed retrieval path with the same citation rules.
- MLflow runs for regression and evaluation tracking.
