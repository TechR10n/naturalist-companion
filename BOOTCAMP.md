# 2-Day Intensive Bootcamp: Agentic Wikipedia (Vertex AI → Databricks)

Goal: In **2 days**, build the same **Wikipedia-grounded RAG pipeline** twice:
1. **Day 1:** GCP + **Vertex AI** (Gemini + Vertex embeddings)
2. **Day 2:** **Databricks** (Model Serving + Vector Search + evaluation workflow)

This repo includes:
- `docs/agentic_wikipedia_gcp_spec.md` (current GCP spec)
- `docs/agentic_wikipedia_dbrx_spec.md` (Databricks roadmap)
- `docs/diagrams/README.md` (architecture diagrams)

Databricks workflow reference:
- [Databricks Generative AI developer workflow](https://docs.databricks.com/aws/en/generative-ai/tutorials/ai-cookbook/genai-developer-workflow)

---

## Success Criteria (use these both days)

**P0 (must-have)**
- A notebook runs end-to-end: **Wikipedia slice → chunk → embed → retrieve → answer with citations**.
- Outputs include **Wikipedia-only citations**.
- A small evaluation set (≈10 questions) runs with a simple scorecard.

**P1 (nice-to-have)**
- Swap local FAISS/Chroma for managed vector search.
- Add a minimal API layer for a shareable demo.
- Log runs and evaluation metrics (MLflow).

---

## Prework (30–60 min, before Day 1)

1. Pick a topic slice and lock it:
- `WIKIPEDIA_QUERY`
- `WIKIPEDIA_MAX_DOCS`
- `WIKIPEDIA_TOP_K`

2. Create a parity checklist:
- Same Wikipedia slice across both days
- Same chunking + retrieval parameters
- Same eval questions + rubric

3. Decide your execution environment:
- Vertex AI: local Jupyter or Vertex AI Workbench
- Databricks: workspace + cluster / serverless

---

## Day 1 (Vertex AI): Build + Evaluate

### 09:00–10:00 — GCP project + auth (P0)
- Enable Vertex AI API and choose a region.
- Authenticate (ADC or service account).
- Verify embeddings + LLM calls work.

**Deliverable:** you can run the sanity checks in `docs/agentic_wikipedia_gcp_spec.md`.

### 10:00–12:00 — Ingest + retrieval (P0)
- Load a small Wikipedia slice.
- Chunk and embed.
- Build a local vector store.
- Retrieve top-k chunks for a test question.

**Deliverable:** retrieval returns Wikipedia-backed chunks with metadata.

### 13:00–15:00 — Answer + citations (P0)
- Generate answers using Vertex AI.
- Enforce Wikipedia-only citations.

**Deliverable:** answers with valid citations.

### 15:00–17:00 — Eval loop (P0)
- Run a small evaluation set.
- Record accuracy + citation validity + latency.

**Deliverable:** simple scorecard + notes.

---

## Day 2 (Databricks): Port + Evolve

### 09:00–10:00 — Workspace + endpoints (P0)
- Confirm Model Serving endpoints for LLM and embeddings.
- Validate a “hello world” call.

**Deliverable:** model calls run in the Databricks environment.

### 10:00–12:00 — Parity port (P0)
- Port the workflow to a Databricks notebook.
- Use FAISS or Chroma for fast parity.

**Deliverable:** retrieval and answers match baseline behavior.

### 13:00–15:00 — Vector Search + eval (P1)
- Persist vectors in Vector Search.
- Run the same evaluation set.

**Deliverable:** scorecard and comparison to Day 1.

### 15:00–17:00 — Optional API layer (P1)
- Add a minimal API wrapper.
- Keep response schema identical to baseline outputs.

**Deliverable:** endpoint callable by another user.

---

## Quick “Gotchas” Checklist

- Keep the Wikipedia slice small at first.
- Always return citations; don’t trust the LLM without grounding.
- Log retrieved page IDs and URLs for debugging.
- Separate “answer model” and “judge model” if you can.
