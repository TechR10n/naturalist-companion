# 2-Day Intensive Bootcamp: Agentic Wikipedia (Vertex AI → Databricks)

Goal: In **2 days**, build the same “Agentic Wikipedia” proof-of-concept twice:
1) **Day 1:** GCP + **Vertex AI** (Gemini + Vertex embeddings)  
2) **Day 2:** **Databricks** (Model Serving + Agent tooling + evaluation workflow)

This repo already includes two specs to follow:
- `docs/agentic_wikipedia_gcp_spec.md` (Vertex AI)
- `docs/agentic_wikipedia_dbrx_spec.md` (Databricks)

Databricks workflow reference:
- [Databricks Generative AI developer workflow](https://docs.databricks.com/aws/en/generative-ai/tutorials/ai-cookbook/genai-developer-workflow)

---

## Success Criteria (use these both days)

**P0 (must-have)**
- A notebook runs end-to-end: **load Wikipedia docs → embed → retrieve → answer**.
- An **agentic workflow** (LangGraph) calls at least one tool (retriever/search) and returns an answer with **citations** (URLs/titles).
- A small **evaluation set** (≈10 questions) runs and produces a simple scorecard (quality + latency).

**P1 (nice-to-have)**
- Swap FAISS for a managed vector store (Vertex AI Vector Search / Databricks Vector Search).
- Add feedback capture + iterative improvement loop.
- Package into a small API (FastAPI) for “pre-prod” sharing.

---

## Prework (30–60 min, before Day 1)

1) Pick a use case and lock it:
- Define `query_terms` (what Wikipedia topics you’ll index)
- Define `example_question` and 10 eval questions
- Set `max_docs` to **10–50** initially (keep cost + iteration time low)

2) Create a parity checklist (so Day 2 is a true comparison):
- Same `query_terms`, `max_docs`, `k`, and prompt style
- Same eval question set
- Same “answer format” (citations required)

3) Decide your execution environment:
- Vertex: local Jupyter **or** Vertex AI Workbench
- Databricks: workspace + a cluster / serverless

---

## Day 1 (Vertex AI): Build + Evaluate a Working Agent

### 09:00–10:00 — GCP project + auth (P0)
- Ensure Vertex AI API is enabled and pick a region (e.g., `us-central1`).
- Decide auth approach:
  - **Vertex AI Workbench** (simplest), or
  - Local dev with `gcloud auth application-default login`
- Confirm a “hello world” Gemini call works (`ChatVertexAI`).

**Deliverable:** you can run the LLM call cell in `docs/agentic_wikipedia_gcp_spec.md`.

### 10:00–11:30 — Data load + retrieval baseline (P0)
- Run the Wikipedia loader with a tiny set (10–50 docs).
- Build embeddings with `VertexAIEmbeddings("text-embedding-004")`.
- Create FAISS index and validate `similarity_search()` returns relevant docs.

**Deliverable:** retrieval prints relevant snippets + metadata.

### 11:30–13:00 — Agentic workflow in LangGraph (P0)
Implement a minimal agent loop:
- Tool: `retrieve(query) -> top_k docs (content + metadata)`
- Agent policy: “If question needs facts, retrieve; then answer with citations”
- Output schema: `{answer, citations:[{title, source}]}` (or similar)

**Deliverable:** calling your graph on `example_question` retrieves + answers with citations.

### 14:00–15:30 — Add quality checks + an eval set (P0)
- Create an eval set of ~10 questions for your topic.
- Add a lightweight judge:
  - Simple rubric (0–2): **relevance**, **factuality**, **citation quality**
  - Optional: Gemini-as-judge (same model or a “judge” model)
- Track:
  - latency per query
  - retrieval hit rate (did it retrieve something useful?)

**Deliverable:** a table/JSON of eval results with scores + notes.

### 15:30–17:00 — “Prototype → Production” reflection (P0)
Write your reflection (required by both specs):
- What would you improve with more time?
- What would it take to productionize? (security, monitoring, evals, feedback, cost controls)

**Deliverable:** completed reflection section.

### Stretch (if time): managed vector search or API (P1)
- Replace FAISS with Vertex AI Vector Search, or
- Wrap as a small API (FastAPI on Cloud Run) for shareable pre-prod feedback.

---

## Day 2 (Databricks): Repeat + Adopt the Databricks Workflow

Day 2 mirrors Day 1, but you’ll align with the Databricks “developer workflow” concepts:
**build → trace/log → evaluate → deploy (pre-prod) → feedback → iterate → monitor**.

### 09:00–10:00 — Workspace + endpoints (P0)
- Confirm you have:
  - a cluster / serverless compute
  - a Model Serving endpoint for an LLM (`ChatDatabricks`)
  - an embeddings endpoint (`DatabricksEmbeddings`)
- Validate a “hello world” LLM call works.

**Deliverable:** the LLM call cell runs in the Databricks environment.

### 10:00–12:00 — Data + retrieval baseline (P0)
Start with parity:
- Same `query_terms`, `max_docs`, `k` as Day 1
- Load Wikipedia docs
- Build embeddings + FAISS index (fastest path)

**Optional (preferred Databricks-style, P1):**
- Store docs in Delta + create a Vector Search index
- Use `VectorSearchRetrieverTool` instead of FAISS directly

**Deliverable:** `similarity_search()` (or Vector Search retriever) works and returns metadata.

### 13:00–14:30 — Agentic workflow (LangGraph) (P0)
- Recreate the same agentic policy from Day 1
- Keep output schema identical to compare

**Deliverable:** graph answers with citations, same as Day 1.

### 14:30–16:00 — Trace/log + evaluation loop (P0)
Align to the Databricks workflow:
- Add tracing/logging for each run (inputs, retrieved docs, output, latency)
- Run your same ~10-question eval set and produce a scorecard
- Capture at least one improvement iteration (prompt/tool behavior) and re-run eval

**Deliverable:** “before vs after” eval summary (even if small).

### 16:00–17:30 — Pre-prod deploy + feedback hook (P1)
- Deploy to a shareable endpoint (Model Serving / agent endpoint)
- Add a minimal feedback payload (thumbs up/down + comment + question)
- Store feedback (Delta table or a simple log)

**Deliverable:** someone else can call the endpoint and leave feedback.

### Wrap-up (30 min): compare stacks + next steps (P0)
Create a short comparison:
- Dev ergonomics
- Eval/monitoring maturity
- Deployment + governance
- Cost/performance notes

---

## Quick “Gotchas” Checklist

- Keep `max_docs` small while iterating; scale only after the loop is correct.
- Always return citations; don’t trust the LLM without retrieval grounding.
- Separate “answer model” and “judge model” if you can (reduces bias).
- Log retrieved doc IDs/URLs for debugging hallucinations.
- Add timeouts/retries for API calls (especially in agent loops).

