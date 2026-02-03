# 2-Day Intensive Bootcamp: Roadside Geology (Vertex AI → Databricks)

Goal: In **2 days**, build the same “Roadside Geology”-style (Wikipedia-only) proof-of-concept twice:
1) **Day 1:** GCP + **Vertex AI** (Gemini + Vertex embeddings)  
2) **Day 2:** **Databricks** (Model Serving + Agent tooling + evaluation workflow)

This repo includes:
- `docs/roadside_geology_v0_mvp_spec.md` (the v0 product spec to implement)
- `docs/agentic_wikipedia_gcp_spec.md` (Vertex AI wiring template)
- `docs/agentic_wikipedia_dbrx_spec.md` (Databricks wiring template)

Databricks workflow reference:
- [Databricks Generative AI developer workflow](https://docs.databricks.com/aws/en/generative-ai/tutorials/ai-cookbook/genai-developer-workflow)

---

## Success Criteria (use these both days)

**P0 (must-have)**
- A notebook runs end-to-end: **route → Wikipedia GeoSearch → retrieve → generate stop cards with citations**.
- An **agentic workflow** (LangGraph) uses tools (`geosearch`, `retrieve`, `write_stop_card`) and enforces **Wikipedia-only citations**.
- Outputs: `guide.json` + `guide.md` with **5–12 stops** for a short test route (20–100 km).
- A small **evaluation set** (≈10 questions) runs and produces a simple scorecard (quality + latency + citation validity).

**P1 (nice-to-have)**
- Swap FAISS for managed vector search (Vertex AI Vector Search / Databricks Vector Search).
- Add a “Drive Mode” output (short summaries / optional audio script) with offline-friendly caching.
- Add feedback capture (thumbs up/down + comment) and an improvement loop.
- Package into a small API for pre-prod sharing.

---

## Prework (30–60 min, before Day 1)

1) Pick a test route and lock it (for parity across days):
- Short route (20–100 km) with known scenery/formation interest
- v0 input format: **GPX** or a pasted list of `{lat, lon}` points

2) Lock your config (from `docs/roadside_geology_v0_mvp_spec.md`):
- `sample_every_m`, `geosearch_radius_m`, `geosearch_limit`
- `max_stops`, `min_stop_spacing_m`, `language`

3) Create a parity checklist (so Day 2 is a true comparison):
- Same route polyline
- Same sampling + GeoSearch params
- Same stop-card JSON schema + citation rules
- Same eval questions + judge rubric

4) Decide your execution environment:
- Vertex: local Jupyter **or** Vertex AI Workbench
- Databricks: workspace + a cluster / serverless

---

## Day 1 (Vertex AI): Build + Evaluate the Roadside Geology Agent

### 09:00–10:00 — GCP project + auth (P0)
- Ensure Vertex AI API is enabled and pick a region (e.g., `us-central1`).
- Decide auth approach:
  - **Vertex AI Workbench** (simplest), or
  - Local dev with `gcloud auth application-default login`
- Confirm a “hello world” Gemini call + embedding call works.

**Deliverable:** you can run the LLM + embedding sanity checks in `docs/agentic_wikipedia_gcp_spec.md`.

### 10:00–11:30 — Route ingest + Wikipedia GeoSearch baseline (P0)
- Ingest your test route (GPX/JSON) and compute cumulative distance.
- Sample points (`sample_every_m`) and call Wikipedia GeoSearch around each sample point.
- Fetch page summaries/extracts for top candidates and apply a basic geology filter.

**Deliverable:** you can print candidate geology pages per sampled point (with Wikipedia URLs).

### 11:30–13:00 — Retrieval + stop selection (P0)
- Chunk Wikipedia extracts, embed with `VertexAIEmbeddings("text-embedding-004")`, and build a FAISS index.
- Implement stop selection (dedupe + spacing + max stops).
- Validate you can produce a deterministic shortlist of stop “seeds” (each with pageid/title/url + route_km).

**Deliverable:** a stable shortlist of 5–12 stops for your route.

### 14:00–15:30 — Agentic workflow in LangGraph (P0)
- Create tools: `geosearch`, `fetch_page`, `retrieve`, `write_stop_card`, `validate_stop_card`.
- Implement LangGraph: route → candidates → stops → stop cards → validate → render outputs.

**Deliverable:** `guide.json` + `guide.md` generated; all citations are Wikipedia URLs.

### 15:30–17:00 — Eval + “Prototype → Production” reflection (P0)
- Create an eval set of ~10 questions about your route (e.g., “Which stop best shows volcanic features?”).
- Add a lightweight judge + validators:
  - Rubric (0–2): **usefulness**, **Wikipedia-grounded factuality**, **citation quality**
  - Hard checks: citations exist + `wikipedia.org` only + schema validity
- Do one improvement iteration (prompt/tool behavior) and re-run eval.

**Deliverable:** eval scorecard + notes (what you’d change next).

Write a short reflection:
- What would you improve with more time?
- What would it take to productionize? (security, monitoring, evals, feedback, cost controls)

---

## Day 2 (Databricks): Repeat + Adopt the Databricks Workflow

Day 2 mirrors Day 1, but you’ll align with the Databricks “developer workflow” concepts:
**build → trace/log → evaluate → deploy (pre-prod) → feedback → iterate → monitor**.

### 09:00–10:00 — Workspace + endpoints (P0)
- Confirm you have:
  - a cluster / serverless compute
  - a Model Serving endpoint for an LLM (`ChatDatabricks`)
  - an embeddings endpoint (`DatabricksEmbeddings`)
- Validate a “hello world” LLM + embedding call works.

**Deliverable:** the LLM call cell runs in the Databricks environment.

### 10:00–12:00 — Route + retrieval parity (P0)
Start with parity:
- Same route polyline + same sampling + GeoSearch params as Day 1
- Same Wikipedia fetch + chunking logic
- Same embeddings + FAISS index (fastest parity path)

**Optional (preferred Databricks-style, P1):**
- Store fetched Wikipedia extracts/chunks in Delta + create a Vector Search index
- Use `VectorSearchRetrieverTool` as the retriever tool

**Deliverable:** retrieval returns Wikipedia-backed chunks with metadata (title/url/pageid).

### 13:00–14:30 — Agentic workflow (LangGraph) (P0)
- Recreate the same graph (route → stops → stop cards → validate → render)
- Keep output schema identical to Day 1 for comparison

**Deliverable:** `guide.json` + `guide.md` for the same route, comparable to Day 1 output.

### 14:30–16:00 — Trace/log + evaluation loop (P0)
Align to the Databricks workflow:
- Add tracing/logging for each run (inputs, retrieved docs, output, latency)
- Run the same eval set and produce a scorecard
- Do one improvement iteration and re-run eval

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

- Keep your test route short while iterating; scale route length only after the loop is correct.
- Always return citations; don’t trust the LLM without Wikipedia grounding.
- Separate “answer model” and “judge model” if you can (reduces bias).
- Log retrieved page IDs/URLs + retrieved chunks to debug hallucinations.
- Add timeouts/retries for API calls (especially in agent loops).
- Don’t encourage collecting samples; keep guidance to observation/photography and respect local rules.
