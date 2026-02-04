# Interview demo script (Agentic Wikipedia)

This is a lightweight talk track you can use while screen-sharing the repo. It’s organized around `docs/diagrams/` so you can “zoom out” to architecture, then “zoom in” to the notebook.

## Pre-demo checklist (5 minutes)

- `cp .env.example .env` and set `GOOGLE_CLOUD_PROJECT` (and optionally `GOOGLE_CLOUD_LOCATION`).
- `./scripts/bootstrap_vertex.sh` → `source .venv/bin/activate`
- `python scripts/smoke_vertex_ai.py --api` (verifies auth + model access; small billable calls)
- Open the notebook: `jupyter lab` → `notebooks/agentic_wikipedia_gcp.ipynb`
- Keep it fast: start with `WIKIPEDIA_MAX_DOCS=5` and a tight `WIKIPEDIA_QUERY`.

## 5–7 minute talk track (recommended order)

### 1) What problem this solves (30s)

- LLMs are great at synthesis, but they need **grounding** to avoid “confident nonsense”.
- This repo is a small, narrow-scope **Wikipedia-grounded RAG** POC that’s easy to iterate on.

Open: `docs/diagrams/rendered/01_system_context.svg`

### 2) The system context (60s)

- Local notebook orchestrates the workflow.
- External dependencies are explicit: Wikipedia for content; Vertex AI for embeddings + LLM.
- Vector store is local (Chroma by default; FAISS optional) to keep iteration tight.

Open: `docs/diagrams/rendered/02_rag_components.svg`

### 3) The RAG pipeline (60–90s)

- Ingestion: Wikipedia → chunk → embed → vector store.
- Runtime: question → embed → retrieve top-k chunks → LLM prompt with strict citation rules.
- Key point: **metadata travels with chunks** so citations are automatic and testable.

Open: `docs/diagrams/rendered/03_question_answer_sequence.svg`

### 4) Demo the notebook (2–3 min)

- Show the query seed (example: `docs/use_case_appalachia_i81.md`).
- Run the ingest/index cells once, then ask 2–3 questions.
- As you answer:
  - show retrieved chunks’ `metadata` (title + source URL)
  - emphasize “answer must be supported by retrieved text”

### 5) “Agentic” next step: a tool-using workflow (60–90s)

If asked “where does the agentic part come in?”, open:

- `docs/diagrams/rendered/04_roadside_geology_langgraph.svg`

Talking points:
- LangGraph is the control plane: tools + retries + validators.
- Validators enforce “Wikipedia-only citations” and schema correctness (so output is testable).

### 6) Contracts make it shippable (60s)

Open:
- `docs/diagrams/rendered/05_stopcard_contract.svg` (Roadside Geology output)
- `docs/diagrams/rendered/06_vision_contract.svg` (camera classification output)

Talking points:
- Structured outputs enable automated checks, evals, and regression testing.
- Narration is optional UI sugar; contracts are the product.

### 7) Future direction: camera → grounded explanation (30–60s)

Open: `docs/diagrams/rendered/07_camera_vision_architecture.svg`

Talking points:
- Multimodal model proposes hypotheses/features; retrieval provides grounding + citations.
- Privacy defaults: don’t store photos unless opt-in (see `docs/architecture_camera_vision.md`).

## Common interviewer questions (quick answers)

- “How do you reduce hallucinations?” → retrieval + strict citation rules + validators + small scope.
- “How do you evaluate quality?” → citation coverage + schema validity + small human rubric (see `docs/roadside_geology_v0_mvp_spec.md`).
- “How would you productionize?” → API wrapper, caching, observability, eval loop, cost controls, and a feedback capture mechanism.

---

## Optional variant: investor-style “AI Naturalist” framing (2 minutes)

If you want to demo this as an iOS “Naturalist Flâneur” concept (vegetation + ecology + geology), keep the same talk track but switch the Wikipedia slice and questions.

### Quick setup

- Set a naturalist slice:
  - `export WIKIPEDIA_QUERY='Shenandoah National Park OR Appalachian Mountains OR Riparian zone OR Wetland OR Deciduous forest OR Karst OR Shale'`
  - `export WIKIPEDIA_MAX_DOCS=10`
  - `export WIKIPEDIA_TOP_K=3`
- Re-run the notebook ingest/index cells (only once per slice).

### Demo questions that map to the product vision

- “What is a riparian zone and why does it matter for vegetation and wildlife?”
- “What is karst, and how can it shape local hydrology and habitats?”
- “What are typical characteristics of temperate deciduous forests in the Appalachians?”
- “Given this corridor, what are 3 ‘look for this’ prompts you would give a beginner naturalist?”

For the investor framing + demo story, see `docs/investor_demo_naturalist_flaneur.md`.
