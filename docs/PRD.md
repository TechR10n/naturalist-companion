# Agentic Naturalist Companion - PRD

## Summary

Agentic Naturalist Companion is a pocket field guide for route-based and camera-based exploration. It grounds answers in citations, keeps uncertainty explicit, and helps people observe what they are seeing without overpromising.

## Vision

Create a trustworthy naturalist companion that feels like a field expert in your pocket: it can identify likely candidates, explain what to look for next, and provide grounded context from a constrained knowledge base.

## Target platform

- iOS first, with an App Store launch target
- Mac and web are optional future surfaces for development and evaluation

## Product principles

- Grounded over clever: citations first, speculation last.
- Calibrated confidence: always present uncertainty and ask for the next best observation.
- Privacy by default: keep photos local unless users opt in.
- Fast iteration: local-first developer workflow with cloud paths when quality requires it.

## Personas

- Curious traveler: wants a quick, accurate explanation of a roadside feature.
- Amateur naturalist: wants deeper context, citations, and follow-up prompts.
- Educator/guide: wants a structured, citation-friendly summary for a route.

## Primary use cases

- Route-based guide: "What should I look for on this drive?"
- Camera-based identification: "What am I looking at?"
- Naturalist flaneur: "Help me explore what is around me." 

See detailed scenarios:
- `docs/use_case_appalachia_i81.md`
- `docs/use_case_camera_roadcuts.md`
- `docs/use_case_naturalist_flaneur.md`

Wireframes (low fidelity):
- `docs/wireframes.md`

## MVP definition (next milestone)

The next milestone is **basic MVP functionality**: a working route-based guide pipeline, plus a camera results contract that can be exercised with stubbed data (UI wiring later).

### MVP must-haves

- Route guide generation works end-to-end using the existing offline MVP graph (toy data) and produces:
  - `guide.json` stop cards with Wikipedia citations
  - a human-readable `guide.md` equivalent
- A thin API surface exists for development:
  - `POST /api/mvp` returns `guide` and a trace
- A minimal evaluation loop exists:
  - schema validation and citation validity checks

### MVP nice-to-haves

- Route input supports GPX and/or pasted JSON points.
- Local RAG smoke test can run against a small Wikipedia slice (when internet is available) and return citations.
- A camera classification stub returns JSON matching the vision schema so UI can be built against it.

## Scope

### In scope (v0 - v1)

- Wikipedia-only grounding.
- Route-to-stop-card workflow with citations.
- Camera-based classification with follow-up prompts.
- Local and GCP/Vertex AI execution paths.

### Out of scope (for now)

- Safety-critical guidance (rockfall, slope stability).
- Non-Wikipedia sources unless explicitly added and validated.
- Full offline iOS release (possible later).

## Product requirements

### Functional requirements

- Route intake (JSON or GPX), sampling, and GeoSearch.
- Candidate filtering and stop selection.
- Retrieval and grounded generation with citations.
- Camera capture flow with structured output:
  - top hypotheses
  - visible features
  - follow-up prompts
  - citations
- Human-readable narrative and a machine-readable contract.

### Non-functional requirements

- Latency target for interactive Q&A: < 5 seconds at p95 (cloud path).
- Citation validity: 100 percent of factual claims must be linked to Wikipedia URLs.
- Privacy: no photo storage by default.
- Observability: log retrieval inputs, sources, and output IDs for debugging.

## Architecture options

### Local (fast iteration)

- Local embeddings and vector store, optional local LLM.
- No cloud credentials required.

### GCP / Vertex AI (current)

- Vertex AI embeddings + Gemini for generation.
- Local vector store (Chroma/FAISS) initially.

### Databricks (future)

- Model Serving, Vector Search, Jobs/Workflows.
- MLflow for eval and regression tracking.

See diagrams:
- `docs/diagrams/01_local_architecture.puml`
- `docs/diagrams/02_gcp_architecture.puml`
- `docs/diagrams/03_future_databricks_architecture.puml`

## Data sources

- Phase 0-1: Wikipedia only.
- Phase 2+: optional curated sources (USGS, field guides) after validation.

## Safety and policy

- Provide warnings for uncertainty.
- Avoid advice that implies legal or safety authority.
- Encourage observation, not collection.

## Metrics

- Answer quality (human rubric) and citation validity.
- User satisfaction (thumbs up/down, comments).
- Latency and cost per request.
- Retention: 7-day and 30-day return rates in beta.

## Roadmap

### Phase 0 - Local RAG baseline

- Local embeddings + vector store + optional local LLM.
- Offline smoke tests and contracts.

### Phase 1 - GCP production path

- Vertex AI embeddings + Gemini.
- Reliable citation validation and evaluation harness.

### Phase 2 - iOS beta

- Camera capture flow, structured output, and RAG grounding.
- Basic privacy controls and opt-in storage.

### Phase 3 - Databricks migration

- Vector Search and Model Serving for scale.
- MLflow-backed regression and eval dashboards.

### Phase 4 - App Store launch

- Polished onboarding, privacy, safety, and a stable API.

## Open questions

- What is the first launch geography and content slice?
- What is the minimum data retention policy for debugging and evaluation?
- Do we need a human curation step for high-traffic corridors?

## Related specs

- `docs/anc_spec_gcp.md`
- `docs/anc_spec_dbrx.md`
- `docs/anc_spec.ollama.md`
- `docs/roadside_geology_v0_mvp_spec.md`
- `docs/roadside_naturalist_v0_mvp_spec.md`
- `docs/vision_prompt_schema.md`
- `docs/vision_prompt_schema_naturalist.md`
