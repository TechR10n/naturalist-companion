# PlantUML diagrams (interview/demo)

These diagrams are meant to be quick to open during a demo and easy to narrate in an interview (context → pipeline → agentic workflow → data contracts → camera extension).

## Diagrams (recommended order)

1. `01_system_context.puml` — notebook + Vertex AI + Wikipedia + local vector store
2. `02_rag_components.puml` — ingest vs runtime components (RAG)
3. `03_question_answer_sequence.puml` — grounded Q&A sequence (with an “index not built yet” branch)
4. `04_roadside_geology_langgraph.puml` — LangGraph workflow for the “Roadside Geology” spec
5. `05_stopcard_contract.puml` — `Guide` / `StopCard` output contract (testable, citation-friendly)
6. `06_vision_contract.puml` — camera classification JSON contract (structured + calibratable)
7. `07_camera_vision_architecture.puml` — iPhone → Cloud Run → Gemini Vision → RAG → citations

## Render locally

- If `docs/diagrams/rendered/` already exists, you can just open the `.svg` files directly.
- Render SVGs into `docs/diagrams/rendered/`:
  - `mkdir -p docs/diagrams/rendered && plantuml -tsvg -o rendered docs/diagrams/[0-9][0-9]_*.puml`
- Render PNGs instead:
  - `mkdir -p docs/diagrams/rendered && plantuml -tpng -o rendered docs/diagrams/[0-9][0-9]_*.puml`
