# PlantUML diagrams

These diagrams cover local iteration paths plus a single ANC Databricks target-state architecture.

## Diagrams (recommended order)

1. `01_local_architecture.puml` — local-only RAG loop (fast iteration, no cloud)
2. `02_gcp_architecture.puml` — current GCP/Vertex AI path (notebook + local vector store)
3. `04_langgraph_route_guide_flow.puml` — minimal LangGraph node flow (offline baseline)
4. `06_anc_databricks_production_app.puml` — production Databricks full-app target architecture

## Render locally

- Render SVGs into `docs/diagrams/rendered/`:
  - `mkdir -p docs/diagrams/rendered && plantuml -tsvg -o rendered docs/diagrams/[0-9][0-9]_*.puml`
- Render PNGs instead:
  - `mkdir -p docs/diagrams/rendered && plantuml -tpng -o rendered docs/diagrams/[0-9][0-9]_*.puml`
