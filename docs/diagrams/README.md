# PlantUML diagrams (current architecture)

These diagrams are focused on the **current delivery path**: local iteration, Vertex AI on GCP, and a forward-looking Databricks migration.

## Diagrams (recommended order)

1. `01_local_architecture.puml` — local-only RAG loop (fast iteration, no cloud)
2. `02_gcp_architecture.puml` — current GCP/Vertex AI path (notebook + local vector store)
3. `03_future_databricks_architecture.puml` — future Databricks target, with baseline notebook retained
4. `04_langgraph_mvp_flow.puml` — minimal LangGraph node flow (offline baseline)
5. `05_anc_dbrx_current_state.puml` — current Databricks notebook execution path from exported HTML
6. `06_anc_databricks_production_app.puml` — production Databricks full-app target architecture

## Render locally

- Render SVGs into `docs/diagrams/rendered/`:
  - `mkdir -p docs/diagrams/rendered && plantuml -tsvg -o rendered docs/diagrams/[0-9][0-9]_*.puml`
- Render PNGs instead:
  - `mkdir -p docs/diagrams/rendered && plantuml -tpng -o rendered docs/diagrams/[0-9][0-9]_*.puml`
