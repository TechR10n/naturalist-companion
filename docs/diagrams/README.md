# PlantUML diagrams

These diagrams cover local iteration paths, Databricks target-state architecture, and target-app C4 views.

## Diagrams (recommended order)

1. `01_local_architecture.puml` — local-only RAG loop (fast iteration, no cloud)
2. `02_gcp_architecture.puml` — current GCP/Vertex AI path (notebook + local vector store)
3. `04_langgraph_route_guide_flow.puml` — minimal LangGraph node flow (offline baseline)
4. `06_anc_databricks_production_app.puml` — production Databricks full-app target architecture
5. `07_anc_dbrx_component_architecture.puml` — ANC DBRX component architecture (Databricks target)
6. `08_target_app_c4_context.puml` — C1 system context for target app
7. `09_target_app_c4_container.puml` — C2 container view for target app
8. `10_target_app_c4_component.puml` — C3 backend component view for target app
9. `11_target_app_c4_deployment.puml` — C4 deployment view for target app
10. `12_anc_dbrx_current_notebook_touchpoints.puml` — current ANC DBRX notebook touchpoints (`anc_dbrx.ipynb` as-run view)

## Render locally

- Render SVGs into `docs/diagrams/rendered/`:
  - `mkdir -p docs/diagrams/rendered && plantuml -tsvg -o rendered docs/diagrams/[0-9][0-9]_*.puml`
- Render PNGs instead:
  - `mkdir -p docs/diagrams/rendered && plantuml -tpng -o rendered docs/diagrams/[0-9][0-9]_*.puml`
