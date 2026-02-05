# ANC Spec - DBRX / Databricks Path

## Status

Roadmap-oriented proof-of-concept companion. Minimal runnable baseline is in `/Users/ryan/Developer/naturalist-companion/notebooks/anc_dbrx.ipynb`.

## Objective

Maintain parity with the GCP baseline while defining the Databricks migration target.

## Target Architecture

- Notebook-based development/debugging
- Jobs/Workflows for ingest refresh
- Model Serving for embeddings + generation
- Vector Search for retrieval
- MLflow for evaluations/regression tracking

## Guardrails

- Wikipedia-only sourcing for factual claims
- Allow a small number of uploaded notebook pictures when they come from Wikipedia/Wikimedia files
- Explicit uncertainty when retrieval evidence is weak
- Stable response schema across notebook/provider variants

## Phases

1. Baseline parity harness with minimal LangGraph behavior.
2. Databricks provider integration notebook path.
3. Vector Search + Model Serving + MLflow integration.
4. Optional API layer while preserving output schema parity.

## Development Policy

After notebook baselines are proven, feature work shifts to the localhost Flask app.
After notebook completion, expand to additional public sources (for example USGS) in app/flask paths.
iOS planning begins after Flask functionality is mature.
