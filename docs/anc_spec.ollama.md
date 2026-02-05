# ANC Spec - Ollama Local Path

## Status

New local-first proof-of-concept path. Canonical baseline notebook: `/Users/ryan/Developer/naturalist-companion/notebooks/anc_ollama.ipynb`.

## Objective

Provide a minimal LangGraph workflow that can run without cloud credentials while keeping notebook grounding constrained to Wikipedia.

## Architecture Intent

- Notebook orchestration: LangGraph
- Notebook retrieval/generation: minimal deterministic stubs
- Notebook evidence policy: Wikipedia-only for factual grounding
- Web/app path: may use Ollama and broader capabilities
- Output: concise answer + citation placeholders

## Guardrails

- Keep notebook behavior deterministic and small.
- Keep notebook factual grounding Wikipedia-only.
- Allow a small number of uploaded notebook pictures when they are sourced from Wikipedia/Wikimedia files.
- Use provider stubs by default for portability.
- Keep citation-first policy for factual claims in full mode.
- Put non-Wikipedia and multimodal expansion in Flask/app.

## Development Policy

1. Prove notebook baseline in PyCharm and online notebook runtimes.
2. Do not continue feature expansion in notebooks after proof.
3. Build next-phase capabilities in localhost Flask.
4. After notebook completion, expand to additional public data sources (for example USGS) in app/flask paths.
5. Start iOS planning after Flask capability depth is sufficient.
