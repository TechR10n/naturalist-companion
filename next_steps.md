# Next steps

This is the canonical walkthrough for local dev, PyCharm, and CI.

## 1) Local environment (uv)

From repo root:

```bash
uv --version
make setup
```

What this does:
- Uses Python 3.12 and syncs `.venv` from `pyproject.toml` + `uv.lock`.
- Installs base + `dev` extras.

Provider-specific variants:

```bash
make setup-gcp
make setup-ollama
make setup-dbrx
```

## 2) Core local checks (Makefile)

Run tests:

```bash
make test
```

Run offline smoke:

```bash
make smoke
```

Run both (same gate used by CI):

```bash
make check
```

Run app:

```bash
make web
```

Clean local outputs/caches:

```bash
make clean
```

Refresh lockfile (after dependency changes):

```bash
make lock
```

## 3) Optional smoke tools

Local RAG smoke using fallback data:

```bash
make smoke-rag
```

Vertex import smoke (no billable calls):

```bash
make smoke-vertex
```

## 4) StateGraph production controls (local observability)

Build persistent partition indexes (batch embeddings, not per request):

```bash
make stategraph-refresh
make stategraph-refresh-real
```

Run eval harness using persistent retrieval backend:

```bash
make stategraph-eval
make stategraph-cache-observe
```

Run release gate (promotion blocker):

```bash
make stategraph-gate
make stategraph-promote
```

Install macOS launchd schedule for daily refresh:

```bash
make stategraph-launchd-install
```

Expected artifacts:
- `out/stategraph_store/refresh_reports/*.json`
- `out/stategraph_eval/**/eval_report.json`
- `out/stategraph_eval/**/eval_report.md`
- `release_gate.json` in the gate run artifact root

## 5) PyCharm setup and run configurations

1. Set interpreter to `./.venv/bin/python`.
2. Use shared run configs in `.idea/runConfigurations/`:
- `Web App`
- `Unit Tests`
- `Smoke: Route Guide Graph`
- `Smoke: Local RAG (Fallback Data)`
- `Smoke: Local RAG (Ollama)`
- `Smoke: Live Wikipedia (Ollama)`
- `Smoke: Vertex AI`
- `Render: Route Guide Graph (Mermaid)`
3. Recommended: in each run config, add a **Before Launch** external tool or shell step:
- `make setup` (or profile-specific setup target)

## 6) CI (GitHub Actions)

CI file: `.github/workflows/ci.yml`

Current CI gate runs on Python 3.12:

```bash
make test
make smoke
```

using:

```bash
uv sync --frozen --python 3.12 --extra dev
```

## 7) Legacy bootstrap path (still available)

If you prefer the existing shell bootstrap:

```bash
./scripts/bootstrap_vertex.sh
source .venv/bin/activate
```

Then run the same make targets.
