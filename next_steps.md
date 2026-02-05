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

Local RAG smoke using toy data:

```bash
make smoke-rag
```

Vertex import smoke (no billable calls):

```bash
make smoke-vertex
```

## 4) PyCharm setup and run configurations

1. Set interpreter to `./.venv/bin/python`.
2. Use shared run configs in `.idea/runConfigurations/`:
- `Web App`
- `Unit Tests`
- `Smoke: LangGraph MVP`
- `Smoke: Local RAG (Toy Data)`
- `Smoke: Local RAG (Ollama)`
- `Smoke: Live Wikipedia (Ollama)`
- `Smoke: Vertex AI`
- `Render: LangGraph MVP (Mermaid)`
3. Recommended: in each run config, add a **Before Launch** external tool or shell step:
- `make setup` (or profile-specific setup target)

## 5) CI (GitHub Actions)

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

## 6) Legacy bootstrap path (still available)

If you prefer the existing shell bootstrap:

```bash
./scripts/bootstrap_vertex.sh
source .venv/bin/activate
```

Then run the same make targets.
