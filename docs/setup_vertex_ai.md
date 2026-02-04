# Vertex AI (GCP) Local Setup (macOS)

This project is notebook-first. The goal is a short, uneventful drive:
1) a Python **3.12** environment that installs cleanly
2) Google Cloud auth that works locally (ADC)
3) a quick “smoke test” lap before you scale up

## Mile 0: Tickets and keys (Prereqs)

- A Google Cloud project with billing enabled
- Vertex AI API enabled (`aiplatform.googleapis.com`)
- `gcloud` CLI installed and on your PATH

## Mile 1: Pick your engine (Python 3.12)

Python 3.13 is still “early” for many scientific wheels. If installs fail (commonly `faiss-cpu`), prefer Python 3.12.

### Option A (recommended): Homebrew `python@3.12`

1) Install:
- `brew install python@3.12`

2) Make it your default `python3`:
- Add to `~/.zshrc` (pick one approach):
  - Portable: `export PATH="$(brew --prefix python@3.12)/bin:$PATH"`
  - Apple Silicon: `export PATH="/opt/homebrew/opt/python@3.12/bin:$PATH"`
  - Intel: `export PATH="/usr/local/opt/python@3.12/bin:$PATH"`
- Restart your shell (or open a new terminal).

Verify:
- `which -a python3`
- `python3 --version` → `Python 3.12...`

### Removing Homebrew Python 3.13 (optional)

If you previously installed Homebrew’s `python` formula (often Python 3.13.x), you can remove it:

- Keep it installed but out of the way:
  - `brew unlink python`
- Remove it entirely:
  - `brew uninstall python`

Note: Homebrew may refuse to uninstall `python` if other formula depend on it. In that case, keep it installed and rely on the PATH ordering above.

Verify:
- `python3 --version` → `Python 3.12...`

## Mile 2: Load the trunk (venv + deps)

From the repo root:

- `cp .env.example .env` and edit values
- `./scripts/bootstrap_vertex.sh`
- `source .venv/bin/activate`

Optional (FAISS vector store):
- `python -m pip install -r requirements-faiss.txt`

## Mile 3: Get your local permit (ADC)

Recommended for local dev:
- `gcloud auth application-default login`

If you prefer a service account JSON instead, set:
- `GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/key.json`

## Mile 4: One quick lap (smoke test)

Imports only (no billable calls):
- `python scripts/smoke_vertex_ai.py`

Make a tiny API call (billable):
- `python scripts/smoke_vertex_ai.py --api`

## Mile 5: Open the map (run the notebook)

- `jupyter lab`
- Open `notebooks/agentic_wikipedia_gcp.ipynb`
- Start with small values: `WIKIPEDIA_MAX_DOCS=5` and a tight `WIKIPEDIA_QUERY`

## Troubleshooting

- **403 / permission denied**: ensure your user/service-account has Vertex AI permissions and the API is enabled.
- **Model not found**: model IDs can be region/preview-specific; set `VERTEX_LLM_MODEL` / `VERTEX_EMBEDDING_MODEL` in `.env`.
- **FAISS on macOS**: FAISS can be finicky to install on macOS via pip. The notebook defaults to Chroma; only install FAISS if you need it (`python -m pip install -r requirements-faiss.txt`).
