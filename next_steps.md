# Next steps

Run these steps from the repo root to see the project working. Pick either:
- **Offline demo** (fastest, no GCP / no API calls), or
- **Vertex AI demo** (full flow, requires a GCP project; makes billable calls).

## 0) Prereqs

- Python **3.12** available as `python3.12` (recommended on macOS).
- For the **Vertex AI** path: a GCP project with billing enabled + the `gcloud` CLI.

## 1) Create a venv + install dependencies

```bash
cp .env.example .env
./scripts/bootstrap_vertex.sh
source .venv/bin/activate
python --version
```

Install the dependency profile for the target you are running:

```bash
# Vertex/GCP notebooks + scripts
python -m pip install -r requirements-gcp-dev.txt

# Ollama notebook path
python -m pip install -r requirements-ollama-dev.txt

# Databricks notebook path
python -m pip install -r requirements-dbrx-dev.txt
```

If `./scripts/bootstrap_vertex.sh` can’t find `python3.12`, rerun it with a different interpreter:

```bash
PYTHON_BIN=python3 ./scripts/bootstrap_vertex.sh
```

Each profile already includes FAISS + local RAG dependencies.

## 2) Fastest demo: Offline LangGraph MVP (no GCP / no API calls)

Run the offline smoke test (writes output files):

```bash
source .venv/bin/activate
python scripts/smoke_langgraph_mvp.py
```

You should see `LangGraph MVP smoke run` and it should write:
- `out/mvp/guide.md`
- `out/mvp/guide.json`

Open the generated guide:

```bash
sed -n '1,200p' out/mvp/guide.md
```

Optional: run without writing files (just prints a summary):

```bash
python scripts/smoke_langgraph_mvp.py --no-write
```

Optional: run unit tests:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## 2.5) Local RAG smoke test (SentenceTransformers + Chroma; optional Ollama answer step)

This runs a “real” local RAG loop (embeddings + vector store + retrieval) without GCP/Databricks.

Note: the first run will download the embedding model weights (internet required). Wikipedia loading also requires internet unless you use `--toy-data`.

```bash
source .venv/bin/activate
python scripts/smoke_local_rag.py
```

Optional: run fully offline using the repo’s tiny toy dataset:

```bash
python scripts/smoke_local_rag.py --toy-data
```

Optional: generate an answer with a local Ollama model (requires Ollama installed + running):

```bash
python scripts/smoke_local_rag.py --ollama --ollama-model llama3.2:3b
```

## 3) Full demo: Vertex AI + Wikipedia notebook (billable)

### 3.1 Configure `.env`

Edit `.env` and set at least:
- `GOOGLE_CLOUD_PROJECT=your-gcp-project-id`
- `GOOGLE_CLOUD_LOCATION=us-central1` (or your preferred region)

If you want a quick/cheap first run, also set:
- `WIKIPEDIA_MAX_DOCS=5`
- keep `WIKIPEDIA_QUERY` tight

### 3.2 Authenticate (ADC)

```bash
gcloud auth application-default login
```

### 3.3 Enable Vertex AI API (one-time per project)

```bash
gcloud config set project your-gcp-project-id
gcloud services enable aiplatform.googleapis.com
```

### 3.4 Smoke test Vertex AI wiring

Imports only (no billable calls):

```bash
source .venv/bin/activate
python scripts/smoke_vertex_ai.py
```

Tiny live checks (billable calls to embeddings + LLM):

```bash
python scripts/smoke_vertex_ai.py --api
```

You should see:
- `embeddings: ok (dim=...)`
- `llm: ok (response='ok')`

### 3.5 Run the notebook

```bash
source .venv/bin/activate
jupyter lab
```

Open `notebooks/anc_gcp.ipynb` and **Run All Cells**.
