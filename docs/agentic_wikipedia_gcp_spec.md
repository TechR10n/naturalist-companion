# Agentic Wikipedia — GCP / Vertex AI Spec (Current)

## Vision (current)

We are building a **Wikipedia-grounded, citation-first RAG pipeline** that can run in three modes:

- **Local-only** for fast iteration (no cloud credentials required).
- **GCP / Vertex AI** for higher-quality embeddings and LLM responses (current primary path).
- **Databricks** for future migration (see `docs/agentic_wikipedia_dbrx_spec.md`).

This document describes the **current GCP path** and its near-term plan.

## Architecture summary

- **Entry point**: `notebooks/agentic_wikipedia_gcp.ipynb`.
- **Source of truth**: Wikipedia API only.
- **Embeddings + LLM**: Vertex AI (`VertexAIEmbeddings`, `ChatVertexAI`).
- **Vector store**: local Chroma (default) or FAISS (optional).
- **Output**: answers with **Wikipedia-only citations**.

See diagrams:
- `docs/diagrams/01_local_architecture.puml`
- `docs/diagrams/02_gcp_architecture.puml`

## Data source and metadata

The Wikipedia loader returns LangChain `Document` objects with:

- `title`: Wikipedia page title
- `summary`: short extract
- `source`: canonical Wikipedia URL

These fields must survive chunking and retrieval so citations can be traced.

## Workflow (GCP path)

1. **Ingest**: query Wikipedia, load a constrained slice of pages.
2. **Chunk**: split pages into ~500–1,000 token chunks and attach metadata.
3. **Embed**: use Vertex AI embeddings for chunks and queries.
4. **Store**: upsert vectors into a local vector store (Chroma/FAISS).
5. **Retrieve**: similarity search for top-k chunks.
6. **Generate**: call Vertex AI Gemini via `ChatVertexAI` with grounding rules.
7. **Validate**: ensure Wikipedia-only citations and schema compliance.

## Configuration

Environment variables in `.env` (see `.env.example`):

- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION`
- `VERTEX_LLM_MODEL`
- `VERTEX_EMBEDDING_MODEL`
- `WIKIPEDIA_QUERY`
- `WIKIPEDIA_MAX_DOCS`
- `WIKIPEDIA_TOP_K`
- `VECTORSTORE` (`chroma` or `faiss`)
- `CHROMA_PERSIST_DIR`

Authentication:
- Preferred: `gcloud auth application-default login`
- Optional: `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json`

## Dependencies (minimal set)

- `langchain-google-vertexai`
- `google-cloud-aiplatform`
- `langchain` + `langgraph`
- `chromadb` (default vector store)
- `faiss-cpu` (optional)
- `wikipedia` (loader)
- `sentence-transformers` (local-only mode)

## Success criteria (GCP path)

- Answers are **grounded in retrieved Wikipedia text**.
- Every factual claim has a **Wikipedia URL citation**.
- The notebook completes end-to-end with a small slice (e.g., 5–10 pages).

## Near-term plan (GCP)

1. Keep the notebook path stable and reproducible.
2. Tighten chunking + citation validation.
3. Optional: move vector storage to a managed service only when necessary.
