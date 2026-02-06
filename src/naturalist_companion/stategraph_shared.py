"""Provider-agnostic StateGraph backend: routed RAG with retries, gates, and evals.

This module supports two runtime modes:

- `deterministic`: offline-friendly corpus + deterministic answer templating.
- `realistic`: live Wikipedia retrieval + provider model generation.

Both modes exercise the same graph behavior:

- Typed state (`TypedDict`) plus structured outputs (`pydantic` models)
- Routing node with three outcomes:
  - answerable now
  - needs clarification
  - needs retrieval retry
- Parallel retrieval (FAISS + keyword BM25-style) followed by reranking
- Persistent partitioned retrieval adapters with refresh cadence controls
- Strict top-k/context budgets and cache-backed repeated-query acceleration
- Quality gate with citation coverage, hallucination checks, and safe-stop rules
- Retry loop with query rewriting and max-iteration cap
- Eval harness for 20 fixed I-81 questions
- Run artifact persistence (timings, snapshots, final JSON)

The graph behavior is shared across providers (`ollama`, `vertex`, `databricks`)
so notebooks and scripts can use a single orchestration implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

try:
    # Pydantic on Python < 3.12 requires TypedDict from typing_extensions.
    from typing_extensions import TypedDict
except ImportError:  # pragma: no cover - fallback for minimal environments.
    from typing import TypedDict

import faiss
import numpy as np
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from . import fallback_data
from .wikipedia_tools import wikipedia_api_get, wikipedia_page_url

TOKEN_RE = re.compile(r"[a-z0-9]+")
SAFE_STOP_NOTE = (
    "Use only legal pull-offs and marked overlooks. Do not stand in travel lanes "
    "or cross active traffic for photos."
)
UNSAFE_PHRASES = (
    "stand in the lane",
    "cross active traffic",
    "trespass",
    "ignore posted signs",
)
CORRIDOR_TERMS = (
    "i-81",
    "interstate 81",
    "appalachian",
    "hagerstown",
    "virginia",
    "maryland",
    "pennsylvania",
    "new york",
    "tennessee",
    "west virginia",
    "basalt",
    "ridge",
    "geology",
)
ProviderName = Literal["ollama", "vertex", "databricks"]
RunMode = Literal["deterministic", "realistic"]
SUPPORTED_PROVIDERS: tuple[ProviderName, ...] = ("ollama", "vertex", "databricks")
PROVIDER_DISPLAY: dict[ProviderName, str] = {
    "ollama": "Ollama",
    "vertex": "Vertex AI",
    "databricks": "Databricks",
}


class GraphConfig(TypedDict):
    """Runtime knobs for the stategraph workflow."""

    provider: str
    top_k_faiss: int
    top_k_keyword: int
    top_k_rerank: int
    embedding_dim: int
    citation_coverage_threshold: float
    min_support_overlap: float
    min_rerank_score: float
    max_retrieval_attempts: int
    snapshot_text_limit: int
    artifact_root: str
    runtime_mode: str
    llm_temperature: float
    llm_model: str
    llm_small_model: str
    llm_large_model: str
    use_large_model_for_eval: bool
    large_model_min_retry_attempt: int
    eval_mode: bool
    ollama_base_url: str
    live_max_docs: int
    wikipedia_language: str
    wikipedia_user_agent: str
    store_root: str
    retrieval_backend: str
    allow_on_demand_index_build: bool
    refresh_cadence: str
    active_partition: str
    context_token_budget: int
    strict_top_k_cap: int
    allow_top_k_experiments: bool
    retrieval_cache_ttl_s: int
    response_cache_ttl_s: int
    cache_max_entries: int
    eval_min_pass_rate_pct: float
    eval_min_citation_validity_pct: float
    eval_min_quality_rate_pct: float


class RetrievalHit(TypedDict):
    """Single retrieval candidate from a retriever or reranker."""

    chunk_id: str
    pageid: int
    title: str
    url: str
    text: str
    source: Literal["faiss", "keyword"]
    score: float
    overlap: float
    rerank_score: float


class NodeTiming(TypedDict):
    """Per-node timing entry captured for artifacts."""

    node: str
    duration_ms: float
    retrieval_attempt: int


class StateGraphState(TypedDict, total=False):
    """Typed state shared across graph nodes."""

    provider: str
    question: str
    config: GraphConfig
    run_id: str
    artifact_dir: str
    trace: list[str]
    snapshot_seq: int
    node_timings: list[NodeTiming]
    active_query: str
    query_history: list[str]
    active_partition: str
    retrieval_attempt: int
    max_retrieval_attempts: int
    last_retrieval_status: Literal["not_run", "ok", "empty", "low_relevance"]
    route_decision: dict[str, Any]
    clarification_prompt: str
    retrieval_candidates: list[RetrievalHit]
    reranked_docs: list[RetrievalHit]
    retrieval_runtime: RetrievalRuntime
    answer: dict[str, Any]
    quality_report: dict[str, Any]
    cache_events: list[str]
    cache_final_hit: bool
    cache_retrieval_hit: bool
    final_status: str
    final_output: dict[str, Any]


class CitationModel(BaseModel):
    """Structured citation entry."""

    title: str
    url: str
    pageid: int | None = None


class ClaimModel(BaseModel):
    """Structured claim with citation linkage + support metadata."""

    text: str
    citation_urls: list[str] = Field(default_factory=list)
    supported: bool | None = None
    support_overlap: float | None = None


class RouteDecisionModel(BaseModel):
    """Structured routing decision."""

    decision: Literal["answerable_now", "needs_clarification", "needs_retrieval_retry"]
    reason: str
    clarification_question: str | None = None


class AnswerModel(BaseModel):
    """Structured answer payload."""

    provider: str
    question: str
    response: str
    claims: list[ClaimModel] = Field(default_factory=list)
    citations: list[CitationModel] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)
    retrieval_attempt: int
    mode: Literal["deterministic", "realistic"] = "deterministic"


class QualityGateModel(BaseModel):
    """Structured quality report for answer gating."""

    passed: bool
    citation_coverage: float
    citation_threshold: float
    citation_validity_pct: float
    unsupported_claims: int
    hallucination_flags: list[str] = Field(default_factory=list)
    safe_stop_ok: bool
    reasons: list[str] = Field(default_factory=list)


class FinalOutputModel(BaseModel):
    """Final per-run JSON payload for review."""

    provider: str
    run_id: str
    question: str
    active_query: str
    route_decision: RouteDecisionModel
    answer: AnswerModel
    quality: QualityGateModel
    retrieval_attempts: int
    query_history: list[str]
    trace: list[str]
    node_timings: list[NodeTiming]
    generated_at: str


@dataclass(frozen=True)
class ChunkRecord:
    """Corpus chunk with source metadata."""

    chunk_id: str
    pageid: int
    title: str
    url: str
    text: str
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalRuntime:
    """Retrieval resources pre-built once and reused by node closures."""

    chunks: tuple[ChunkRecord, ...]
    chunk_by_id: dict[str, ChunkRecord]
    embedding_dim: int
    embedding_matrix: np.ndarray
    faiss_index: faiss.IndexFlatIP
    bm25_tfs: tuple[dict[str, int], ...]
    bm25_lengths: tuple[int, ...]
    bm25_idf: dict[str, float]
    bm25_avg_len: float


def _default_config() -> GraphConfig:
    return {
        "provider": "ollama",
        "top_k_faiss": 6,
        "top_k_keyword": 6,
        "top_k_rerank": 6,
        "embedding_dim": 256,
        "citation_coverage_threshold": 0.80,
        "min_support_overlap": 0.20,
        "min_rerank_score": 0.12,
        "max_retrieval_attempts": 3,
        "snapshot_text_limit": 300,
        "artifact_root": "out/stategraph",
        "runtime_mode": "deterministic",
        "llm_temperature": 0.25,
        "llm_model": "",
        "llm_small_model": "",
        "llm_large_model": "",
        "use_large_model_for_eval": True,
        "large_model_min_retry_attempt": 2,
        "eval_mode": False,
        "ollama_base_url": "",
        "live_max_docs": 12,
        "wikipedia_language": "en",
        "wikipedia_user_agent": "naturalist-companion-stategraph/0.1 (realistic mode)",
        "store_root": "out/stategraph_store",
        "retrieval_backend": "persistent",
        "allow_on_demand_index_build": True,
        "refresh_cadence": "daily",
        "active_partition": "corridor_i81",
        "context_token_budget": 900,
        "strict_top_k_cap": 8,
        "allow_top_k_experiments": False,
        "retrieval_cache_ttl_s": 900,
        "response_cache_ttl_s": 3600,
        "cache_max_entries": 2000,
        "eval_min_pass_rate_pct": 90.0,
        "eval_min_citation_validity_pct": 100.0,
        "eval_min_quality_rate_pct": 90.0,
    }


def _normalize_provider(raw: Any) -> ProviderName:
    value = str(raw or "").strip().lower()
    if value in SUPPORTED_PROVIDERS:
        return value  # type: ignore[return-value]
    return "ollama"


def _normalize_runtime_mode(raw: Any) -> RunMode:
    value = str(raw or "").strip().lower()
    if value in {"deterministic", "realistic"}:
        return value  # type: ignore[return-value]
    return "deterministic"


def _merge_config(overrides: dict[str, Any] | None) -> GraphConfig:
    cfg = _default_config()
    if not overrides:
        return cfg

    int_keys = {
        "top_k_faiss",
        "top_k_keyword",
        "top_k_rerank",
        "embedding_dim",
        "max_retrieval_attempts",
        "snapshot_text_limit",
        "live_max_docs",
        "large_model_min_retry_attempt",
        "context_token_budget",
        "strict_top_k_cap",
        "retrieval_cache_ttl_s",
        "response_cache_ttl_s",
        "cache_max_entries",
    }
    float_keys = {
        "citation_coverage_threshold",
        "min_support_overlap",
        "min_rerank_score",
        "llm_temperature",
        "eval_min_pass_rate_pct",
        "eval_min_citation_validity_pct",
        "eval_min_quality_rate_pct",
    }
    bool_keys = {
        "use_large_model_for_eval",
        "eval_mode",
        "allow_on_demand_index_build",
        "allow_top_k_experiments",
    }

    for key, value in overrides.items():
        if key not in cfg:
            continue
        if key == "provider":
            cfg[key] = _normalize_provider(value)  # type: ignore[typeddict-item]
            continue
        if key == "runtime_mode":
            cfg[key] = _normalize_runtime_mode(value)  # type: ignore[typeddict-item]
            continue
        if key in bool_keys:
            cfg[key] = str(value).strip().lower() in {"1", "true", "yes", "on"}  # type: ignore[typeddict-item]
            continue
        if key in int_keys:
            try:
                x = int(value)
            except Exception:
                continue
            if x > 0:
                cfg[key] = x  # type: ignore[typeddict-item]
            continue
        if key in float_keys:
            try:
                x = float(value)
            except Exception:
                continue
            if key == "llm_temperature":
                if 0.0 <= x <= 2.0:
                    cfg[key] = x  # type: ignore[typeddict-item]
                continue
            if key in {
                "citation_coverage_threshold",
                "min_support_overlap",
                "min_rerank_score",
            } and 0.0 <= x <= 1.5:
                cfg[key] = x  # type: ignore[typeddict-item]
                continue
            if key in {
                "eval_min_pass_rate_pct",
                "eval_min_citation_validity_pct",
                "eval_min_quality_rate_pct",
            } and 0.0 <= x <= 100.0:
                cfg[key] = x  # type: ignore[typeddict-item]
            continue
        if key in {
            "artifact_root",
            "llm_model",
            "llm_small_model",
            "llm_large_model",
            "ollama_base_url",
            "wikipedia_language",
            "wikipedia_user_agent",
            "store_root",
            "active_partition",
        }:
            text = str(value).strip()
            if text:
                cfg[key] = text  # type: ignore[typeddict-item]
            continue
        if key == "retrieval_backend":
            mode = str(value or "").strip().lower()
            if mode in {"persistent", "in_memory"}:
                cfg[key] = mode  # type: ignore[typeddict-item]
            continue
        if key == "refresh_cadence":
            cadence = str(value or "").strip().lower()
            if cadence in {"daily", "weekly"}:
                cfg[key] = cadence  # type: ignore[typeddict-item]

    cap = max(1, int(cfg["strict_top_k_cap"]))
    if not bool(cfg.get("allow_top_k_experiments")):
        cfg["top_k_faiss"] = min(max(1, int(cfg["top_k_faiss"])), cap)
        cfg["top_k_keyword"] = min(max(1, int(cfg["top_k_keyword"])), cap)
        cfg["top_k_rerank"] = min(max(1, int(cfg["top_k_rerank"])), cap)
    return cfg


def _model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[return-value]
    return model.dict()  # pragma: no cover


def _coerce_answer(
    payload: dict[str, Any] | None,
    *,
    provider: str,
    question: str,
    retrieval_attempt: int,
    mode: RunMode = "deterministic",
) -> AnswerModel:
    merged = {
        "provider": provider,
        "question": question,
        "response": "",
        "claims": [],
        "citations": [],
        "safety_notes": [SAFE_STOP_NOTE],
        "retrieval_attempt": retrieval_attempt,
        "mode": mode,
    }
    if payload:
        merged.update(payload)
    merged["provider"] = provider
    merged["mode"] = mode
    return AnswerModel(**merged)


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def _stable_bucket(token: str, dim: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % dim


def _embed_tokens(tokens: list[str], dim: int) -> np.ndarray:
    vec = np.zeros((dim,), dtype=np.float32)
    if not tokens:
        return vec
    for tok in tokens:
        vec[_stable_bucket(tok, dim)] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32)


PARTITION_SEED_QUERIES: dict[str, tuple[str, ...]] = {
    "corridor_i81": (
        "Interstate 81 geology",
        "Appalachian Mountains along I-81",
        "Valley and Ridge roadside geology",
    ),
    "topic_basalt": (
        "Basalt geology",
        "Columnar jointing",
        "igneous roadside outcrops",
    ),
    "geo_appalachia": (
        "Appalachian Mountains geology",
        "Valley and Ridge province",
        "Blue Ridge geology",
    ),
}
PARTITION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "corridor_i81": ("i-81", "interstate 81", "roadside", "corridor", "hagerstown"),
    "topic_basalt": ("basalt", "igneous", "lava", "columnar", "volcanic"),
    "geo_appalachia": ("appalachian", "ridge", "valley", "tectonic", "orogeny"),
}
REFRESH_CADENCE_SECONDS: dict[str, int] = {
    "daily": 24 * 60 * 60,
    "weekly": 7 * 24 * 60 * 60,
}


def _normalize_refresh_cadence(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    if value in REFRESH_CADENCE_SECONDS:
        return value
    return "daily"


def _resolve_active_partition(question: str, cfg: GraphConfig) -> str:
    explicit = str(cfg.get("active_partition") or "").strip()
    if explicit in PARTITION_SEED_QUERIES:
        return explicit
    q = (question or "").strip().lower()
    scores: dict[str, int] = {}
    for partition, terms in PARTITION_KEYWORDS.items():
        scores[partition] = sum(1 for term in terms if term in q)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if ranked and ranked[0][1] > 0:
        return ranked[0][0]
    return "corridor_i81"


def _partition_dir(cfg: GraphConfig, runtime_mode: RunMode, partition: str) -> Path:
    root = Path(str(cfg.get("store_root") or "out/stategraph_store"))
    return root / runtime_mode / partition


def _partition_paths(cfg: GraphConfig, runtime_mode: RunMode, partition: str) -> dict[str, Path]:
    base = _partition_dir(cfg, runtime_mode, partition)
    return {
        "base": base,
        "manifest": base / "manifest.json",
        "chunks": base / "chunks.jsonl",
        "embeddings": base / "embeddings.npy",
        "faiss": base / "faiss.index",
        "bm25": base / "bm25.json",
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_partition_manifest(
    cfg: GraphConfig,
    *,
    runtime_mode: RunMode,
    partition: str,
) -> dict[str, Any] | None:
    manifest = _partition_paths(cfg, runtime_mode, partition)["manifest"]
    if not manifest.exists():
        return None
    try:
        payload = _load_json(manifest)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _cadence_due(
    cfg: GraphConfig,
    *,
    runtime_mode: RunMode,
    partition: str,
    now_s: float | None = None,
) -> bool:
    manifest = _load_partition_manifest(cfg, runtime_mode=runtime_mode, partition=partition)
    if not manifest:
        return True
    now = float(now_s if now_s is not None else time.time())
    refreshed_at = float(manifest.get("refreshed_at_unix") or 0.0)
    cadence = _normalize_refresh_cadence(cfg.get("refresh_cadence"))
    return now - refreshed_at >= float(REFRESH_CADENCE_SECONDS[cadence])


def _build_runtime_from_pages(
    *,
    pages: dict[int, dict[str, Any]],
    embedding_dim: int,
) -> RetrievalRuntime:
    chunks: list[ChunkRecord] = []
    for pageid in sorted(pages.keys()):
        page = pages[pageid]
        lines: list[str] = []
        summary = str(page.get("summary") or "").strip()
        if summary:
            lines.append(summary)
        for raw in str(page.get("content") or "").splitlines():
            line = raw.strip(" -\t")
            if line:
                lines.append(line)
        for idx, line in enumerate(lines, start=1):
            chunk_id = f"{pageid}_{idx:02d}"
            tokens = tuple(_tokenize(line))
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    pageid=int(pageid),
                    title=str(page["title"]),
                    url=str(page["url"]),
                    text=line,
                    tokens=tokens,
                )
            )

    if not chunks:
        raise RuntimeError("No chunks available to build retrieval runtime.")

    embeddings = np.vstack([_embed_tokens(list(c.tokens), embedding_dim) for c in chunks]).astype(np.float32)
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)

    tf_docs: list[dict[str, int]] = []
    lengths: list[int] = []
    df: dict[str, int] = {}
    for chunk in chunks:
        counter = Counter(chunk.tokens)
        tf_docs.append(dict(counter))
        dl = int(sum(counter.values()))
        lengths.append(dl)
        for term in counter.keys():
            df[term] = df.get(term, 0) + 1

    n_docs = len(chunks)
    idf: dict[str, float] = {}
    for term, freq in df.items():
        # BM25 idf with +1 for numerical stability.
        idf[term] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
    avg_len = float(sum(lengths)) / float(n_docs)

    return RetrievalRuntime(
        chunks=tuple(chunks),
        chunk_by_id={c.chunk_id: c for c in chunks},
        embedding_dim=embedding_dim,
        embedding_matrix=embeddings,
        faiss_index=index,
        bm25_tfs=tuple(tf_docs),
        bm25_lengths=tuple(lengths),
        bm25_idf=idf,
        bm25_avg_len=avg_len,
    )


def _persist_runtime_partition(
    runtime: RetrievalRuntime,
    *,
    cfg: GraphConfig,
    runtime_mode: RunMode,
    partition: str,
    source: str,
) -> None:
    paths = _partition_paths(cfg, runtime_mode, partition)
    base = paths["base"]
    base.mkdir(parents=True, exist_ok=True)

    chunks = []
    for chunk in runtime.chunks:
        chunks.append(
            {
                "chunk_id": chunk.chunk_id,
                "pageid": chunk.pageid,
                "title": chunk.title,
                "url": chunk.url,
                "text": chunk.text,
                "tokens": list(chunk.tokens),
            }
        )
    paths["chunks"].write_text("\n".join(json.dumps(x) for x in chunks) + ("\n" if chunks else ""))
    np.save(paths["embeddings"], runtime.embedding_matrix)
    faiss.write_index(runtime.faiss_index, str(paths["faiss"]))
    bm25_payload = {
        "tfs": list(runtime.bm25_tfs),
        "lengths": list(runtime.bm25_lengths),
        "idf": runtime.bm25_idf,
        "avg_len": runtime.bm25_avg_len,
    }
    paths["bm25"].write_text(json.dumps(bm25_payload, indent=2) + "\n")
    manifest = {
        "partition": partition,
        "runtime_mode": runtime_mode,
        "embedding_dim": runtime.embedding_dim,
        "chunk_count": len(runtime.chunks),
        "source": source,
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "refreshed_at_unix": time.time(),
        "refresh_cadence": _normalize_refresh_cadence(cfg.get("refresh_cadence")),
        "top_k_cap": int(cfg.get("strict_top_k_cap", 8)),
        "context_token_budget": int(cfg.get("context_token_budget", 900)),
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")


def _load_runtime_partition(
    cfg: GraphConfig,
    *,
    runtime_mode: RunMode,
    partition: str,
) -> RetrievalRuntime | None:
    paths = _partition_paths(cfg, runtime_mode, partition)
    required = [paths["chunks"], paths["embeddings"], paths["faiss"], paths["bm25"], paths["manifest"]]
    if any(not p.exists() for p in required):
        return None

    chunks: list[ChunkRecord] = []
    try:
        for line in paths["chunks"].read_text().splitlines():
            row = json.loads(line)
            chunks.append(
                ChunkRecord(
                    chunk_id=str(row["chunk_id"]),
                    pageid=int(row["pageid"]),
                    title=str(row["title"]),
                    url=str(row["url"]),
                    text=str(row["text"]),
                    tokens=tuple(str(x) for x in row.get("tokens") or []),
                )
            )
        embeddings = np.load(paths["embeddings"]).astype(np.float32)
        index = faiss.read_index(str(paths["faiss"]))
        bm25 = _load_json(paths["bm25"])
    except Exception:
        return None

    if not chunks:
        return None
    if embeddings.shape[0] != len(chunks):
        return None
    tf_docs = tuple({str(k): int(v) for k, v in doc.items()} for doc in (bm25.get("tfs") or []))
    lengths = tuple(int(x) for x in (bm25.get("lengths") or []))
    if len(tf_docs) != len(chunks) or len(lengths) != len(chunks):
        return None
    idf = {str(k): float(v) for k, v in (bm25.get("idf") or {}).items()}
    avg_len = float(bm25.get("avg_len") or 1.0)
    return RetrievalRuntime(
        chunks=tuple(chunks),
        chunk_by_id={c.chunk_id: c for c in chunks},
        embedding_dim=int(embeddings.shape[1]),
        embedding_matrix=embeddings,
        faiss_index=index,
        bm25_tfs=tf_docs,
        bm25_lengths=lengths,
        bm25_idf=idf,
        bm25_avg_len=max(1.0, avg_len),
    )


def _cache_db_path(cfg: GraphConfig) -> Path:
    root = Path(str(cfg.get("store_root") or "out/stategraph_store"))
    root.mkdir(parents=True, exist_ok=True)
    return root / "stategraph_cache.sqlite3"


def _cache_table_name(table: str) -> str:
    if table not in {"retrieval_cache", "response_cache"}:
        raise ValueError(f"Unsupported cache table: {table}")
    return table


def _cache_connect(cfg: GraphConfig) -> sqlite3.Connection:
    path = _cache_db_path(cfg)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS retrieval_cache (
            key TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS response_cache (
            key TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_retrieval_cache_expiry ON retrieval_cache(expires_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_response_cache_expiry ON response_cache(expires_at)")
    return conn


def _cache_get(cfg: GraphConfig, *, table: str, key: str) -> dict[str, Any] | None:
    table_name = _cache_table_name(table)
    now = float(time.time())
    with _cache_connect(cfg) as conn:
        row = conn.execute(
            f"SELECT payload_json, expires_at FROM {table_name} WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        payload_json, expires_at = row
        if float(expires_at) < now:
            conn.execute(f"DELETE FROM {table_name} WHERE key = ?", (key,))
            conn.commit()
            return None
    try:
        payload = json.loads(str(payload_json))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _cache_put(
    cfg: GraphConfig,
    *,
    table: str,
    key: str,
    payload: dict[str, Any],
    ttl_s: int,
    max_entries: int,
) -> None:
    table_name = _cache_table_name(table)
    now = float(time.time())
    expires_at = now + float(max(1, ttl_s))
    payload_json = json.dumps(payload)
    with _cache_connect(cfg) as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO {table_name}(key, payload_json, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (key, payload_json, now, expires_at),
        )
        conn.execute(f"DELETE FROM {table_name} WHERE expires_at < ?", (now,))
        limit = max(10, int(max_entries))
        conn.execute(
            f"DELETE FROM {table_name} WHERE key NOT IN (SELECT key FROM {table_name} ORDER BY created_at DESC LIMIT ?)",
            (limit,),
        )
        conn.commit()


def _normalize_cache_key(text: str) -> str:
    tokens = _tokenize(text)
    return " ".join(tokens).strip()


def _retrieval_cache_key(provider: ProviderName, partition: str, query: str, cfg: GraphConfig) -> str:
    norm = _normalize_cache_key(query)
    return "|".join(
        [
            "retrieval",
            provider,
            str(cfg.get("runtime_mode") or "deterministic"),
            partition,
            str(cfg.get("top_k_faiss", 6)),
            str(cfg.get("top_k_keyword", 6)),
            str(cfg.get("top_k_rerank", 6)),
            hashlib.md5(norm.encode("utf-8")).hexdigest(),
        ]
    )


def _response_cache_key(provider: ProviderName, partition: str, question: str, cfg: GraphConfig) -> str:
    norm = _normalize_cache_key(question)
    return "|".join(
        [
            "response",
            provider,
            str(cfg.get("runtime_mode") or "deterministic"),
            partition,
            str(cfg.get("llm_small_model") or ""),
            str(cfg.get("llm_large_model") or ""),
            hashlib.md5(norm.encode("utf-8")).hexdigest(),
        ]
    )


def _rough_token_count(text: str) -> int:
    return len(_tokenize(text))


def list_default_partitions() -> tuple[str, ...]:
    """Return default partition ids used by persistent retrieval indexes."""

    return tuple(sorted(PARTITION_SEED_QUERIES.keys()))


def refresh_retrieval_partitions(
    *,
    config: dict[str, Any] | None = None,
    runtime_mode: RunMode = "realistic",
    partitions: list[str] | None = None,
    force: bool = False,
    max_docs: int | None = None,
) -> dict[str, Any]:
    """Batch-refresh one or more retrieval partitions and persist their indexes."""

    cfg = _merge_config(config)
    mode = _normalize_runtime_mode(runtime_mode)
    targets = partitions or list(list_default_partitions())
    out_rows: list[dict[str, Any]] = []
    now_s = float(time.time())

    for partition in targets:
        if partition not in PARTITION_SEED_QUERIES:
            out_rows.append(
                {
                    "partition": partition,
                    "status": "skipped_unknown_partition",
                    "reason": "Partition is not defined in PARTITION_SEED_QUERIES.",
                }
            )
            continue
        run_cfg: GraphConfig = {**cfg}
        if max_docs is not None:
            run_cfg["live_max_docs"] = max(4, int(max_docs))

        due = _cadence_due(run_cfg, runtime_mode=mode, partition=partition, now_s=now_s)
        if (not force) and (not due):
            manifest = _load_partition_manifest(run_cfg, runtime_mode=mode, partition=partition) or {}
            out_rows.append(
                {
                    "partition": partition,
                    "status": "skipped_fresh",
                    "refreshed_at": manifest.get("refreshed_at"),
                    "refresh_cadence": manifest.get("refresh_cadence", run_cfg["refresh_cadence"]),
                }
            )
            continue

        queries = list(PARTITION_SEED_QUERIES[partition])
        if mode == "realistic":
            pages = _load_realistic_pages(
                f"{partition} geology",
                run_cfg,
                seed_queries=queries,
            )
            source = "live_wikipedia"
        else:
            pages = fallback_data.fallback_wiki_pages()
            source = "fallback_data"

        runtime = _build_runtime_from_pages(pages=pages, embedding_dim=int(run_cfg["embedding_dim"]))
        _persist_runtime_partition(
            runtime,
            cfg=run_cfg,
            runtime_mode=mode,
            partition=partition,
            source=source,
        )
        out_rows.append(
            {
                "partition": partition,
                "status": "refreshed",
                "chunk_count": len(runtime.chunks),
                "embedding_dim": runtime.embedding_dim,
                "source": source,
            }
        )

    summary = {
        "runtime_mode": mode,
        "store_root": str(Path(cfg["store_root"]).resolve()),
        "refresh_cadence": cfg["refresh_cadence"],
        "total": len(out_rows),
        "refreshed": sum(1 for row in out_rows if row.get("status") == "refreshed"),
        "skipped": sum(1 for row in out_rows if str(row.get("status", "")).startswith("skipped_")),
    }
    return {"summary": summary, "rows": out_rows}


def _live_search_titles(
    query: str,
    *,
    max_titles: int,
    language: str,
    user_agent: str,
) -> list[str]:
    payload = wikipedia_api_get(
        {
            "action": "query",
            "list": "search",
            "format": "json",
            "formatversion": 2,
            "srlimit": max(1, min(int(max_titles), 20)),
            "srsearch": query,
        },
        language=language,
        user_agent=user_agent,
        timeout_s=12.0,
    )
    out: list[str] = []
    for item in ((payload.get("query") or {}).get("search") or []):
        title = str(item.get("title") or "").strip()
        if title:
            out.append(title)
    return out


def _live_fetch_pages_for_titles(
    titles: list[str],
    *,
    language: str,
    user_agent: str,
) -> dict[int, dict[str, Any]]:
    pages_by_id: dict[int, dict[str, Any]] = {}

    for title in titles:
        payload = wikipedia_api_get(
            {
                "action": "query",
                "prop": "extracts|categories|info|coordinates",
                "inprop": "url",
                "explaintext": 1,
                "exsectionformat": "plain",
                "cllimit": 50,
                "redirects": 1,
                "titles": title,
                "format": "json",
                "formatversion": 2,
            },
            language=language,
            user_agent=user_agent,
            timeout_s=12.0,
        )

        for page in ((payload.get("query") or {}).get("pages") or []):
            if not isinstance(page, dict) or page.get("missing"):
                continue
            pageid = int(page.get("pageid") or 0)
            if pageid <= 0:
                continue

            extract = str(page.get("extract") or "").strip()
            if not extract:
                continue
            summary = extract.split("\n", 1)[0].strip()

            cats: list[str] = []
            for c in page.get("categories") or []:
                raw = str(c.get("title") or "")
                if raw.startswith("Category:"):
                    raw = raw[len("Category:") :]
                raw = raw.strip()
                if raw:
                    cats.append(raw)

            coords = page.get("coordinates") or []
            lat = None
            lon = None
            if coords and isinstance(coords[0], dict):
                lat = coords[0].get("lat")
                lon = coords[0].get("lon")

            resolved_title = str(page.get("title") or title).strip() or title
            url = str(page.get("canonicalurl") or page.get("fullurl") or "").strip()
            if not url:
                url = wikipedia_page_url(resolved_title, language=language)

            pages_by_id[pageid] = {
                "pageid": pageid,
                "title": resolved_title,
                "url": url,
                "summary": summary,
                "content": extract,
                "categories": cats,
                "lat": float(lat) if lat is not None else None,
                "lon": float(lon) if lon is not None else None,
            }

    return pages_by_id


def _load_realistic_pages(
    question: str,
    cfg: GraphConfig,
    *,
    seed_queries: list[str] | None = None,
) -> dict[int, dict[str, Any]]:
    language = str(cfg.get("wikipedia_language") or "en").strip() or "en"
    user_agent = (
        str(cfg.get("wikipedia_user_agent") or "").strip()
        or "naturalist-companion-stategraph/0.1 (realistic mode)"
    )
    max_docs = max(4, int(cfg.get("live_max_docs", 12)))

    queries = list(seed_queries or [])
    if not queries:
        queries = [
            str(question or "").strip(),
            "Interstate 81 geology",
            "Appalachian Mountains geology",
            "Valley and Ridge geology",
            "roadcut geology",
        ]
    queries = [q for q in queries if q]

    titles: list[str] = []
    seen: set[str] = set()
    for query in queries:
        remaining = max_docs - len(titles)
        if remaining <= 0:
            break
        found = _live_search_titles(
            query,
            max_titles=min(remaining, 4),
            language=language,
            user_agent=user_agent,
        )
        for title in found:
            key = title.lower()
            if key in seen:
                continue
            seen.add(key)
            titles.append(title)
            if len(titles) >= max_docs:
                break

    if not titles:
        raise RuntimeError(
            "Realistic mode could not find live Wikipedia titles. "
            "Check network access and your question text."
        )

    pages = _live_fetch_pages_for_titles(titles, language=language, user_agent=user_agent)
    if not pages:
        raise RuntimeError(
            "Realistic mode could not fetch live Wikipedia page content. "
            "Check network access and Wikipedia API availability."
        )
    return pages


def _build_runtime(
    *,
    embedding_dim: int,
    runtime_mode: RunMode,
    question: str,
    cfg: GraphConfig,
) -> RetrievalRuntime:
    partition = _resolve_active_partition(question, cfg)
    backend = str(cfg.get("retrieval_backend") or "persistent").strip().lower()

    if backend == "persistent":
        cached = _load_runtime_partition(cfg, runtime_mode=runtime_mode, partition=partition)
        if cached is not None:
            return cached
        if not bool(cfg.get("allow_on_demand_index_build", True)):
            raise RuntimeError(
                "Persistent retrieval index is missing for partition "
                f"{partition!r} in mode {runtime_mode!r}. "
                "Run scripts/stategraph_refresh_index.py before online requests."
            )

    if runtime_mode == "realistic":
        queries = [question, *PARTITION_SEED_QUERIES.get(partition, ())]
        pages = _load_realistic_pages(question, cfg, seed_queries=[q for q in queries if q])
        source = "live_wikipedia"
    else:
        pages = fallback_data.fallback_wiki_pages()
        source = "fallback_data"

    runtime = _build_runtime_from_pages(pages=pages, embedding_dim=embedding_dim)
    if backend == "persistent":
        _persist_runtime_partition(
            runtime,
            cfg=cfg,
            runtime_mode=runtime_mode,
            partition=partition,
            source=source,
        )
    return runtime


def _faiss_retrieve(runtime: RetrievalRuntime, query: str, k: int) -> list[RetrievalHit]:
    tokens = _tokenize(query)
    if not tokens:
        return []
    query_vec = _embed_tokens(tokens, runtime.embedding_dim).reshape(1, -1)
    top_k = min(max(1, int(k)), len(runtime.chunks))
    scores, idxs = runtime.faiss_index.search(query_vec, top_k)

    out: list[RetrievalHit] = []
    qset = set(tokens)
    for score, i in zip(scores[0], idxs[0], strict=True):
        if i < 0 or i >= len(runtime.chunks):
            continue
        chunk = runtime.chunks[int(i)]
        cset = set(chunk.tokens)
        overlap = float(len(qset & cset)) / float(len(qset) or 1)
        out.append(
            {
                "chunk_id": chunk.chunk_id,
                "pageid": chunk.pageid,
                "title": chunk.title,
                "url": chunk.url,
                "text": chunk.text,
                "source": "faiss",
                "score": float(score),
                "overlap": overlap,
                "rerank_score": 0.0,
            }
        )
    return out


def _keyword_retrieve(runtime: RetrievalRuntime, query: str, k: int) -> list[RetrievalHit]:
    terms = _tokenize(query)
    if not terms:
        return []

    k1 = 1.5
    b = 0.75
    scores: list[tuple[float, int]] = []
    for doc_idx, tf in enumerate(runtime.bm25_tfs):
        dl = float(runtime.bm25_lengths[doc_idx])
        norm = k1 * (1.0 - b + b * (dl / runtime.bm25_avg_len))
        score = 0.0
        for term in terms:
            f = float(tf.get(term, 0))
            if f <= 0.0:
                continue
            idf = runtime.bm25_idf.get(term, 0.0)
            score += idf * (f * (k1 + 1.0)) / (f + norm)
        if score > 0.0:
            scores.append((score, doc_idx))

    scores.sort(key=lambda x: x[0], reverse=True)
    top = scores[: min(max(1, int(k)), len(scores))]
    qset = set(terms)

    out: list[RetrievalHit] = []
    for score, i in top:
        chunk = runtime.chunks[i]
        cset = set(chunk.tokens)
        overlap = float(len(qset & cset)) / float(len(qset) or 1)
        out.append(
            {
                "chunk_id": chunk.chunk_id,
                "pageid": chunk.pageid,
                "title": chunk.title,
                "url": chunk.url,
                "text": chunk.text,
                "source": "keyword",
                "score": float(score),
                "overlap": overlap,
                "rerank_score": 0.0,
            }
        )
    return out


def _rerank(query: str, candidates: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
    if not candidates:
        return []

    max_by_source: dict[str, float] = {"faiss": 1e-8, "keyword": 1e-8}
    for hit in candidates:
        src = hit["source"]
        max_by_source[src] = max(max_by_source[src], float(hit["score"]))

    merged: dict[str, RetrievalHit] = {}
    for hit in candidates:
        key = hit["chunk_id"]
        normalized = float(hit["score"]) / max_by_source[hit["source"]]
        source_weight = 0.60 if hit["source"] == "faiss" else 0.40
        blended = source_weight * normalized + 0.25 * float(hit["overlap"])

        existing = merged.get(key)
        if existing is None:
            merged[key] = {**hit, "rerank_score": blended}
            continue
        # Keep strongest score while preserving whichever source scored highest.
        if blended > existing["rerank_score"]:
            merged[key] = {**hit, "rerank_score": blended}
        else:
            existing["rerank_score"] = max(existing["rerank_score"], blended)

    ranked = sorted(merged.values(), key=lambda x: x["rerank_score"], reverse=True)
    return ranked[: max(1, int(top_k))]


def _needs_clarification(question: str) -> tuple[bool, str]:
    q = (question or "").strip()
    if not q:
        return True, "Question is empty."

    terms = _tokenize(q)
    if len(terms) < 5:
        return True, "Question is too short for grounded retrieval."

    ql = q.lower()
    vague_tokens = {"this", "that", "there", "it", "here"}
    has_vague = any(tok in vague_tokens for tok in terms)
    has_anchor = any(term in ql for term in CORRIDOR_TERMS)
    if has_vague and not has_anchor:
        return True, "Question is ambiguous and misses I-81 context anchors."

    return False, ""


def _trim_text(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        return value[:limit] + "...<trimmed>"
    if isinstance(value, list):
        return [_trim_text(v, limit) for v in value]
    if isinstance(value, dict):
        return {k: _trim_text(v, limit) for k, v in value.items()}
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _cache_event_list(state: StateGraphState, event: str) -> list[str]:
    return [*state.get("cache_events", []), event]


def _write_snapshot(state: StateGraphState, node: str, seq: int) -> None:
    artifact_dir = str(state.get("artifact_dir") or "").strip()
    if not artifact_dir:
        return
    cfg = state.get("config") or _default_config()
    limit = int(cfg.get("snapshot_text_limit", 300))

    snapshot_payload = {
        "provider": state.get("provider"),
        "run_id": state.get("run_id"),
        "node": node,
        "seq": seq,
        "question": state.get("question"),
        "active_query": state.get("active_query"),
        "active_partition": state.get("active_partition"),
        "route_decision": state.get("route_decision"),
        "retrieval_attempt": state.get("retrieval_attempt", 0),
        "max_retrieval_attempts": state.get("max_retrieval_attempts", 0),
        "last_retrieval_status": state.get("last_retrieval_status"),
        "retrieval_candidates": _trim_text(state.get("retrieval_candidates", []), limit),
        "reranked_docs": _trim_text(state.get("reranked_docs", []), limit),
        "answer": _trim_text(state.get("answer", {}), limit),
        "quality_report": state.get("quality_report"),
        "cache_events": state.get("cache_events", []),
        "cache_final_hit": bool(state.get("cache_final_hit")),
        "cache_retrieval_hit": bool(state.get("cache_retrieval_hit")),
        "final_status": state.get("final_status"),
        "trace": state.get("trace", []),
        "latest_timing": (state.get("node_timings") or [])[-1] if state.get("node_timings") else None,
    }
    out = Path(artifact_dir) / "state_snapshots" / f"{seq:03d}_{node}.json"
    _write_json(out, snapshot_payload)


def _instrument_node(name: str, node_fn):
    def _wrapped(state: StateGraphState) -> StateGraphState:
        start = time.perf_counter()
        updates = node_fn(state)
        duration_ms = (time.perf_counter() - start) * 1000.0

        trace = [*state.get("trace", []), name]
        attempt = int(updates.get("retrieval_attempt", state.get("retrieval_attempt", 0)))
        node_timings: list[NodeTiming] = [
            *state.get("node_timings", []),
            {
                "node": name,
                "duration_ms": round(duration_ms, 3),
                "retrieval_attempt": attempt,
            },
        ]
        seq = int(state.get("snapshot_seq", 0)) + 1

        updates = {**updates, "trace": trace, "node_timings": node_timings, "snapshot_seq": seq}
        merged = {**state, **updates}
        _write_snapshot(merged, name, seq)
        return updates

    return _wrapped


def _init_run_node(state: StateGraphState) -> StateGraphState:
    question = str(state.get("question") or "").strip()
    if not question:
        raise ValueError("question must be non-empty")

    cfg = _merge_config(state.get("config"))
    provider = _normalize_provider(cfg.get("provider"))
    runtime_mode = _normalize_runtime_mode(cfg.get("runtime_mode"))
    partition = _resolve_active_partition(question, cfg)
    cfg["active_partition"] = partition
    run_id = str(state.get("run_id") or "")
    if not run_id:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        qhash = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
        run_id = f"run_{stamp}_{qhash}"

    artifact_dir = Path(cfg["artifact_root"]) / provider / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    runtime = _build_runtime(
        embedding_dim=cfg["embedding_dim"],
        runtime_mode=runtime_mode,
        question=question,
        cfg=cfg,
    )
    response_key = _response_cache_key(provider, partition, question, cfg)
    response_cached = _cache_get(cfg, table="response_cache", key=response_key)
    cache_events: list[str] = []
    if _cadence_due(cfg, runtime_mode=runtime_mode, partition=partition):
        cache_events.append("partition_refresh_due")
    else:
        cache_events.append("partition_refresh_not_due")
    cache_final_hit = False
    answer_payload: dict[str, Any] = {}
    quality_payload: dict[str, Any] = {}
    reranked_docs: list[RetrievalHit] = []
    retrieval_candidates: list[RetrievalHit] = []
    last_retrieval_status: Literal["not_run", "ok", "empty", "low_relevance"] = "not_run"
    if response_cached:
        answer_raw = response_cached.get("answer")
        quality_raw = response_cached.get("quality_report")
        if isinstance(answer_raw, dict) and isinstance(quality_raw, dict):
            answer_payload = answer_raw
            quality_payload = quality_raw
            reranked_docs = list(response_cached.get("reranked_docs") or [])
            retrieval_candidates = list(response_cached.get("retrieval_candidates") or [])
            last_retrieval_status = "ok"
            cache_final_hit = True
            cache_events.append("response_cache_hit")
        else:
            cache_events.append("response_cache_malformed")
    else:
        cache_events.append("response_cache_miss")

    return {
        "provider": provider,
        "question": question,
        "config": cfg,
        "run_id": run_id,
        "artifact_dir": str(artifact_dir),
        "active_query": question,
        "active_partition": partition,
        "query_history": [question],
        "retrieval_attempt": 0,
        "max_retrieval_attempts": int(cfg["max_retrieval_attempts"]),
        "last_retrieval_status": last_retrieval_status,
        "retrieval_candidates": retrieval_candidates,
        "reranked_docs": reranked_docs,
        "retrieval_runtime": runtime,
        "answer": answer_payload,
        "quality_report": quality_payload,
        "cache_events": cache_events,
        "cache_final_hit": cache_final_hit,
        "cache_retrieval_hit": False,
        "final_status": "running",
    }


def _route_query_node(state: StateGraphState) -> StateGraphState:
    if bool(state.get("cache_final_hit")):
        decision = RouteDecisionModel(
            decision="answerable_now",
            reason="Response cache hit: using prior grounded output.",
            clarification_question=None,
        )
        return {
            "route_decision": _model_dump(decision),
            "cache_events": _cache_event_list(state, "route_used_response_cache"),
        }

    needs_clar, reason = _needs_clarification(state["question"])
    if needs_clar:
        decision = RouteDecisionModel(
            decision="needs_clarification",
            reason=reason,
            clarification_question=(
                "Can you include your corridor context (for example: I-81 segment, nearby "
                "city, and what observation you want to verify)?"
            ),
        )
        return {
            "route_decision": _model_dump(decision),
            "clarification_prompt": str(decision.clarification_question or ""),
        }

    if state.get("last_retrieval_status") in {"empty", "low_relevance"}:
        decision = RouteDecisionModel(
            decision="needs_retrieval_retry",
            reason="Prior retrieval did not provide enough grounded evidence.",
            clarification_question=None,
        )
        return {"route_decision": _model_dump(decision)}

    decision = RouteDecisionModel(
        decision="answerable_now",
        reason="Question has enough context and can proceed to retrieval.",
        clarification_question=None,
    )
    return {"route_decision": _model_dump(decision)}


def _route_query_edge(
    state: StateGraphState,
) -> Literal["answerable_now", "needs_clarification", "needs_retrieval_retry"]:
    route = state.get("route_decision") or {}
    decision = route.get("decision")
    if decision == "needs_clarification":
        return "needs_clarification"
    if decision == "needs_retrieval_retry":
        return "needs_retrieval_retry"
    return "answerable_now"


def _clarification_node(state: StateGraphState) -> StateGraphState:
    prompt = str(state.get("clarification_prompt") or "Please provide more context.")
    cfg = state.get("config") or _default_config()
    runtime_mode = _normalize_runtime_mode(cfg.get("runtime_mode"))
    answer = AnswerModel(
        provider=str(state.get("provider") or "ollama"),
        question=state["question"],
        response=f"Need clarification before retrieval: {prompt}",
        claims=[],
        citations=[],
        safety_notes=[SAFE_STOP_NOTE],
        retrieval_attempt=int(state.get("retrieval_attempt", 0)),
        mode=runtime_mode,
    )
    quality = QualityGateModel(
        passed=False,
        citation_coverage=0.0,
        citation_threshold=float(state["config"]["citation_coverage_threshold"]),
        citation_validity_pct=0.0,
        unsupported_claims=0,
        hallucination_flags=[],
        safe_stop_ok=True,
        reasons=["needs_clarification"],
    )
    return {
        "answer": _model_dump(answer),
        "quality_report": _model_dump(quality),
        "final_status": "needs_clarification",
    }


def _rewrite_query_node(state: StateGraphState) -> StateGraphState:
    attempt = int(state.get("retrieval_attempt", 0))
    suffixes = [
        "I-81 geology context",
        "Appalachian Mountains evidence",
        "Basalt and stratigraphy",
        "safe roadside observation",
    ]
    suffix = " ".join(suffixes[: min(len(suffixes), attempt + 1)])
    rewritten = f"{state['question']} {suffix}".strip()
    return {
        "active_query": rewritten,
        "query_history": [*state.get("query_history", []), rewritten],
    }


def _retrieve_parallel_node(state: StateGraphState) -> StateGraphState:
    cfg = state["config"]
    runtime_mode = _normalize_runtime_mode(cfg.get("runtime_mode"))
    provider = _normalize_provider(state.get("provider"))
    partition = str(state.get("active_partition") or _resolve_active_partition(state["question"], cfg))

    if bool(state.get("cache_final_hit")) and (state.get("retrieval_candidates") or state.get("reranked_docs")):
        return {
            "retrieval_attempt": int(state.get("retrieval_attempt", 0)),
            "retrieval_candidates": list(state.get("retrieval_candidates") or []),
            "last_retrieval_status": "ok",
            "cache_retrieval_hit": True,
            "cache_events": _cache_event_list(state, "retrieval_skipped_due_to_response_cache"),
        }

    runtime = state.get("retrieval_runtime")
    if runtime is None:
        runtime = _build_runtime(
            embedding_dim=cfg["embedding_dim"],
            runtime_mode=runtime_mode,
            question=state["question"],
            cfg=cfg,
        )

    query = str(state.get("active_query") or state["question"])
    faiss_k = int(cfg["top_k_faiss"])
    keyword_k = int(cfg["top_k_keyword"])
    attempt = int(state.get("retrieval_attempt", 0)) + 1
    rkey = _retrieval_cache_key(provider, partition, query, cfg)
    cached = _cache_get(cfg, table="retrieval_cache", key=rkey)
    if cached and isinstance(cached.get("candidates"), list):
        candidates_cached = list(cached["candidates"])
        status_cached: Literal["ok", "empty"] = "ok" if candidates_cached else "empty"
        return {
            "retrieval_attempt": attempt,
            "retrieval_candidates": candidates_cached,
            "last_retrieval_status": status_cached,
            "retrieval_runtime": runtime,
            "cache_retrieval_hit": True,
            "cache_events": _cache_event_list(state, "retrieval_cache_hit"),
        }

    with ThreadPoolExecutor(max_workers=2) as executor:
        faiss_future = executor.submit(_faiss_retrieve, runtime, query, faiss_k)
        keyword_future = executor.submit(_keyword_retrieve, runtime, query, keyword_k)
        faiss_hits = faiss_future.result()
        keyword_hits = keyword_future.result()

    candidates = [*faiss_hits, *keyword_hits]
    _cache_put(
        cfg,
        table="retrieval_cache",
        key=rkey,
        payload={"candidates": candidates},
        ttl_s=int(cfg.get("retrieval_cache_ttl_s", 900)),
        max_entries=int(cfg.get("cache_max_entries", 2000)),
    )
    status: Literal["ok", "empty"] = "ok" if candidates else "empty"
    return {
        "retrieval_attempt": attempt,
        "retrieval_candidates": candidates,
        "last_retrieval_status": status,
        "retrieval_runtime": runtime,
        "cache_retrieval_hit": False,
        "cache_events": _cache_event_list(state, "retrieval_cache_miss"),
    }


def _rerank_node(state: StateGraphState) -> StateGraphState:
    cfg = state["config"]
    if bool(state.get("cache_final_hit")) and state.get("reranked_docs"):
        return {"reranked_docs": list(state.get("reranked_docs") or []), "last_retrieval_status": "ok"}
    candidates = state.get("retrieval_candidates") or []
    reranked = _rerank(state.get("active_query", state["question"]), candidates, cfg["top_k_rerank"])
    if not reranked:
        return {"reranked_docs": [], "last_retrieval_status": "empty"}

    top_score = float(reranked[0]["rerank_score"])
    status: Literal["ok", "low_relevance"] = (
        "ok" if top_score >= float(cfg["min_rerank_score"]) else "low_relevance"
    )
    return {"reranked_docs": reranked, "last_retrieval_status": status}


def _claim_from_doc_text(text: str) -> str:
    sentence = (text or "").strip()
    sentence = sentence.replace("Fallback extract:", "").strip()
    return sentence[:220].rstrip(". ") + "."


def _build_context_block(top_docs: list[RetrievalHit], *, token_budget: int) -> str:
    lines: list[str] = []
    used_tokens = 0
    budget = max(120, int(token_budget))
    for idx, hit in enumerate(top_docs, start=1):
        snippet = str(hit["text"] or "").strip().replace("\n", " ")
        snippet = re.sub(r"\s+", " ", snippet)
        snippet_tokens = _tokenize(snippet)
        if not snippet_tokens:
            continue
        remaining = budget - used_tokens
        if remaining <= 50 and lines:
            break
        snippet_trimmed = " ".join(snippet_tokens[: max(1, min(remaining, 120))])
        entry = (
            f"[{idx}] {hit['title']} ({hit['url']})\n"
            f"source={hit['source']}, rerank_score={float(hit['rerank_score']):.3f}\n"
            f"text={snippet_trimmed}"
        )
        entry_tokens = _rough_token_count(entry)
        if used_tokens + entry_tokens > budget and lines:
            break
        lines.append(entry)
        used_tokens += entry_tokens
    return "\n\n".join(lines)


def _resolve_model_for_request(
    provider: ProviderName,
    cfg: GraphConfig,
    *,
    retrieval_attempt: int,
) -> tuple[str, str]:
    explicit = str(cfg.get("llm_model") or "").strip()
    if explicit:
        return explicit, "override"

    if provider == "ollama":
        default_small = os.environ.get("OLLAMA_LLM_MODEL_SMALL") or os.environ.get(
            "OLLAMA_LLM_MODEL", "llama3.1:8b"
        )
        default_large = os.environ.get("OLLAMA_LLM_MODEL_LARGE", default_small)
    elif provider == "vertex":
        default_small = os.environ.get("VERTEX_LLM_MODEL_SMALL") or os.environ.get(
            "VERTEX_LLM_MODEL", "gemini-1.5-flash"
        )
        default_large = os.environ.get("VERTEX_LLM_MODEL_LARGE", default_small)
    else:
        default_small = os.environ.get(
            "DATABRICKS_LLM_ENDPOINT_SMALL",
            os.environ.get("DATABRICKS_LLM_ENDPOINT", "databricks-meta-llama-3-1-8b-instruct"),
        )
        default_large = os.environ.get(
            "DATABRICKS_LLM_ENDPOINT_LARGE",
            default_small,
        )

    small = str(cfg.get("llm_small_model") or default_small).strip() or default_small
    large = str(cfg.get("llm_large_model") or default_large).strip() or default_large

    if bool(cfg.get("eval_mode")) and bool(cfg.get("use_large_model_for_eval")):
        return large, "large_eval"
    if retrieval_attempt >= int(cfg.get("large_model_min_retry_attempt", 2)):
        return large, "large_retry"
    return small, "small_online"


def _invoke_provider_llm(
    provider: ProviderName,
    prompt: str,
    cfg: GraphConfig,
    *,
    model_name: str,
) -> str:
    temperature = float(cfg.get("llm_temperature", 0.25))

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover - provider package optional in test env.
            raise RuntimeError(
                "Realistic mode for provider='ollama' requires `langchain-ollama`."
            ) from exc

        model = model_name
        base_url = (
            str(cfg.get("ollama_base_url") or "").strip()
            or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
    elif provider == "vertex":
        try:
            from langchain_google_vertexai import ChatVertexAI
        except ImportError as exc:  # pragma: no cover - provider package optional in test env.
            raise RuntimeError(
                "Realistic mode for provider='vertex' requires `langchain-google-vertexai`."
            ) from exc

        model = model_name
        kwargs: dict[str, Any] = {}
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get("GCP_LOCATION")
        if project:
            kwargs["project"] = project
        if location:
            kwargs["location"] = location
        try:
            llm = ChatVertexAI(model_name=model, temperature=temperature, **kwargs)
        except TypeError:
            llm = ChatVertexAI(model=model, temperature=temperature, **kwargs)
    else:
        try:
            from databricks_langchain import ChatDatabricks
        except ImportError as exc:  # pragma: no cover - provider package optional in test env.
            raise RuntimeError(
                "Realistic mode for provider='databricks' requires `databricks-langchain`."
            ) from exc

        endpoint = model_name
        llm = ChatDatabricks(endpoint=endpoint, temperature=temperature)

    response = llm.invoke(prompt)
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
                if text:
                    parts.append(text)
            else:
                text = str(item).strip()
                if text:
                    parts.append(text)
        return " ".join(parts).strip()
    return str(content or "").strip()


def _generate_realistic_response(
    provider: ProviderName,
    question: str,
    top_docs: list[RetrievalHit],
    claims: list[ClaimModel],
    cfg: GraphConfig,
    *,
    retrieval_attempt: int,
) -> str:
    context_block = _build_context_block(
        top_docs,
        token_budget=int(cfg.get("context_token_budget", 900)),
    )
    model_name, model_tier = _resolve_model_for_request(
        provider,
        cfg,
        retrieval_attempt=retrieval_attempt,
    )
    claim_block = "\n".join(f"- {claim.text}" for claim in claims)
    prompt = (
        "You are a roadside geology field assistant.\n"
        "Write a grounded answer that is realistic, practical, and concise.\n"
        "Use only the evidence in Context.\n"
        "Do not invent locations or citations.\n"
        "Include uncertainty when evidence is weak.\n"
        "End with this exact safety note:\n"
        f"{SAFE_STOP_NOTE}\n\n"
        f"Question:\n{question}\n\n"
        f"Candidate claims:\n{claim_block}\n\n"
        f"Model tier: {model_tier}\n\n"
        f"Context:\n{context_block}\n"
    )
    response = _invoke_provider_llm(provider, prompt, cfg, model_name=model_name)
    if not response:
        raise RuntimeError("Provider returned an empty response in realistic mode.")
    if SAFE_STOP_NOTE.lower() not in response.lower():
        response = response.rstrip() + f" {SAFE_STOP_NOTE}"
    return response.strip()


def _generate_answer_node(state: StateGraphState) -> StateGraphState:
    cfg = state["config"]
    if bool(state.get("cache_final_hit")) and isinstance(state.get("answer"), dict) and state.get("answer"):
        return {
            "answer": dict(state.get("answer") or {}),
            "cache_events": _cache_event_list(state, "answer_reused_from_response_cache"),
        }

    docs = state.get("reranked_docs") or []
    attempt = int(state.get("retrieval_attempt", 0))
    provider = _normalize_provider(state.get("provider"))
    provider_label = PROVIDER_DISPLAY[provider]
    runtime_mode = _normalize_runtime_mode(cfg.get("runtime_mode"))

    if not docs:
        answer = AnswerModel(
            provider=provider,
            question=state["question"],
            response=(
                f"[{provider_label}] I could not retrieve grounded evidence yet. "
                "I will retry retrieval with a rewritten query."
            ),
            claims=[],
            citations=[],
            safety_notes=[SAFE_STOP_NOTE],
            retrieval_attempt=attempt,
            mode=runtime_mode,
        )
        return {"answer": _model_dump(answer)}

    top_docs = docs[:3]
    claims: list[ClaimModel] = []
    citations: list[CitationModel] = []
    seen_urls: set[str] = set()
    for hit in top_docs:
        url = str(hit["url"])
        claims.append(ClaimModel(text=_claim_from_doc_text(hit["text"]), citation_urls=[url]))
        if url not in seen_urls:
            citations.append(
                CitationModel(title=str(hit["title"]), url=url, pageid=int(hit["pageid"]))
            )
            seen_urls.add(url)

    if runtime_mode == "realistic":
        response = _generate_realistic_response(
            provider=provider,
            question=state["question"],
            top_docs=top_docs,
            claims=claims,
            cfg=cfg,
            retrieval_attempt=attempt,
        )
    else:
        bullets = " ".join([f"- {claim.text}" for claim in claims])
        response = (
            f"[{provider_label}] Grounded summary for '{state['question']}': {bullets} "
            f"{SAFE_STOP_NOTE}"
        )
    answer = AnswerModel(
        provider=provider,
        question=state["question"],
        response=response,
        claims=claims,
        citations=citations,
        safety_notes=[SAFE_STOP_NOTE],
        retrieval_attempt=attempt,
        mode=runtime_mode,
    )
    return {"answer": _model_dump(answer)}


def _token_overlap_ratio(text_a: str, text_b: str) -> float:
    ta = set(_tokenize(text_a))
    tb = set(_tokenize(text_b))
    if not ta:
        return 0.0
    return float(len(ta & tb)) / float(len(ta))


def _safe_stop_check(response: str, safety_notes: list[str]) -> tuple[bool, list[str]]:
    """Validate safety guidance while ignoring the canonical safe-note text."""

    response_l = str(response or "").lower()
    canonical_note = SAFE_STOP_NOTE.lower()
    notes_l = [str(note or "").strip().lower() for note in safety_notes if str(note or "").strip()]

    has_safe_guidance = (
        canonical_note in response_l
        or "legal pull-offs" in response_l
        or any("legal pull-offs" in note for note in notes_l)
    )

    # Remove the canonical note to avoid false unsafe hits from its warning phrases.
    scrubbed_response = response_l.replace(canonical_note, " ")
    unsafe_hits = [phrase for phrase in UNSAFE_PHRASES if phrase in scrubbed_response]
    return has_safe_guidance and not unsafe_hits, unsafe_hits


def _quality_gate_node(state: StateGraphState) -> StateGraphState:
    cfg = state["config"]
    if (
        bool(state.get("cache_final_hit"))
        and isinstance(state.get("quality_report"), dict)
        and state.get("quality_report")
    ):
        return {
            "answer": dict(state.get("answer") or {}),
            "quality_report": dict(state.get("quality_report") or {}),
            "cache_events": _cache_event_list(state, "quality_reused_from_response_cache"),
        }

    runtime_mode = _normalize_runtime_mode(cfg.get("runtime_mode"))
    answer_payload = state.get("answer") or {}
    answer = _coerce_answer(
        answer_payload,
        provider=str(state.get("provider") or "ollama"),
        question=state["question"],
        retrieval_attempt=int(state.get("retrieval_attempt", 0)),
        mode=runtime_mode,
    )
    docs = state.get("reranked_docs") or []
    docs_by_url: dict[str, list[str]] = {}
    for doc in docs:
        docs_by_url.setdefault(str(doc["url"]), []).append(str(doc["text"]))

    claims = []
    hallucination_flags: list[str] = []
    unsupported = 0
    claims_with_citations = 0
    for claim in answer.claims:
        evidence_text = " ".join(
            " ".join(docs_by_url.get(url, [])) for url in (claim.citation_urls or [])
        )
        if claim.citation_urls:
            claims_with_citations += 1
        overlap = _token_overlap_ratio(claim.text, evidence_text)
        supported = overlap >= float(cfg["min_support_overlap"])
        if not supported:
            unsupported += 1
            hallucination_flags.append(f"unsupported_claim:{claim.text[:80]}")
        claims.append(
            ClaimModel(
                text=claim.text,
                citation_urls=list(claim.citation_urls),
                supported=supported,
                support_overlap=round(overlap, 3),
            )
        )

    citation_coverage = float(claims_with_citations) / float(len(answer.claims) or 1)
    citations_total = len(answer.citations)
    citations_valid = 0
    for citation in answer.citations:
        url = str(citation.url)
        if url.startswith("https://en.wikipedia.org/wiki/"):
            citations_valid += 1
    citation_validity_pct = (
        float(citations_valid) / float(citations_total) if citations_total else 0.0
    )

    safe_stop_ok, unsafe_hits = _safe_stop_check(answer.response, list(answer.safety_notes))
    for phrase in unsafe_hits:
        hallucination_flags.append(f"unsafe_phrase:{phrase}")

    reasons: list[str] = []
    if citation_coverage < float(cfg["citation_coverage_threshold"]):
        reasons.append("citation_coverage_below_threshold")
    if citation_validity_pct < 1.0:
        reasons.append("invalid_citation_url")
    if unsupported > 0:
        reasons.append("unsupported_claims")
    if not safe_stop_ok:
        reasons.append("safe_stop_constraint_failed")

    passed = not reasons

    updated_answer = AnswerModel(
        provider=answer.provider,
        question=answer.question,
        response=answer.response,
        claims=claims,
        citations=answer.citations,
        safety_notes=answer.safety_notes,
        retrieval_attempt=answer.retrieval_attempt,
        mode=answer.mode,
    )
    report = QualityGateModel(
        passed=passed,
        citation_coverage=round(citation_coverage, 3),
        citation_threshold=float(cfg["citation_coverage_threshold"]),
        citation_validity_pct=round(citation_validity_pct, 3),
        unsupported_claims=unsupported,
        hallucination_flags=hallucination_flags,
        safe_stop_ok=safe_stop_ok,
        reasons=reasons,
    )

    updates: StateGraphState = {
        "answer": _model_dump(updated_answer),
        "quality_report": _model_dump(report),
    }
    if not passed:
        updates["last_retrieval_status"] = "low_relevance"
    return updates


def _quality_edge(state: StateGraphState) -> Literal["pass", "retry", "stop"]:
    quality = state.get("quality_report") or {}
    if bool(quality.get("passed")):
        return "pass"
    attempts = int(state.get("retrieval_attempt", 0))
    max_attempts = int(state.get("max_retrieval_attempts", 0))
    if attempts < max_attempts:
        return "retry"
    return "stop"


def _safe_stop_node(state: StateGraphState) -> StateGraphState:
    cfg = state.get("config") or _default_config()
    runtime_mode = _normalize_runtime_mode(cfg.get("runtime_mode"))
    answer_payload = state.get("answer") or {}
    answer = _coerce_answer(
        answer_payload,
        provider=str(state.get("provider") or "ollama"),
        question=state["question"],
        retrieval_attempt=int(state.get("retrieval_attempt", 0)),
        mode=runtime_mode,
    )

    answer.response = (
        "Safe stop: quality gates did not pass after maximum retrieval retries. "
        "No final grounded answer was emitted."
    )
    answer.safety_notes = [*answer.safety_notes, SAFE_STOP_NOTE]
    return {"answer": _model_dump(answer), "final_status": "safe_stop"}


def _finalize_node(state: StateGraphState) -> StateGraphState:
    provider = _normalize_provider(state.get("provider"))
    cfg = state.get("config") or _default_config()
    runtime_mode = _normalize_runtime_mode(cfg.get("runtime_mode"))
    route_payload = state.get("route_decision") or {
        "decision": "answerable_now",
        "reason": "fallback",
        "clarification_question": None,
    }
    answer_payload = state.get("answer") or {}
    quality_payload = state.get("quality_report") or {
        "passed": False,
        "citation_coverage": 0.0,
        "citation_threshold": float(state["config"]["citation_coverage_threshold"]),
        "citation_validity_pct": 0.0,
        "unsupported_claims": 0,
        "hallucination_flags": [],
        "safe_stop_ok": False,
        "reasons": ["missing_quality_report"],
    }

    final = FinalOutputModel(
        provider=provider,
        run_id=str(state["run_id"]),
        question=state["question"],
        active_query=str(state.get("active_query") or state["question"]),
        route_decision=RouteDecisionModel(**route_payload),
        answer=_coerce_answer(
            answer_payload,
            provider=provider,
            question=state["question"],
            retrieval_attempt=int(state.get("retrieval_attempt", 0)),
            mode=runtime_mode,
        ),
        quality=QualityGateModel(**quality_payload),
        retrieval_attempts=int(state.get("retrieval_attempt", 0)),
        query_history=list(state.get("query_history", [])),
        trace=list(state.get("trace", [])),
        node_timings=list(state.get("node_timings", [])),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    final_payload = _model_dump(final)
    partition = str(state.get("active_partition") or _resolve_active_partition(state["question"], cfg))

    artifact_dir = Path(str(state.get("artifact_dir") or ""))
    if artifact_dir:
        _write_json(artifact_dir / "final_output.json", final_payload)
        _write_json(artifact_dir / "node_timings.json", state.get("node_timings", []))
        _write_json(
            artifact_dir / "run_summary.json",
            {
                "provider": final_payload["provider"],
                "run_id": final_payload["run_id"],
                "question": final_payload["question"],
                "route_decision": final_payload["route_decision"]["decision"],
                "passed": final_payload["quality"]["passed"],
                "retrieval_attempts": final_payload["retrieval_attempts"],
                "active_partition": partition,
                "cache_events": list(state.get("cache_events", [])),
                "cache_final_hit": bool(state.get("cache_final_hit")),
                "cache_retrieval_hit": bool(state.get("cache_retrieval_hit")),
                "artifact_dir": str(artifact_dir),
            },
        )

    if bool(final_payload.get("quality", {}).get("passed")):
        response_key = _response_cache_key(provider, partition, state["question"], cfg)
        _cache_put(
            cfg,
            table="response_cache",
            key=response_key,
            payload={
                "answer": dict(state.get("answer") or {}),
                "quality_report": dict(state.get("quality_report") or {}),
                "retrieval_candidates": list(state.get("retrieval_candidates") or []),
                "reranked_docs": list(state.get("reranked_docs") or []),
            },
            ttl_s=int(cfg.get("response_cache_ttl_s", 3600)),
            max_entries=int(cfg.get("cache_max_entries", 2000)),
        )

    raw_status = str(state.get("final_status") or "").strip().lower()
    if raw_status in {"", "running"}:
        status = "passed" if final.quality.passed else "failed"
    else:
        status = raw_status
    return {
        "final_output": final_payload,
        "final_status": status,
        "cache_events": list(state.get("cache_events") or []),
    }


def build_stategraph_app(
    *,
    provider: ProviderName = "ollama",
    config: dict[str, Any] | None = None,
):
    """Build and compile the advanced shared StateGraph app."""

    cfg = _merge_config({"provider": provider, **(config or {})})

    graph: StateGraph = StateGraph(StateGraphState)
    graph.add_node("init_run", _instrument_node("init_run", _init_run_node))
    graph.add_node("route_query", _instrument_node("route_query", _route_query_node))
    graph.add_node("ask_clarification", _instrument_node("ask_clarification", _clarification_node))
    graph.add_node("rewrite_query", _instrument_node("rewrite_query", _rewrite_query_node))
    graph.add_node(
        "retrieve_parallel",
        _instrument_node("retrieve_parallel", _retrieve_parallel_node),
    )
    graph.add_node("rerank_results", _instrument_node("rerank_results", _rerank_node))
    graph.add_node(
        "generate_structured_answer",
        _instrument_node("generate_structured_answer", _generate_answer_node),
    )
    graph.add_node("quality_gate", _instrument_node("quality_gate", _quality_gate_node))
    graph.add_node("safe_stop", _instrument_node("safe_stop", _safe_stop_node))
    graph.add_node("finalize", _instrument_node("finalize", _finalize_node))

    graph.add_edge(START, "init_run")
    graph.add_edge("init_run", "route_query")
    graph.add_conditional_edges(
        "route_query",
        _route_query_edge,
        {
            "answerable_now": "retrieve_parallel",
            "needs_clarification": "ask_clarification",
            "needs_retrieval_retry": "rewrite_query",
        },
    )
    graph.add_edge("rewrite_query", "retrieve_parallel")
    graph.add_edge("retrieve_parallel", "rerank_results")
    graph.add_edge("rerank_results", "generate_structured_answer")
    graph.add_edge("generate_structured_answer", "quality_gate")
    graph.add_conditional_edges(
        "quality_gate",
        _quality_edge,
        {"pass": "finalize", "retry": "route_query", "stop": "safe_stop"},
    )
    graph.add_edge("ask_clarification", "finalize")
    graph.add_edge("safe_stop", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


def run_stategraph(
    question: str,
    *,
    provider: ProviderName = "ollama",
    config: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> StateGraphState:
    """Run one question through the advanced shared StateGraph workflow."""

    cfg = _merge_config({"provider": provider, **(config or {})})
    app = build_stategraph_app(provider=provider, config=cfg)
    initial: StateGraphState = {
        "question": question,
        "config": cfg,
        "provider": provider,
    }
    if run_id:
        initial["run_id"] = run_id
    result: StateGraphState = app.invoke(initial)  # type: ignore[assignment]
    return result


I81_EVAL_QUESTIONS: tuple[str, ...] = (
    "Which states does Interstate 81 pass through from south to north?",
    "I am driving I-81 near Hagerstown, what Appalachian geology context should I notice?",
    "What does basalt tell me about volcanic history along an I-81 style geology stop?",
    "Near I-81 in Virginia, how can I describe Valley and Ridge landscape patterns?",
    "Give me a short, citation-backed geology briefing for an I-81 roadside stop.",
    "What safe pull-off guidance should I follow before observing roadcut rock layers on I-81?",
    "How would you explain Appalachians uplift and erosion in plain language for I-81 travelers?",
    "What two field observations can confirm mixed rock types in the Appalachian region?",
    "Summarize why Interstate 81 is relevant when discussing eastern U.S. corridor geology.",
    "If I only have 10 minutes, what geology evidence should I prioritize near I-81?",
    "How can I compare igneous versus sedimentary cues using I-81 roadcut examples?",
    "What citation-backed facts should a student note about Appalachian tectonic history?",
    "Give me a safe, grounded script to narrate a quick roadside geology stop near I-81.",
    "Explain rounded ridges and valleys along the Appalachians for first-time observers.",
    "What does columnar jointing in basalt indicate, and why does it matter for interpretation?",
    "Which I-81 corridor states are most relevant for a geology-themed road trip outline?",
    "Provide three short claims with Wikipedia citations about Appalachian geology and I-81.",
    "How should I answer uncertainty if evidence is weak during an I-81 geology Q&A?",
    "Create a concise answer with citations and safety note for an I-81 geology question.",
    "What should I look for first at a legal pull-off when discussing Appalachian rock history?",
)


def run_i81_eval_harness(
    *,
    provider: ProviderName = "ollama",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run 20 fixed I-81 questions and emit eval metrics + artifacts."""

    cfg = _merge_config({"provider": provider, **(config or {})})
    eval_id = datetime.now(timezone.utc).strftime("eval_%Y%m%dT%H%M%SZ")
    eval_root = Path(cfg["artifact_root"]) / provider / "eval_runs" / eval_id
    eval_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    citation_validity_scores: list[float] = []
    pass_count = 0

    for idx, question in enumerate(I81_EVAL_QUESTIONS, start=1):
        run_name = f"q{idx:02d}"
        run_cfg = {**cfg, "artifact_root": str(eval_root / "questions")}
        start = time.perf_counter()
        state = run_stategraph(question, provider=provider, config=run_cfg, run_id=run_name)
        latency_ms = (time.perf_counter() - start) * 1000.0

        final = state.get("final_output") or {}
        quality = final.get("quality") or {}
        route = (final.get("route_decision") or {}).get("decision", "")
        answer = final.get("answer") or {}
        citations = answer.get("citations") or []
        valid = 0
        for c in citations:
            if str(c.get("url", "")).startswith("https://en.wikipedia.org/wiki/"):
                valid += 1
        citation_validity = float(valid) / float(len(citations) or 1)
        passed = bool(quality.get("passed")) and route == "answerable_now"

        latencies_ms.append(latency_ms)
        citation_validity_scores.append(citation_validity)
        if passed:
            pass_count += 1

        row = {
            "id": run_name,
            "question": question,
            "latency_ms": round(latency_ms, 2),
            "route_decision": route,
            "quality_passed": bool(quality.get("passed")),
            "citation_validity_pct": round(citation_validity * 100.0, 2),
            "pass": passed,
            "artifact_dir": str((eval_root / "questions" / run_name).resolve()),
        }
        rows.append(row)

    summary = {
        "provider": provider,
        "eval_id": eval_id,
        "question_count": len(I81_EVAL_QUESTIONS),
        "avg_latency_ms": round(statistics.mean(latencies_ms), 2),
        "median_latency_ms": round(statistics.median(latencies_ms), 2),
        "citation_validity_pct": round(statistics.mean(citation_validity_scores) * 100.0, 2),
        "pass_rate_pct": round((pass_count / len(I81_EVAL_QUESTIONS)) * 100.0, 2),
        "pass_count": pass_count,
        "fail_count": len(I81_EVAL_QUESTIONS) - pass_count,
    }

    report = {"summary": summary, "rows": rows}
    _write_json(eval_root / "eval_report.json", report)

    # Human-friendly markdown summary for notebook reviews.
    md_lines = [
        f"# I-81 Eval Report ({provider}, {eval_id})",
        "",
        f"- Questions: {summary['question_count']}",
        f"- Avg latency: {summary['avg_latency_ms']} ms",
        f"- Median latency: {summary['median_latency_ms']} ms",
        f"- Citation validity: {summary['citation_validity_pct']}%",
        f"- Pass rate: {summary['pass_rate_pct']}% ({summary['pass_count']}/{summary['question_count']})",
        "",
        "| id | latency_ms | route | citation_validity_pct | pass |",
        "|---|---:|---|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['id']} | {row['latency_ms']} | {row['route_decision']} | "
            f"{row['citation_validity_pct']} | {str(row['pass']).lower()} |"
        )
    (eval_root / "eval_report.md").write_text("\n".join(md_lines).rstrip() + "\n")

    report["artifact_root"] = str(eval_root.resolve())
    return report


def _quality_pass_rate_pct(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    passed = sum(1 for row in rows if bool(row.get("quality_passed")))
    return round((passed / len(rows)) * 100.0, 2)


def _net_gain_vs_baseline(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    cand = candidate.get("summary") or {}
    base = baseline.get("summary") or {}
    cand_rows = list(candidate.get("rows") or [])
    base_rows = list(baseline.get("rows") or [])
    cand_quality = _quality_pass_rate_pct(cand_rows)
    base_quality = _quality_pass_rate_pct(base_rows)

    cand_pass = float(cand.get("pass_rate_pct") or 0.0)
    base_pass = float(base.get("pass_rate_pct") or 0.0)
    cand_citation = float(cand.get("citation_validity_pct") or 0.0)
    base_citation = float(base.get("citation_validity_pct") or 0.0)
    cand_latency = float(cand.get("avg_latency_ms") or 1.0)
    base_latency = float(base.get("avg_latency_ms") or 1.0)
    latency_ratio = cand_latency / max(base_latency, 1.0)

    pass_gain = cand_pass - base_pass
    quality_gain = cand_quality - base_quality
    citation_ok = cand_citation >= base_citation
    latency_ok = latency_ratio <= 1.10
    gained = (pass_gain >= 0.5) or (quality_gain >= 0.5)
    accepted = gained and citation_ok and latency_ok
    details = {
        "candidate_pass_rate_pct": round(cand_pass, 2),
        "baseline_pass_rate_pct": round(base_pass, 2),
        "candidate_quality_rate_pct": round(cand_quality, 2),
        "baseline_quality_rate_pct": round(base_quality, 2),
        "candidate_citation_validity_pct": round(cand_citation, 2),
        "baseline_citation_validity_pct": round(base_citation, 2),
        "candidate_avg_latency_ms": round(cand_latency, 2),
        "baseline_avg_latency_ms": round(base_latency, 2),
        "latency_ratio": round(latency_ratio, 3),
        "pass_gain_pct": round(pass_gain, 2),
        "quality_gain_pct": round(quality_gain, 2),
        "citation_non_regression": citation_ok,
        "latency_within_10pct": latency_ok,
        "accepted": accepted,
    }
    return accepted, details


def run_real_data_release_gate(
    *,
    provider: ProviderName = "ollama",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a hard release gate with realistic retrieval and threshold checks."""

    base_cfg = _merge_config(
        {
            "provider": provider,
            "runtime_mode": "realistic",
            "eval_mode": True,
            "retrieval_backend": "persistent",
            "allow_on_demand_index_build": False,
            **(config or {}),
        }
    )
    candidate_report = run_i81_eval_harness(provider=provider, config=base_cfg)
    candidate_summary = candidate_report["summary"]
    candidate_quality_rate = _quality_pass_rate_pct(candidate_report["rows"])

    thresholds = {
        "min_pass_rate_pct": float(base_cfg.get("eval_min_pass_rate_pct", 90.0)),
        "min_citation_validity_pct": float(base_cfg.get("eval_min_citation_validity_pct", 100.0)),
        "min_quality_rate_pct": float(base_cfg.get("eval_min_quality_rate_pct", 90.0)),
    }
    checks = {
        "pass_rate": float(candidate_summary["pass_rate_pct"]) >= thresholds["min_pass_rate_pct"],
        "citation_validity": float(candidate_summary["citation_validity_pct"])
        >= thresholds["min_citation_validity_pct"],
        "quality_rate": float(candidate_quality_rate) >= thresholds["min_quality_rate_pct"],
    }

    strict_cap = int(base_cfg.get("strict_top_k_cap", 8))
    candidate_max_k = max(
        int(base_cfg.get("top_k_faiss", 1)),
        int(base_cfg.get("top_k_keyword", 1)),
        int(base_cfg.get("top_k_rerank", 1)),
    )
    top_k_increase_check = {"required": False, "accepted": True}
    if candidate_max_k > strict_cap:
        top_k_increase_check["required"] = True
        baseline_cfg = _merge_config(
            {
                **base_cfg,
                "allow_top_k_experiments": False,
                "top_k_faiss": strict_cap,
                "top_k_keyword": strict_cap,
                "top_k_rerank": strict_cap,
            }
        )
        baseline_report = run_i81_eval_harness(provider=provider, config=baseline_cfg)
        accepted, details = _net_gain_vs_baseline(candidate_report, baseline_report)
        top_k_increase_check = {
            "required": True,
            "accepted": accepted,
            "candidate_max_top_k": candidate_max_k,
            "strict_top_k_cap": strict_cap,
            "details": details,
            "baseline_artifact_root": baseline_report.get("artifact_root"),
        }
        checks["top_k_net_gain"] = accepted
    else:
        checks["top_k_net_gain"] = True

    failed_checks = [name for name, ok in checks.items() if not bool(ok)]
    passed = not failed_checks
    gate = {
        "provider": provider,
        "passed": passed,
        "failed_checks": failed_checks,
        "thresholds": thresholds,
        "metrics": {
            "pass_rate_pct": float(candidate_summary["pass_rate_pct"]),
            "citation_validity_pct": float(candidate_summary["citation_validity_pct"]),
            "quality_rate_pct": float(candidate_quality_rate),
            "avg_latency_ms": float(candidate_summary["avg_latency_ms"]),
            "median_latency_ms": float(candidate_summary["median_latency_ms"]),
        },
        "checks": checks,
        "top_k_increase_check": top_k_increase_check,
        "candidate_artifact_root": candidate_report.get("artifact_root"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    artifact_root = Path(str(candidate_report.get("artifact_root") or ""))
    if artifact_root:
        _write_json(artifact_root / "release_gate.json", gate)
    return gate


def run_i81_eval_harness_all_providers(
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run eval harness for ollama + vertex + databricks and aggregate results."""

    reports: dict[str, dict[str, Any]] = {}
    for provider in SUPPORTED_PROVIDERS:
        reports[provider] = run_i81_eval_harness(provider=provider, config=config)

    aggregate_rows: list[dict[str, Any]] = []
    for provider, report in reports.items():
        summary = report["summary"]
        aggregate_rows.append(
            {
                "provider": provider,
                "question_count": summary["question_count"],
                "avg_latency_ms": summary["avg_latency_ms"],
                "citation_validity_pct": summary["citation_validity_pct"],
                "pass_rate_pct": summary["pass_rate_pct"],
                "pass_count": summary["pass_count"],
                "fail_count": summary["fail_count"],
                "artifact_root": report["artifact_root"],
            }
        )

    return {"providers": reports, "aggregate": aggregate_rows}
