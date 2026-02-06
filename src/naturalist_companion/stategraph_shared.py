"""Provider-agnostic StateGraph backend: routed RAG with retries, gates, and evals.

This module is intentionally offline-friendly so it can run in local environments
without external APIs while still exercising real graph behavior:

- Typed state (`TypedDict`) plus structured outputs (`pydantic` models)
- Routing node with three outcomes:
  - answerable now
  - needs clarification
  - needs retrieval retry
- Parallel retrieval (FAISS + keyword BM25-style) followed by reranking
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
import re
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

import faiss
import numpy as np
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from . import mvp_data

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
    retrieval_attempt: int
    max_retrieval_attempts: int
    last_retrieval_status: Literal["not_run", "ok", "empty", "low_relevance"]
    route_decision: dict[str, Any]
    clarification_prompt: str
    retrieval_candidates: list[RetrievalHit]
    reranked_docs: list[RetrievalHit]
    answer: dict[str, Any]
    quality_report: dict[str, Any]
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
    mode: Literal["deterministic"] = "deterministic"


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
    }


def _normalize_provider(raw: Any) -> ProviderName:
    value = str(raw or "").strip().lower()
    if value in SUPPORTED_PROVIDERS:
        return value  # type: ignore[return-value]
    return "ollama"


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
    }
    float_keys = {"citation_coverage_threshold", "min_support_overlap", "min_rerank_score"}

    for key, value in overrides.items():
        if key not in cfg:
            continue
        if key == "provider":
            cfg[key] = _normalize_provider(value)  # type: ignore[typeddict-item]
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
            if 0.0 <= x <= 1.5:
                cfg[key] = x  # type: ignore[typeddict-item]
            continue
        if key == "artifact_root":
            text = str(value).strip()
            if text:
                cfg[key] = text  # type: ignore[typeddict-item]

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
) -> AnswerModel:
    merged = {
        "provider": provider,
        "question": question,
        "response": "",
        "claims": [],
        "citations": [],
        "safety_notes": [SAFE_STOP_NOTE],
        "retrieval_attempt": retrieval_attempt,
        "mode": "deterministic",
    }
    if payload:
        merged.update(payload)
    merged["provider"] = provider
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


def _build_runtime(*, embedding_dim: int) -> RetrievalRuntime:
    pages = mvp_data.minimal_wiki_pages()
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
        "route_decision": state.get("route_decision"),
        "retrieval_attempt": state.get("retrieval_attempt", 0),
        "max_retrieval_attempts": state.get("max_retrieval_attempts", 0),
        "last_retrieval_status": state.get("last_retrieval_status"),
        "retrieval_candidates": _trim_text(state.get("retrieval_candidates", []), limit),
        "reranked_docs": _trim_text(state.get("reranked_docs", []), limit),
        "answer": _trim_text(state.get("answer", {}), limit),
        "quality_report": state.get("quality_report"),
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
    run_id = str(state.get("run_id") or "")
    if not run_id:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        qhash = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
        run_id = f"run_{stamp}_{qhash}"

    artifact_dir = Path(cfg["artifact_root"]) / provider / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    return {
        "provider": provider,
        "question": question,
        "config": cfg,
        "run_id": run_id,
        "artifact_dir": str(artifact_dir),
        "active_query": question,
        "query_history": [question],
        "retrieval_attempt": 0,
        "max_retrieval_attempts": int(cfg["max_retrieval_attempts"]),
        "last_retrieval_status": "not_run",
        "retrieval_candidates": [],
        "reranked_docs": [],
        "final_status": "running",
    }


def _route_query_node(state: StateGraphState) -> StateGraphState:
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
    answer = AnswerModel(
        provider=str(state.get("provider") or "ollama"),
        question=state["question"],
        response=f"Need clarification before retrieval: {prompt}",
        claims=[],
        citations=[],
        safety_notes=[SAFE_STOP_NOTE],
        retrieval_attempt=int(state.get("retrieval_attempt", 0)),
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


def _retrieve_parallel_node(runtime: RetrievalRuntime):
    def _node(state: StateGraphState) -> StateGraphState:
        cfg = state["config"]
        query = str(state.get("active_query") or state["question"])
        faiss_k = int(cfg["top_k_faiss"])
        keyword_k = int(cfg["top_k_keyword"])
        attempt = int(state.get("retrieval_attempt", 0)) + 1

        with ThreadPoolExecutor(max_workers=2) as executor:
            faiss_future = executor.submit(_faiss_retrieve, runtime, query, faiss_k)
            keyword_future = executor.submit(_keyword_retrieve, runtime, query, keyword_k)
            faiss_hits = faiss_future.result()
            keyword_hits = keyword_future.result()

        candidates = [*faiss_hits, *keyword_hits]
        status: Literal["ok", "empty"] = "ok" if candidates else "empty"
        return {
            "retrieval_attempt": attempt,
            "retrieval_candidates": candidates,
            "last_retrieval_status": status,
        }

    return _node


def _rerank_node(state: StateGraphState) -> StateGraphState:
    cfg = state["config"]
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
    sentence = sentence.replace("Toy extract:", "").strip()
    return sentence[:220].rstrip(". ") + "."


def _generate_answer_node(state: StateGraphState) -> StateGraphState:
    docs = state.get("reranked_docs") or []
    attempt = int(state.get("retrieval_attempt", 0))
    provider = _normalize_provider(state.get("provider"))
    provider_label = PROVIDER_DISPLAY[provider]

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
    )
    return {"answer": _model_dump(answer)}


def _token_overlap_ratio(text_a: str, text_b: str) -> float:
    ta = set(_tokenize(text_a))
    tb = set(_tokenize(text_b))
    if not ta:
        return 0.0
    return float(len(ta & tb)) / float(len(ta))


def _quality_gate_node(state: StateGraphState) -> StateGraphState:
    cfg = state["config"]
    answer_payload = state.get("answer") or {}
    answer = _coerce_answer(
        answer_payload,
        provider=str(state.get("provider") or "ollama"),
        question=state["question"],
        retrieval_attempt=int(state.get("retrieval_attempt", 0)),
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

    response_l = answer.response.lower()
    safe_stop_ok = "legal pull-offs" in response_l
    if safe_stop_ok:
        for phrase in UNSAFE_PHRASES:
            if phrase in response_l:
                safe_stop_ok = False
                hallucination_flags.append(f"unsafe_phrase:{phrase}")
                break

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
    answer_payload = state.get("answer") or {}
    answer = _coerce_answer(
        answer_payload,
        provider=str(state.get("provider") or "ollama"),
        question=state["question"],
        retrieval_attempt=int(state.get("retrieval_attempt", 0)),
    )

    answer.response = (
        "Safe stop: quality gates did not pass after maximum retrieval retries. "
        "No final grounded answer was emitted."
    )
    answer.safety_notes = [*answer.safety_notes, SAFE_STOP_NOTE]
    return {"answer": _model_dump(answer), "final_status": "safe_stop"}


def _finalize_node(state: StateGraphState) -> StateGraphState:
    provider = _normalize_provider(state.get("provider"))
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
        ),
        quality=QualityGateModel(**quality_payload),
        retrieval_attempts=int(state.get("retrieval_attempt", 0)),
        query_history=list(state.get("query_history", [])),
        trace=list(state.get("trace", [])),
        node_timings=list(state.get("node_timings", [])),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    final_payload = _model_dump(final)

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
                "artifact_dir": str(artifact_dir),
            },
        )

    status = str(state.get("final_status") or ("passed" if final.quality.passed else "failed"))
    return {"final_output": final_payload, "final_status": status}


def build_stategraph_app(
    *,
    provider: ProviderName = "ollama",
    config: dict[str, Any] | None = None,
):
    """Build and compile the advanced shared StateGraph app."""

    cfg = _merge_config({"provider": provider, **(config or {})})
    runtime = _build_runtime(embedding_dim=cfg["embedding_dim"])

    graph: StateGraph = StateGraph(StateGraphState)
    graph.add_node("init_run", _instrument_node("init_run", _init_run_node))
    graph.add_node("route_query", _instrument_node("route_query", _route_query_node))
    graph.add_node("ask_clarification", _instrument_node("ask_clarification", _clarification_node))
    graph.add_node("rewrite_query", _instrument_node("rewrite_query", _rewrite_query_node))
    graph.add_node(
        "retrieve_parallel",
        _instrument_node("retrieve_parallel", _retrieve_parallel_node(runtime)),
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
