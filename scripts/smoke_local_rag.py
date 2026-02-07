"""Local RAG smoke test for offline embeddings and retrieval."""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _maybe_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    val = os.environ.get(name, "").strip()
    return val if val else default


def _as_docs_from_wikipedia_loader(query: str, max_docs: int):
    from langchain_community.document_loaders import WikipediaLoader  # type: ignore

    return WikipediaLoader(query=query, load_max_docs=max_docs).load()


def _split_query_terms(query: str) -> list[str]:
    if " OR " not in query:
        return [query.strip()]
    parts = [p.strip() for p in query.split(" OR ")]
    return [p for p in parts if p]


@dataclass
class RequestBudget:
    """Simple request budget tracker."""

    max_requests: int
    used: int = 0

    def consume(self, n: int = 1) -> bool:
        if self.max_requests <= 0:
            return True
        if self.used + n > self.max_requests:
            return False
        self.used += n
        return True


def _as_docs_from_wikipedia_pkg(
    query: str,
    max_docs: int,
    *,
    min_interval_s: float,
    budget: RequestBudget | None,
    counts: dict[str, int],
):
    from langchain_core.documents import Document

    import wikipedia  # type: ignore

    titles: list[str] = []
    seen: set[str] = set()
    for term in _split_query_terms(query):
        if budget is not None and not budget.consume():
            break
        try:
            results = wikipedia.search(term, results=max_docs)
        except Exception:
            continue
        counts["search"] += 1
        for t in results:
            if t in seen:
                continue
            titles.append(t)
            seen.add(t)
            if len(titles) >= max_docs:
                break
        if len(titles) >= max_docs:
            break

    docs: list[Document] = []
    for t in titles[:max_docs]:
        if budget is not None and not budget.consume():
            break
        try:
            page = wikipedia.page(t, auto_suggest=False)
        except Exception:
            continue
        counts["page"] += 1
        content = getattr(page, "content", "") or ""
        summary = getattr(page, "summary", "") or ""
        url = getattr(page, "url", "") or ""
        title = getattr(page, "title", t) or t
        docs.append(
            Document(
                page_content=content.strip() or summary.strip(),
                metadata={"title": title, "summary": summary, "source": url},
            )
        )
        if min_interval_s > 0:
            time.sleep(min_interval_s)

    return docs


def _as_docs_from_fallback_data():
    from langchain_core.documents import Document

    from naturalist_companion import fallback_data

    pages = fallback_data.fallback_wiki_pages()
    return [
        Document(
            page_content=f"{p.get('summary','')}\n\n{p.get('content','')}".strip(),
            metadata={
                "title": str(p.get("title") or ""),
                "summary": str(p.get("summary") or ""),
                "source": str(p.get("url") or ""),
                "pageid": int(p.get("pageid") or 0),
            },
        )
        for p in pages.values()
    ]


def _load_docs(
    query: str,
    max_docs: int,
    *,
    fallback_data: bool,
    prefer_wikipedia_pkg: bool,
    min_interval_s: float,
    budget: RequestBudget | None,
    counts: dict[str, int],
) -> list[Any]:
    if fallback_data:
        return _as_docs_from_fallback_data()

    if prefer_wikipedia_pkg:
        return _as_docs_from_wikipedia_pkg(
            query, max_docs, min_interval_s=min_interval_s, budget=budget, counts=counts
        )

    try:
        return _as_docs_from_wikipedia_loader(query, max_docs)
    except Exception:
        # Fallback to the lightweight `wikipedia` package path if langchain loader
        # isn't available or errors locally.
        return _as_docs_from_wikipedia_pkg(
            query, max_docs, min_interval_s=min_interval_s, budget=budget, counts=counts
        )


def _build_vector_store(docs: list[Any], *, embedding_model: str, collection_name: str, persist_dir: str | None):
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )

    kwargs: dict[str, Any] = {"collection_name": collection_name}
    if persist_dir:
        kwargs["persist_directory"] = persist_dir

    return Chroma.from_documents(docs, embeddings, **kwargs)


def _print_results(results: Iterable[Any]) -> list[dict[str, str]]:
    citations: list[dict[str, str]] = []
    for i, doc in enumerate(results, start=1):
        meta = getattr(doc, "metadata", {}) or {}
        title = str(meta.get("title") or meta.get("page") or f"result_{i}")
        url = str(meta.get("source") or meta.get("url") or "")
        citations.append({"n": str(i), "title": title, "url": url})

        snippet = " ".join(str(getattr(doc, "page_content", "") or "").split())
        snippet = textwrap.shorten(snippet, width=240, placeholder="…")
        print(f"[{i}] {title}")
        if url:
            print(f"    {url}")
        if snippet:
            print(f"    {snippet}")
        print("")

    return citations


def _ollama_generate(*, base_url: str, model: str, prompt: str, timeout_s: float) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")
        except Exception:
            pass
        raise RuntimeError(f"Ollama HTTP {e.code}: {detail or e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach Ollama at {base_url} ({e.reason})") from e

    try:
        parsed = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Unexpected Ollama response: {raw[:200]!r}") from e

    return str(parsed.get("response") or "").strip()


def _build_context(results: list[Any], *, max_chars: int) -> tuple[str, list[dict[str, str]]]:
    blocks: list[str] = []
    citations: list[dict[str, str]] = []
    used = 0

    for i, doc in enumerate(results, start=1):
        meta = getattr(doc, "metadata", {}) or {}
        title = str(meta.get("title") or meta.get("page") or f"result_{i}")
        url = str(meta.get("source") or meta.get("url") or "")
        text = str(getattr(doc, "page_content", "") or "").strip()

        header = f"[{i}] {title}\nURL: {url}\n"
        remaining = max_chars - used
        if remaining <= 0:
            break
        body = text[: max(0, remaining - len(header))]
        block = (header + body).strip()
        if not block:
            continue
        blocks.append(block)
        citations.append({"n": str(i), "title": title, "url": url})
        used += len(block) + 2

    return "\n\n".join(blocks).strip(), citations


def main() -> int:
    """Run the local RAG pipeline and print a summary."""
    parser = argparse.ArgumentParser(
        description="Local smoke test: Wikipedia → sentence-transformers embeddings → Chroma → retrieval; optional answer via Ollama."
    )
    parser.add_argument(
        "--wiki-query",
        default="",
        help="Wikipedia search query (defaults to env WIKIPEDIA_QUERY or a small demo query).",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Max Wikipedia docs to load (defaults to env WIKIPEDIA_MAX_DOCS or 5).",
    )
    parser.add_argument(
        "--question",
        default="",
        help="Question to run through retrieval (defaults to env LOCAL_RAG_QUESTION or a demo question).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Retriever top-k (defaults to env WIKIPEDIA_TOP_K or 3).",
    )
    parser.add_argument(
        "--embedding-model",
        default="",
        help="Sentence-transformers model name (defaults to env LOCAL_EMBEDDING_MODEL or all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--collection",
        default="naturalist-companion-local-smoke",
        help="Chroma collection name (default: naturalist-companion-local-smoke).",
    )
    parser.add_argument(
        "--persist-dir",
        default="",
        help="Persist Chroma to this directory (default: in-memory; set e.g. ./.chroma-local-smoke).",
    )
    parser.add_argument(
        "--fallback-data",
        action="store_true",
        help="Use the repo's offline fallback dataset instead of hitting Wikipedia.",
    )
    parser.add_argument(
        "--prefer-wikipedia-package",
        action="store_true",
        help="Use the `wikipedia` package instead of LangChain WikipediaLoader (lets us rate-limit).",
    )
    parser.add_argument(
        "--min-interval-s",
        type=float,
        default=1.5,
        help="Minimum seconds between Wikipedia page fetches when using the `wikipedia` package (default: 1.5).",
    )
    parser.add_argument(
        "--request-budget",
        type=int,
        default=20,
        help="Max Wikipedia requests per run when using the `wikipedia` package (default: 20). Use 0 to disable.",
    )

    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Also ask a local Ollama model to answer using retrieved context.",
    )
    parser.add_argument(
        "--ollama-url",
        default="",
        help="Ollama base URL (default: env OLLAMA_URL or http://localhost:11434).",
    )
    parser.add_argument(
        "--ollama-model",
        default="",
        help="Ollama model name (default: env OLLAMA_MODEL or llama3.2:3b).",
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=6000,
        help="Max characters of retrieved context to send to Ollama (default: 6000).",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=60.0,
        help="Timeout for the Ollama HTTP request (default: 60s).",
    )
    args = parser.parse_args()

    _maybe_load_dotenv()

    wiki_query = (args.wiki_query or _env_str("WIKIPEDIA_QUERY", "")).strip()
    if not wiki_query:
        wiki_query = "Interstate 81 OR Shenandoah Valley OR Roanoke OR Appalachian Mountains"

    max_docs = args.max_docs or _env_int("WIKIPEDIA_MAX_DOCS", 5)
    top_k = args.top_k or _env_int("WIKIPEDIA_TOP_K", 3)
    question = (args.question or _env_str("LOCAL_RAG_QUESTION", "")).strip()
    if not question:
        question = "What states does Interstate 81 run through?"

    embedding_model = (args.embedding_model or _env_str("LOCAL_EMBEDDING_MODEL", "")).strip()
    if not embedding_model:
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    persist_dir = args.persist_dir.strip() or None

    print("Local RAG smoke test")
    print(f"- wiki_query: {wiki_query!r}" if not args.fallback_data else "- wiki_query: (fallback data)")
    print(f"- max_docs: {max_docs}")
    print(f"- embedding_model: {embedding_model}")
    print(f"- chroma: {'persist=' + str(Path(persist_dir)) if persist_dir else 'in-memory'}")
    print(f"- collection: {args.collection}")
    print(f"- top_k: {top_k}")
    print(f"- question: {question!r}")

    t0 = time.time()
    request_counts = {"search": 0, "page": 0}
    budget = None
    if args.prefer_wikipedia_package and not args.fallback_data:
        budget = RequestBudget(max_requests=max(0, args.request_budget))

    docs = _load_docs(
        wiki_query,
        max_docs,
        fallback_data=args.fallback_data,
        prefer_wikipedia_pkg=args.prefer_wikipedia_package,
        min_interval_s=max(0.0, args.min_interval_s),
        budget=budget,
        counts=request_counts,
    )
    if not docs:
        print(
            "\nERROR: loaded 0 documents. Try adjusting --wiki-query/--max-docs or run with --fallback-data."
        )
        return 2
    print(f"\nLoaded documents: {len(docs)} ({time.time() - t0:.2f}s)")
    if args.prefer_wikipedia_package and not args.fallback_data:
        budget_note = (
            f"{budget.used}/{budget.max_requests}"
            if budget is not None and budget.max_requests > 0
            else "unlimited"
        )
        print(
            f"Wikipedia request counts: search={request_counts['search']} "
            f"page={request_counts['page']} (budget {budget_note})"
        )
    elif not args.fallback_data:
        print("Wikipedia request counts: (not tracked; use --prefer-wikipedia-package to enable)")

    t1 = time.time()
    vector_store = _build_vector_store(
        docs,
        embedding_model=embedding_model,
        collection_name=args.collection,
        persist_dir=persist_dir,
    )
    print(f"Built vector store ({time.time() - t1:.2f}s)")

    t2 = time.time()
    results = vector_store.similarity_search(question, k=top_k)
    print(f"\nTop {min(top_k, len(results))} retrieval results ({time.time() - t2:.2f}s)\n")
    citations = _print_results(results)

    use_ollama = bool(args.ollama or args.ollama_model or os.environ.get("OLLAMA_MODEL"))
    if not use_ollama:
        print("Skipped Ollama. Re-run with `--ollama` to generate an answer locally.\n")
        return 0

    base_url = (args.ollama_url or _env_str("OLLAMA_URL", "")).strip() or "http://localhost:11434"
    model = (args.ollama_model or _env_str("OLLAMA_MODEL", "")).strip() or "llama3.2:3b"

    context, used_citations = _build_context(results, max_chars=max(0, args.context_chars))
    if not context:
        print("\nERROR: no context available to send to Ollama.")
        return 2

    prompt = f"""You are a helpful assistant.

Answer the question using ONLY the context below. If the answer is not in the context, say you don't know based on the provided context.

Question:
{question}

Context:
{context}

Rules:
- Do not use outside knowledge.
- Keep it concise.
- End with a line like: Citations: [1], [2]
"""

    print(f"Ollama answer (model={model!r}, url={base_url!r})\n")
    try:
        answer = _ollama_generate(
            base_url=base_url, model=model, prompt=prompt, timeout_s=max(1.0, args.timeout_s)
        )
    except Exception as e:
        print(f"ERROR: {e}")
        print("\nTips:")
        print("- Install Ollama: https://ollama.com")
        print("- Start it: `ollama serve`")
        print(f"- Pull the model: `ollama pull {model}`")
        return 2

    print(answer.strip() + "\n")
    print("Sources (retrieval)")
    for c in used_citations or citations:
        if c.get("url"):
            print(f"- [{c['n']}] {c.get('title','')} — {c['url']}")
        else:
            print(f"- [{c['n']}] {c.get('title','')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
