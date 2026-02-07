"""Smoke test for Vertex AI + LangChain wiring."""

from __future__ import annotations

import argparse
import os
import platform
import sys


def _maybe_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv()


def main() -> int:
    """Run local import checks and optional Vertex AI API calls."""
    parser = argparse.ArgumentParser(
        description="Sanity checks for local Vertex AI + LangChain wiring."
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Make tiny (billable) API calls to embeddings + LLM.",
    )
    args = parser.parse_args()

    _maybe_load_dotenv()

    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip() or "us-central1"
    llm_model = os.environ.get("VERTEX_LLM_MODEL", "").strip() or "gemini-flash-latest"
    embedding_model = (
        os.environ.get("VERTEX_EMBEDDING_MODEL", "").strip() or "text-embedding-005"
    )

    print("Environment")
    print(f"- python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"- GOOGLE_CLOUD_PROJECT: {project or '(unset)'}")
    print(f"- GOOGLE_CLOUD_LOCATION: {location}")
    print(f"- VERTEX_LLM_MODEL: {llm_model}")
    print(f"- VERTEX_EMBEDDING_MODEL: {embedding_model}")

    print("\nImports")
    import vertexai  # noqa: F401
    from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings  # noqa: F401

    print("- vertexai: ok")
    print("- langchain_google_vertexai: ok")

    if not args.api:
        print("\nSkipped API calls. Re-run with `--api` to do a tiny live check.")
        return 0

    if not project:
        print(
            "\nERROR: GOOGLE_CLOUD_PROJECT is required for --api. "
            "Set it in your environment or in a local .env file."
        )
        return 2

    import vertexai
    from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

    vertexai.init(project=project, location=location)

    print("\nAPI calls (billable)")
    embeddings = VertexAIEmbeddings(model_name=embedding_model)
    vec = embeddings.embed_query("hello world")
    print(f"- embeddings: ok (dim={len(vec)})")

    llm = ChatVertexAI(model_name=llm_model)
    msg = llm.invoke("Reply with exactly: ok")
    content = getattr(msg, "content", str(msg))
    print(f"- llm: ok (response={content!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
