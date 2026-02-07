"""Run repeated StateGraph queries and print cache events for observability."""

from __future__ import annotations

import argparse

from naturalist_companion.stategraph_shared import run_stategraph


def main() -> int:
    parser = argparse.ArgumentParser(description="Observe StateGraph retrieval/response cache behavior.")
    parser.add_argument(
        "--provider",
        choices=("ollama", "vertex", "databricks"),
        default="ollama",
        help="Provider to run (default: ollama).",
    )
    parser.add_argument(
        "--question",
        default="I am on I-81 near Hagerstown and need citation backed geology context.",
        help="Question to execute repeatedly.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Number of repeated runs (default: 2).",
    )
    parser.add_argument(
        "--runtime-mode",
        choices=("deterministic", "realistic"),
        default="deterministic",
        help="Runtime mode (default: deterministic).",
    )
    parser.add_argument(
        "--store-root",
        default="out/stategraph_store",
        help="Persistent retrieval/cache root (default: out/stategraph_store).",
    )
    parser.add_argument(
        "--artifact-root",
        default="out/stategraph_cache_demo",
        help="Artifact root for repeated runs (default: out/stategraph_cache_demo).",
    )
    args = parser.parse_args()

    cfg = {
        "runtime_mode": args.runtime_mode,
        "retrieval_backend": "persistent",
        "allow_on_demand_index_build": False,
        "store_root": args.store_root,
        "artifact_root": args.artifact_root,
    }

    print("StateGraph cache observability run")
    print(f"- provider: {args.provider}")
    print(f"- runtime mode: {args.runtime_mode}")
    print(f"- question: {args.question}")
    print(f"- runs: {max(1, int(args.runs))}")

    for idx in range(1, max(1, int(args.runs)) + 1):
        run_id = f"cache_obs_{idx:02d}"
        state = run_stategraph(args.question, provider=args.provider, config=cfg, run_id=run_id)
        events = list(state.get("cache_events") or [])
        print(f"  run {idx}: final_status={state.get('final_status')} cache_events={events}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
