"""Run the advanced shared StateGraph eval harness."""

from __future__ import annotations

import argparse

from naturalist_companion.stategraph_shared import (
    run_i81_eval_harness,
    run_i81_eval_harness_all_providers,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run stategraph eval harness (20 fixed I-81 questions).")
    parser.add_argument(
        "--artifact-root",
        default="out/stategraph",
        help="Root directory for run artifacts (default: out/stategraph).",
    )
    parser.add_argument(
        "--provider",
        choices=("ollama", "vertex", "databricks"),
        default="ollama",
        help="Provider mode to evaluate (default: ollama).",
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
        help="Persistent retrieval store root (default: out/stategraph_store).",
    )
    parser.add_argument(
        "--retrieval-backend",
        choices=("persistent", "in_memory"),
        default="persistent",
        help="Retrieval backend mode (default: persistent).",
    )
    parser.add_argument(
        "--allow-on-demand-index-build",
        action="store_true",
        help="Allow request-time index build if partition index is missing.",
    )
    parser.add_argument(
        "--all-providers",
        action="store_true",
        help="Run evals for ollama + vertex + databricks.",
    )
    parser.add_argument(
        "--max-retrieval-attempts",
        type=int,
        default=3,
        help="Maximum retrieval attempts per question (default: 3).",
    )
    parser.add_argument(
        "--citation-coverage-threshold",
        type=float,
        default=0.80,
        help="Quality gate threshold for claim citation coverage (default: 0.80).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Top-k for FAISS/keyword/rerank (default: 6).",
    )
    parser.add_argument(
        "--strict-top-k-cap",
        type=int,
        default=8,
        help="Strict top-k cap for online path (default: 8).",
    )
    parser.add_argument(
        "--context-token-budget",
        type=int,
        default=900,
        help="Context token budget (default: 900).",
    )
    args = parser.parse_args()

    top_k = max(1, int(args.top_k))
    cfg = {
        "artifact_root": args.artifact_root,
        "store_root": args.store_root,
        "runtime_mode": args.runtime_mode,
        "retrieval_backend": args.retrieval_backend,
        "allow_on_demand_index_build": bool(args.allow_on_demand_index_build),
        "max_retrieval_attempts": max(1, args.max_retrieval_attempts),
        "citation_coverage_threshold": max(0.0, min(1.0, args.citation_coverage_threshold)),
        "top_k_faiss": top_k,
        "top_k_keyword": top_k,
        "top_k_rerank": top_k,
        "strict_top_k_cap": max(1, int(args.strict_top_k_cap)),
        "context_token_budget": max(120, int(args.context_token_budget)),
    }
    print("StateGraph I-81 eval harness")
    if args.all_providers:
        report = run_i81_eval_harness_all_providers(config=cfg)
        for row in report["aggregate"]:
            print(
                f"- provider={row['provider']} questions={row['question_count']} "
                f"avg_latency_ms={row['avg_latency_ms']} citation_validity_pct={row['citation_validity_pct']} "
                f"pass_rate_pct={row['pass_rate_pct']} pass/fail={row['pass_count']}/{row['fail_count']}"
            )
            print(f"  artifact root: {row['artifact_root']}")
        return 0

    report = run_i81_eval_harness(provider=args.provider, config=cfg)
    summary = report["summary"]
    print(f"- provider: {summary['provider']}")
    print(f"- runtime mode: {args.runtime_mode}")
    print(f"- retrieval backend: {args.retrieval_backend}")
    print(f"- artifact root: {report['artifact_root']}")
    print(f"- questions: {summary['question_count']}")
    print(f"- avg latency (ms): {summary['avg_latency_ms']}")
    print(f"- median latency (ms): {summary['median_latency_ms']}")
    print(f"- citation validity (%): {summary['citation_validity_pct']}")
    print(f"- pass rate (%): {summary['pass_rate_pct']}")
    print(f"- pass/fail: {summary['pass_count']}/{summary['fail_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
