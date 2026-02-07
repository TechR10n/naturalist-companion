"""Run real-data release gate for the shared StateGraph pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from naturalist_companion.stategraph_shared import run_real_data_release_gate


def main() -> int:
    if sys.platform == "darwin":
        # macOS toolchains can load duplicate OpenMP runtimes across FAISS/torch stacks.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    parser = argparse.ArgumentParser(description="Run StateGraph real-data release gate.")
    parser.add_argument(
        "--provider",
        choices=("ollama", "vertex", "databricks"),
        default="ollama",
        help="Provider to gate (default: ollama).",
    )
    parser.add_argument(
        "--artifact-root",
        default="out/stategraph_release_gate",
        help="Artifact root for eval outputs (default: out/stategraph_release_gate).",
    )
    parser.add_argument(
        "--store-root",
        default="out/stategraph_store",
        help="Persistent retrieval store root (default: out/stategraph_store).",
    )
    parser.add_argument(
        "--max-retrieval-attempts",
        type=int,
        default=3,
        help="Max retrieval attempts per question (default: 3).",
    )
    parser.add_argument(
        "--citation-coverage-threshold",
        type=float,
        default=0.80,
        help="Quality gate threshold for claim citation coverage (default: 0.80).",
    )
    parser.add_argument(
        "--min-pass-rate-pct",
        type=float,
        default=90.0,
        help="Release threshold for pass rate percentage (default: 90).",
    )
    parser.add_argument(
        "--min-citation-validity-pct",
        type=float,
        default=100.0,
        help="Release threshold for citation validity percentage (default: 100).",
    )
    parser.add_argument(
        "--min-quality-rate-pct",
        type=float,
        default=90.0,
        help="Release threshold for quality-pass percentage (default: 90).",
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
        help="Strict online top-k cap (default: 8).",
    )
    parser.add_argument(
        "--allow-top-k-experiments",
        action="store_true",
        help="Allow top-k above cap; promotion still requires net-gain check.",
    )
    parser.add_argument(
        "--context-token-budget",
        type=int,
        default=900,
        help="Context token budget for generation prompt (default: 900).",
    )
    parser.add_argument(
        "--refresh-cadence",
        choices=("daily", "weekly"),
        default="daily",
        help="Cadence used by partition freshness checks (default: daily).",
    )
    args = parser.parse_args()

    top_k = max(1, int(args.top_k))
    cfg = {
        "artifact_root": args.artifact_root,
        "store_root": args.store_root,
        "retrieval_backend": "persistent",
        "allow_on_demand_index_build": False,
        "max_retrieval_attempts": max(1, int(args.max_retrieval_attempts)),
        "citation_coverage_threshold": max(0.0, min(1.0, float(args.citation_coverage_threshold))),
        "eval_min_pass_rate_pct": max(0.0, min(100.0, float(args.min_pass_rate_pct))),
        "eval_min_citation_validity_pct": max(0.0, min(100.0, float(args.min_citation_validity_pct))),
        "eval_min_quality_rate_pct": max(0.0, min(100.0, float(args.min_quality_rate_pct))),
        "top_k_faiss": top_k,
        "top_k_keyword": top_k,
        "top_k_rerank": top_k,
        "strict_top_k_cap": max(1, int(args.strict_top_k_cap)),
        "allow_top_k_experiments": bool(args.allow_top_k_experiments),
        "context_token_budget": max(120, int(args.context_token_budget)),
        "refresh_cadence": args.refresh_cadence,
    }
    try:
        gate = run_real_data_release_gate(provider=args.provider, config=cfg)
    except Exception as e:
        print("StateGraph release gate failed to execute")
        print(f"- error: {e}")
        print(
            "- hint: refresh realistic indexes first (`make stategraph-refresh-real`) "
            "and ensure provider endpoint credentials/runtime are available."
        )
        return 2

    print("StateGraph release gate")
    print(f"- provider: {gate['provider']}")
    print(f"- passed: {str(gate['passed']).lower()}")
    print(
        f"- metrics: pass_rate={gate['metrics']['pass_rate_pct']} "
        f"citation_validity={gate['metrics']['citation_validity_pct']} "
        f"quality_rate={gate['metrics']['quality_rate_pct']} "
        f"avg_latency_ms={gate['metrics']['avg_latency_ms']}"
    )
    print(
        f"- thresholds: pass_rate>={gate['thresholds']['min_pass_rate_pct']} "
        f"citation_validity>={gate['thresholds']['min_citation_validity_pct']} "
        f"quality_rate>={gate['thresholds']['min_quality_rate_pct']}"
    )
    if gate.get("failed_checks"):
        print(f"- failed checks: {', '.join(gate['failed_checks'])}")
    topk = gate.get("top_k_increase_check") or {}
    if topk.get("required"):
        print(
            f"- top-k increase check: accepted={str(topk.get('accepted')).lower()} "
            f"(candidate={topk.get('candidate_max_top_k')} cap={topk.get('strict_top_k_cap')})"
        )
        details = topk.get("details") or {}
        print(
            "  details: "
            f"pass_gain={details.get('pass_gain_pct')} "
            f"quality_gain={details.get('quality_gain_pct')} "
            f"latency_ratio={details.get('latency_ratio')}"
        )

    artifact_root = str(gate.get("candidate_artifact_root") or "").strip()
    if artifact_root:
        path = Path(artifact_root).resolve()
        print(f"- candidate artifacts: {path}")
        gate_path = path / "release_gate.json"
        if gate_path.exists():
            print(f"- gate report: {gate_path}")
    else:
        path = Path(args.artifact_root).resolve()
        path.mkdir(parents=True, exist_ok=True)
        gate_path = path / "release_gate.latest.json"
        gate_path.write_text(json.dumps(gate, indent=2) + "\n")
        print(f"- gate report: {gate_path}")

    return 0 if bool(gate.get("passed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
