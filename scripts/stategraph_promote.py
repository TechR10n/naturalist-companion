"""Promotion command that requires a passing real-data StateGraph release gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from naturalist_companion.stategraph_shared import run_real_data_release_gate


def main() -> int:
    if sys.platform == "darwin":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    parser = argparse.ArgumentParser(description="Promote StateGraph only when release gate passes.")
    parser.add_argument(
        "--provider",
        choices=("ollama", "vertex", "databricks"),
        default="ollama",
        help="Provider to evaluate for promotion (default: ollama).",
    )
    parser.add_argument(
        "--artifact-root",
        default="out/stategraph_release_gate",
        help="Gate artifact root (default: out/stategraph_release_gate).",
    )
    parser.add_argument(
        "--store-root",
        default="out/stategraph_store",
        help="Persistent retrieval store root (default: out/stategraph_store).",
    )
    args = parser.parse_args()

    try:
        gate = run_real_data_release_gate(
            provider=args.provider,
            config={
                "artifact_root": args.artifact_root,
                "store_root": args.store_root,
                "retrieval_backend": "persistent",
                "allow_on_demand_index_build": False,
            },
        )
    except Exception as e:
        print("Promotion blocked: release gate failed to execute.")
        print(f"- error: {e}")
        return 2
    if not bool(gate.get("passed")):
        print("Promotion blocked: release gate did not pass.")
        print(f"- failed checks: {', '.join(gate.get('failed_checks') or [])}")
        candidate_root = gate.get("candidate_artifact_root")
        if candidate_root:
            print(f"- gate artifacts: {candidate_root}")
        return 1

    promotion = {
        "provider": gate["provider"],
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "gate_metrics": gate.get("metrics", {}),
        "thresholds": gate.get("thresholds", {}),
        "candidate_artifact_root": gate.get("candidate_artifact_root"),
        "release_gate_passed": True,
    }
    promote_root = Path(args.artifact_root).expanduser().resolve() / "promotions"
    promote_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = promote_root / f"promotion_{args.provider}_{stamp}.json"
    out_path.write_text(json.dumps(promotion, indent=2) + "\n")
    print("Promotion approved.")
    print(f"- record: {out_path}")
    print(f"- gate artifacts: {gate.get('candidate_artifact_root')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
