"""Batch refresh persistent StateGraph retrieval indexes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from naturalist_companion.stategraph_shared import (
    list_default_partitions,
    refresh_retrieval_partitions,
)


def _parse_partitions(raw: str) -> list[str]:
    text = str(raw or "").strip().lower()
    if not text or text == "all":
        return list(list_default_partitions())
    out = []
    for part in text.split(","):
        p = part.strip()
        if p:
            out.append(p)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh persistent StateGraph retrieval indexes.")
    parser.add_argument(
        "--runtime-mode",
        choices=("deterministic", "realistic"),
        default="realistic",
        help="Source mode for index refresh (default: realistic).",
    )
    parser.add_argument(
        "--partitions",
        default="all",
        help="Comma-separated partition ids or 'all' (default: all).",
    )
    parser.add_argument(
        "--store-root",
        default="out/stategraph_store",
        help="Persistent index/cache root (default: out/stategraph_store).",
    )
    parser.add_argument(
        "--refresh-cadence",
        choices=("daily", "weekly"),
        default="daily",
        help="Refresh cadence guard used to skip fresh partitions (default: daily).",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=24,
        help="Live-doc budget per partition for realistic refresh (default: 24).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force refresh even when cadence says partition is fresh.",
    )
    args = parser.parse_args()

    partitions = _parse_partitions(args.partitions)
    cfg = {
        "store_root": args.store_root,
        "refresh_cadence": args.refresh_cadence,
        "retrieval_backend": "persistent",
        "allow_on_demand_index_build": False,
        "live_max_docs": max(4, int(args.max_docs)),
    }
    report = refresh_retrieval_partitions(
        config=cfg,
        runtime_mode=args.runtime_mode,
        partitions=partitions,
        force=bool(args.force),
        max_docs=max(4, int(args.max_docs)),
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = (
        Path(args.store_root).expanduser().resolve() / "refresh_reports" / f"refresh_{args.runtime_mode}_{stamp}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    summary = report["summary"]
    print("StateGraph retrieval index refresh")
    print(f"- runtime mode: {summary['runtime_mode']}")
    print(f"- store root: {summary['store_root']}")
    print(f"- cadence: {summary['refresh_cadence']}")
    print(f"- refreshed: {summary['refreshed']}/{summary['total']}")
    for row in report["rows"]:
        status = row.get("status", "unknown")
        partition = row.get("partition", "?")
        if status == "refreshed":
            print(
                f"  - {partition}: refreshed, chunks={row.get('chunk_count')} source={row.get('source')}"
            )
        elif status == "skipped_fresh":
            print(f"  - {partition}: skipped (fresh, refreshed_at={row.get('refreshed_at')})")
        else:
            print(f"  - {partition}: {status}")
    print(f"- report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
