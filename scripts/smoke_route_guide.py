"""Offline smoke test runner for the route-guide LangGraph flow."""

from __future__ import annotations

import argparse
from pathlib import Path

from naturalist_companion.route_guide import run_route_guide


def main() -> int:
    """Run the offline route-guide graph and optionally write outputs to disk."""
    parser = argparse.ArgumentParser(
        description="Offline smoke test: run the minimal Roadside Geology route-guide graph."
    )
    parser.add_argument(
        "--live-wikipedia",
        action="store_true",
        help="Use the real Wikipedia API (network required) instead of fallback offline data.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Wikipedia language code (default: en). Only used with --live-wikipedia.",
    )
    parser.add_argument(
        "--user-agent",
        default="naturalist-companion (local dev)",
        help="User-Agent string to send to Wikipedia (default: naturalist-companion (local dev)).",
    )
    parser.add_argument(
        "--min-interval-s",
        type=float,
        default=1.25,
        help="Minimum seconds between Wikipedia requests (default: 1.25).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for Wikipedia 429/5xx responses (default: 3).",
    )
    parser.add_argument(
        "--backoff-s",
        type=float,
        default=1.5,
        help="Base backoff seconds for Wikipedia retries (default: 1.5).",
    )
    parser.add_argument(
        "--sample-every-m",
        type=int,
        default=10_000,
        help="Route sampling interval in meters (default: 10000).",
    )
    parser.add_argument(
        "--geosearch-radius-m",
        type=int,
        default=15_000,
        help="Wikipedia GeoSearch radius in meters (default: 15000).",
    )
    parser.add_argument(
        "--geosearch-limit",
        type=int,
        default=8,
        help="Max GeoSearch results per point (default: 8).",
    )
    parser.add_argument(
        "--max-stops",
        type=int,
        default=5,
        help="Max stops to select (default: 5).",
    )
    parser.add_argument(
        "--min-stop-spacing-m",
        type=int,
        default=15_000,
        help="Minimum spacing between stops in meters (default: 15000).",
    )
    parser.add_argument(
        "--out-dir",
        default="out/guide",
        help="Directory to write guide.json and guide.md (default: out/guide).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Don't write output files; just run the graph and print a summary.",
    )
    args = parser.parse_args()

    tools = None
    if args.live_wikipedia:
        from naturalist_companion.wikipedia_tools import wikipedia_tools

        tools = wikipedia_tools(
            language=args.language,
            user_agent=args.user_agent,
            min_interval_s=max(0.0, args.min_interval_s),
            max_retries=max(0, args.max_retries),
            backoff_s=max(0.1, args.backoff_s),
        )

    config = {
        "sample_every_m": max(1, args.sample_every_m),
        "geosearch_radius_m": max(1000, args.geosearch_radius_m),
        "geosearch_limit": max(1, args.geosearch_limit),
        "max_stops": max(1, args.max_stops),
        "min_stop_spacing_m": max(0, args.min_stop_spacing_m),
        "language": args.language,
    }

    result = run_route_guide(
        out_dir=None if args.no_write else Path(args.out_dir),
        tools=tools,
        config=config,
    )

    print("LangGraph route-guide smoke run")
    print(f"- nodes executed: {len(result.get('trace', []))}")
    print(f"- trace: {', '.join(result.get('trace', []))}")
    guide = result["guide"]
    print(f"- stops: {len(guide['stops'])}")
    if not args.no_write:
        print(f"- wrote: {Path(args.out_dir) / 'guide.json'}")
        print(f"- wrote: {Path(args.out_dir) / 'guide.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
