"""Smoke tests for the offline LangGraph MVP."""

from __future__ import annotations

import unittest
from datetime import datetime

from naturalist_companion.mvp import build_mvp_app


class TestLangGraphMVPSmoke(unittest.TestCase):
    """Smoke tests for the offline LangGraph MVP graph."""

    def test_smoke_executes_all_nodes(self) -> None:
        app = build_mvp_app()
        result = app.invoke(
            {
                "config": {
                    "sample_every_m": 10_000,
                    "geosearch_radius_m": 15_000,
                    "geosearch_limit": 20,
                    "max_stops": 5,
                    "min_stop_spacing_m": 15_000,
                    "language": "en",
                },
                "route_name": "unit_test_route",
            }
        )

        trace = result.get("trace", [])
        self.assertTrue(trace, "expected a non-empty execution trace")

        expected_nodes = [
            "ingest_route",
            "sample_points",
            "discover_candidates",
            "fetch_and_filter_pages",
            "select_stops",
            "build_index",
            "retrieve",
            "write_stop_card",
            "validate_stop_card",
            "accumulate_and_advance",
            "render_outputs",
        ]
        for node in expected_nodes:
            self.assertIn(node, trace, f"expected node {node!r} to execute at least once")

        guide = result["guide"]
        self.assertGreaterEqual(len(guide["stops"]), 1)
        datetime.fromisoformat(guide["generated_at"])  # parseable

        for stop in guide["stops"]:
            citations = stop.get("citations") or []
            self.assertTrue(citations, "expected >=1 citation per stop")
            for c in citations:
                self.assertIn("wikipedia.org/wiki/", c.get("url", ""))


if __name__ == "__main__":
    unittest.main()
