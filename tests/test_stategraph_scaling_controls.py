"""Tests for StateGraph production scaling controls and persistence behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from naturalist_companion.stategraph_shared import (
    _merge_config,
    refresh_retrieval_partitions,
    run_stategraph,
)


class TestStateGraphScalingControls(unittest.TestCase):
    def test_top_k_is_capped_when_experiments_disabled(self) -> None:
        cfg = _merge_config(
            {
                "top_k_faiss": 99,
                "top_k_keyword": 42,
                "top_k_rerank": 11,
                "strict_top_k_cap": 7,
                "allow_top_k_experiments": False,
            }
        )
        self.assertEqual(cfg["top_k_faiss"], 7)
        self.assertEqual(cfg["top_k_keyword"], 7)
        self.assertEqual(cfg["top_k_rerank"], 7)

    def test_top_k_can_exceed_cap_in_experiment_mode(self) -> None:
        cfg = _merge_config(
            {
                "top_k_faiss": 12,
                "top_k_keyword": 10,
                "top_k_rerank": 9,
                "strict_top_k_cap": 6,
                "allow_top_k_experiments": True,
            }
        )
        self.assertEqual(cfg["top_k_faiss"], 12)
        self.assertEqual(cfg["top_k_keyword"], 10)
        self.assertEqual(cfg["top_k_rerank"], 9)

    def test_partition_refresh_skips_when_cadence_not_due(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_root = str(Path(tmp) / "store")
            config = {
                "store_root": store_root,
                "refresh_cadence": "daily",
            }
            first = refresh_retrieval_partitions(
                config=config,
                runtime_mode="deterministic",
                partitions=["corridor_i81"],
                force=True,
            )
            self.assertEqual(first["summary"]["refreshed"], 1)

            second = refresh_retrieval_partitions(
                config=config,
                runtime_mode="deterministic",
                partitions=["corridor_i81"],
                force=False,
            )
            self.assertEqual(second["summary"]["refreshed"], 0)
            self.assertEqual(second["rows"][0]["status"], "skipped_fresh")

            manifest_path = Path(store_root) / "deterministic" / "corridor_i81" / "manifest.json"
            self.assertTrue(manifest_path.exists())

    def test_response_cache_hit_is_observable_on_repeat_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_root = str(Path(tmp) / "store")
            artifact_root = str(Path(tmp) / "artifacts")
            cfg = {
                "runtime_mode": "deterministic",
                "store_root": store_root,
                "artifact_root": artifact_root,
                "retrieval_backend": "persistent",
                "allow_on_demand_index_build": True,
                "response_cache_ttl_s": 3600,
            }
            question = "I am on I-81 near Hagerstown and need citation backed geology context."

            first = run_stategraph(question, provider="ollama", config=cfg, run_id="cache_q1_first")
            first_events = list(first.get("cache_events") or [])
            self.assertIn("response_cache_miss", first_events)

            second = run_stategraph(question, provider="ollama", config=cfg, run_id="cache_q1_second")
            second_events = list(second.get("cache_events") or [])
            self.assertIn("response_cache_hit", second_events)
            self.assertIn("route_used_response_cache", second_events)


if __name__ == "__main__":
    unittest.main()
