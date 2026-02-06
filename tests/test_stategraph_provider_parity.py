"""Provider-parity smoke tests for the shared StateGraph pipeline."""

from __future__ import annotations

import unittest

from naturalist_companion.stategraph_shared import (
    run_i81_eval_harness,
    run_stategraph,
)


class TestStateGraphProviderParity(unittest.TestCase):
    """Smoke tests to ensure ollama/vertex/databricks share graph behavior."""

    def test_single_question_runs_for_all_providers(self) -> None:
        question = "I am on I-81 near Hagerstown. What geology should I notice?"
        providers = ("ollama", "vertex", "databricks")
        for provider in providers:
            with self.subTest(provider=provider):
                state = run_stategraph(
                    question,
                    provider=provider,
                    config={
                        "artifact_root": "out/test_stategraph/providers",
                        "max_retrieval_attempts": 2,
                    },
                )
                final = state["final_output"]
                self.assertEqual(final["provider"], provider)
                self.assertIn(
                    final["route_decision"]["decision"],
                    {"answerable_now", "needs_clarification", "needs_retrieval_retry"},
                )
                self.assertIn("quality", final)
                self.assertIn("answer", final)

    def test_eval_harness_runs_for_vertex_mode(self) -> None:
        report = run_i81_eval_harness(
            provider="vertex",
            config={
                "artifact_root": "out/test_stategraph/eval",
                "max_retrieval_attempts": 2,
            },
        )
        summary = report["summary"]
        self.assertEqual(summary["provider"], "vertex")
        self.assertEqual(summary["question_count"], 20)
        self.assertGreaterEqual(summary["citation_validity_pct"], 0.0)
        self.assertLessEqual(summary["citation_validity_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
