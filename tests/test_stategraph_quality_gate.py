"""Quality-gate unit tests for safety-note handling."""

from __future__ import annotations

import unittest

from naturalist_companion.stategraph_shared import (
    SAFE_STOP_NOTE,
    _default_config,
    _quality_gate_node,
)


def _base_state(response: str) -> dict:
    citation_url = "https://en.wikipedia.org/wiki/Appalachian_Mountains"
    claim_text = "The Appalachian Mountains are a mountain range in eastern North America."
    return {
        "question": "What geology should I notice near I-81?",
        "provider": "ollama",
        "config": _default_config(),
        "retrieval_attempt": 1,
        "reranked_docs": [
            {
                "url": citation_url,
                "text": claim_text,
            }
        ],
        "answer": {
            "provider": "ollama",
            "question": "What geology should I notice near I-81?",
            "response": response,
            "claims": [{"text": claim_text, "citation_urls": [citation_url]}],
            "citations": [{"title": "Appalachian Mountains", "url": citation_url, "pageid": 4718}],
            "safety_notes": [SAFE_STOP_NOTE],
            "retrieval_attempt": 1,
            "mode": "deterministic",
        },
    }


class TestStateGraphQualityGate(unittest.TestCase):
    def test_canonical_safe_note_does_not_fail_unsafe_phrase_filter(self) -> None:
        response = (
            "Grounded summary for the stop. "
            f"{SAFE_STOP_NOTE}"
        )
        updates = _quality_gate_node(_base_state(response))
        quality = updates["quality_report"]
        self.assertTrue(quality["safe_stop_ok"])
        self.assertTrue(quality["passed"])
        self.assertNotIn("safe_stop_constraint_failed", quality["reasons"])

    def test_unsafe_phrase_outside_safe_note_still_fails(self) -> None:
        response = (
            "Grounded summary for the stop. "
            "Some visitors stand in the lane for photos. "
            f"{SAFE_STOP_NOTE}"
        )
        updates = _quality_gate_node(_base_state(response))
        quality = updates["quality_report"]
        self.assertFalse(quality["safe_stop_ok"])
        self.assertFalse(quality["passed"])
        self.assertIn("safe_stop_constraint_failed", quality["reasons"])


if __name__ == "__main__":
    unittest.main()
