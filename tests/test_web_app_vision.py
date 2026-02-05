"""Picture-focused tests for the Flask camera/vision endpoint."""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import unittest
from unittest import mock

from naturalist_companion.web import create_app

_TINY_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7S5XQAAAAASUVORK5CYII="
_TINY_PNG_BYTES = base64.b64decode(_TINY_PNG_BASE64)


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        return None

    def read(self) -> bytes:
        return self._raw


class TestWebAppVision(unittest.TestCase):
    """Tests for `/api/vision` request/response behavior."""

    def setUp(self) -> None:
        self.app = create_app()
        self.client = self.app.test_client()

    def test_vision_multipart_stub(self) -> None:
        resp = self.client.post(
            "/api/vision",
            data={
                "domain": "geology",
                "use_ollama": "false",
                "images": [
                    (io.BytesIO(_TINY_PNG_BYTES), "wide.png"),
                    (io.BytesIO(_TINY_PNG_BYTES), "closeup.png"),
                ],
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json() or {}
        self.assertEqual(payload.get("provider"), "stub")
        self.assertEqual(payload.get("image_count"), 2)
        vision = payload.get("vision") or {}
        self.assertEqual(vision.get("domain"), "geology")
        self.assertTrue(vision.get("hypotheses"))
        citations = vision.get("citations") or []
        self.assertTrue(citations)
        for c in citations:
            self.assertIn("wikipedia.org/wiki/", str(c.get("url", "")))

    def test_vision_json_base64_stub(self) -> None:
        resp = self.client.post(
            "/api/vision",
            json={
                "domain": "flora",
                "images": [
                    {
                        "mime_type": "image/png",
                        "data_base64": _TINY_PNG_BASE64,
                    }
                ],
            },
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json() or {}
        self.assertEqual(payload.get("provider"), "stub")
        vision = payload.get("vision") or {}
        self.assertEqual(vision.get("domain"), "flora")
        self.assertTrue(vision.get("features"))

    def test_vision_rejects_invalid_base64(self) -> None:
        resp = self.client.post(
            "/api/vision",
            json={
                "domain": "geology",
                "images": [{"mime_type": "image/png", "data_base64": "%%%not-base64%%%"}],
            },
        )
        self.assertEqual(resp.status_code, 400)
        payload = resp.get_json() or {}
        self.assertIn("error", payload)

    def test_vision_ollama_success_path(self) -> None:
        ollama_json = {
            "version": "1.0",
            "domain": "geology",
            "hypotheses": [
                {
                    "label": "Basalt",
                    "confidence": 0.67,
                    "why": ["Dark fine-grained texture"],
                    "disambiguators": ["Capture a sharper close-up"],
                }
            ],
            "features": [
                {"label": "Dark matrix", "confidence": 0.7, "why": ["Low visible grain size"]}
            ],
            "follow_up": {
                "questions": ["Do you see vesicles?"],
                "photo_requests": ["Take an oblique close-up with scale"],
            },
            "safety_notes": ["Stay clear of traffic."],
            "citations": [
                {"title": "Basalt", "url": "https://en.wikipedia.org/wiki/Basalt"},
                {"title": "Not allowed", "url": "https://example.com/nope"},
            ],
        }
        raw = {"message": {"content": json.dumps(ollama_json)}}

        with mock.patch(
            "naturalist_companion.ollama_vision.urllib.request.urlopen",
            return_value=_FakeHTTPResponse(raw),
        ):
            resp = self.client.post(
                "/api/vision",
                json={
                    "domain": "geology",
                    "use_ollama": True,
                    "images": [{"mime_type": "image/png", "data_base64": _TINY_PNG_BASE64}],
                },
            )

        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json() or {}
        self.assertEqual(payload.get("provider"), "ollama")
        self.assertEqual(payload.get("model"), "llava:7b")
        vision = payload.get("vision") or {}
        self.assertEqual(vision.get("hypotheses")[0].get("label"), "Basalt")
        citations = vision.get("citations") or []
        self.assertEqual(len(citations), 1)
        self.assertIn("wikipedia.org/wiki/", str(citations[0].get("url", "")))

    def test_vision_ollama_failure_falls_back_to_stub(self) -> None:
        with mock.patch(
            "naturalist_companion.ollama_vision.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            resp = self.client.post(
                "/api/vision",
                json={
                    "domain": "mixed",
                    "use_ollama": True,
                    "images": [{"mime_type": "image/png", "data_base64": _TINY_PNG_BASE64}],
                },
            )

        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json() or {}
        self.assertEqual(payload.get("provider"), "stub")
        self.assertIn("fallback_reason", payload)
        self.assertTrue(str(payload.get("fallback_reason") or "").strip())
        vision = payload.get("vision") or {}
        self.assertEqual(vision.get("domain"), "mixed")


if __name__ == "__main__":
    unittest.main()
