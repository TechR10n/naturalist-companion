"""Smoke tests for the Flask web app."""

from __future__ import annotations

import unittest

from naturalist_companion.web import create_app


class TestWebAppSmoke(unittest.TestCase):
    """Smoke tests for the Flask app endpoints."""

    def test_healthz(self) -> None:
        app = create_app()
        client = app.test_client()
        resp = client.get("/healthz")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), {"ok": True})

    def test_mvp_endpoint_returns_guide(self) -> None:
        app = create_app()
        client = app.test_client()
        resp = client.post("/api/mvp", json={"route_name": "unit_test_web"})
        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json() or {}
        self.assertIn("guide", payload)
        guide = payload["guide"]
        self.assertIn("stops", guide)

    def test_tour_page_loads(self) -> None:
        app = create_app()
        client = app.test_client()
        resp = client.get("/tour")
        self.assertEqual(resp.status_code, 200)
        body = (resp.data or b"").decode("utf-8", errors="replace")
        self.assertIn("Self-guided geology tour", body)

    def test_tour_endpoint_returns_story_and_route(self) -> None:
        app = create_app()
        client = app.test_client()
        resp = client.post("/api/tour", json={"route_name": "unit_test_tour"})
        self.assertEqual(resp.status_code, 200)
        payload = resp.get_json() or {}
        self.assertIn("guide", payload)
        self.assertIn("story", payload)
        self.assertIsInstance(payload.get("story"), str)
        self.assertIn("route_points", payload)
        self.assertIsInstance(payload.get("route_points"), list)
        guide = payload["guide"]
        self.assertIn("stops", guide)


if __name__ == "__main__":
    unittest.main()
