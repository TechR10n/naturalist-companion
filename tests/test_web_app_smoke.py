from __future__ import annotations

import unittest

from anc.web import create_app


class TestWebAppSmoke(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

