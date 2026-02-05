from __future__ import annotations

import argparse
import os
from typing import Any

from flask import Flask, jsonify, render_template, request

from anc.mvp import run_mvp


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            brand_name="Agentic Naturalist",
            tagline=(
                "A companion team of data + storytelling: intellectual sherpas and logisticians "
                "for field learning and route-based exploration."
            ),
        )

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True})

    @app.post("/api/mvp")
    def api_mvp():
        payload: dict[str, Any] = request.get_json(silent=True) or {}
        route_name = str(payload.get("route_name") or "web_mvp").strip() or "web_mvp"

        result = run_mvp(route_name=route_name, out_dir=None)
        return jsonify({"guide": result["guide"], "trace": result.get("trace", [])})

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agentic Naturalist Companion (Flask)")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument(
        "--debug",
        action="store_true",
        default=(os.environ.get("FLASK_DEBUG", "").strip() in {"1", "true", "True"}),
    )
    args = parser.parse_args(argv)

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

