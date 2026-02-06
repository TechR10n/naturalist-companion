"""Flask web app entrypoints for Agentic Naturalist Companion."""

from __future__ import annotations

import argparse
import base64
import os
from typing import Any

from flask import Flask, jsonify, render_template, request

from .route_guide import run_route_guide
from .ollama_vision import VisionImage, classify_images


def _story_from_guide(guide: dict[str, Any]) -> str:
    route = guide.get("route") if isinstance(guide.get("route"), dict) else {}
    route_name = str(route.get("name") or "self_guided_route")
    length_km = route.get("length_km")
    try:
        length_km_s = f"{float(length_km):.1f}"
    except Exception:
        length_km_s = "?"

    stops = guide.get("stops") if isinstance(guide.get("stops"), list) else []

    lines: list[str] = [
        f"Self-guided geology tour — {route_name}",
        "",
        f"Route length: {length_km_s} km • Stops: {len(stops)}",
        "",
        "How to use this tour",
        "- Read ahead before you drive; don’t interact with the map while moving.",
        "- Stop only where it’s legal and safe (pull-offs, parking areas, trailheads).",
        "- If this route follows a highway corridor, treat stops as “nearby” context, not roadside instructions.",
        "",
        "Story",
    ]

    if not stops:
        lines.extend(["", "No geology stops were generated for this route.", ""])
        return "\n".join(lines).rstrip() + "\n"

    lines.extend(
        [
            "",
            "Today’s thread is simple: watch how rocks and landforms change as you move, and keep asking "
            "“what process made this?” (volcanism, sedimentation, metamorphism, uplift, erosion).",
            "",
        ]
    )

    for i, stop in enumerate(stops, start=1):
        if not isinstance(stop, dict):
            continue
        stop_id = str(stop.get("stop_id") or f"stop_{i:02d}")
        title = str(stop.get("title") or "Geology stop")
        route_km = stop.get("route_km")
        try:
            route_km_s = f"{float(route_km):.1f}"
        except Exception:
            route_km_s = "?"

        why_stop = str(stop.get("why_stop") or "").strip()
        what_to_look_for = stop.get("what_to_look_for") if isinstance(stop.get("what_to_look_for"), list) else []
        key_facts = stop.get("key_facts") if isinstance(stop.get("key_facts"), list) else []

        lines.append(f"{stop_id} — {title} (around km {route_km_s})")
        if why_stop:
            lines.append(f"{why_stop}")
        if what_to_look_for:
            lines.append("")
            lines.append("What to look for:")
            for b in what_to_look_for[:6]:
                lines.append(f"- {str(b)}")
        if key_facts:
            lines.append("")
            lines.append("A few quick facts:")
            for b in key_facts[:5]:
                lines.append(f"- {str(b)}")
        lines.append("")

    lines.extend(
        [
            "Wrap-up",
            "Take one last look back at the route as a whole. If you had to explain the landscape in "
            "three processes, which would you pick—and what evidence did you see for each?",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _route_geojson(route: list[dict[str, Any]], *, name: str) -> dict[str, Any]:
    coords: list[list[float]] = []
    for pt in route:
        try:
            coords.append([float(pt["lon"]), float(pt["lat"])])
        except Exception:
            continue
    return {
        "type": "Feature",
        "properties": {"name": name},
        "geometry": {"type": "LineString", "coordinates": coords},
    }


def _stops_geojson(stops: list[dict[str, Any]]) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    for stop in stops:
        center = stop.get("center") if isinstance(stop.get("center"), dict) else {}
        try:
            lon = float(center["lon"])
            lat = float(center["lat"])
        except Exception:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {k: v for k, v in stop.items() if k != "center"},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _as_timeout_seconds(value: Any, *, default: float) -> float:
    try:
        timeout_s = float(value)
    except Exception:
        return default
    return max(1.0, min(300.0, timeout_s))


def _decode_base64_image(raw: str) -> bytes:
    payload = str(raw or "").strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload, validate=True)
    except Exception as e:
        raise ValueError("Invalid base64 image payload in `images`.") from e


def _resolve_wikipedia_tools(*, payload: dict[str, Any], config: Any):
    live_wikipedia = _as_bool(payload.get("live_wikipedia"))
    if not live_wikipedia:
        return None

    from .wikipedia_tools import wikipedia_tools

    language = "en"
    if isinstance(config, dict) and isinstance(config.get("language"), str):
        language = str(config["language"])
    user_agent = str(payload.get("user_agent") or "naturalist-companion (local dev)")
    return wikipedia_tools(language=language, user_agent=user_agent)


def create_app() -> Flask:
    """Create the Flask application instance."""
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

    @app.get("/tour")
    def tour():
        return render_template(
            "tour.html",
            brand_name="Agentic Naturalist",
            title="Self-guided geology tour",
        )

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True})

    @app.post("/api/guide")
    def api_guide():
        payload: dict[str, Any] = request.get_json(silent=True) or {}
        route_name = str(payload.get("route_name") or "web_guide").strip() or "web_guide"

        route_points = payload.get("route_points")
        config = payload.get("config")
        tools = _resolve_wikipedia_tools(payload=payload, config=config)

        result = run_route_guide(
            route_name=route_name,
            route_points=route_points if isinstance(route_points, list) else None,
            config=config if isinstance(config, dict) else None,
            out_dir=None,
            tools=tools,
        )
        return jsonify(
            {
                "guide": result["guide"],
                "guide_markdown": result.get("guide_markdown", ""),
                "trace": result.get("trace", []),
            }
        )

    @app.post("/api/tour")
    def api_tour():
        payload: dict[str, Any] = request.get_json(silent=True) or {}
        route_name = str(payload.get("route_name") or "web_tour").strip() or "web_tour"

        route_points = payload.get("route_points")
        config = payload.get("config")
        tools = _resolve_wikipedia_tools(payload=payload, config=config)

        result = run_route_guide(
            route_name=route_name,
            route_points=route_points if isinstance(route_points, list) else None,
            config=config if isinstance(config, dict) else None,
            out_dir=None,
            tools=tools,
        )

        guide = result["guide"]
        route = result.get("route") or []
        stops = guide.get("stops") or []
        if not isinstance(route, list):
            route = []
        if not isinstance(stops, list):
            stops = []

        return jsonify(
            {
                "guide": guide,
                "route_points": route,
                "route_geojson": _route_geojson(route, name=str(guide.get("route", {}).get("name") or route_name)),
                "stops_geojson": _stops_geojson([s for s in stops if isinstance(s, dict)]),
                "story": _story_from_guide(guide if isinstance(guide, dict) else {}),
                "guide_markdown": result.get("guide_markdown", ""),
                "trace": result.get("trace", []),
            }
        )

    @app.post("/api/vision")
    def api_vision():
        images: list[VisionImage] = []
        domain = "geology"
        note = ""
        use_ollama = False
        ollama_url: str | None = None
        ollama_model: str | None = None
        timeout_s = 60.0

        content_type = (request.content_type or "").lower()
        is_multipart = "multipart/form-data" in content_type

        if is_multipart:
            for uploaded in request.files.getlist("images"):
                data = uploaded.read()
                if not data:
                    continue
                images.append(
                    {
                        "mime_type": str(uploaded.mimetype or "application/octet-stream"),
                        "data": data,
                    }
                )
            domain = str(request.form.get("domain") or "geology")
            note = str(request.form.get("note") or "")
            use_ollama = _as_bool(request.form.get("use_ollama"))
            ollama_url = str(request.form.get("ollama_url") or "").strip() or None
            ollama_model = str(request.form.get("ollama_model") or "").strip() or None
            timeout_s = _as_timeout_seconds(request.form.get("timeout_s"), default=60.0)
        else:
            payload: dict[str, Any] = request.get_json(silent=True) or {}
            raw_images = payload.get("images")
            if isinstance(raw_images, list):
                for raw in raw_images:
                    mime_type = "application/octet-stream"
                    data_b64 = ""
                    if isinstance(raw, dict):
                        mime_type = str(raw.get("mime_type") or mime_type)
                        data_b64 = str(raw.get("data_base64") or raw.get("data") or "").strip()
                    elif isinstance(raw, str):
                        data_b64 = raw.strip()
                    if not data_b64:
                        continue
                    try:
                        image_bytes = _decode_base64_image(data_b64)
                    except ValueError as e:
                        return jsonify({"error": str(e)}), 400
                    images.append({"mime_type": mime_type, "data": image_bytes})

            domain = str(payload.get("domain") or "geology")
            note = str(payload.get("note") or "")
            use_ollama = _as_bool(payload.get("use_ollama"))
            ollama_url = str(payload.get("ollama_url") or "").strip() or None
            ollama_model = str(payload.get("ollama_model") or "").strip() or None
            timeout_s = _as_timeout_seconds(payload.get("timeout_s"), default=60.0)

        if not images:
            return (
                jsonify(
                    {
                        "error": (
                            "No images were provided. Send multipart files as `images` "
                            "or JSON `images[].data_base64`."
                        )
                    }
                ),
                400,
            )

        if len(images) > 4:
            images = images[:4]

        result = classify_images(
            images=images,
            domain=domain,
            note=note,
            use_ollama=use_ollama,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
            timeout_s=timeout_s,
        )
        return jsonify(result)

    return app


def main(argv: list[str] | None = None) -> int:
    """Run the Flask development server."""
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
