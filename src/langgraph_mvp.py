"""Offline LangGraph MVP for the roadside geology workflow."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict

from langgraph.graph import END, StateGraph

from . import mvp_data


class RoutePoint(TypedDict):
    """Route point along the path with cumulative distance metadata."""

    i: int
    lat: float
    lon: float
    cum_dist_m: float


class WikiCandidate(TypedDict):
    """Candidate Wikipedia page near a sampled route point."""

    pageid: int
    title: str
    lat: float
    lon: float
    dist_m: float
    route_km: float


class WikiPage(TypedDict):
    """Fetched Wikipedia page payload used by the MVP graph."""

    pageid: int
    title: str
    url: str
    summary: str
    content: str
    categories: list[str]
    lat: float
    lon: float


class Chunk(TypedDict):
    """Text chunk with source metadata used for retrieval."""

    text: str
    pageid: int
    title: str
    url: str
    route_km: float


class StopSeed(TypedDict):
    """Seed metadata for a candidate stop card."""

    stop_id: str
    pageid: int
    title: str
    url: str
    center: dict[str, float]
    route_km: float


class StopCard(TypedDict):
    """Structured stop card produced by the MVP."""

    stop_id: str
    route_km: float
    center: dict[str, float]
    title: str
    why_stop: str
    what_to_look_for: list[str]
    key_facts: list[str]
    photo_prompts: list[str]
    confidence: float
    citations: list[dict[str, Any]]


class Guide(TypedDict):
    """Guide output for a route with generated stop cards."""

    route: dict[str, Any]
    generated_at: str
    config: dict[str, Any]
    stops: list[StopCard]


class MVPConfig(TypedDict):
    """Configuration options for the offline MVP."""

    sample_every_m: int
    geosearch_radius_m: int
    geosearch_limit: int
    max_stops: int
    min_stop_spacing_m: int
    language: str


class GraphState(TypedDict, total=False):
    """Shared state container passed between LangGraph nodes."""

    trace: list[str]
    config: MVPConfig
    route_name: str
    route_points: list[dict[str, float]]

    route: list[RoutePoint]
    route_length_km: float
    sample_points: list[RoutePoint]

    candidates: list[WikiCandidate]
    pages: dict[int, WikiPage]

    stop_seeds: list[StopSeed]
    chunks: list[Chunk]

    current_stop_idx: int
    current_chunks: list[Chunk]
    current_stop_card: StopCard
    _validation: dict[str, Any]
    stop_cards: list[StopCard]

    guide: Guide
    guide_markdown: str


GeosearchFn = Callable[[float, float, int, int, str], list[dict[str, Any]]]
FetchPageFn = Callable[[int, str], dict[str, Any]]


@dataclass(frozen=True)
class Tools:
    """Callable hooks for geosearch and page fetching."""

    geosearch: GeosearchFn
    fetch_page: FetchPageFn


def _haversine_m(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    r = 6371000.0
    phi1 = math.radians(a_lat)
    phi2 = math.radians(b_lat)
    dphi = math.radians(b_lat - a_lat)
    dlambda = math.radians(b_lon - a_lon)

    x = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * r * math.atan2(math.sqrt(x), math.sqrt(1.0 - x))


def _append_trace(state: GraphState, node: str) -> list[str]:
    return [*state.get("trace", []), node]


def _default_config() -> MVPConfig:
    return {
        "sample_every_m": 10_000,
        "geosearch_radius_m": 15_000,
        "geosearch_limit": 20,
        "max_stops": 5,
        "min_stop_spacing_m": 15_000,
        "language": "en",
    }


def _ingest_route(state: GraphState) -> GraphState:
    route_points = state.get("route_points") or mvp_data.minimal_route_points()
    if len(route_points) < 2:
        raise ValueError("route_points must include at least 2 points")

    route: list[RoutePoint] = []
    cum = 0.0
    prev = route_points[0]
    route.append({"i": 0, "lat": prev["lat"], "lon": prev["lon"], "cum_dist_m": 0.0})
    for i, pt in enumerate(route_points[1:], start=1):
        cum += _haversine_m(prev["lat"], prev["lon"], pt["lat"], pt["lon"])
        route.append({"i": i, "lat": pt["lat"], "lon": pt["lon"], "cum_dist_m": cum})
        prev = pt

    return {
        "trace": _append_trace(state, "ingest_route"),
        "route": route,
        "route_length_km": round(cum / 1000.0, 3),
    }


def _sample_points(state: GraphState) -> GraphState:
    route = state["route"]
    sample_every_m = state["config"]["sample_every_m"]
    if sample_every_m <= 0:
        raise ValueError("sample_every_m must be > 0")

    total_m = route[-1]["cum_dist_m"]
    targets = [0.0]
    x = float(sample_every_m)
    while x < total_m:
        targets.append(x)
        x += float(sample_every_m)
    if targets[-1] != total_m:
        targets.append(total_m)

    sample_points: list[RoutePoint] = []
    j = 0
    for t in targets:
        while j + 1 < len(route) and route[j + 1]["cum_dist_m"] < t:
            j += 1
        sample_points.append(route[j])

    # De-dupe identical indices (route might be too short for the chosen sampling).
    uniq: list[RoutePoint] = []
    seen: set[int] = set()
    for pt in sample_points:
        if pt["i"] in seen:
            continue
        uniq.append(pt)
        seen.add(pt["i"])

    return {"trace": _append_trace(state, "sample_points"), "sample_points": uniq}


def _discover_candidates(tools: Tools) -> Callable[[GraphState], GraphState]:
    def _node(state: GraphState) -> GraphState:
        cfg = state["config"]
        candidates: list[WikiCandidate] = []
        for pt in state["sample_points"]:
            raw = tools.geosearch(
                pt["lat"],
                pt["lon"],
                cfg["geosearch_radius_m"],
                cfg["geosearch_limit"],
                cfg["language"],
            )
            for c in raw:
                candidates.append(
                    {
                        "pageid": int(c["pageid"]),
                        "title": str(c["title"]),
                        "lat": float(c["lat"]),
                        "lon": float(c["lon"]),
                        "dist_m": float(c.get("dist_m") or 0.0),
                        "route_km": round(pt["cum_dist_m"] / 1000.0, 3),
                    }
                )

        # De-dupe by pageid (keep closest occurrence).
        best_by_pageid: dict[int, WikiCandidate] = {}
        for c in candidates:
            prev = best_by_pageid.get(c["pageid"])
            if prev is None or c["dist_m"] < prev["dist_m"]:
                best_by_pageid[c["pageid"]] = c

        return {
            "trace": _append_trace(state, "discover_candidates"),
            "candidates": list(best_by_pageid.values()),
        }

    return _node


_GEO_KEYWORDS = (
    "fault",
    "basalt",
    "sandstone",
    "granite",
    "formation",
    "stratigraphy",
    "glacier",
    "moraine",
    "volcano",
    "caldera",
    "igneous",
    "metamorphic",
    "sedimentary",
    "fossil",
    "tectonic",
    "orogeny",
    "geology",
)


def _fetch_and_filter_pages(tools: Tools) -> Callable[[GraphState], GraphState]:
    def _node(state: GraphState) -> GraphState:
        pages: dict[int, WikiPage] = {}
        kept: list[WikiCandidate] = []
        for c in state["candidates"]:
            raw = tools.fetch_page(c["pageid"], state["config"]["language"])
            page: WikiPage = {
                "pageid": int(raw["pageid"]),
                "title": str(raw["title"]),
                "url": str(raw["url"]),
                "summary": str(raw.get("summary") or ""),
                "content": str(raw.get("content") or ""),
                "categories": list(raw.get("categories") or []),
                "lat": float(raw.get("lat") or c["lat"]),
                "lon": float(raw.get("lon") or c["lon"]),
            }

            text = " ".join([page["title"], page["summary"], " ".join(page["categories"])])
            text_l = text.lower()
            if any(k in text_l for k in _GEO_KEYWORDS):
                pages[page["pageid"]] = page
                kept.append(c)

        return {
            "trace": _append_trace(state, "fetch_and_filter_pages"),
            "pages": pages,
            "candidates": kept,
        }

    return _node


def _select_stops(state: GraphState) -> GraphState:
    cfg = state["config"]
    max_stops = cfg["max_stops"]
    min_spacing_m = float(cfg["min_stop_spacing_m"])

    scored: list[tuple[float, WikiCandidate]] = []
    for c in state["candidates"]:
        page = state["pages"].get(c["pageid"])
        if not page:
            continue
        text = " ".join([page["title"], page["summary"], " ".join(page["categories"])])
        score = sum(1.0 for k in _GEO_KEYWORDS if k in text.lower())
        score += max(0.0, (cfg["geosearch_radius_m"] - c["dist_m"]) / cfg["geosearch_radius_m"])
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)

    seeds: list[StopSeed] = []
    for _, c in scored:
        if len(seeds) >= max_stops:
            break
        if any(abs(c["route_km"] - s["route_km"]) * 1000.0 < min_spacing_m for s in seeds):
            continue
        page = state["pages"][c["pageid"]]
        stop_id = f"stop_{len(seeds) + 1:02d}"
        seeds.append(
            {
                "stop_id": stop_id,
                "pageid": page["pageid"],
                "title": page["title"],
                "url": page["url"],
                "center": {"lat": page["lat"], "lon": page["lon"]},
                "route_km": c["route_km"],
            }
        )

    if not seeds:
        raise RuntimeError("No stops selected. Try loosening filters/radius in config.")

    return {"trace": _append_trace(state, "select_stops"), "stop_seeds": seeds}


def _build_index(state: GraphState) -> GraphState:
    chunks: list[Chunk] = []
    for seed in state["stop_seeds"]:
        page = state["pages"][seed["pageid"]]
        # Minimal chunking: split on lines, keep non-empty.
        for line in page["content"].splitlines():
            text = line.strip(" -\t")
            if not text:
                continue
            chunks.append(
                {
                    "text": text,
                    "pageid": page["pageid"],
                    "title": page["title"],
                    "url": page["url"],
                    "route_km": seed["route_km"],
                }
            )

    return {
        "trace": _append_trace(state, "build_index"),
        "chunks": chunks,
        "current_stop_idx": 0,
        "stop_cards": [],
    }


def _retrieve(state: GraphState) -> GraphState:
    idx = state["current_stop_idx"]
    seed = state["stop_seeds"][idx]
    chunks = [c for c in state["chunks"] if c["pageid"] == seed["pageid"]]
    if not chunks:
        chunks = state["chunks"][:3]
    return {"trace": _append_trace(state, "retrieve"), "current_chunks": chunks}


def _write_stop_card(state: GraphState) -> GraphState:
    idx = state["current_stop_idx"]
    seed = state["stop_seeds"][idx]
    chunks = state["current_chunks"]
    source_titles = sorted({c["title"] for c in chunks})
    why = f"Wikipedia-backed stop highlighting {seed['title']} near this route segment."

    def _bullets(prefix: str, limit: int) -> list[str]:
        bullets: list[str] = []
        for c in chunks:
            if len(bullets) >= limit:
                break
            text = c["text"].strip()
            if not text:
                continue
            bullets.append(f"{prefix}{text[:160]}")
        return bullets or [f"{prefix}(No retrieved text for this stop in the MVP dataset.)"]

    card: StopCard = {
        "stop_id": seed["stop_id"],
        "route_km": seed["route_km"],
        "center": seed["center"],
        "title": f"{seed['title']} (MVP stop)",
        "why_stop": why,
        "what_to_look_for": _bullets("", 3),
        "key_facts": _bullets("", 3),
        "photo_prompts": [
            "Wide shot of the landscape/context.",
            "Close-up texture detail (if safely visible).",
        ],
        "confidence": 0.5,
        "citations": [
            {"title": t, "url": seed["url"], "pageid": seed["pageid"]} for t in source_titles
        ]
        or [{"title": seed["title"], "url": seed["url"], "pageid": seed["pageid"]}],
    }
    return {"trace": _append_trace(state, "write_stop_card"), "current_stop_card": card}


def _validate_stop_card(state: GraphState) -> GraphState:
    card = state["current_stop_card"]
    errors: list[str] = []

    required_keys = (
        "stop_id",
        "route_km",
        "center",
        "title",
        "why_stop",
        "what_to_look_for",
        "key_facts",
        "photo_prompts",
        "citations",
    )
    for k in required_keys:
        if k not in card:
            errors.append(f"missing:{k}")

    citations = card.get("citations") or []
    if not citations:
        errors.append("citations:empty")
    for c in citations:
        url = str(c.get("url") or "")
        if "wikipedia.org/wiki/" not in url:
            errors.append(f"citation:not_wikipedia:{url}")

    ok = not errors
    card["confidence"] = 0.8 if ok else 0.2
    return {
        "trace": _append_trace(state, "validate_stop_card"),
        "current_stop_card": card,
        "_validation": {"ok": ok, "errors": errors},
    }


def _accumulate_and_advance(state: GraphState) -> GraphState:
    cards = [*state.get("stop_cards", []), state["current_stop_card"]]
    return {
        "trace": _append_trace(state, "accumulate_and_advance"),
        "stop_cards": cards,
        "current_stop_idx": state["current_stop_idx"] + 1,
    }


def _should_continue(state: GraphState) -> Literal["continue", "done"]:
    return "continue" if state["current_stop_idx"] < len(state["stop_seeds"]) else "done"


def _render_outputs(state: GraphState) -> GraphState:
    guide: Guide = {
        "route": {
            "name": state.get("route_name") or "mvp_route",
            "num_points": len(state["route"]),
            "length_km": state["route_length_km"],
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": dict(state["config"]),
        "stops": state["stop_cards"],
    }

    md_lines: list[str] = [
        f"# Roadside Geology (MVP) — {guide['route']['name']}",
        "",
        f"- Length: {guide['route']['length_km']} km",
        f"- Stops: {len(guide['stops'])}",
        "",
    ]
    for stop in guide["stops"]:
        md_lines.extend(
            [
                f"## {stop['stop_id']}: {stop['title']}",
                "",
                f"Route km: {stop['route_km']}",
                "",
                f"**Why stop**: {stop['why_stop']}",
                "",
                "**What to look for**",
                *[f"- {b}" for b in stop["what_to_look_for"]],
                "",
                "**Key facts**",
                *[f"- {b}" for b in stop["key_facts"]],
                "",
                "**Photo prompts**",
                *[f"- {b}" for b in stop["photo_prompts"]],
                "",
                "**Citations**",
                *[f"- {c['url']}" for c in stop["citations"]],
                "",
            ]
        )

    return {
        "trace": _append_trace(state, "render_outputs"),
        "guide": guide,
        "guide_markdown": "\n".join(md_lines).rstrip() + "\n",
    }


def build_mvp_app(tools: Tools | None = None):
    """Build the offline MVP LangGraph app."""
    tools = tools or offline_tools()

    graph: StateGraph = StateGraph(GraphState)
    graph.add_node("ingest_route", _ingest_route)
    graph.add_node("sample_points", _sample_points)
    graph.add_node("discover_candidates", _discover_candidates(tools))
    graph.add_node("fetch_and_filter_pages", _fetch_and_filter_pages(tools))
    graph.add_node("select_stops", _select_stops)
    graph.add_node("build_index", _build_index)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("write_stop_card", _write_stop_card)
    graph.add_node("validate_stop_card", _validate_stop_card)
    graph.add_node("accumulate_and_advance", _accumulate_and_advance)
    graph.add_node("render_outputs", _render_outputs)

    graph.set_entry_point("ingest_route")
    graph.add_edge("ingest_route", "sample_points")
    graph.add_edge("sample_points", "discover_candidates")
    graph.add_edge("discover_candidates", "fetch_and_filter_pages")
    graph.add_edge("fetch_and_filter_pages", "select_stops")
    graph.add_edge("select_stops", "build_index")
    graph.add_edge("build_index", "retrieve")
    graph.add_edge("retrieve", "write_stop_card")
    graph.add_edge("write_stop_card", "validate_stop_card")
    graph.add_edge("validate_stop_card", "accumulate_and_advance")

    graph.add_conditional_edges(
        "accumulate_and_advance",
        _should_continue,
        {"continue": "retrieve", "done": "render_outputs"},
    )
    graph.add_edge("render_outputs", END)

    return graph.compile()


def offline_tools() -> Tools:
    """Return offline-only tool implementations for the MVP."""
    pages = mvp_data.minimal_wiki_pages()
    # Return different candidates for early vs late route points so the per-stop
    # loop runs more than once in the MVP.
    geosearch_early = [
        {"pageid": 2001, "title": "Interstate 81", "lat": 38.15, "lon": -79.07, "dist_m": 1200},
        {"pageid": 1001, "title": "Basalt", "lat": 38.46, "lon": -78.88, "dist_m": 6200},
    ]
    geosearch_late = [
        {"pageid": 2001, "title": "Interstate 81", "lat": 38.15, "lon": -79.07, "dist_m": 1200},
        {
            "pageid": 1002,
            "title": "Appalachian Mountains",
            "lat": 38.75,
            "lon": -78.67,
            "dist_m": 9100,
        },
    ]

    def geosearch(
        lat: float, lon: float, radius_m: int, limit: int, language: str
    ) -> list[dict[str, Any]]:
        _ = (lat, lon, radius_m, language)
        results = geosearch_early if lat < 38.6 else geosearch_late
        return results[:limit]

    def fetch_page(pageid: int, language: str) -> dict[str, Any]:
        _ = language
        if pageid not in pages:
            raise KeyError(f"Unknown pageid: {pageid}")
        return pages[pageid]

    return Tools(geosearch=geosearch, fetch_page=fetch_page)


def run_mvp(
    *,
    route_points: list[dict[str, float]] | None = None,
    config: MVPConfig | None = None,
    route_name: str = "mvp_route",
    out_dir: str | Path | None = None,
    tools: Tools | None = None,
) -> GraphState:
    """Execute the offline MVP graph and return the resulting state."""
    cfg = config or _default_config()
    app = build_mvp_app(tools=tools)
    state: GraphState = {"config": cfg, "route_name": route_name}
    if route_points is not None:
        state["route_points"] = route_points

    result: GraphState = app.invoke(state)  # type: ignore[assignment]

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "guide.json").write_text(json.dumps(result["guide"], indent=2) + "\n")
        (out_path / "guide.md").write_text(result["guide_markdown"])

    return result
