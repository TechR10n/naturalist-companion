# Roadside Naturalist (Wikipedia-only) — v0 MVP Spec

Build a “Roadside Naturalist”-style guide for **any walkable/drivable route** by using **only Wikipedia content** as the factual source of truth.

This is a direct sibling of `docs/roadside_geology_v0_mvp_spec.md`, but expands “what to notice” to include:
- **vegetation** (plants, forests, notable species)
- **ecology** (habitats, ecoregions, communities)
- **geology** (when present/obvious from Wikipedia geo pages)

The core idea remains the same: keep v0 narrow and testable, then broaden sources in v1+.

---

## 1) MVP Goal

Given a route polyline (list of lat/lon points), generate a structured set of **stop cards** like:
- “Stop 4 — Riparian corridor + floodplain ecology near X”
- “Stop 7 — Mixed hardwood slope forest + understory plants”
- “Stop 9 — Roadcut exposures (sedimentary bedding) near X”

### Core workflow (must implement)
`route → sample points → Wikipedia GeoSearch → retrieve source text → generate stop cards with citations`

---

## 2) Non-Goals (explicitly out of scope for v0)

- Building a full iOS app. (v0 accepts route input from files/JSON; iPhone integration is v1.)
- Guaranteed species identification. (v0 is “guide + prompts”, not a taxonomic authority.)
- Using non-Wikipedia factual sources. (No iNaturalist/GBIF/USGS in v0.)
- Safety-critical guidance (terrain, water crossings, wildlife safety).

---

## 3) Inputs / Outputs

### Inputs

**A. Route polyline (required)**
- `route_points`: ordered list of `{lat, lon}` from start → end

**Accepted v0 formats**
- JSON list (paste into notebook)
- GPX track file (parse into points)

**B. Config (required)**
- `sample_every_m`: e.g., `3000` (sample every 3 km)
- `geosearch_radius_m`: e.g., `15000`
- `geosearch_limit`: e.g., `50`
- `max_stops`: e.g., `12`
- `min_stop_spacing_m`: e.g., `10000`
- `language`: e.g., `"en"`

**C. User intent (optional, recommended)**
- `theme`: e.g., `"spring wildflowers"`, `"wetlands"`, `"forest succession"`, `"karst"`
- `audience_level`: `"beginner" | "enthusiast"`
- `season`: `"spring" | "summer" | "fall" | "winter"` (v0 is narrative only; no hard facts without citations)

### Outputs

**A. `guide.json` (required)**
- Machine-readable stop cards

**B. `guide.md` (required)**
- Human-readable guide

---

## 4) Data Contracts

### RoutePoint
```json
{
  "i": 0,
  "lat": 39.73915,
  "lon": -104.98470,
  "cum_dist_m": 0
}
```

### WikiCandidate (from GeoSearch)
```json
{
  "pageid": 12345,
  "title": "Riparian zone",
  "lat": 39.7,
  "lon": -105.0,
  "dist_m": 8200
}
```

### StopCard (primary output)
```json
{
  "stop_id": "stop_03",
  "route_km": 42.7,
  "center": {"lat": 39.70, "lon": -105.00},
  "title": "Riparian corridor ecology near <place>",
  "domains": ["ecology", "vegetation"],
  "why_stop": "1–2 sentences, Wikipedia-grounded.",
  "what_to_look_for": ["bullets..."],
  "key_facts": ["bullets..."],
  "photo_prompts": ["bullets..."],
  "confidence": 0.0,
  "citations": [
    {"title": "Riparian zone", "url": "https://en.wikipedia.org/wiki/Riparian_zone", "pageid": 12345}
  ]
}
```

### Guide
```json
{
  "route": {"name": "optional", "num_points": 1234, "length_km": 210.3},
  "generated_at": "2026-02-03T12:34:56Z",
  "config": {"sample_every_m": 3000, "geosearch_radius_m": 15000, "max_stops": 12},
  "stops": [/* StopCard[] */]
}
```

---

## 5) Wikipedia-Only Sourcing Rules (hard requirement)

The LLM must follow these constraints:
- **Only** use facts present in the retrieved Wikipedia text snippets.
- Every factual bullet must be supported by at least one citation URL.
- If sources are insufficient, explicitly say: **“Wikipedia sources retrieved for this route don’t contain enough detail to answer.”**
- No external URLs in citations (must match `*.wikipedia.org/wiki/...`).

---

## 6) Route → Stops Algorithm (v0)

Identical to the geology v0 pipeline, but with a broader relevance filter.

### Step 1 — Ingest route
Input a polyline (GPX/JSON) and compute cumulative distance.

### Step 2 — Sample points
Sample by `sample_every_m`.

### Step 3 — GeoSearch Wikipedia near each sample point
GeoSearch around each sample point to get candidate pages.

### Step 4 — Fetch source text for candidates
Fetch URL + extract/summary (+ optional categories).

### Step 5 — Filter for “naturalist relevance”
Start heuristic and tighten with eval.

**Keyword seed (examples)**
- vegetation/botany: `tree`, `shrub`, `wildflower`, `flora`, `forest`, `oak`, `maple`, `pine`, `rhododendron`, `fern`
- ecology: `ecology`, `habitat`, `wetland`, `riparian`, `meadow`, `ecoregion`, `watershed`, `succession`
- geology (keep): `formation`, `limestone`, `shale`, `sandstone`, `fault`, `karst`

**Exclusions**
- obvious non-naturalist pages (sports, people) unless the summary contains strong ecology/botany/geology terms.

### Step 6 — Select stops (avoid duplicates + clustering)
Same greedy spacing approach as geology v0:
- dedupe by `pageid`
- score by relevance + proximity + diversity of domains
- greedy pick with `min_stop_spacing_m` until `max_stops`

---

## 7) Retrieval + Generation (RAG)

Same shape as geology v0:
- chunk extracts (~500–1,000 tokens) with metadata (`title`, `url`, `pageid`, `route_km`)
- embed + local vector store (FAISS/Chroma)
- generate StopCards with a strict JSON schema and citation rules

---

## 8) Agentic Workflow (LangGraph)

Minimum viable tools:
- `geosearch(lat, lon, radius_m, limit) -> WikiCandidate[]`
- `fetch_page(pageid) -> {title, url, summary, content}`
- `retrieve(stop_context) -> chunks[]`
- `write_stop_card(stop_context, chunks) -> StopCard`
- `validate_stop_card(stop_card) -> {ok, errors[]}`

Graph outline:
`ingest_route → sample_points → discover_candidates → filter_and_select_stops → (retrieve → write → validate)* → render_outputs`

---

## 9) Evaluation (v0)

### Automated checks (required)
- Citation coverage: every StopCard has `>= 1` citation and all URLs are `wikipedia.org`.
- Schema validity: output matches StopCard schema.
- Dupes: no duplicate `pageid` across stops.
- Domain diversity: ensure not all stops collapse into “town pages” or a single topic category.

### Small human eval (recommended)
Score each stop (0–2):
- relevance to a naturalist observer
- Wikipedia-grounded factuality
- usefulness of photo prompts

---

## 10) iPhone Input (v1 direction, not required for v0)

v0 accepts GPX/JSON. iOS v1 can add:
- route import + offline “corridor packs”
- camera capture + observation notes
- on-device coarse labels (privacy + cost), cloud for detailed grounding

---

## 11) Definition of Done (v0)

- For a short test route (20–100 km), the notebook produces:
  - `guide.json` with 5–12 stops across vegetation/ecology/geology
  - `guide.md` with readable stop cards
  - all facts are cited with Wikipedia URLs only

