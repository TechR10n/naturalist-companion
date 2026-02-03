# Roadside Geology (Wikipedia-only) — v0 MVP Spec

Build a “Roadside Geology”-style guide for **any drivable route** by using **only Wikipedia content** as the factual source of truth.

This spec is designed to be implemented using the same stack already used in this repo:
- **Day 1 / Vertex AI:** `langchain-google-vertexai`, `ChatVertexAI`, `VertexAIEmbeddings`, LangGraph
- **Day 2 / Databricks:** `databricks-langchain`, `ChatDatabricks`, `DatabricksEmbeddings`, LangGraph (+ optional Vector Search / MLflow)

---

## 1) MVP Goal

Given a route polyline (list of lat/lon points), generate a structured set of **stop cards** like:
- “Stop 3 — Basalt flows near X”
- “What you’re looking at”
- “Key facts (Wikipedia-sourced)”
- “Photo prompts”
- “Citations” (Wikipedia URLs)

### Core workflow (must implement)
`route → sample points → Wikipedia GeoSearch → retrieve source text → generate stop cards with citations`

---

## 2) Non-Goals (explicitly out of scope for v0)

- Building a full iOS app. (v0 accepts route input from files/JSON; iPhone integration is v1.)
- Guaranteeing every stop is roadside-accessible/safe/legal. (v0 includes disclaimers; later you can add land-access layers.)
- Using non-Wikipedia factual sources. (No external geology databases in v0.)

---

## 3) Inputs / Outputs

### Inputs

**A. Route polyline (required)**
- `route_points`: ordered list of `{lat, lon}` from start → end

**Accepted v0 formats**
- JSON list (paste into notebook)
- GPX track file (parse into points)

**B. Config (required)**
- `sample_every_m`: e.g., `5000` (sample every 5 km)
- `geosearch_radius_m`: e.g., `15000` (15 km)
- `geosearch_limit`: e.g., `50`
- `max_stops`: e.g., `12`
- `min_stop_spacing_m`: e.g., `15000` (avoid 5 stops clustered together)
- `language`: e.g., `"en"`

**C. User intent (optional, but recommended)**
- `theme`: e.g., `"volcanism"`, `"glaciation"`, `"fossils"`, `"tectonics"`
- `audience_level`: `"beginner" | "enthusiast"`

### Outputs

**A. `guide.json` (required)**
- Machine-readable stop cards

**B. `guide.md` (required)**
- Human-readable “Roadside Geology” guide

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
  "title": "Basalt",
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
  "title": "Basalt flows near <place>",
  "why_stop": "1–2 sentences, Wikipedia-grounded.",
  "what_to_look_for": ["bullets..."],
  "key_facts": ["bullets..."],
  "photo_prompts": ["bullets..."],
  "confidence": 0.0,
  "citations": [
    {"title": "Basalt", "url": "https://en.wikipedia.org/wiki/Basalt", "pageid": 12345}
  ]
}
```

### Guide
```json
{
  "route": {"name": "optional", "num_points": 1234, "length_km": 210.3},
  "generated_at": "2026-02-03T12:34:56Z",
  "config": {"sample_every_m": 5000, "geosearch_radius_m": 15000, "max_stops": 12},
  "stops": [/* StopCard[] */]
}
```

---

## 5) Wikipedia-Only Sourcing Rules (hard requirement)

The LLM must follow these constraints:
- **Only** use facts present in the retrieved Wikipedia text snippets.
- Every factual bullet must be supported by at least one citation URL.
- If sources are insufficient, explicitly say **“Wikipedia sources retrieved for this route don’t contain enough detail to answer.”**
- No external URLs in citations (must match `*.wikipedia.org/wiki/...`).

---

## 6) Route → Stops Algorithm (v0)

### Step 1 — Ingest route
Input a polyline:
- For v0: accept GPX or JSON points.
- Compute cumulative distance along the route (Haversine between consecutive points).

### Step 2 — Sample points
Create `sample_points` by taking the closest route point to each multiple of `sample_every_m`.

### Step 3 — GeoSearch Wikipedia near each sample point
Call Wikipedia API GeoSearch around each sample point to get candidate pages.

**Endpoint (conceptual)**
- `action=query&list=geosearch&gscoord={lat}|{lon}&gsradius={meters}&gslimit={n}&format=json`

### Step 4 — Fetch source text for candidates
For each candidate page (top N by distance):
- Fetch:
  - canonical Wikipedia URL
  - short summary/extract
  - (optional) categories
  - (optional) intro section / plaintext extract

### Step 5 — Filter for geology relevance
v0 filtering is allowed to be heuristic and imperfect.

**Heuristic options**
- Keyword filter on title/summary/categories:
  - geology terms: `fault`, `basalt`, `sandstone`, `granite`, `formation`, `stratigraphy`, `glacier`, `moraine`, `volcano`, `caldera`, `igneous`, `metamorphic`, `sedimentary`, `fossil`, `tectonic`, `orogeny`
- Exclude obvious non-geology pages (cities, sports teams, people) unless the summary contains strong geology terms.

### Step 6 — Select “stops” (avoid duplicates + clustering)
Goal: produce `<= max_stops` items that are:
- geology-relevant
- spread along the route (`min_stop_spacing_m`)
- each backed by strong Wikipedia content

**Simple approach (good enough for v0)**
1) Build a pool of candidates with a relevance score.
2) Deduplicate by `pageid` (keep best-scoring occurrence).
3) Sort by score desc.
4) Greedy pick: add stop if it’s at least `min_stop_spacing_m` from existing stops.
5) If you end up with too few stops, loosen spacing or increase radius.

---

## 7) Retrieval + Generation (RAG)

### Chunking
- Convert page extracts into chunks (~500–1,000 tokens each) with metadata:
  - `pageid`, `title`, `url`, `lat`, `lon`, `dist_m`, `route_km`

### Embeddings + Vector Store
- v0: FAISS (fastest), consistent with existing specs.
- Optional later:
  - Vertex AI Vector Search (Day 1 stretch)
  - Databricks Vector Search (Day 2 stretch)

### Generation prompt (pattern)
For each stop, provide:
- route context (route_km, lat/lon)
- a few retrieved chunks (top-k)
- a required JSON schema
- citation rules (“only Wikipedia URLs”)

**Stop card writing guidance**
- Keep it practical: what you can see from the road/turnouts.
- Avoid “go collect samples” language; keep it “observe/photograph”.
- If the Wikipedia sources don’t clearly describe a visible feature, say so.

---

## 8) Agentic Workflow (LangGraph)

Minimum viable graph nodes/tools:

**Tools**
- `geosearch(lat, lon, radius_m, limit) -> WikiCandidate[]`
- `fetch_page(pageid) -> {title, url, summary, content}`
- `retrieve(stop_context) -> chunks[]` (vector store similarity)
- `write_stop_card(stop_context, chunks) -> StopCard`
- `validate_stop_card(stop_card) -> {ok, errors[]}` (citations present, wikipedia-only, schema)

**Graph outline**
1) `ingest_route`
2) `sample_points`
3) `discover_candidates` (GeoSearch)
4) `filter_and_select_stops`
5) For each stop:
   - `retrieve`
   - `write_stop_card`
   - `validate_stop_card` (retry once with stricter prompt if invalid)
6) `render_outputs` (JSON + Markdown)

---

## 9) Evaluation (v0)

### Automated checks (required)
- **Citation coverage:** every `StopCard` has `>= 1` citation and all URLs are `wikipedia.org`.
- **Schema validity:** output matches the StopCard schema.
- **Dupes:** no duplicate `pageid` across stops (unless explicitly allowed).
- **On-topic:** stop titles/sections contain geology keywords above a threshold.

### Small human eval (recommended)
Create 5–10 route questions like:
- “What rock types are common along this segment?”
- “What geologic processes shaped this area?”
- “Which stop is best for seeing volcanic features?”

Score (0–2) each:
- Relevance
- Wikipedia-grounded factuality
- Usefulness for a roadside observer

---

## 10) iPhone Input (v1 direction, not required for v0)

v0 accepts GPX/JSON. For an iPhone-first UX later:
- Accept an **Apple Maps share link** and compute the route server-side, or
- Provide a **Share Sheet** extension to export a GPX/GeoJSON route into the app

Keep the core pipeline the same: route polyline → sample points → GeoSearch → stops.

---

## 11) Definition of Done (v0)

- Given a short test route (20–100 km), the notebook produces:
  - `guide.json` with 5–12 stops
  - `guide.md` with readable stop cards
  - All stop facts are cited with Wikipedia URLs
- Same route can be run on Vertex AI (Day 1) and Databricks (Day 2) with comparable output.

