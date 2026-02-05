# Roadside Geology (Wikipedia-only) - v0 MVP Spec

Build a short, citation-grounded field guide for any drivable route using only Wikipedia content.

## Goal

Given a route polyline, generate a set of stop cards with citations and a readable guide.

## Inputs

- Route points: list of `{ lat, lon }` or GPX track.
- Config:
  - `sample_every_m`
  - `geosearch_radius_m`
  - `geosearch_limit`
  - `max_stops`
  - `min_stop_spacing_m`
  - `language`
- Optional theme (volcanism, fossils, glaciation)

## Outputs

- `guide.json` (machine-readable)
- `guide.md` (human-readable)

## Core workflow

`route -> sample points -> Wikipedia GeoSearch -> retrieve -> write stop cards -> validate -> render`

## Data contracts

StopCard (minimal)
```json
{
  "stop_id": "stop_03",
  "route_km": 42.7,
  "center": {"lat": 39.70, "lon": -105.00},
  "title": "Basalt flows near <place>",
  "why_stop": "1-2 sentences grounded in citations.",
  "what_to_look_for": ["bullets"],
  "key_facts": ["bullets"],
  "photo_prompts": ["bullets"],
  "confidence": 0.0,
  "citations": [
    {"title": "Basalt", "url": "https://en.wikipedia.org/wiki/Basalt", "pageid": 12345}
  ]
}
```

Guide (minimal)
```json
{
  "route": {"name": "optional", "num_points": 1234, "length_km": 210.3},
  "generated_at": "2026-02-05T12:34:56Z",
  "config": {"sample_every_m": 5000, "geosearch_radius_m": 15000, "max_stops": 12},
  "stops": []
}
```

## Wikipedia-only sourcing rules

- Only use facts present in retrieved Wikipedia text.
- Every factual bullet must have a Wikipedia URL citation.
- If sources are insufficient, say so explicitly.

## Algorithm (v0)

1. Ingest route and compute cumulative distance.
2. Sample points every `sample_every_m`.
3. GeoSearch Wikipedia around each sample point.
4. Fetch page summaries/extracts and filter for geology relevance.
5. Deduplicate, score, and select stops with spacing constraints.
6. Chunk, embed, and retrieve top-k chunks per stop.
7. Generate stop cards with citations.
8. Validate schema and citation rules.

## Non-goals

- Guaranteeing roadside access or safety.
- Using non-Wikipedia sources.
