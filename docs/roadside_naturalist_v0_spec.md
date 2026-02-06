# Roadside Naturalist (Wikipedia-only) - v0 Baseline Spec

This spec extends the geology guide into a broader naturalist experience (flora, fauna, ecology, and geology), still grounded in Wikipedia.

## Goal

Given a route, generate a set of stop cards with naturalist context and citations.

## Inputs

- Route points: `{ lat, lon }` list or GPX track
- Config:
  - `sample_every_m`
  - `geosearch_radius_m`
  - `geosearch_limit`
  - `max_stops`
  - `min_stop_spacing_m`
  - `language`
- Optional theme:
  - `flora`, `fauna`, `geology`, `ecology`, `mixed`

## Outputs

- `guide.json`
- `guide.md`

## Stop card fields (extended)

- `title`
- `why_stop`
- `what_to_look_for` (visual cues)
- `key_facts` (short bullets)
- `photo_prompts`
- `best_time_of_day` (optional)
- `citations` (Wikipedia URLs)

## Wikipedia-only rules

- All facts must be grounded in retrieved Wikipedia text.
- Citations must be `*.wikipedia.org/wiki/...`.
- If sources are insufficient, say so.

## Non-goals

- Precise species identification from images.
- Safety-critical guidance.
- Non-Wikipedia sources.
