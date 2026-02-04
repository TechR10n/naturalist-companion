# Starter use-case: I-81 through the Appalachians

If you want a place to start, start with a single ribbon of pavement: I‑81 threading through the long folds of the Appalachians. It’s familiar enough to picture, specific enough to retrieve, and broad enough to bump into geography, history, and a few towns with stories to tell.

## Mile 0: Why this slice?

It’s narrow enough to debug quickly, but rich enough to keep your retriever honest:
- geology/topography (Appalachians, Shenandoah Valley)
- cities/regions along the corridor
- highway route/history/interchanges

## Mile 1: What “done” looks like

We’re going to stay close to the road: answer Wikipedia-grounded questions about **Interstate 81** (plus nearby regional context) with citations back to the retrieved Wikipedia pages.

Out of scope (for now):
- live traffic, weather, closures, routing (needs non-Wikipedia data)
- turn-by-turn directions

## Mile 2: The first search sign (Wikipedia query seed)

Use this as a starting point (adjust freely). If retrieval gets “chatty”, tighten the query; if it feels thin, broaden it.

- `Interstate 81 OR Shenandoah Valley OR Roanoke OR Appalachian Mountains`

If you want “tighter”:
- `Interstate 81`

If you want “broader”:
- `Interstate 81 OR Interstate 77 OR Interstate 64 OR Blue Ridge Mountains OR Great Valley`

## Mile 3: A short road test (question set)

1) What states does Interstate 81 run through?
2) Which major cities does I-81 serve in Virginia?
3) How does I-81 relate to the Great Valley / Shenandoah Valley region?
4) What are the main interstates that intersect I-81 in Virginia?
5) Where does I-81 begin and end?
6) Why was I-81 routed where it is (historical/engineering context)?
7) What are notable mountain ranges or valleys near I-81 in VA/WV?
8) How does I-81 connect to I-40 / I-77 for access to the Smokies region?
9) What is the relationship between I-81 and US Route 11?
10) What national parks/forests are near segments of I-81 (as described on Wikipedia)?
11) What are the major freight/logistics roles attributed to I-81 (if described)?
12) What are the most common alternative routes mentioned for major sections (if described)?
13) How does I-81 connect to the Northeast corridor highway network (I-78, I-83, etc.)?
14) What are notable geographic features near Roanoke / Winchester that affect roadway alignment?
15) What major bridges/tunnels (if any) are on I-81 in the region (Wikipedia-described)?

## Mile 4: Signs you’re on the right road (MVP)

- Retrieval returns relevant pages for most questions (top-3 includes the “right” page)
- The final answer is grounded in retrieved text (no invented facts)
- Runtime is fast enough for iteration (small `max_docs`, then scale)
