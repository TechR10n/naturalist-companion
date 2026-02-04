# Use-case: Naturalist Flâneur (Vegetation + Ecology + Geology) (Draft)

The “flâneur” version of a field guide is less about search and more about *noticing*: the app helps you name what you’re seeing, connect it to the surrounding ecosystem and substrate, and gently coach you toward the next observation that reduces uncertainty.

## The scene

- The user is walking a trail, wandering a neighborhood greenway, or driving a scenic corridor.
- They see something interesting:
  - a leaf + bark combo they don’t recognize
  - a wet meadow edge with unfamiliar grasses
  - a roadcut with layered rock and a seep line
- They want: “What is this? What should I look at next? Why is it here?”

## What the system should do (MVP)

1) Identify across domains (top‑k, not single answers):
   - **Vegetation:** plant hypotheses at the best-supported rank (family/genus/species)
   - **Ecology:** habitat + community context hypotheses (riparian edge, meadow, mixed hardwoods, etc.)
   - **Geology:** coarse rock/material + visible structure hypotheses when relevant
2) Explain *why* (grounded):
   - concrete visual cues (shape, venation, texture, layering)
   - short definitions and context with citations (v0: Wikipedia only)
3) Ask for the next best evidence:
   - “Photo the underside / buds / twig”
   - “Wide shot of the stand + close-up with scale”
   - “Add a second angle to show layering”
4) Create a field note artifact:
   - observation summary + hypotheses + photos + citations
   - export/share (later: community verification)

## What the user hears (narration)

Something like:

> “This looks consistent with a maple-type leaf (opposite arrangement, palmate veins). I’m not sure on species yet—can you photograph the buds or the underside of the leaf? This area also looks like a moist hardwood slope, which often supports spring ephemerals. If you want, take one wide shot of the stand so I can use the habitat as a soft prior.”

## MVP success criteria

- The system frequently says “I’m not sure yet” when it should.
- The follow‑up photo requests measurably increase confidence or narrow hypotheses.
- Explanations are grounded with citations (no invented facts).
- The UI produces a reusable field note (even if IDs remain coarse).

