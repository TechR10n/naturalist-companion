# Use-case: Roadcut Geology From an iPhone Camera (Draft)

The classic roadside guide works because it does two things at once: it tells you *what you’re seeing* and it hints at *why it’s there*. A camera-assisted version should do the same, but with better manners about uncertainty.

## The scene

- The user is traveling a corridor (say, I‑81).
- They spot an exposed roadcut—layered rock, a folded seam, a cliff of broken blocks.
- They take a wide photo, then a close-up, and (if they listen) they tuck a pen or coin into the frame for scale.

## What the system should do (MVP)

1) Identify coarse categories:
   - likely rock family (sedimentary vs igneous vs metamorphic)
   - a few plausible rock names (top‑3)
2) Identify visible features:
   - bedding / foliation / joints / folds, etc.
3) Ask for the next best evidence:
   - “Get closer to show grain size”
   - “Take a second photo from the side to show layering”
4) Provide a short explanation with citations:
   - basic definitions and context (Wikipedia to start)

## What the user hears (narration)

Something like:

> “This looks consistent with thin, layered sedimentary rock—possibly shale. The parallel bands are a good sign of bedding. If you can, take a close-up with a coin for scale and another shot at a shallow angle to show whether the layers split into plates.”

## MVP success criteria

- The system frequently says “I’m not sure yet” when it should.
- The follow‑up photo requests actually improve confidence.
- The explanation cites retrieved pages (not just model intuition).

