# Investor Demo - Naturalist Flaneur

## One-liner

A pocket naturalist that turns any drive or walk into a guided field trip: it identifies what you are seeing, explains why it matters, and backs claims with citations.

## Product story

- People are curious about their surroundings but do not have a field guide in their pocket.
- Existing tools overpromise on identification and rarely show sources.
- Agentic Naturalist is built to be grounded, honest about uncertainty, and helpful in the moment.

## The experience (demo flow)

1. User opens the app on a drive or hike.
2. The app suggests a few nearby stops or questions.
3. The user takes a photo of a rock outcrop or plant.
4. The app returns:
   - top hypotheses
   - visible features
   - follow-up questions
   - citations
5. The user learns what to look for next.

## Why now

- Multimodal models can describe images.
- Retrieval plus citations can reduce hallucinations.
- Mobile cameras and location metadata make the experience contextual.

## Differentiation

- Grounded answers with citations.
- Calibrated confidence, not single-answer guesses.
- A guided exploration flow instead of a static answer box.

## Roadmap

- v0: local and GCP RAG pipeline with Wikipedia grounding.
- v1: camera-based classification with structured output.
- v2: polished iOS beta.
- v3: App Store launch.

## Risks and mitigations

- Incorrect identification -> always present top-k and follow-up questions.
- Source quality -> start with Wikipedia, then validate new sources.
- Privacy -> do not store photos by default.
