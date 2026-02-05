# Use Case - Camera Roadcuts

## Scenario

A user pulls over safely at a roadcut and takes two photos: a wide shot and a close-up. They ask, "What am I looking at?"

## Goals

- Return top-k hypotheses (not a single answer).
- Explain visible features.
- Ask for the next best photo if uncertain.
- Provide citations for background explanations.

## Flow

1. User captures images.
2. Preflight checks for focus, exposure, and framing.
3. Vision model proposes hypotheses and features.
4. RAG retrieves Wikipedia citations.
5. App returns structured output + narration.

## Success criteria

- Top hypothesis is plausible in >= 70 percent of test cases.
- Follow-up prompt is actionable.
- Citations are present and valid.
