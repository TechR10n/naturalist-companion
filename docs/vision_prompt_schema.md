# Vision Prompt Schema (Geology)

Use this schema for camera-based geology classification. The model must return **only** JSON that validates to this shape.

## JSON shape

```json
{
  "version": "1.0",
  "hypotheses": [
    {
      "label": "Sandstone",
      "confidence": 0.62,
      "why": ["Layered bedding visible", "Grainy texture"],
      "disambiguators": ["Check for cemented grains", "Look for cross-bedding"]
    }
  ],
  "features": [
    {
      "label": "Layering",
      "confidence": 0.74,
      "why": ["Parallel bands across the outcrop"]
    }
  ],
  "follow_up": {
    "questions": ["Is the surface gritty to the touch?"],
    "photo_requests": ["Take a close-up with a coin for scale"]
  },
  "safety_notes": ["Do not approach unstable slopes."],
  "citations": [
    {"title": "Sandstone", "url": "https://en.wikipedia.org/wiki/Sandstone", "pageid": 12345}
  ]
}
```

## Rules

- Return **top-k hypotheses**, not a single answer.
- Confidence must be calibrated (0.0 to 1.0).
- Citations must be Wikipedia URLs only.
- If sources are insufficient, say so in `safety_notes` or a top-level `limitations` string (optional).
