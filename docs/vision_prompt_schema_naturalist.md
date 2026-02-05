# Vision Prompt Schema (Naturalist)

Use this schema for camera-based naturalist identification (flora, fauna, geology). The model must return **only** JSON that validates to this shape.

## JSON shape

```json
{
  "version": "1.0",
  "domain": "flora",
  "hypotheses": [
    {
      "label": "Red oak",
      "confidence": 0.55,
      "why": ["Lobed leaves", "Bark texture"],
      "disambiguators": ["Check for pointed lobes", "Look for acorns"]
    }
  ],
  "features": [
    {
      "label": "Lobed leaves",
      "confidence": 0.71,
      "why": ["Distinct lobes visible on leaf edges"]
    }
  ],
  "follow_up": {
    "questions": ["Are the leaves opposite or alternate?"],
    "photo_requests": ["Photo of leaf underside"]
  },
  "safety_notes": ["Do not touch unknown plants."],
  "citations": [
    {"title": "Oak", "url": "https://en.wikipedia.org/wiki/Oak", "pageid": 12345}
  ]
}
```

## Rules

- `domain` must be one of: `flora`, `fauna`, `geology`, `mixed`.
- Return **top-k hypotheses** and show uncertainty.
- Citations must be Wikipedia URLs only.
- If uncertain, ask for the next best photo.
