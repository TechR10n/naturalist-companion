# Vision Classification Prompt + Output Schema (Draft)

The goal is to get **structured, testable output** from a multimodal model. Narration comes later.

## Output schema (JSON)

Return JSON only (no markdown fences). Keep it short.

```json
{
  "version": "0.1",
  "hypotheses": [
    {
      "label": "shale",
      "confidence": 0.0,
      "why": ["short visual cues"],
      "disambiguators": ["what to check next"]
    }
  ],
  "features": [
    {
      "label": "bedding planes",
      "confidence": 0.0,
      "why": ["short visual cues"]
    }
  ],
  "follow_up": {
    "questions": ["short questions the user can answer"],
    "photo_requests": ["wide shot", "close-up with scale", "angle change"]
  },
  "safety_notes": [
    "Do not use this for safety-critical decisions."
  ]
}
```

### Guidance

- `confidence` is **0.0–1.0**, calibrated (not vibes).
- Keep `why` concrete and visual (“thin, parallel layers”, “visible grains”, “glassy luster”).
- Prefer coarse labels if uncertain (“sedimentary rock (fine‑grained)”) instead of a confident specific ID.

## Prompt template (multimodal)

**System**
- You are a field-assistant for roadside geology.
- You must not claim certainty when the image is ambiguous.
- You must return JSON that matches the schema.

**User**
- Context: (optional) corridor, state, “roadcut”, elevation, etc.
- Images: (1–3 photos)
- Question: “What rock/feature is this and what should I photograph next to be sure?”

**Developer (optional)**
- Provide an allowed label set for the MVP:
  - rock types: `shale`, `sandstone`, `limestone`, `granite`, `basalt`, `gneiss`, `schist`, `quartzite`, `conglomerate`, `unknown`
  - features: `bedding`, `cross_bedding`, `foliation`, `joints`, `folding`, `faulting`, `veins`, `weathering_rind`, `unknown`

Start small; widen the label set only after you have evaluation data.

