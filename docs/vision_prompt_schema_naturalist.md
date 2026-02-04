# Vision Classification Prompt + Output Schema (Naturalist Flâneur) (Draft)

Goal: get **structured, testable output** from a multimodal model for vegetation/ecology/geology observations. Narration comes later.

## Output schema (JSON)

Return JSON only (no markdown fences). Keep it short.

```json
{
  "version": "0.1",
  "domains": ["vegetation", "ecology", "geology"],
  "taxa_hypotheses": [
    {
      "rank": "family|genus|species|unknown",
      "label": "acer (maples)",
      "confidence": 0.0,
      "why": ["short visual cues"],
      "disambiguators": ["what to check next"]
    }
  ],
  "habitat_hypotheses": [
    {
      "label": "riparian edge",
      "confidence": 0.0,
      "why": ["short visual cues"]
    }
  ],
  "geology_hypotheses": [
    {
      "label": "sedimentary rock (layered)",
      "confidence": 0.0,
      "why": ["short visual cues"],
      "disambiguators": ["what to check next"]
    }
  ],
  "follow_up": {
    "questions": ["short questions the user can answer"],
    "photo_requests": ["wide shot", "close-up", "underside", "scale reference"]
  },
  "safety_notes": ["Do not use this for safety-critical decisions."],
  "notes": ["optional short freeform note"]
}
```

### Guidance

- `confidence` is **0.0–1.0**, calibrated (not vibes).
- Prefer the best-supported taxonomic rank; if species is not supportable, return genus/family.
- Keep `why` concrete and visual (venation, margin, bark texture, layering, grain size).
- If nothing is identifiable, return `unknown` and ask for the next best photo.

## Prompt template (multimodal)

**System**
- You are a field-assistant for naturalist observation (plants + habitat + geology).
- You must not claim certainty when the image is ambiguous.
- You must return JSON that matches the schema.

**User**
- Context: (optional) location, season, “trail edge”, “roadcut”, elevation, etc.
- Images: (1–3 photos)
- Question: “What am I looking at and what should I photograph next to be sure?”

**Developer (optional)**
- Provide a constrained label set for MVP:
  - taxa: a small list of common plants for one region (plus `unknown`)
  - habitats: a small list of coarse habitat types
  - geology: coarse rock families + a few common rocks/features

