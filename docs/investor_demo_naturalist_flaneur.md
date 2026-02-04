# Investor Demo Frame: AI Naturalist Flâneur (iOS)

Treat the “Agentic Wikipedia / Roadside Geology” assignment as a **fundable product demo**: an iPhone app that turns any walk/drive into a guided field experience across **vegetation + ecology + geology** (and eventually birds/wildlife).

The assignment’s core strength (agentic workflow + grounded citations) becomes the product’s credibility layer: *“Don’t just guess—show your sources, ask for better evidence, and keep uncertainty explicit.”*

---

## 1) The one‑liner

**AI Naturalist Flâneur** is a pocket field guide that uses camera + location + retrieval to identify what you’re seeing, explain how it fits into the local ecosystem/geology, and coach you to capture the *next best* observation when it’s unsure.

---

## 2) The “why” (problem → wedge)

People notice things outside—plants, strange rock, a wetland edge—but:
- IDs are often **narrow** (birds *or* plants *or* rocks) and don’t connect them.
- Apps frequently feel **overconfident**, with weak grounding and little teaching.
- What users really want is “**What is this? What should I look for next? Why is it here?**”

Wedge: start with a **single corridor / park / trail system** (like the repo’s I‑81/Appalachia slice) so retrieval + evaluation can be made excellent before going broad.

---

## 3) The demo (3–5 minutes, investor‑style)

### Act 1 — “Flâneur Mode” (serendipity, not search)
**User opens the app on a walk.**
- App shows 3 “today’s field prompts” based on (optional) location + season:
  - *Vegetation:* “Look for opposite branching + smooth gray bark (maple candidates).”
  - *Ecology:* “This slope aspect + understory suggests a mesic hardwood community—watch for ferns and spring ephemerals.”
  - *Geology:* “Roadcut ahead often exposes layered sedimentary units; capture a wide shot + close‑up with scale.”

### Act 2 — “Point & Ask” (camera + uncertainty + next best evidence)
**User snaps a leaf + bark photo.**
- App returns **top‑3 hypotheses** (species/genus/family as appropriate), with:
  - concrete visual cues (“serrated margin”, “palmate venation”, “opposite leaves”)
  - an explicit confidence score
  - a *single* follow‑up request that reduces uncertainty (“photo the underside”, “capture buds/twigs”, “include a coin for scale”)

### Act 3 — “Explain with receipts” (RAG citations)
**User taps “Why is this here?”**
- App gives a short explanation grounded in retrieved references (v0: Wikipedia):
  - species/habitat basics
  - likely ecosystem context (edge vs interior, riparian indicators, disturbance)
  - (if relevant) local geology context as a *soft prior* (“limestone areas often…”, with citations)

Close: “This is the same core engine you’re seeing in the assignment: tools + retrieval + citation‑checked generation. The product wraps it in camera UX + field‑note ergonomics.”

---

## 4) What’s already here (map the repo to the product)

Current repo direction already demonstrates the core investor‑credible pieces:
- **Grounded retrieval** over a constrained slice (Wikipedia → embeddings → local vector store).
- **Agentic workflow** (LangGraph) where tools are first‑class and outputs are validated.
- A clear path to camera input (see `docs/architecture_camera_vision.md` and `docs/vision_prompt_schema.md`).

Investor framing: v0 proves the “**truth layer**” (retrieval + citations + eval). v1 adds “**senses**” (camera + location) and “**habit building**” (flâneur prompts + notes).

---

## 5) Differentiation / moat (what makes this more than a demo)

- **Uncertainty‑aware UX:** top‑k hypotheses + calibration + “next best observation” coaching.
- **Contextual intelligence:** combine observation with habitat + geology context (without pretending location is certainty).
- **Grounded explanations:** citations and “no‑answer” behavior when sources are thin.
- **Compounding data:** (later) user‑consented field notes + feedback improve prompts, calibration, and on‑device models.

---

## 6) MVP scope (tight, shippable)

**MVP (6–10 weeks)**
- iOS capture + notes + “ask a question” UX
- Cloud endpoint: vision classification → retrieval → grounded response
- Start with a constrained label set:
  - coarse plant IDs (family/genus) + a handful of common species in one corridor/park
  - coarse habitat types (riparian, meadow edge, mixed hardwood forest, conifer stand)
  - geology coarse classes + a few common rocks/features
- Evaluation: calibration + citation validity + user usefulness

**Demo‑ready metric targets**
- >90% schema validity (structured output)
- >95% citation validity (whitelist sources in v0)
- “next photo” prompt improves confidence in a measurable subset of cases

---

## 7) Prototype → production (what investors care about)

- **Privacy default:** don’t store photos by default; location opt‑in; retention controls.
- **Cost controls:** on‑device first pass + cloud only when needed; caching of corridor packs.
- **Quality:** eval harness + regression tests + human spot checks; explicit “unknown” and “not enough evidence” outcomes.
- **Trust:** consistent citations, safe wording, and clear non‑goals (no safety‑critical guidance).

---

## 8) The ask (template)

“We’re raising a seed round to ship an iOS v1 with camera identification + grounded explanations for a single high‑quality region, then expand coverage and add community/partner packs.”

Use of funds (typical):
- iOS + design + UX polish (capture, notes, offline packs)
- ML/LLM engineering (multimodal + calibration + eval)
- Domain expertise (botany/ecology/geology advisory + dataset curation)

