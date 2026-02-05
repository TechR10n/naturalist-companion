# Camera -> Geology (iPhone -> Cloud) Architecture Notes

This document describes a camera-first workflow for the future iOS app. The core idea is to keep the camera experience fast and to keep the explanation grounded in retrieved sources.

## What we are trying to do

Input
- One or more photos (wide shot + close-up).
- Optional metadata: GPS (with user consent), compass heading, timestamp, short user note.

Output
- Structured classification:
  - top hypotheses with confidence
  - visible features (texture, layering, fractures)
  - follow-up questions and photo requests
  - citations for background explanations
- A short narration derived from the structured result

Non-goals
- Safety-critical decisions (slope stability, rockfall risk).
- Guaranteed identification from a single photo.

## Reference architecture (cloud-first)

```plantuml
@startuml
left to right direction

rectangle "iPhone (On-device)" as D {
  rectangle "Camera capture (1-3 photos)" as Cam
  rectangle "GPS/EXIF + user note (optional)" as Meta
  rectangle "Preflight (resize + quality checks)" as Pre
}

rectangle "Cloud (GCP or future DBX)" as C {
  rectangle "API layer\nAuth + rate limits" as API
  database "Object store\noptional image blobs" as Store
  rectangle "Multimodal model\nclassification + features" as VLM
  rectangle "RAG orchestrator\nretrieval + grounding" as RAG
  database "Vector store\nWikipedia slice" as VS
  rectangle "Observability\nlogs + traces" as Obs
}

Cam --> Pre
Meta --> Pre
Pre --> API : HTTPS
API --> Store
API --> VLM
API --> RAG
RAG <--> VS
VLM --> API
RAG --> API
API --> D : JSON result + narration
API --> Obs

@enduml
```

## Where agentic Wikipedia fits

1. Vision model proposes top candidates and visible features.
2. Orchestrator generates Wikipedia queries from those candidates.
3. Retrieval returns a small slice of citations.
4. Final explanation is grounded in retrieved sources.

This separation lets the vision model suggest possibilities without making ungrounded claims.

## Request and response shape (conceptual)

Request
- `images[]`: JPEG/HEIC bytes or signed URLs
- `location`: `{ lat, lon, accuracy_m }` (optional)
- `context`: `{ corridor: "I-81", note: "roadcut near Roanoke" }` (optional)
- `preferences`: `{ store_images: false }`

Response
- `hypotheses[]`: labels with `confidence` and `why`
- `features[]`: visible structures with confidence
- `follow_up`: questions and photo requests
- `citations[]`: Wikipedia URLs
- `narration`: short text for the UI

## Guardrails

- Always return top-k hypotheses, not a single claim.
- Ask for the next best photo when uncertain.
- Use location as a soft prior only with consent.
- Default to not storing user photos.
