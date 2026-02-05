"""Ollama-backed vision helpers for the camera classification flow."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Any, TypedDict

ALLOWED_DOMAINS = {"flora", "fauna", "geology", "mixed"}
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llava:7b"


class VisionImage(TypedDict):
    """Image payload for a single camera input."""

    mime_type: str
    data: bytes


def coerce_domain(value: Any) -> str:
    """Normalize a requested domain to supported values."""
    raw = str(value or "").strip().lower()
    return raw if raw in ALLOWED_DOMAINS else "geology"


def _clamp_confidence(value: Any, *, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        v = default
    return max(0.0, min(1.0, round(v, 3)))


def _string_list(value: Any, *, max_items: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        s = str(item).strip()
        if not s:
            continue
        out.append(s)
        if len(out) >= max_items:
            break
    return out


def _normalize_hypotheses(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        out.append(
            {
                "label": label,
                "confidence": _clamp_confidence(item.get("confidence"), default=0.25),
                "why": _string_list(item.get("why"), max_items=6),
                "disambiguators": _string_list(item.get("disambiguators"), max_items=6),
            }
        )
        if len(out) >= 5:
            break
    return out


def _normalize_features(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        out.append(
            {
                "label": label,
                "confidence": _clamp_confidence(item.get("confidence"), default=0.25),
                "why": _string_list(item.get("why"), max_items=6),
            }
        )
        if len(out) >= 8:
            break
    return out


def _normalize_follow_up(value: Any) -> dict[str, list[str]]:
    src = value if isinstance(value, dict) else {}
    return {
        "questions": _string_list(src.get("questions"), max_items=6),
        "photo_requests": _string_list(src.get("photo_requests"), max_items=6),
    }


def _normalize_citations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if "wikipedia.org/wiki/" not in url:
            continue
        title = str(item.get("title") or "").strip() or url.rsplit("/", 1)[-1].replace("_", " ")
        pageid_raw = item.get("pageid")
        citation: dict[str, Any] = {"title": title, "url": url}
        try:
            citation["pageid"] = int(pageid_raw)
        except Exception:
            pass
        out.append(citation)
        if len(out) >= 8:
            break
    return out


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline >= 0:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("No JSON object found in Ollama response.")

    try:
        parsed = json.loads(text[start : end + 1])
    except Exception as e:
        raise RuntimeError("Could not parse Ollama JSON payload.") from e
    if not isinstance(parsed, dict):
        raise RuntimeError("Ollama payload was not a JSON object.")
    return parsed


def _stub_result(*, domain: str, note: str, image_count: int) -> dict[str, Any]:
    note_suffix = f" User note: {note.strip()}" if str(note or "").strip() else ""
    common = {
        "version": "1.0",
        "domain": domain,
        "follow_up": {
            "questions": [
                "Can you share whether this site is dry, wet, shaded, or exposed?",
            ],
            "photo_requests": [
                "Take one wide context shot from farther back.",
                "Take one close-up with a coin or hand for scale.",
            ],
        },
        "safety_notes": [
            "Use legal pull-offs and do not stand near unstable slopes.",
            "This is a best-effort guess from photos; verify in person.",
        ],
        "citations": [
            {"title": "Geology", "url": "https://en.wikipedia.org/wiki/Geology"},
        ],
        "limitations": (
            f"Stub analysis for {image_count} image(s). Use Ollama for model-generated output."
            f"{note_suffix}"
        ),
    }

    if domain == "flora":
        return {
            **common,
            "hypotheses": [
                {
                    "label": "Oak (Quercus, likely)",
                    "confidence": 0.49,
                    "why": ["Lobed leaf silhouette", "Coarse bark texture"],
                    "disambiguators": ["Check if lobes are pointed", "Photograph underside of leaves"],
                },
                {
                    "label": "Maple (Acer, possible)",
                    "confidence": 0.28,
                    "why": ["Potential palmate leaf pattern"],
                    "disambiguators": ["Capture branch node arrangement"],
                },
            ],
            "features": [
                {
                    "label": "Lobed leaves",
                    "confidence": 0.62,
                    "why": ["Distinct edge segmentation in the photo"],
                }
            ],
            "citations": [
                {"title": "Oak", "url": "https://en.wikipedia.org/wiki/Oak"},
                {"title": "Maple", "url": "https://en.wikipedia.org/wiki/Maple"},
            ],
        }

    if domain == "fauna":
        return {
            **common,
            "hypotheses": [
                {
                    "label": "Songbird (family-level guess)",
                    "confidence": 0.41,
                    "why": ["Small body profile", "Perching posture"],
                    "disambiguators": ["Capture side profile", "Capture tail and beak shape"],
                },
                {
                    "label": "Squirrel (possible)",
                    "confidence": 0.24,
                    "why": ["Mammal-shaped outline in scene"],
                    "disambiguators": ["Take a sharper close-up of head and tail"],
                },
            ],
            "features": [
                {
                    "label": "Small moving subject",
                    "confidence": 0.58,
                    "why": ["Subject appears compact and elevated above ground"],
                }
            ],
            "citations": [
                {"title": "Bird", "url": "https://en.wikipedia.org/wiki/Bird"},
                {"title": "Squirrel", "url": "https://en.wikipedia.org/wiki/Squirrel"},
            ],
        }

    if domain == "mixed":
        return {
            **common,
            "hypotheses": [
                {
                    "label": "Layered sedimentary outcrop",
                    "confidence": 0.44,
                    "why": ["Horizontally repeated bands"],
                    "disambiguators": ["Get a close-up of grain texture"],
                },
                {
                    "label": "Vegetated slope context",
                    "confidence": 0.31,
                    "why": ["Plant cover mixed with exposed rock"],
                    "disambiguators": ["Photograph leaf and bark details separately"],
                },
            ],
            "features": [
                {
                    "label": "Layering",
                    "confidence": 0.63,
                    "why": ["Parallel visual bands in exposed section"],
                },
                {
                    "label": "Mixed substrate",
                    "confidence": 0.49,
                    "why": ["Rock exposure interleaved with vegetation"],
                },
            ],
            "citations": [
                {"title": "Sedimentary rock", "url": "https://en.wikipedia.org/wiki/Sedimentary_rock"},
                {"title": "Roadcut", "url": "https://en.wikipedia.org/wiki/Roadcut"},
            ],
        }

    return {
        **common,
        "hypotheses": [
            {
                "label": "Sandstone",
                "confidence": 0.56,
                "why": ["Layered bedding", "Granular surface texture"],
                "disambiguators": ["Look for cemented grains", "Check for cross-bedding at close range"],
            },
            {
                "label": "Shale",
                "confidence": 0.23,
                "why": ["Possible thin lamination"],
                "disambiguators": ["Check if rock splits into thin sheets"],
            },
            {
                "label": "Limestone",
                "confidence": 0.14,
                "why": ["Possible massive bedding intervals"],
                "disambiguators": ["Inspect for fossil fragments or carbonate texture"],
            },
        ],
        "features": [
            {
                "label": "Layering",
                "confidence": 0.74,
                "why": ["Repeated horizontal to sub-horizontal bands"],
            },
            {
                "label": "Grainy texture",
                "confidence": 0.58,
                "why": ["Visible medium-grain texture in exposed face"],
            },
        ],
        "citations": [
            {"title": "Sandstone", "url": "https://en.wikipedia.org/wiki/Sandstone"},
            {"title": "Shale", "url": "https://en.wikipedia.org/wiki/Shale"},
            {"title": "Roadcut", "url": "https://en.wikipedia.org/wiki/Roadcut"},
        ],
    }


def _build_prompt(*, domain: str, note: str, image_count: int) -> str:
    note_text = str(note or "").strip() or "none"
    return (
        "You are a cautious naturalist assistant. Analyze the provided photo(s) and return only JSON. "
        "No markdown.\n"
        "Required keys: version, domain, hypotheses, features, follow_up, safety_notes, citations.\n"
        "Constraints:\n"
        "- domain must be one of flora, fauna, geology, mixed.\n"
        "- hypotheses: top-k candidates with calibrated confidence 0..1.\n"
        "- features: visible signals and confidence 0..1.\n"
        "- follow_up: include both questions[] and photo_requests[].\n"
        "- citations: Wikipedia URLs only.\n"
        f"Requested domain: {domain}\n"
        f"Image count: {image_count}\n"
        f"User note: {note_text}\n"
    )


def _ollama_vision_chat(
    *,
    images: list[VisionImage],
    domain: str,
    note: str,
    base_url: str,
    model: str,
    timeout_s: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": "Return valid JSON only."},
            {
                "role": "user",
                "content": _build_prompt(domain=domain, note=note, image_count=len(images)),
                "images": [base64.b64encode(img["data"]).decode("ascii") for img in images],
            },
        ],
    }
    url = base_url.rstrip("/") + "/api/chat"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")
        except Exception:
            pass
        raise RuntimeError(f"Ollama HTTP {e.code}: {detail or e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach Ollama at {base_url} ({e.reason})") from e

    try:
        parsed = json.loads(raw)
    except Exception as e:
        raise RuntimeError("Ollama response was not valid JSON.") from e

    content: Any = ""
    if isinstance(parsed, dict):
        message = parsed.get("message")
        if isinstance(message, dict):
            content = message.get("content")
        if not content:
            content = parsed.get("response")
        if isinstance(content, dict):
            return content
        if isinstance(content, str) and content.strip():
            return _extract_json_object(content)

    raise RuntimeError("Ollama response did not contain a usable JSON message.")


def _normalize_vision_output(raw: dict[str, Any], *, requested_domain: str) -> dict[str, Any]:
    raw_domain = str(raw.get("domain") or "").strip().lower()
    domain = raw_domain if raw_domain in ALLOWED_DOMAINS else requested_domain
    hypotheses = _normalize_hypotheses(raw.get("hypotheses"))
    if not hypotheses:
        hypotheses = _normalize_hypotheses(_stub_result(domain=requested_domain, note="", image_count=1)["hypotheses"])
    features = _normalize_features(raw.get("features"))
    follow_up = _normalize_follow_up(raw.get("follow_up"))
    safety_notes = _string_list(raw.get("safety_notes"), max_items=8)
    citations = _normalize_citations(raw.get("citations"))
    limitations = str(raw.get("limitations") or "").strip()

    return {
        "version": str(raw.get("version") or "1.0"),
        "domain": domain,
        "hypotheses": hypotheses,
        "features": features,
        "follow_up": follow_up,
        "safety_notes": safety_notes,
        "citations": citations,
        "limitations": limitations,
    }


def classify_images(
    *,
    images: list[VisionImage],
    domain: Any = "geology",
    note: str = "",
    use_ollama: bool = False,
    ollama_url: str | None = None,
    ollama_model: str | None = None,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Classify one or more images with Ollama or deterministic fallback."""
    if not images:
        raise ValueError("At least one image is required.")

    requested_domain = coerce_domain(domain)
    base_url = (ollama_url or os.environ.get("OLLAMA_URL", "")).strip() or DEFAULT_OLLAMA_URL
    model = (ollama_model or os.environ.get("OLLAMA_MODEL", "")).strip() or DEFAULT_OLLAMA_MODEL
    try:
        timeout_s = float(timeout_s)
    except Exception:
        timeout_s = 60.0
    timeout_s = max(1.0, min(300.0, timeout_s))

    if not use_ollama:
        stub = _stub_result(domain=requested_domain, note=note, image_count=len(images))
        return {
            "vision": _normalize_vision_output(stub, requested_domain=requested_domain),
            "provider": "stub",
            "model": None,
            "image_count": len(images),
            "fallback_reason": "",
        }

    try:
        raw = _ollama_vision_chat(
            images=images,
            domain=requested_domain,
            note=note,
            base_url=base_url,
            model=model,
            timeout_s=timeout_s,
        )
        return {
            "vision": _normalize_vision_output(raw, requested_domain=requested_domain),
            "provider": "ollama",
            "model": model,
            "image_count": len(images),
            "fallback_reason": "",
        }
    except Exception as e:
        stub = _stub_result(domain=requested_domain, note=note, image_count=len(images))
        stub["limitations"] = (
            f"{stub.get('limitations', '')} "
            f"Ollama fallback reason: {str(e).strip()}"
        ).strip()
        return {
            "vision": _normalize_vision_output(stub, requested_domain=requested_domain),
            "provider": "stub",
            "model": model,
            "image_count": len(images),
            "fallback_reason": str(e).strip(),
        }
