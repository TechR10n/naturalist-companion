"""Live Wikipedia API tool implementations.

These are optional helpers to run the LangGraph MVP against the real Wikipedia API.
The offline MVP remains the default.

Notes
- Wikipedia asks API clients to send a descriptive User-Agent.
- Network access is required.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator

if TYPE_CHECKING:
    from .langgraph_mvp import Tools

WIKIPEDIA_IMAGE_USER_AGENT = "naturalist-companion/0.1 (notebook image preview)"


@dataclass(frozen=True)
class WikipediaAPIConfig:
    """Configuration for Wikipedia API calls."""

    language: str = "en"
    user_agent: str = "naturalist-companion (local dev)"
    timeout_s: float = 15.0
    min_interval_s: float = 1.25
    max_retries: int = 3
    backoff_s: float = 1.5
    max_backoff_s: float = 20.0


def _api_url(cfg: WikipediaAPIConfig, params: dict[str, Any]) -> str:
    base = f"https://{cfg.language}.wikipedia.org/w/api.php"
    qs = urllib.parse.urlencode(params)
    return f"{base}?{qs}"


def wikipedia_page_url(title: str, *, language: str = "en") -> str:
    encoded = urllib.parse.quote(title.replace(" ", "_"), safe="()!$*,-./:;@_~")
    return f"https://{language}.wikipedia.org/wiki/{encoded}"


def title_from_wikipedia_url(url: str) -> str | None:
    parsed = urllib.parse.urlparse(str(url or ""))
    marker = "/wiki/"
    if marker not in parsed.path:
        return None
    title = urllib.parse.unquote(parsed.path.split(marker, 1)[1]).replace("_", " ").strip()
    return title or None


def _get_json(cfg: WikipediaAPIConfig, params: dict[str, Any]) -> dict[str, Any]:
    url = _api_url(cfg, params)
    req = urllib.request.Request(url, headers={"User-Agent": cfg.user_agent})
    with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def wikipedia_api_get(
    params: dict[str, Any],
    *,
    language: str = "en",
    user_agent: str = WIKIPEDIA_IMAGE_USER_AGENT,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    cfg = WikipediaAPIConfig(language=language, user_agent=user_agent, timeout_s=timeout_s)
    try:
        return _get_json(cfg, params)
    except Exception:
        return {}


def wikipedia_title_from_search(
    query: str,
    *,
    language: str = "en",
    user_agent: str = WIKIPEDIA_IMAGE_USER_AGENT,
    timeout_s: float = 10.0,
) -> str | None:
    payload = wikipedia_api_get(
        {
            "action": "query",
            "list": "search",
            "format": "json",
            "formatversion": 2,
            "srlimit": 1,
            "srsearch": query,
        },
        language=language,
        user_agent=user_agent,
        timeout_s=timeout_s,
    )
    results = (payload.get("query") or {}).get("search") or []
    if not results:
        return None
    title = str(results[0].get("title") or "").strip()
    return title or None


def iter_wikipedia_page_refs(
    items: Iterable[Any] | None,
    *,
    language: str = "en",
    user_agent: str = WIKIPEDIA_IMAGE_USER_AGENT,
    timeout_s: float = 10.0,
    title_resolver: Callable[[str], str | None] | None = None,
) -> Iterator[dict[str, str]]:
    if title_resolver is None:
        title_resolver = lambda raw: wikipedia_title_from_search(
            raw,
            language=language,
            user_agent=user_agent,
            timeout_s=timeout_s,
        )

    for item in items or []:
        if isinstance(item, str):
            raw = item.strip()
            if not raw:
                continue
            title = title_from_wikipedia_url(raw)
            if not title:
                title = title_resolver(raw)
            if title:
                yield {"title": title, "url": wikipedia_page_url(title, language=language)}
            continue

        if isinstance(item, dict):
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or item.get("source") or "").strip()
            if not title and url:
                title = title_from_wikipedia_url(url) or ""
            if title:
                yield {"title": title, "url": url}
            continue

        metadata = getattr(item, "metadata", None) or {}
        title = str(metadata.get("title") or "").strip()
        url = str(metadata.get("source") or "").strip()
        if not title and url:
            title = title_from_wikipedia_url(url) or ""
        if title:
            yield {"title": title, "url": url}


def wikipedia_thumbnail_for_title(
    title: str,
    *,
    thumb_px: int = 640,
    language: str = "en",
    user_agent: str = WIKIPEDIA_IMAGE_USER_AGENT,
    timeout_s: float = 10.0,
) -> str | None:
    payload = wikipedia_api_get(
        {
            "action": "query",
            "prop": "pageimages",
            "format": "json",
            "formatversion": 2,
            "redirects": 1,
            "piprop": "thumbnail|original",
            "pithumbsize": int(thumb_px),
            "titles": title,
        },
        language=language,
        user_agent=user_agent,
        timeout_s=timeout_s,
    )
    pages = (payload.get("query") or {}).get("pages") or []
    for page in pages:
        if not isinstance(page, dict):
            continue
        thumb = page.get("thumbnail") or {}
        original = page.get("original") or {}
        source = thumb.get("source") or original.get("source")
        if source:
            return str(source)
    return None


def display_wikipedia_images_for_pages(
    items: Iterable[Any] | None,
    *,
    max_images: int = 4,
    thumb_px: int = 640,
    language: str = "en",
    user_agent: str = WIKIPEDIA_IMAGE_USER_AGENT,
    timeout_s: float = 10.0,
) -> int:
    shown = 0
    seen: set[str] = set()

    try:
        from IPython.display import Image as IPyImage
        from IPython.display import Markdown as IPyMarkdown
        from IPython.display import display as ipy_display
    except Exception:
        IPyImage = None
        IPyMarkdown = None
        ipy_display = None

    for ref in iter_wikipedia_page_refs(
        items,
        language=language,
        user_agent=user_agent,
        timeout_s=timeout_s,
    ):
        title = ref["title"]
        if title in seen:
            continue
        seen.add(title)

        image_url = wikipedia_thumbnail_for_title(
            title,
            thumb_px=thumb_px,
            language=language,
            user_agent=user_agent,
            timeout_s=timeout_s,
        )
        if not image_url:
            continue

        shown += 1
        page_url = ref.get("url") or wikipedia_page_url(title, language=language)
        if ipy_display and IPyImage and IPyMarkdown:
            ipy_display(IPyMarkdown(f"**Wikipedia image preview: {title}**"))
            ipy_display(IPyImage(url=image_url, width=min(int(thumb_px), 720)))
            ipy_display(IPyMarkdown(f"[Open page]({page_url})"))
        else:
            print(f"Wikipedia image preview: {title}")
            print(f"  image: {image_url}")
            print(f"  page: {page_url}")

        if shown >= int(max_images):
            break

    if shown == 0:
        print("[wiki-images] No thumbnail images found for the selected pages.")
    return shown


# Backward-compatible aliases for notebook helper names.
def _wiki_api_get(params: dict[str, Any]) -> dict[str, Any]:
    return wikipedia_api_get(params)


def _title_from_wikipedia_url(url: str) -> str | None:
    return title_from_wikipedia_url(url)


def _wiki_title_from_search(query: str) -> str | None:
    return wikipedia_title_from_search(query)


def _iter_page_refs(items: Iterable[Any] | None) -> Iterator[dict[str, str]]:
    return iter_wikipedia_page_refs(items)


def _wiki_thumbnail_for_title(title: str, thumb_px: int = 640) -> str | None:
    return wikipedia_thumbnail_for_title(title, thumb_px=thumb_px)


def wikipedia_tools(
    *,
    language: str = "en",
    user_agent: str = "naturalist-companion (local dev)",
    min_interval_s: float = 1.25,
    max_retries: int = 3,
    backoff_s: float = 1.5,
    max_backoff_s: float = 20.0,
) -> "Tools":
    """Build live Tools for Wikipedia GeoSearch + page fetch."""
    from .langgraph_mvp import Tools

    cfg = WikipediaAPIConfig(
        language=language,
        user_agent=user_agent,
        min_interval_s=min_interval_s,
        max_retries=max_retries,
        backoff_s=backoff_s,
        max_backoff_s=max_backoff_s,
    )

    geosearch_cache: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    page_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    last_request_time: float | None = None

    def _rate_limit() -> None:
        nonlocal last_request_time
        now = time.monotonic()
        if last_request_time is not None:
            wait = cfg.min_interval_s - (now - last_request_time)
            if wait > 0:
                time.sleep(wait)
        last_request_time = time.monotonic()

    def _request_json(api_cfg: WikipediaAPIConfig, params: dict[str, Any]) -> dict[str, Any]:
        attempt = 0
        while True:
            _rate_limit()
            try:
                return _get_json(api_cfg, params)
            except urllib.error.HTTPError as e:
                retryable = e.code in {429, 502, 503, 504}
                if not retryable or attempt >= cfg.max_retries:
                    raise
                retry_after = e.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except Exception:
                        delay = cfg.backoff_s * (2**attempt)
                else:
                    delay = cfg.backoff_s * (2**attempt)
                time.sleep(min(cfg.max_backoff_s, delay))
            except urllib.error.URLError:
                if attempt >= cfg.max_retries:
                    raise
                delay = cfg.backoff_s * (2**attempt)
                time.sleep(min(cfg.max_backoff_s, delay))
            attempt += 1

    def geosearch(
        lat: float, lon: float, radius_m: int, limit: int, language: str
    ) -> list[dict[str, Any]]:
        # Prefer the runtime language argument if provided by the graph.
        api_cfg = cfg if language == cfg.language else WikipediaAPIConfig(language=language, user_agent=cfg.user_agent)

        key = (api_cfg.language, round(lat, 5), round(lon, 5), int(radius_m), int(limit))
        if key in geosearch_cache:
            return geosearch_cache[key]

        data = _request_json(
            api_cfg,
            {
                "action": "query",
                "list": "geosearch",
                "gscoord": f"{lat}|{lon}",
                "gsradius": int(radius_m),
                "gslimit": int(limit),
                "format": "json",
                "formatversion": 2,
            }
        )

        results: list[dict[str, Any]] = []
        for item in (data.get("query") or {}).get("geosearch") or []:
            results.append(
                {
                    "pageid": int(item.get("pageid") or 0),
                    "title": str(item.get("title") or ""),
                    "lat": float(item.get("lat") or 0.0),
                    "lon": float(item.get("lon") or 0.0),
                    "dist_m": float(item.get("dist") or 0.0),
                }
            )
        geosearch_cache[key] = results
        return results

    def fetch_page(pageid: int, language: str) -> dict[str, Any]:
        api_cfg = cfg if language == cfg.language else WikipediaAPIConfig(language=language, user_agent=cfg.user_agent)

        key = (api_cfg.language, int(pageid))
        if key in page_cache:
            return page_cache[key]

        data = _request_json(
            api_cfg,
            {
                "action": "query",
                "pageids": int(pageid),
                "prop": "extracts|categories|info|coordinates",
                "inprop": "url",
                "explaintext": 1,
                "exsectionformat": "plain",
                "cllimit": 50,
                "format": "json",
                "formatversion": 2,
            }
        )

        pages = (data.get("query") or {}).get("pages") or []
        if not pages:
            raise RuntimeError(f"Wikipedia returned no page for pageid={pageid}")

        p = pages[0]
        extract = str(p.get("extract") or "")
        summary = extract.strip().split("\n", 1)[0].strip()
        coords = (p.get("coordinates") or [])
        lat = None
        lon = None
        if coords:
            lat = coords[0].get("lat")
            lon = coords[0].get("lon")

        cats: list[str] = []
        for c in p.get("categories") or []:
            title = str(c.get("title") or "")
            if title.startswith("Category:"):
                title = title[len("Category:") :]
            if title:
                cats.append(title)

        payload = {
            "pageid": int(p.get("pageid") or pageid),
            "title": str(p.get("title") or ""),
            "url": str(p.get("canonicalurl") or p.get("fullurl") or ""),
            "summary": summary,
            "content": extract,
            "categories": cats,
            "lat": float(lat) if lat is not None else None,
            "lon": float(lon) if lon is not None else None,
        }
        page_cache[key] = payload
        return payload

    return Tools(geosearch=geosearch, fetch_page=fetch_page)
