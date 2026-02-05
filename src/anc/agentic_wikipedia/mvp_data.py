"""Toy Wikipedia and route data for the offline MVP."""

from __future__ import annotations

from typing import Any


def minimal_route_points() -> list[dict[str, float]]:
    """Tiny route polyline for the offline MVP.

    This is intentionally minimal so the graph can be smoke-tested without
    external APIs or large datasets.
    """

    # Roughly along the I-81 corridor in Virginia (toy coordinates).
    return [
        {"lat": 38.1496, "lon": -79.0717},
        {"lat": 38.4596, "lon": -78.8717},
        {"lat": 38.7496, "lon": -78.6717},
    ]


def minimal_wiki_pages() -> dict[int, dict[str, Any]]:
    """Toy Wikipedia page payloads (not verbatim Wikipedia text)."""

    return {
        1001: {
            "pageid": 1001,
            "title": "Basalt",
            "url": "https://en.wikipedia.org/wiki/Basalt",
            "summary": (
                "Toy extract: Basalt is a dark, fine-grained igneous rock that forms "
                "from rapidly cooled lava at or near Earth's surface."
            ),
            "content": (
                "Toy extract:\n"
                "- Basalt commonly forms in volcanic settings and can cool quickly into a fine-grained texture.\n"
                "- It is rich in iron and magnesium compared to many other common rocks.\n"
                "- Columnar jointing can form as lava flows cool and contract.\n"
            ),
            "categories": ["Igneous rock", "Volcanism"],
            "lat": 38.46,
            "lon": -78.88,
        },
        1002: {
            "pageid": 1002,
            "title": "Appalachian Mountains",
            "url": "https://en.wikipedia.org/wiki/Appalachian_Mountains",
            "summary": (
                "Toy extract: The Appalachian Mountains are an ancient mountain range "
                "with complex geology shaped by multiple mountain-building events."
            ),
            "content": (
                "Toy extract:\n"
                "- The range records a long history of tectonic collisions and uplift.\n"
                "- Erosion over time has produced rounded ridges and valleys in many areas.\n"
                "- Rock types vary widely, including metamorphic, igneous, and sedimentary units.\n"
            ),
            "categories": ["Mountain range", "Geology"],
            "lat": 38.75,
            "lon": -78.67,
        },
        2001: {
            "pageid": 2001,
            "title": "Interstate 81",
            "url": "https://en.wikipedia.org/wiki/Interstate_81",
            "summary": (
                "Toy extract: Interstate 81 is a major north–south highway in the eastern United States."
            ),
            "content": (
                "Toy extract:\n"
                "- The highway runs through several states and serves regional freight traffic.\n"
                "- It connects a number of metropolitan areas and rural corridors.\n"
            ),
            "categories": ["Interstate Highway System"],
            "lat": 38.15,
            "lon": -79.07,
        },
    }


def minimal_geosearch_index() -> list[dict[str, Any]]:
    """Toy GeoSearch results returned for any sampled point."""

    # Includes an intentionally off-topic page (Interstate 81) to exercise filtering.
    return [
        {"pageid": 2001, "title": "Interstate 81", "lat": 38.15, "lon": -79.07, "dist_m": 1200},
        {"pageid": 1001, "title": "Basalt", "lat": 38.46, "lon": -78.88, "dist_m": 6200},
        {
            "pageid": 1002,
            "title": "Appalachian Mountains",
            "lat": 38.75,
            "lon": -78.67,
            "dist_m": 9100,
        },
    ]
