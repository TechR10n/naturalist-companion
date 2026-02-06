"""Unit tests for shared Wikipedia notebook helper functions."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from naturalist_companion import wikipedia_tools


class _Doc:
    def __init__(self, *, title: str, source: str) -> None:
        self.metadata = {"title": title, "source": source}


class TestWikipediaToolsHelpers(unittest.TestCase):
    def test_title_from_wikipedia_url_decodes_title(self) -> None:
        self.assertEqual(
            wikipedia_tools.title_from_wikipedia_url("https://en.wikipedia.org/wiki/Hagerstown,_Maryland"),
            "Hagerstown, Maryland",
        )
        self.assertEqual(
            wikipedia_tools.title_from_wikipedia_url("https://en.wikipedia.org/wiki/Carboniferous"),
            "Carboniferous",
        )
        self.assertIsNone(wikipedia_tools.title_from_wikipedia_url("https://example.com/not-a-wiki-page"))

    def test_iter_wikipedia_page_refs_handles_strings_dicts_and_docs(self) -> None:
        items = [
            "https://en.wikipedia.org/wiki/Carboniferous",
            "fold mountain geology",
            {"title": "Basalt", "url": "https://en.wikipedia.org/wiki/Basalt"},
            {"source": "https://en.wikipedia.org/wiki/Shale"},
            _Doc(title="Roadcut", source="https://en.wikipedia.org/wiki/Roadcut"),
        ]

        refs = list(
            wikipedia_tools.iter_wikipedia_page_refs(
                items,
                title_resolver=lambda raw: "Fold mountain" if raw == "fold mountain geology" else None,
            )
        )
        self.assertEqual(
            refs,
            [
                {"title": "Carboniferous", "url": "https://en.wikipedia.org/wiki/Carboniferous"},
                {"title": "Fold mountain", "url": "https://en.wikipedia.org/wiki/Fold_mountain"},
                {"title": "Basalt", "url": "https://en.wikipedia.org/wiki/Basalt"},
                {"title": "Shale", "url": "https://en.wikipedia.org/wiki/Shale"},
                {"title": "Roadcut", "url": "https://en.wikipedia.org/wiki/Roadcut"},
            ],
        )

    def test_display_wikipedia_images_for_pages_respects_max_images(self) -> None:
        refs = [
            {"title": "A", "url": "https://en.wikipedia.org/wiki/A"},
            {"title": "B", "url": "https://en.wikipedia.org/wiki/B"},
            {"title": "C", "url": "https://en.wikipedia.org/wiki/C"},
        ]
        with patch(
            "naturalist_companion.wikipedia_tools.iter_wikipedia_page_refs",
            return_value=refs,
        ), patch(
            "naturalist_companion.wikipedia_tools.wikipedia_thumbnail_for_title",
            side_effect=lambda title, **kwargs: f"https://img.example/{title}.jpg",
        ), patch(
            "IPython.display.display",
            return_value=None,
        ):
            shown = wikipedia_tools.display_wikipedia_images_for_pages([], max_images=2)

        self.assertEqual(shown, 2)

    def test_wikipedia_api_get_returns_empty_dict_on_error(self) -> None:
        with patch("naturalist_companion.wikipedia_tools._get_json", side_effect=RuntimeError("boom")):
            payload = wikipedia_tools.wikipedia_api_get({"action": "query"})
        self.assertEqual(payload, {})


if __name__ == "__main__":
    unittest.main()
