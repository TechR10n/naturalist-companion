"""Render the offline route-guide graph to a Mermaid markdown file.

This is intended for one-click usage via a PyCharm run configuration.
"""

from __future__ import annotations

from pathlib import Path


def _strip_yaml_front_matter(mermaid: str) -> str:
    """LangChain may emit a YAML front matter config block; strip it for GitHub."""
    text = mermaid.lstrip()
    if not text.startswith("---"):
        return mermaid.strip()

    lines = text.splitlines()
    if not lines:
        return ""

    # Find the second '---' line and drop everything up to it (inclusive).
    delim_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            delim_idx = i
            break
    if delim_idx is None:
        return mermaid.strip()

    return "\n".join(lines[delim_idx + 1 :]).strip()


def main() -> int:
    from naturalist_companion.route_guide import build_route_guide_app

    app = build_route_guide_app()
    mermaid = app.get_graph().draw_mermaid()
    mermaid = _strip_yaml_front_matter(mermaid)

    out_path = Path("docs/diagrams/route_guide_graph.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(
            [
                "# Route Guide Graph - Mermaid",
                "",
                "Generated from `naturalist_companion.route_guide.build_route_guide_app()`.",
                "",
                "```mermaid",
                mermaid,
                "```",
                "",
            ]
        )
    )

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
