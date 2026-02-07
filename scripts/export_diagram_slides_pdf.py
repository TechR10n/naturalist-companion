"""Build a one-slide-per-page PDF deck from rendered PlantUML diagrams.

Deck structure:
- Title slide
- One diagram slide per rendered diagram

This implementation uses Pillow so each diagram is guaranteed to fit a single page.
"""

from __future__ import annotations

import argparse
import html
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


@dataclass(frozen=True)
class DiagramSlide:
    path: Path
    title: str


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export rendered PlantUML diagrams to a PDF slide deck.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("docs/diagrams/rendered"),
        help="Directory containing rendered diagram files (default: docs/diagrams/rendered).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/diagrams/rendered/diagram_slides.pdf"),
        help="Output PDF path (default: docs/diagrams/rendered/diagram_slides.pdf).",
    )
    parser.add_argument(
        "--title",
        default="Naturalist Companion Architecture Diagrams",
        help="Deck title for the first slide.",
    )
    parser.add_argument(
        "--subtitle",
        default="Rendered PlantUML diagram set",
        help="Deck subtitle for the first slide.",
    )
    parser.add_argument(
        "--page-width",
        default="13.333in",
        help="PDF page width (default: 13.333in, 16:9).",
    )
    parser.add_argument(
        "--page-height",
        default="7.5in",
        help="PDF page height (default: 7.5in, 16:9).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Raster DPI for slide canvas sizing (default: 150).",
    )
    parser.add_argument(
        "--margin",
        default="0.2in",
        help="Slide margin (default: 0.2in).",
    )
    return parser.parse_args(argv)


def _parse_inches(value: str) -> float:
    raw = value.strip().lower()
    if raw.endswith("in"):
        return float(raw[:-2])
    if raw.endswith("pt"):
        return float(raw[:-2]) / 72.0
    if raw.endswith("px"):
        return float(raw[:-2]) / 96.0
    return float(raw)


def _px(value: str, dpi: int) -> int:
    return max(1, int(round(_parse_inches(value) * dpi)))


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _pretty_title_from_stem(stem: str) -> str:
    parts = stem.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        stem = parts[1]
    return stem.replace("_", " ")


def _ensure_png_for_stem(source_dir: Path, stem: str) -> Path | None:
    png_path = source_dir / f"{stem}.png"
    if png_path.exists():
        return png_path

    puml_path = source_dir.parent / f"{stem}.puml"
    if not puml_path.exists():
        return None

    cmd = ["plantuml", "-tpng", "-o", str(source_dir.name), str(puml_path)]
    proc = subprocess.run(cmd, cwd=source_dir.parent.parent, check=False)
    if proc.returncode != 0:
        return None
    return png_path if png_path.exists() else None


def _discover_slides(source_dir: Path) -> list[DiagramSlide]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    stems: set[str] = set()
    for p in source_dir.glob("[0-9][0-9]_*.png"):
        stems.add(p.stem)
    for p in source_dir.glob("[0-9][0-9]_*.svg"):
        stems.add(p.stem)

    def sort_key(stem: str) -> tuple[int, str]:
        prefix = stem.split("_", 1)[0]
        return (int(prefix) if prefix.isdigit() else 999, stem)

    slides: list[DiagramSlide] = []
    for stem in sorted(stems, key=sort_key):
        png_path = _ensure_png_for_stem(source_dir, stem)
        if png_path is None:
            # Skip stems that cannot be materialized as PNG.
            continue
        slides.append(DiagramSlide(path=png_path, title=_pretty_title_from_stem(stem)))
    return slides


def _draw_centered(draw: ImageDraw.ImageDraw, text: str, box: tuple[int, int, int, int], font: ImageFont.ImageFont, fill: tuple[int, int, int]) -> None:
    left, top, right, bottom = box
    tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    x = left + max(0, (right - left - tw) // 2)
    y = top + max(0, (bottom - top - th) // 2)
    draw.text((x, y), text, font=font, fill=fill)


def _make_title_slide(
    *,
    title: str,
    subtitle: str,
    source_dir: Path,
    slide_count: int,
    page_w: int,
    page_h: int,
    margin: int,
) -> Image.Image:
    img = Image.new("RGB", (page_w, page_h), color=(255, 255, 255))
    d = ImageDraw.Draw(img)

    card_left = margin * 2
    card_top = margin * 2
    card_right = page_w - margin * 2
    card_bottom = page_h - margin * 2

    d.rounded_rectangle((card_left, card_top, card_right, card_bottom), radius=30, fill=(248, 250, 252), outline=(203, 213, 225), width=3)

    title_font = _font(56)
    subtitle_font = _font(30)
    meta_font = _font(22)

    _draw_centered(d, title, (card_left + 40, card_top + 70, card_right - 40, card_top + 220), title_font, (15, 23, 42))
    _draw_centered(d, subtitle, (card_left + 40, card_top + 220, card_right - 40, card_top + 300), subtitle_font, (71, 85, 105))

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    metas = [
        f"Generated: {generated}",
        f"Source: {source_dir}",
        f"Slides: {slide_count}",
    ]
    y = card_top + 340
    for line in metas:
        _draw_centered(d, line, (card_left + 40, y, card_right - 40, y + 50), meta_font, (71, 85, 105))
        y += 56

    return img


def _make_diagram_slide(*, slide: DiagramSlide, index: int, total: int, page_w: int, page_h: int, margin: int) -> Image.Image:
    img = Image.new("RGB", (page_w, page_h), color=(255, 255, 255))
    d = ImageDraw.Draw(img)

    shell = (margin, margin, page_w - margin, page_h - margin)
    d.rounded_rectangle(shell, radius=18, fill=(255, 255, 255), outline=(203, 213, 225), width=2)

    header_h = max(56, page_h // 12)
    left, top, right, bottom = shell
    header = (left + 1, top + 1, right - 1, top + header_h)
    d.rectangle(header, fill=(248, 250, 252), outline=(226, 232, 240), width=1)

    title_font = _font(24)
    badge_font = _font(16)
    text_color = (11, 58, 83)
    d.text((left + 22, top + 16), slide.title.title(), font=title_font, fill=text_color)

    badge = f"{index}/{total} | {slide.path.name}"
    badge_bbox = d.textbbox((0, 0), badge, font=badge_font)
    bw = badge_bbox[2] - badge_bbox[0]
    bh = badge_bbox[3] - badge_bbox[1]
    bx2 = right - 18
    bx1 = bx2 - bw - 20
    by1 = top + 16
    by2 = by1 + max(24, bh + 8)
    d.rounded_rectangle((bx1, by1, bx2, by2), radius=12, fill=(241, 245, 249), outline=(203, 213, 225), width=1)
    d.text((bx1 + 10, by1 + 4), badge, font=badge_font, fill=(30, 41, 59))

    # Diagram canvas area
    pad = 20
    canvas = (left + pad, top + header_h + pad, right - pad, bottom - pad)
    d.rectangle(canvas, fill=(255, 255, 255), outline=(226, 232, 240), width=1)

    src = Image.open(slide.path).convert("RGB")
    target_w = max(1, canvas[2] - canvas[0] - 8)
    target_h = max(1, canvas[3] - canvas[1] - 8)
    fitted = ImageOps.contain(src, (target_w, target_h), method=Image.Resampling.LANCZOS)

    px = canvas[0] + (target_w - fitted.width) // 2 + 4
    py = canvas[1] + (target_h - fitted.height) // 2 + 4
    img.paste(fitted, (px, py))

    return img


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    source_dir = args.source_dir.expanduser().resolve()
    output_pdf = args.output.expanduser().resolve()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    page_w = _px(args.page_width, args.dpi)
    page_h = _px(args.page_height, args.dpi)
    margin = _px(args.margin, args.dpi)

    slides = _discover_slides(source_dir)
    if not slides:
        print(
            f"[diagram-slides] No diagrams found (or convertible to PNG) under {source_dir}",
            file=sys.stderr,
        )
        return 2

    pages: list[Image.Image] = []
    pages.append(
        _make_title_slide(
            title=html.unescape(args.title),
            subtitle=html.unescape(args.subtitle),
            source_dir=source_dir,
            slide_count=len(slides),
            page_w=page_w,
            page_h=page_h,
            margin=margin,
        )
    )

    total = len(slides)
    for idx, slide in enumerate(slides, start=1):
        pages.append(_make_diagram_slide(slide=slide, index=idx, total=total, page_w=page_w, page_h=page_h, margin=margin))

    first, rest = pages[0], pages[1:]
    first.save(output_pdf, "PDF", save_all=True, append_images=rest, resolution=float(args.dpi))

    print(f"[diagram-slides] Slides: {len(slides)}")
    print(f"[diagram-slides] Export complete: {output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
