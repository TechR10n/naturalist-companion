"""Sanitize non-standard notebook output keys and export to PDF via nbconvert."""

from __future__ import annotations

import asyncio
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ALLOWED_OUTPUT_KEYS: dict[str, set[str]] = {
    "stream": {"output_type", "name", "text"},
    "display_data": {"output_type", "data", "metadata"},
    "execute_result": {
        "output_type",
        "execution_count",
        "data",
        "metadata",
    },
    "error": {"output_type", "ename", "evalue", "traceback"},
}


def _sanitize_output(output: dict[str, Any]) -> int:
    """Normalize output objects to standard nbformat keys.

    Returns the number of key-level mutations applied.
    """

    changes = 0
    output_type = str(output.get("output_type", ""))

    # Drop Databricks/JetBrains-specific metadata that breaks strict nbformat validation.
    if output.pop("jetTransient", None) is not None:
        changes += 1

    allowed = ALLOWED_OUTPUT_KEYS.get(output_type)
    if allowed is None:
        return changes

    for key in list(output.keys()):
        if key not in allowed:
            output.pop(key, None)
            changes += 1

    return changes


def sanitize_notebook(notebook_payload: dict[str, Any]) -> int:
    """Sanitize all cell outputs in a notebook payload.

    Returns the number of output-level key mutations.
    """

    changes = 0
    for cell in notebook_payload.get("cells", []):
        outputs = cell.get("outputs")
        if not isinstance(outputs, list):
            continue
        for output in outputs:
            if isinstance(output, dict):
                changes += _sanitize_output(output)
    return changes


def validate_notebook(payload: dict[str, Any]) -> None:
    """Validate notebook schema if nbformat is importable."""

    try:
        import nbformat
    except Exception as exc:  # pragma: no cover - optional runtime dependency.
        print(f"[notebook-export] Skipping validation (nbformat import failed: {exc})")
        return

    notebook = nbformat.from_dict(payload)
    nbformat.validate(notebook)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sanitize non-standard notebook output keys (for example Databricks/JetBrains "
            "jetTransient) before exporting PDF with jupyter nbconvert."
        )
    )
    parser.add_argument("notebook", type=Path, help="Path to .ipynb notebook")
    parser.add_argument(
        "--to",
        default="pdf",
        choices=("pdf", "webpdf"),
        help="nbconvert exporter format (default: pdf)",
    )
    parser.add_argument(
        "--output-basename",
        default=None,
        help="Basename for exported file (without extension). Defaults to notebook stem.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for exported file. Defaults to the notebook directory.",
    )
    parser.add_argument(
        "--keep-sanitized",
        action="store_true",
        help="Keep the generated *.sanitized.ipynb file after export succeeds.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip nbformat schema validation before running nbconvert.",
    )
    parser.add_argument(
        "--hide-input",
        action="store_true",
        help="Hide code cells from the export.",
    )
    parser.add_argument(
        "--hide-prompts",
        action="store_true",
        help="Hide In/Out execution prompts from the export.",
    )
    parser.add_argument(
        "--polished",
        action="store_true",
        help="Apply a report-style preset (hide prompts).",
    )
    parser.add_argument(
        "--landscape",
        action="store_true",
        help="Render webpdf output in landscape orientation.",
    )
    parser.add_argument(
        "--nbconvert-arg",
        action="append",
        default=[],
        help="Extra raw argument to pass through to nbconvert (repeatable).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    notebook_path = args.notebook.expanduser().resolve()
    if not notebook_path.exists():
        print(f"[notebook-export] Notebook not found: {notebook_path}", file=sys.stderr)
        return 2

    with notebook_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    changes = sanitize_notebook(payload)

    sanitized_path = notebook_path.with_name(f"{notebook_path.stem}.sanitized.ipynb")
    with sanitized_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
        fh.write("\n")

    if not args.skip_validate:
        try:
            validate_notebook(payload)
        except Exception as exc:
            print(
                f"[notebook-export] Validation failed for sanitized notebook: {exc}",
                file=sys.stderr,
            )
            print(f"[notebook-export] Kept sanitized notebook: {sanitized_path}", file=sys.stderr)
            return 1

    output_dir = (args.output_dir or notebook_path.parent).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_basename = args.output_basename or notebook_path.stem

    print(f"[notebook-export] Sanitized keys changed: {changes}")

    if args.to == "webpdf" and args.landscape:
        html_basename = f"{output_basename}.landscape_tmp"
        html_path = output_dir / f"{html_basename}.html"
        pdf_path = output_dir / f"{output_basename}.pdf"

        nbconvert_args = [raw for raw in args.nbconvert_arg if raw]
        allow_chromium_download = False
        disable_chromium_sandbox = False
        filtered_nbconvert_args: list[str] = []
        for raw in nbconvert_args:
            if raw == "--allow-chromium-download":
                allow_chromium_download = True
                continue
            if raw == "--disable-chromium-sandbox":
                disable_chromium_sandbox = True
                continue
            filtered_nbconvert_args.append(raw)

        html_cmd = [
            "jupyter",
            "nbconvert",
            "--to",
            "html",
            str(sanitized_path),
            "--output",
            html_basename,
            "--output-dir",
            str(output_dir),
            "--embed-images",
        ]
        if args.hide_input:
            html_cmd.append("--no-input")
        if args.hide_prompts or args.polished:
            html_cmd.append("--TemplateExporter.exclude_input_prompt=True")
            html_cmd.append("--TemplateExporter.exclude_output_prompt=True")
        html_cmd.extend(filtered_nbconvert_args)

        print(f"[notebook-export] Running: {' '.join(html_cmd)}")
        html_proc = subprocess.run(html_cmd, cwd=notebook_path.parent, check=False)
        if html_proc.returncode != 0:
            print(f"[notebook-export] nbconvert html failed with code {html_proc.returncode}", file=sys.stderr)
            print(f"[notebook-export] Kept sanitized notebook: {sanitized_path}", file=sys.stderr)
            return int(html_proc.returncode)

        async def _render_webpdf_landscape() -> None:
            try:
                from playwright.async_api import async_playwright  # type: ignore[import-not-found]
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Playwright is required for landscape webpdf export. Install `nbconvert[webpdf]`."
                ) from exc

            async with async_playwright() as playwright:
                chromium = playwright.chromium
                launch_args: list[str] = []
                if disable_chromium_sandbox:
                    launch_args.append("--no-sandbox")

                try:
                    browser = await chromium.launch(
                        handle_sigint=False,
                        handle_sigterm=False,
                        handle_sighup=False,
                        args=launch_args,
                    )
                except Exception:
                    if not allow_chromium_download:
                        raise
                    install_cmd = [sys.executable, "-m", "playwright", "install", "chromium"]
                    print(f"[notebook-export] Running: {' '.join(install_cmd)}")
                    subprocess.check_call(install_cmd)  # noqa: S603
                    browser = await chromium.launch(
                        handle_sigint=False,
                        handle_sigterm=False,
                        handle_sighup=False,
                        args=launch_args,
                    )

                page = await browser.new_page()
                await page.emulate_media(media="print")
                await page.wait_for_timeout(120)
                await page.goto(html_path.as_uri(), wait_until="networkidle")
                await page.wait_for_timeout(120)
                await page.pdf(path=str(pdf_path), print_background=True, landscape=True)
                await browser.close()

        try:
            asyncio.run(_render_webpdf_landscape())
        except Exception as exc:
            print(f"[notebook-export] landscape webpdf render failed: {exc}", file=sys.stderr)
            print(f"[notebook-export] Kept sanitized notebook: {sanitized_path}", file=sys.stderr)
            return 1

        try:
            html_path.unlink(missing_ok=True)
        except Exception as exc:
            print(f"[notebook-export] Warning: failed to remove {html_path}: {exc}")
    else:
        cmd = [
            "jupyter",
            "nbconvert",
            "--to",
            args.to,
            str(sanitized_path),
            "--output",
            output_basename,
            "--output-dir",
            str(output_dir),
        ]
        if args.to == "webpdf":
            # Force image embedding so local/remote image references are baked into the rendered PDF.
            cmd.append("--embed-images")
        if args.hide_input:
            cmd.append("--no-input")
        if args.hide_prompts or args.polished:
            cmd.append("--TemplateExporter.exclude_input_prompt=True")
            cmd.append("--TemplateExporter.exclude_output_prompt=True")
        for raw_arg in args.nbconvert_arg:
            if raw_arg:
                cmd.append(raw_arg)

        print(f"[notebook-export] Running: {' '.join(cmd)}")

        proc = subprocess.run(cmd, cwd=notebook_path.parent, check=False)

        if proc.returncode != 0:
            print(f"[notebook-export] nbconvert failed with code {proc.returncode}", file=sys.stderr)
            print(f"[notebook-export] Kept sanitized notebook: {sanitized_path}", file=sys.stderr)
            return int(proc.returncode)

    if args.keep_sanitized:
        print(f"[notebook-export] Kept sanitized notebook: {sanitized_path}")
    else:
        try:
            sanitized_path.unlink(missing_ok=True)
        except Exception as exc:
            print(f"[notebook-export] Warning: failed to remove {sanitized_path}: {exc}")

    ext = "pdf"
    print(f"[notebook-export] Export complete: {output_dir / f'{output_basename}.{ext}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
