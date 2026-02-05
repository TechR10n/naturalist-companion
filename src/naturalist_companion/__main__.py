"""Module entrypoint for running the Flask app via `python -m naturalist_companion`."""

from __future__ import annotations

from naturalist_companion.web import main


if __name__ == "__main__":
    raise SystemExit(main())

