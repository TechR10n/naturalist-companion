"""Module entrypoint for running the Flask app via `python -m anc`."""

from __future__ import annotations

from anc.web import main


if __name__ == "__main__":
    raise SystemExit(main())
