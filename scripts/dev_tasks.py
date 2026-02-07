"""Small task runner wrapper for IDE-friendly Make target execution."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ALLOWED_TARGETS = ("setup", "test", "smoke", "clean", "check")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run project tasks via Make targets.")
    parser.add_argument("target", choices=ALLOWED_TARGETS, help="Make target to execute")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(["make", args.target], cwd=repo_root, check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
