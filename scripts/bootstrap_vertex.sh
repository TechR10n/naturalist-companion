#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if [[ "$PYTHON_BIN" == "python3.12" ]] && command -v brew >/dev/null 2>&1; then
    BREW_PYTHON_BIN="$(brew --prefix python@3.12 2>/dev/null)/bin/python3.12"
    if [[ -x "$BREW_PYTHON_BIN" ]]; then
      PYTHON_BIN="$BREW_PYTHON_BIN"
    fi
  fi
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Could not find $PYTHON_BIN." >&2
  echo "" >&2
  echo "Install Python 3.12 (recommended) or re-run with a different interpreter, e.g.:" >&2
  echo "  PYTHON_BIN=python3 ./scripts/bootstrap_vertex.sh" >&2
  echo "" >&2
  echo "If you installed Homebrew python@3.12, you can also run:" >&2
  echo "  PYTHON_BIN=\"\$(brew --prefix python@3.12)/bin/python3.12\" ./scripts/bootstrap_vertex.sh" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate

python -m pip install -q --disable-pip-version-check -U pip
python -m pip install -q --disable-pip-version-check -r requirements-gcp-dev.txt
python -m pip install -q --disable-pip-version-check -e .

python -m ipykernel install --user --name naturalist-companion --display-name "Agentic Naturalist"

echo ""
echo "Done."
echo "- Activate: source .venv/bin/activate"
echo "- Configure: cp .env.example .env"
echo "- Web app: python -m naturalist_companion --debug"
echo "- Smoke test: python scripts/smoke_vertex_ai.py"
echo "- Notebook: jupyter lab"
