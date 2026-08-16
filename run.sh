#!/usr/bin/env bash
# run.sh — one command to run the pipeline, no Python setup knowledge required.
#
#   ./run.sh                    normal run
#   ./run.sh --rebuild-dataset  regenerate model_dataset.csv from raw data
#   ./run.sh --skip-cv          skip cross-validation (the slowest stage)
#
# On first run this creates a virtual environment in .venv and installs everything
# from requirements.txt, which takes a couple of minutes. After that it just runs.

set -euo pipefail
cd "$(dirname "$0")"

# --- find a Python 3.10+ interpreter -----------------------------------------
PY=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
            PY="$candidate"; break
        fi
    fi
done

if [ -z "$PY" ]; then
    echo "ERROR: no Python 3.10 or newer found."
    echo "  macOS ships Python 3.9, which this project does not support."
    echo "  Install a newer one:  brew install python@3.12"
    exit 1
fi

# --- create the virtual environment if it isn't there ------------------------
if [ ! -d ".venv" ]; then
    echo "[run.sh] Creating virtual environment with $PY ..."
    "$PY" -m venv .venv
    ./.venv/bin/pip install --quiet --upgrade pip
    echo "[run.sh] Installing dependencies (a couple of minutes, first run only) ..."
    ./.venv/bin/pip install --quiet -r requirements.txt
fi

# --- check XGBoost can actually load (macOS needs libomp) --------------------
if ! ./.venv/bin/python -c "import xgboost" >/dev/null 2>&1; then
    echo "[run.sh] XGBoost failed to import."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  On macOS this is usually the missing OpenMP runtime. Fix with:"
        echo "      brew install libomp"
    else
        echo "  Try:  ./.venv/bin/pip install -r requirements.txt"
    fi
    exit 1
fi

# --- run ---------------------------------------------------------------------
exec ./.venv/bin/python run_pipeline.py "$@"
