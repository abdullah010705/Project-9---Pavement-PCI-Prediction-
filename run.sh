#!/usr/bin/env bash
# run.sh — one command to run the pipeline, no Python setup knowledge required.
#
#   ./run.sh                     normal run
#   ./run.sh --rebuild-dataset   regenerate the modelling datasets from raw data
#   ./run.sh --stop-on-missing   fail on an unimplemented stage instead of skipping
#   PCI_SEARCH_ITER=40 ./run.sh  thorough hyperparameter search (use for final numbers)
#
# First run creates .venv and installs dependencies (a couple of minutes).
# After that it just runs.

set -euo pipefail
cd "$(dirname "$0")"

PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && \
       "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
        PY="$c"; break
    fi
done
if [ -z "$PY" ]; then
    echo "ERROR: no Python 3.10 or newer found."
    echo "  macOS ships 3.9, which this project does not support."
    echo "  Install one:  brew install python@3.12"
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "[run.sh] Creating virtual environment with $PY ..."
    "$PY" -m venv .venv
    ./.venv/bin/pip install --quiet --upgrade pip
    echo "[run.sh] Installing dependencies (first run only, a couple of minutes) ..."
    ./.venv/bin/pip install --quiet -r requirements.txt
fi

for mod in xgboost shap sklearn pandas matplotlib joblib; do
    if ! ./.venv/bin/python -c "import $mod" >/dev/null 2>&1; then
        echo "[run.sh] '$mod' is missing — installing dependencies ..."
        ./.venv/bin/pip install --quiet -r requirements.txt
        break
    fi
done

if ! ./.venv/bin/python -c "import xgboost" >/dev/null 2>&1; then
    echo "[run.sh] XGBoost still will not import."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  On macOS this is the missing OpenMP runtime. Fix with:"
        echo "      brew install libomp"
    fi
    exit 1
fi

exec ./.venv/bin/python run_pipeline.py "$@"
