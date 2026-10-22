#!/usr/bin/env bash
# run_llm.sh — macOS/Linux equivalent of Luke's run_llm.cmd / run_llm.ps1.
#
#   ./run_llm.sh                                  interactive: choose a road and survey
#   ./run_llm.sh check                            environment checks only
#   ./run_llm.sh brief --road 1-4126-1 --date 1993-03-30
#   ./run_llm.sh brief --road 1-4126-1 --date 1993-03-30 --no-llm   # template, no Ollama
#   ./run_llm.sh list --split test
#   ./run_llm.sh test                             pytest, no Ollama needed
#   ./run_llm.sh eval                             20-road evaluation
#
# Uses the project's own .venv (created by ./run.sh if it does not exist yet).

set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "[run_llm] No .venv yet — creating it ..."
    PY=""
    for c in python3.13 python3.12 python3.11 python3.10 python3; do
        if command -v "$c" >/dev/null 2>&1 && \
           "$c" -c 'import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)' 2>/dev/null; then
            PY="$c"; break
        fi
    done
    if [ -z "$PY" ]; then
        echo "ERROR: need Python 3.10 or newer (macOS ships 3.9)."
        echo "  brew install python@3.12"
        exit 1
    fi
    "$PY" -m venv .venv
    ./.venv/bin/pip install --quiet --upgrade pip
    echo "[run_llm] Installing dependencies (first run only, a few minutes) ..."
    ./.venv/bin/pip install --quiet -r requirements.txt -r llm/requirements.txt
fi

PYBIN="./.venv/bin/python"

# install the LLM layer's extra dependencies if missing
for mod in jinja2 pydantic ollama pytest; do
    if ! $PYBIN -c "import $mod" >/dev/null 2>&1; then
        echo "[run_llm] Installing LLM layer dependencies ..."
        $PYBIN -m pip install --quiet -r requirements.txt -r llm/requirements.txt
        break
    fi
done

CMD="${1:-interactive}"; [ $# -gt 0 ] && shift || true

case "$CMD" in
  test)
    exec $PYBIN -m pytest tests -q -p no:cacheprovider "$@" ;;
  check)
    exec $PYBIN -m llm.check_env "$@" ;;
  eval)
    exec $PYBIN -m llm.eval.run_eval run "$@" ;;
  interactive|pick|ask)
    if ! command -v ollama >/dev/null 2>&1; then
        echo "[run_llm] Ollama not installed — the picker will use the template fallback."
        exec $PYBIN -m llm.interactive --no-llm "$@"
    fi
    if ! curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "[run_llm] Starting Ollama server ..."; (ollama serve >/dev/null 2>&1 &); sleep 3
    fi
    exec $PYBIN -m llm.interactive "$@" ;;
  brief|list|payload)
    # Ollama is only needed for a real brief, not for --no-llm, list or payload
    if [ "$CMD" = "brief" ] && [[ " $* " != *" --no-llm "* ]]; then
        if ! command -v ollama >/dev/null 2>&1; then
            echo "[run_llm] Ollama is not installed."
            echo "  Install it from https://ollama.com, then:  ollama pull qwen3:8b"
            echo "  Or run without it:  ./run_llm.sh brief --road ... --date ... --no-llm"
            exit 1
        fi
        if ! curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo "[run_llm] Starting Ollama server ..."
            (ollama serve >/dev/null 2>&1 &)
            sleep 3
        fi

        # Check the model is actually pulled. Without this the failure surfaces as a
        # 404 traceback from deep inside the ollama client, which is not obvious.
        WANT="${LLM_MODEL:-llama3.1:8b}"
        for ((i=1; i<$#+1; i++)); do
            if [ "${!i}" = "--model" ]; then j=$((i+1)); WANT="${!j:-$WANT}"; fi
        done
        if ! ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$WANT"; then
            echo "[run_llm] Ollama is running, but the model '$WANT' is not downloaded."
            echo
            echo "  Download it (about 5 GB, one time):"
            echo "      ollama pull $WANT"
            echo
            echo "  Models you already have:"
            ollama list 2>/dev/null | tail -n +2 | awk '{print "      " $1}' | grep -v '^      $' || echo "      (none)"
            echo
            echo "  Or skip the model entirely and use the deterministic template:"
            echo "      ./run_llm.sh brief --road 1-4126-1 --date 1993-03-30 --no-llm"
            exit 1
        fi
    fi
    exec $PYBIN -m llm.cli "$CMD" "$@" ;;
  *)
    exec $PYBIN -m llm.cli "$CMD" "$@" ;;
esac
