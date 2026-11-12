#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "========================================================================"
echo "  ME401 Rotary Hybrid Range-Extender Simulation - macOS/Linux Launcher"
echo "========================================================================"

VENV_PY=".venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        SYS_PY="python3"
    elif command -v python >/dev/null 2>&1; then
        SYS_PY="python"
    else
        echo "[ERROR] Python 3 was not found. Install Python 3.11+ from https://www.python.org/downloads/ or via Homebrew/apt."
        exit 1
    fi

    echo "[Setup] Creating virtual environment in .venv ..."
    "$SYS_PY" -m venv .venv
fi

if ! "$VENV_PY" -c "import numpy, scipy, matplotlib, pandas" >/dev/null 2>&1; then
    echo "[Setup] Installing required packages from requirements.txt ..."
    "$VENV_PY" -m pip install --upgrade pip
    "$VENV_PY" -m pip install -r requirements.txt
fi

echo
exec "$VENV_PY" main.py "$@"

