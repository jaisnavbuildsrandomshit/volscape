#!/usr/bin/env bash
# VOLSCAPE launcher (macOS / Linux). Creates .venv next to this file, installs pinned deps, starts the app.
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for cand in python3.12 python3.13 python3.11 python3 python; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' 2>/dev/null; then
    PY="$cand"; break
  fi
done
if [ -z "$PY" ]; then
  echo "[VOLSCAPE] ERROR: Python 3.11+ not found. Install it (macOS: 'brew install python@3.12'; Debian/Ubuntu: 'sudo apt install python3.12 python3.12-venv') and re-run." >&2
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "[VOLSCAPE] Creating virtual environment with $PY ..."
  if ! "$PY" -m venv .venv; then
    echo "[VOLSCAPE] ERROR: venv creation failed. On Debian/Ubuntu install the venv package (e.g. python3.12-venv); delete any partial .venv and retry." >&2
    exit 1
  fi
fi

VPY=.venv/bin/python
if ! cmp -s requirements.txt .venv/.installed-requirements; then
  echo "[VOLSCAPE] Installing pinned dependencies (first run only, needs internet once) ..."
  if ! "$VPY" -m pip install --disable-pip-version-check -q -r requirements.txt; then
    echo "[VOLSCAPE] ERROR: dependency installation failed. Check network/proxy. Python 3.11-3.13 recommended." >&2
    exit 1
  fi
  cp requirements.txt .venv/.installed-requirements
fi

if [ "${1:-}" = "test" ]; then
  exec "$VPY" -m pytest -q
fi

echo "[VOLSCAPE] Starting at http://127.0.0.1:8501 (Ctrl+C to stop)"
exec "$VPY" -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1
