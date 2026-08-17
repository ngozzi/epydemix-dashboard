#!/usr/bin/env bash
# Resilient launcher for the EpyScenario dashboard.
#
# - Forces single-threaded native math (numpy/BLAS/numexpr) to avoid a class of
#   intermittent native segfaults when heavy math runs inside Streamlit's worker
#   threads.
# - Supervises the Streamlit process and restarts it automatically if it exits
#   (e.g. an intermittent native crash), so the app stays available.
#
# Usage:  ./run_dashboard.sh [PORT]      (default port 8501)

set -u
cd "$(dirname "$0")"

PORT="${1:-8501}"

export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

PY="./venv/Scripts/python.exe"
[ -x "$PY" ] || PY="python"

while true; do
  "$PY" -m streamlit run Dashboard.py --server.port "$PORT" --server.headless true
  code=$?
  echo "[supervisor] streamlit exited (code $code) — restarting in 2s..." >&2
  sleep 2
done
