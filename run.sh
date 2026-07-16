#!/usr/bin/env bash
#
# Project ThermoLogic — one-command, zero-intervention pipeline.
#
# Ensures the (only) dependencies are present, then trains the hybrid
# Differentiable Theorem Proving + Energy-Based Model prototype end-to-end.
#
# Usage:
#   ./run.sh                 # defaults
#   ./run.sh --epochs 60     # any train.py flag is forwarded
#
set -euo pipefail

cd "$(dirname "$0")"

PY="${PYTHON:-python3}"

echo ">> Project ThermoLogic pipeline"
echo ">> using interpreter: $("$PY" --version 2>&1)"

# --- dependencies -------------------------------------------------------- #
# torch + numpy are the only requirements. We install from the default PyPI
# index (torch CPU wheels are published there); nothing else is needed.
ensure_pkg() {
  local import_name="$1" pip_name="$2"
  if "$PY" -c "import ${import_name}" >/dev/null 2>&1; then
    echo ">> dependency '${pip_name}' already present"
  else
    echo ">> installing '${pip_name}' ..."
    "$PY" -m pip install --quiet "${pip_name}"
  fi
}

ensure_pkg torch torch
ensure_pkg numpy numpy

# --- run ----------------------------------------------------------------- #
echo ">> launching training"
exec "$PY" train.py "$@"
