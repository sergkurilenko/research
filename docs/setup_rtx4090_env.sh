#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements_rtx4090.txt}"
PIP_PROXY_ARG=()
if [[ -n "${PIP_PROXY:-}" ]]; then
  PIP_PROXY_ARG=(--proxy "$PIP_PROXY")
fi

if [[ -d "$VENV_DIR" ]]; then
  if [[ "${FORCE_RECREATE:-0}" == "1" ]]; then
    rm -rf "$VENV_DIR"
  else
    echo "Refusing to reuse existing $VENV_DIR. Set FORCE_RECREATE=1 to recreate it." >&2
    exit 2
  fi
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python - <<'PY'
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Expected Python 3.12 for the pinned RTX4090 wheel, got {sys.version}")
PY

python -m pip install -U pip "${PIP_PROXY_ARG[@]}"
python -m pip install -r "$REQUIREMENTS_FILE" "${PIP_PROXY_ARG[@]}"

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this environment")
if torch.version.cuda != "12.4":
    raise SystemExit(f"Expected CUDA 12.4 wheel, got {torch.version.cuda}")
PY
