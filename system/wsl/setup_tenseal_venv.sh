#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/mnt/d/PHD/jsupercomp_submission/revision_workspace
TARGET="$WORKSPACE/tmp/wsl_tenseal_site"
WHEELHOUSE="$WORKSPACE/tmp/wsl_tenseal_wheels"

echo UNAME
uname -a
echo DISTRO
grep '^PRETTY_NAME=' /etc/os-release
echo CPU
grep -m1 'model name' /proc/cpuinfo
echo PYTHON_BASE
python3 --version
echo SAGE_BASE
sage --version
echo ACTIVE_SECURITY_PROCESSES_BEFORE
pgrep -af 'sage|estimator' || true

# Ubuntu's base Python lacks ensurepip/python3.10-venv.  A pip --target tree on
# the D-backed workspace provides equivalent disposable isolation without apt,
# a user-site install, or any change to Sage/the base distribution.  Download
# first so the exact wheel filenames and compatibility tags remain archived.
mkdir -p "$TARGET" "$WHEELHOUSE"
python3 -m pip download \
  --only-binary=:all: \
  --no-cache-dir \
  --dest "$WHEELHOUSE" \
  tenseal==0.3.16 \
  numpy

echo DOWNLOADED_BINARY_WHEELS
find "$WHEELHOUSE" -maxdepth 1 -type f -printf '%f\n' | sort

python3 -m pip install \
  --only-binary=:all: \
  --no-index \
  --find-links "$WHEELHOUSE" \
  --target "$TARGET" \
  tenseal==0.3.16 \
  numpy

echo INSTALLED_RUNTIME
PYTHONNOUSERSITE=1 PYTHONPATH="$TARGET" python3 - <<'PY'
import importlib.metadata as metadata
import platform
import sys

import numpy

print("python", sys.version.replace("\n", " "))
print("platform", platform.platform())
print("numpy", numpy.__version__)
print("tenseal", metadata.version("tenseal"))
print("numpy_file", numpy.__file__)
PY
