#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/mnt/d/PHD/jsupercomp_submission/revision_workspace
TARGET="$WORKSPACE/tmp/wsl_tenseal_site"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$TARGET"

cd "$WORKSPACE"
exec python3 system/packed_ckks.py \
  --dimension 672 \
  --candidates 100 \
  --repeats 20 \
  --warmups 2 \
  --seed 20260710 \
  --output results/system_revision/systems_expansion/wsl_linux_ckks_micro.json
