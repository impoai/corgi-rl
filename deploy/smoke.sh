#!/usr/bin/env bash
# Run inside the prepared remote NVIDIA environment. No package installation.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MUJOCO_GL=${MUJOCO_GL:-egl}
export MJLAB_WARP_QUIET=1
export WANDB_MODE=disabled
export GIT_PYTHON_REFRESH=quiet
export OMP_NUM_THREADS=4
PY=${CORGI_PYTHON:-.venv/bin/python}
OUT=${CORGI_RUN_DIR:?Set CORGI_RUN_DIR to a new, persistent run directory}
mkdir -p "$OUT"
"$PY" -m pytest -q -s tests | tee "$OUT/tests.log"
"$PY" -m corgi_rl.cli model
"$PY" -m corgi_rl.cli evaluate --seconds 3 --video "$OUT/home-pd.mp4" > "$OUT/home-pd.log"
for task in stand walk; do
    "$PY" -m corgi_rl.cli train --task "$task" --output "$OUT/$task" \
        --num-envs 32 --steps 24 --iterations 2 2>&1 | tee "$OUT/$task-train.log"
    "$PY" -m corgi_rl.cli evaluate --task "$task" --policy "$OUT/$task/policy.onnx" \
        --seconds 3 --video "$OUT/$task.mp4" > "$OUT/$task-evaluate.log"
done
