#!/usr/bin/env bash
#SBATCH --job-name=corgi-smoke
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00

# Supply IMAGE, SOURCE_ARCHIVE, ENV_ARCHIVE and RESULTS at submission.
# Prepare/lock/install dependencies on the login node before allocating a GPU.
set -euo pipefail
: "${IMAGE:?}" "${SOURCE_ARCHIVE:?}" "${ENV_ARCHIVE:?}" "${RESULTS:?}"
scratch=$(mktemp -d "/var/tmp/corgi-${SLURM_JOB_ID}-XXXXXX")
tar -xf "$SOURCE_ARCHIVE" -C "$scratch"
tar -xf "$ENV_ARCHIVE" -C "$scratch"
export CORGI_RUN_DIR="$RESULTS/job-$SLURM_JOB_ID"
mkdir -p "$CORGI_RUN_DIR"
nvidia-smi > "$CORGI_RUN_DIR/nvidia-smi.txt"
apptainer exec --nv --bind /data0:/data0 --bind /var/tmp:/var/tmp \
    "$IMAGE" bash -c 'set -e; cd "$1"; export PYTHONPATH="$PWD/src"; bash deploy/smoke.sh' _ "$scratch"
# Only remove this job's own scratch directory after success; keep failures for diagnosis.
find "$scratch" -depth -delete
