#!/bin/bash
# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent
#
# Full ensemble: 20 nodes x 4 GPUs, the published configuration.
# --rng legacy-cupy reproduces the dataset seed-for-seed. About 1.1 TB.
#SBATCH --job-name=acropora_mach2000
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=logs/data_%A_task_%a.out

# --- environment: everything comes from the driver (jobs.sh) via REPO -------
: "${REPO:?submit through jobs.sh, or export REPO=/path/to/the/repository}"
source "$REPO/acropora-env.sh"
if [ -n "${MODULES:-}" ]; then module purge; module load $MODULES; fi
source "$VENV/bin/activate"
cd "$REPO/examples/mach2000_jet"
export PYTHONUNBUFFERED=1
mkdir -p logs

export CUPY_CACHE_DIR=/tmp/.cupy_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}; mkdir -p "$CUPY_CACHE_DIR"
NUM_GPUS=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
echo "node $(hostname), array task ${SLURM_ARRAY_TASK_ID}"
python3 run_generation.py \
    --samples 1000 --max-t 0.0035 --num-checkpoints 8 \
    --resolutions 500 1000 2000 4000 \
    --backend gpu --workers "$NUM_GPUS" --num-devices "$NUM_GPUS" \
    --node-id "$SLURM_ARRAY_TASK_ID" --total-nodes 20 \
    --boundary-mode ghost --rng legacy-cupy \
    --data-dir "$OUT_DIR"
rm -rf "$CUPY_CACHE_DIR"
echo "array task $SLURM_ARRAY_TASK_ID completed"
