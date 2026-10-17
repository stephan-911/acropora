#!/bin/bash
# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent
#
# Small regeneration on one GPU node: exercises the full generation path.
# Writes to its own folder; the production ensemble is never touched.
#SBATCH --job-name=acropora_gen_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=logs/gen_test_%j.out

# --- environment: everything comes from the driver (jobs.sh) via REPO -------
: "${REPO:?submit through jobs.sh, or export REPO=/path/to/the/repository}"
source "$REPO/acropora-env.sh"
if [ -n "${MODULES:-}" ]; then module purge; module load $MODULES; fi
source "$VENV/bin/activate"
cd "$REPO/examples/mach2000_jet"
export PYTHONUNBUFFERED=1
mkdir -p logs

export CUPY_CACHE_DIR=/tmp/.cupy_${SLURM_JOB_ID}; mkdir -p "$CUPY_CACHE_DIR"
NUM_GPUS=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
python3 run_generation.py \
    --samples "${GEN_SAMPLES:-20}" --max-t 0.0035 --num-checkpoints 8 \
    --resolutions ${GEN_RES:-500 1000} \
    --backend gpu --workers "$NUM_GPUS" --num-devices "$NUM_GPUS" \
    --node-id 0 --total-nodes 1 --boundary-mode ghost --rng legacy-cupy \
    --data-dir "$OUT_DIR"
rm -rf "$CUPY_CACHE_DIR"
