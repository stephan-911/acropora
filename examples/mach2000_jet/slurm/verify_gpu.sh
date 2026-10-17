#!/bin/bash
# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent
#
# GPU verification: recompute samples with the legacy CuPy RNG and diff them
# against the stored dataset (expect bit-exact on the original hardware and
# CuPy version), then GPU-vs-CPU divergence growth and throughput.
#SBATCH --job-name=acropora_verify_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/verify_gpu_%j.out

# --- environment: everything comes from the driver (jobs.sh) via REPO -------
: "${REPO:?submit through jobs.sh, or export REPO=/path/to/the/repository}"
source "$REPO/acropora-env.sh"
if [ -n "${MODULES:-}" ]; then module purge; module load $MODULES; fi
source "$VENV/bin/activate"
cd "$REPO/examples/mach2000_jet"
export PYTHONUNBUFFERED=1
mkdir -p logs

export CUPY_CACHE_DIR=/tmp/.cupy_${SLURM_JOB_ID}; mkdir -p "$CUPY_CACHE_DIR"
python3 "$REPO/benchmarks/bench_gpu_cluster.py" \
    --paper-npy "$PAPER_NPY" --paper-nx 500 --paper-samples "${PAPER_SAMPLES:-3}" \
    --num-checkpoints 8 --seed-base 42 \
    --cross-nx 250 --speed-nx 1000 --speed-steps 200
rm -rf "$CUPY_CACHE_DIR"
