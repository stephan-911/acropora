#!/bin/bash
# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent
#
# Two-point Wasserstein distance, cauchy protocol (exact EMD; the long job).
#SBATCH --job-name=acropora_w2_cauchy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/w2_cauchy_%j.out

# --- environment: everything comes from the driver (jobs.sh) via REPO -------
: "${REPO:?submit through jobs.sh, or export REPO=/path/to/the/repository}"
source "$REPO/acropora-env.sh"
if [ -n "${MODULES:-}" ]; then module purge; module load $MODULES; fi
source "$VENV/bin/activate"
cd "$REPO/examples/mach2000_jet"
export PYTHONUNBUFFERED=1
mkdir -p logs

python3 run_postprocess.py --convergence \
    --data-dir "$DATA_DIR" --restricted-on 500 --out-dir "$OUT_DIR" \
    --resolutions 500 1000 2000 4000 \
    --samples 1000 --num-checkpoints 8 --max-t 0.0035 \
    --fixedM 1000 --protocols cauchy --marginal 2 --n-per-dir 10
