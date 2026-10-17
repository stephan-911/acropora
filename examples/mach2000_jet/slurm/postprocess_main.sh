#!/bin/bash
# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent
#
# Convergence study on the restricted dataset: one-point Wasserstein,
# strong L1, mean/std, both protocols, coarsest restriction.
#SBATCH --job-name=acropora_main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/main_%j.out

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
    --fixedM 1000 --protocols both --marginal 1
