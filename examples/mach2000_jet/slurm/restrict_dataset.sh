#!/bin/bash
# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent
#
# One-off restriction of the native ensemble onto the coarsest grid.
# I/O bound (reads every frame once); resumable, so resubmit after a timeout.
#SBATCH --job-name=acropora_restrict
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/restrict_%j.out

# --- environment: everything comes from the driver (jobs.sh) via REPO -------
: "${REPO:?submit through jobs.sh, or export REPO=/path/to/the/repository}"
source "$REPO/acropora-env.sh"
if [ -n "${MODULES:-}" ]; then module purge; module load $MODULES; fi
source "$VENV/bin/activate"
cd "$REPO/examples/mach2000_jet"
export PYTHONUNBUFFERED=1
mkdir -p logs

python3 restrict_dataset.py \
    --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" \
    --resolutions 500 1000 2000 4000 --target 500 \
    --samples 1000 --num-checkpoints 8 --max-t 0.0035 --verify 5
