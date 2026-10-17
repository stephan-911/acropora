#!/bin/bash
# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent
#
# Centerline bands, one-point density grids, PDF heatmaps and the
# quantile-band diagnostics. Reads the NATIVE data (resolution-dependent).
#SBATCH --job-name=acropora_centerline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=logs/centerline_%j.out

# --- environment: everything comes from the driver (jobs.sh) via REPO -------
: "${REPO:?submit through jobs.sh, or export REPO=/path/to/the/repository}"
source "$REPO/acropora-env.sh"
if [ -n "${MODULES:-}" ]; then module purge; module load $MODULES; fi
source "$VENV/bin/activate"
cd "$REPO/examples/mach2000_jet"
export PYTHONUNBUFFERED=1
mkdir -p logs

mkdir -p "$OUT_DIR"
python3 run_postprocess.py --centerline --pdf-grid \
    --data-dir "$DATA_DIR" --centerline-out "$OUT_DIR" \
    --resolutions 500 1000 2000 4000 \
    --samples 1000 --num-checkpoints 8 --max-t 0.0035 \
    --sample-index 0 --pdf-nbins 80 --pdf-xslices 60
python3 run_postprocess.py --heatmap \
    --centerline-out "$OUT_DIR" --heatmap-nx 4000 > "$OUT_DIR/heatmap_figure.tex"
python3 run_postprocess.py --check-bands \
    --centerline-out "$OUT_DIR" --head-region 1.4 2.3 > "$OUT_DIR/band_diagnostics.txt"
