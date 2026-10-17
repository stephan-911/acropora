#!/bin/bash
# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent
#
# Submit one stage of the Mach 2000 workflow.
#
#   ./jobs.sh <stage> [--after JOBID] [--dry-run] [-- extra sbatch options]
#
# stages:
#   restrict       restrict the native ensemble onto the coarsest grid (CPU)
#   main           convergence study: W1, strong L1, mean/std, both protocols
#   w2-reference   two-point Wasserstein, reference protocol
#   w2-cauchy      two-point Wasserstein, Cauchy protocol
#   centerline     centerline bands, PDF grids, heatmaps, band diagnostics
#   verify-gpu     recompute samples on the GPU and diff against the dataset
#   gen-test       small regeneration on one GPU node
#   gen-full       the full 20-node regeneration (about 1.1 TB)
#   merge          merge the gen-full part files into samples_{Nx}.npy
#
# Reads <repo>/acropora-env.sh (copy acropora-env.example.sh). Paths, partition
# and account come from there; nothing needs to be exported in your shell.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ENV_FILE="$REPO/acropora-env.sh"
[ -f "$ENV_FILE" ] || { echo "missing $ENV_FILE  (copy acropora-env.example.sh and edit it)"; exit 1; }
# shellcheck source=/dev/null
source "$ENV_FILE"
for v in VENV ACCOUNT PARTITION_CPU PARTITION_GPU DATA_NATIVE DATA_ON500 STATS; do
    [ -n "${!v:-}" ] || { echo "$v is not set in $ENV_FILE"; exit 1; }
done
[ -x "$VENV/bin/python" ] || { echo "no python at $VENV/bin/python"; exit 1; }

stage="${1:-}"; shift || true
after=""; dry=""
while [ $# -gt 0 ]; do
    case "$1" in
        --after)   after="$2"; shift 2 ;;
        --dry-run) dry=1; shift ;;
        --)        shift; break ;;
        *)         echo "unknown option $1"; exit 1 ;;
    esac
done

declare -A SCRIPT PART DIN DOUT
SCRIPT[restrict]=restrict_dataset.sh;    PART[restrict]=$PARTITION_CPU; DIN[restrict]=$DATA_NATIVE; DOUT[restrict]=$DATA_ON500
SCRIPT[main]=postprocess_main.sh;        PART[main]=$PARTITION_CPU;     DIN[main]=$DATA_ON500;      DOUT[main]=$STATS/main
SCRIPT[w2-reference]=postprocess_w2_reference.sh; PART[w2-reference]=$PARTITION_CPU; DIN[w2-reference]=$DATA_ON500; DOUT[w2-reference]=$STATS/w2_reference
SCRIPT[w2-cauchy]=postprocess_w2_cauchy.sh;       PART[w2-cauchy]=$PARTITION_CPU;    DIN[w2-cauchy]=$DATA_ON500;    DOUT[w2-cauchy]=$STATS/w2_cauchy
SCRIPT[centerline]=postprocess_centerline.sh; PART[centerline]=$PARTITION_CPU; DIN[centerline]=$DATA_NATIVE; DOUT[centerline]=$HERE/centerline
SCRIPT[verify-gpu]=verify_gpu.sh;        PART[verify-gpu]=$PARTITION_GPU; DIN[verify-gpu]=$DATA_NATIVE; DOUT[verify-gpu]=""
SCRIPT[gen-test]=generate_test.sh;       PART[gen-test]=$PARTITION_GPU;   DIN[gen-test]="";           DOUT[gen-test]=${DATA_REGEN:-$REPO/examples/mach2000_jet/data_regen}
SCRIPT[gen-full]=generate_data.sh;       PART[gen-full]=$PARTITION_GPU;   DIN[gen-full]="";           DOUT[gen-full]=${DATA_FULL:-$HERE/data}
SCRIPT[merge]=merge_data.sh;             PART[merge]=$PARTITION_CPU;      DIN[merge]=${DATA_FULL:-$HERE/data}; DOUT[merge]=${DATA_FULL:-$HERE/data}

[ -n "${SCRIPT[$stage]:-}" ] || { echo "usage: $0 {restrict|main|w2-reference|w2-cauchy|centerline|verify-gpu|gen-test|gen-full|merge}"; exit 1; }

mkdir -p "$HERE/logs"
opts=(--partition="${PART[$stage]}" --account="$ACCOUNT" --export=ALL)
[ -n "$after" ] && opts+=(--dependency=afterok:"$after")

export REPO VENV MODULES
export DATA_DIR="${DIN[$stage]}" OUT_DIR="${DOUT[$stage]}"
export PAPER_NPY="$DATA_NATIVE/samples_500.npy"

echo "stage      $stage"
echo "script     slurm/${SCRIPT[$stage]}"
echo "partition  ${PART[$stage]}   account $ACCOUNT"
[ -n "$DATA_DIR" ] && echo "reads      $DATA_DIR"
[ -n "$OUT_DIR" ]  && echo "writes     $OUT_DIR"
[ -n "$after" ]    && echo "after      job $after"
cmd=(sbatch "${opts[@]}" "$@" "$HERE/slurm/${SCRIPT[$stage]}")
echo "command    ${cmd[*]}"
if [ -n "$dry" ]; then echo "(dry run, not submitted)"; exit 0; fi
cd "$HERE"
"${cmd[@]}"
