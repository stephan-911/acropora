#!/bin/bash
# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent
#
# Site configuration for the batch jobs. Copy to acropora-env.sh (git-ignored)
# next to this file, at the repository root, and fill in your paths. The job
# driver examples/mach2000_jet/jobs.sh reads it; nothing has to be exported
# in your shell.

# python environment with acropora installed (pip install -e ".[post]")
export VENV=/path/to/acropora-venv

# SLURM
export ACCOUNT=your-project-account
export PARTITION_CPU=cpuonly
export PARTITION_GPU=accelerated
export MODULES="compiler/gnu devel/cuda"      # loaded before activating VENV

# data
export DATA_NATIVE=/path/to/data              # samples_{500,1000,2000,4000}.npy
export DATA_FULL=/path/to/data_full           # output of ./jobs.sh gen-full (~1.1 TB)
export DATA_ON500=/path/to/data_on500         # restricted copy (~51 GB)
export DATA_REGEN=/path/to/data_regen         # regeneration tests
export STATS=/path/to/stats                   # analysis CSVs
