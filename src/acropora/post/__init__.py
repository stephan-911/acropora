# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

from .restrict import coarsen_to
from .metrics import w1_percell, w2_twopoint, strong_l1, moment_l1
from .protocols import steps_for, fit_slope, scan_healthy, load_stack, open_dataset
from .convergence import run_convergence
from .merge import merge_parts
from .centerline import (read_centerline, centerline_stats,
                         centerline_pdf_grid, write_centerline_dat,
                         write_pdf_grid_dat, coarsen_pdf_grid)
from .diagnostics import band_diagnostics, format_diagnostics
from .heatmap import (load_pdf_grid, render_pdf_png, global_scale,
                      panel_tex, colorbar_tex)
