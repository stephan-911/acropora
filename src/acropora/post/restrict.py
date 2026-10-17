# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Grid restriction for cross-resolution comparison.

Pairwise-to-coarse convention: in every comparison the finer field is
block-averaged (volume mean) down to the coarser member's native grid, and
the coarse field is left untouched, so comparisons happen on the coarse
member's own mesh.
"""
from __future__ import annotations

from ..grid import grid_from_nx


def coarsen_to(field, nx_src, nx_tgt):
    """Block-average a (M, Nx, Ny) stack from x-resolution nx_src to nx_tgt."""
    if nx_src == nx_tgt:
        return field
    _, Ny_t, _ = grid_from_nx(nx_tgt)
    sx = nx_src // nx_tgt
    sy = grid_from_nx(nx_src)[1] // Ny_t
    return field[:, :nx_tgt * sx, :Ny_t * sy].reshape(
        -1, nx_tgt, sx, Ny_t, sy).mean(axis=(2, 4))
