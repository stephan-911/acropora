# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Grid geometry from the actual x-resolution.

Resolutions are given as Nx, the number of cells along x, matching the data
files (samples_500.npy, samples_1000.npy, ...). The transverse count and the
mesh width follow from the domain aspect: for the jet domain
[0, 2.5] x [-1, 1] the ladder Nx = 500, 1000, 2000, 4000 gives
Ny = 400, 800, 1600, 3200 and dx = 5.0e-3, 2.5e-3, 1.25e-3, 6.25e-4.
"""
from __future__ import annotations

DOMAIN = (2.5, 2.0)   # (length in x, height in y)


def grid_from_nx(nx, domain=DOMAIN):
    """Return (Nx, Ny, dx) for the given x-resolution."""
    dx = domain[0] / nx
    ny = int(round(domain[1] / dx))
    return int(nx), ny, dx


def nx_ladder_check(nx, domain=DOMAIN):
    """True if Nx yields an integer transverse count on this domain."""
    ratio = domain[1] * nx / domain[0]
    return abs(ratio - round(ratio)) < 1e-9
