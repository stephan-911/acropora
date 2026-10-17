# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Physics sanity tests (CPU): positivity, finiteness, jet advance,
inlet-perturbation statistics."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))
from acropora import (Euler2D, VectorialD2Q5, Mach2000JetBC,
                     draw_coefficients, grid_from_nx)


def _solve(nx=50, seed=42, n=120):
    Nx, Ny, dx = grid_from_nx(nx)
    bc = Mach2000JetBC.make('ghost', Ny, seed=seed)
    s = VectorialD2Q5(Euler2D(5/3), Nx, Ny, dx, xp=np, boundary=bc)
    bc.initial_condition(s)
    for _ in range(n):
        s.step()
    return s


def test_positivity_and_finiteness():
    for seed in (42, 137, 7):
        s = _solve(seed=seed)
        rho = s.U[0]
        assert np.isfinite(s.U).all()
        assert (rho > 0).all(), "density positivity violated"


def test_jet_advances():
    # Nx=100: the jet half-width spans 2 nodes; at Nx=50 it is a single node
    # and over-diffuses (documented resolution floor for this setup)
    s = _solve(nx=100, n=80)
    sx, sy = s.interior()
    rho = np.asarray(s.U[0, sx, sy])
    assert rho.max() > 5.5, "no compression developed"
    ix = np.unravel_index(np.argmax(rho), rho.shape)[0]
    assert ix > 3, "jet head did not advance"


def test_inlet_std_matches_theory():
    """On the axis the perturbation std is A*rho*sqrt(sum k^-4 / 3) ~ 0.30
    (the value quoted in the paper). Monte Carlo over seeds."""
    from acropora import fourier_perturbation
    vals = [float(fourier_perturbation(np.zeros(1), s, 5.0)[0])
            for s in range(4000)]
    std = np.std(vals)
    theory = 0.1*5.0*np.sqrt(sum(k**-4 for k in range(1, 11))/3.0)
    assert abs(std - theory) < 0.02, (std, theory)
