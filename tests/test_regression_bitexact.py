# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Regression: acropora must reproduce the reference implementation
bit-for-bit on CPU.

The baseline is the reference generation script running through a NumPy
shim (baseline/lbm_baseline_numpy.py). The library runs with
rng_mode='legacy-numpy', replaying the shim's coefficient stream, so the full
chain (perturbed inlet -> N steps) must agree to the last bit.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "baseline"))

from acropora import Euler2D, VectorialD2Q5, Mach2000JetBC, grid_from_nx  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "lbm_baseline", os.path.join(ROOT, "baseline", "lbm_baseline_numpy.py"))
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def _run_baseline(nx, seed, n_steps, boundary_mode):
    Nx, Ny, dx = grid_from_nx(nx)
    s = baseline.VectorialLBM_D2Q5(Nx, Ny, dx, boundary_mode=boundary_mode,
                                   gamma=5.0 / 3.0)
    s.init_mach_2000_jet(seed=seed)
    dts = [float(s.step()) for _ in range(n_steps)]
    return np.asarray(s.U), dts


def _run_library(nx, seed, n_steps, boundary_mode):
    Nx, Ny, dx = grid_from_nx(nx)
    bc = Mach2000JetBC.make(boundary_mode, Ny, seed=seed,
                            rng_mode="legacy-numpy")
    s = VectorialD2Q5(Euler2D(gamma=5.0 / 3.0), Nx, Ny, dx,
                      xp=np, boundary=bc)
    bc.initial_condition(s)
    dts = [float(s.step()) for _ in range(n_steps)]
    return np.asarray(s.U), dts


@pytest.mark.parametrize("boundary_mode", ["ghost", "sponge"])
@pytest.mark.parametrize("nx,seed,n_steps", [
    (50, 42, 60),
    (50, 43, 60),
    (100, 42, 40),
    (100, 137, 40),
])
def test_bitexact_full_chain(nx, seed, n_steps, boundary_mode):
    U_b, dts_b = _run_baseline(nx, seed, n_steps, boundary_mode)
    U_l, dts_l = _run_library(nx, seed, n_steps, boundary_mode)
    assert U_b.shape == U_l.shape
    assert dts_b == dts_l, "time-step sequences differ"
    diff = np.abs(U_b - U_l)
    assert diff.max() == 0.0, f"max |diff| = {diff.max():.3e} (not bit-exact)"


def test_initial_condition_bitexact():
    nx, seed = 100, 42
    U_b, _ = _run_baseline(nx, seed, 0, "ghost")
    U_l, _ = _run_library(nx, seed, 0, "ghost")
    assert np.abs(U_b - U_l).max() == 0.0


def test_portable_rng_is_backend_independent():
    from acropora import draw_coefficients
    Y1, Z1 = draw_coefficients(7, 10, "portable")
    Y2, Z2 = draw_coefficients(7, 10, "portable")
    assert np.array_equal(Y1, Y2) and np.array_equal(Z1, Z2)
    Y3, _ = draw_coefficients(8, 10, "portable")
    assert not np.array_equal(Y1, Y3)
    assert np.all(np.abs(Y1) <= 1.0) and np.all(np.abs(Z1) <= 1.0)
