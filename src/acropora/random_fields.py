# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Stochastic inlet perturbation (truncated Fourier / KL-type expansion).

rho(y) = base * (1 + A * sum_k k^-p (Y_k cos(k pi y) + Z_k sin(k pi y)) * W(y)),
W(y) = (1 + cos(pi y)) / 2 on the jet-normalized coordinate y in (-1, 1),
Y_k, Z_k ~ U(-1, 1) i.i.d. The amplitude is applied WITHOUT normalizing by the
sample maximum, preserving the independence and variance of the coefficients
(required for the Wasserstein convergence study).

RNG modes
---------
portable      backend-independent numpy PCG64 stream per seed;
              identical initial data on CPU and GPU. Default.
legacy-cupy   reproduces the published H200 dataset (CuPy global XORWOW,
              draw order Y_1,Z_1,...,Y_K,Z_K). GPU only.
legacy-numpy  same draw order through numpy's legacy MT19937 global state;
              reproduces the CPU baseline port for regression tests.
"""
from __future__ import annotations
import numpy as np


def draw_coefficients(seed: int, num_modes: int, mode: str = "portable", xp=np):
    """Return (Y, Z) each of shape (num_modes,) as float64 numpy arrays."""
    if mode == "portable":
        rng = np.random.Generator(np.random.PCG64(seed))
        yz = rng.uniform(-1.0, 1.0, size=(num_modes, 2))
        return yz[:, 0].copy(), yz[:, 1].copy()
    if mode == "legacy-numpy":
        np.random.seed(seed)
        Y = np.empty(num_modes); Z = np.empty(num_modes)
        for k in range(num_modes):
            Y[k] = np.random.uniform(-1.0, 1.0)
            Z[k] = np.random.uniform(-1.0, 1.0)
        return Y, Z
    if mode == "legacy-cupy":
        cp = xp
        if cp is np:
            raise ValueError("rng mode 'legacy-cupy' requires the GPU backend")
        cp.random.seed(seed)
        Y = np.empty(num_modes); Z = np.empty(num_modes)
        for k in range(num_modes):
            Y[k] = float(cp.random.uniform(-1.0, 1.0))
            Z[k] = float(cp.random.uniform(-1.0, 1.0))
        return Y, Z
    raise ValueError(f"unknown rng mode {mode!r}")


def fourier_perturbation(y_coords, seed: int, base_val: float, *, xp=np,
                         num_modes: int = 10, amplitude: float = 0.1,
                         p: float = 2.0, rng_mode: str = "portable"):
    """Perturbed profile on `y_coords` (jet-normalized, in (-1,1)).

    The accumulation order (k ascending, then windowing) is fixed, so
    results are bit-reproducible given equal coefficients.
    """
    Y, Z = draw_coefficients(seed, num_modes, rng_mode, xp)
    perturbation = xp.zeros_like(y_coords)
    for k in range(1, num_modes + 1):
        scaling = 1.0 / (k ** p)
        perturbation += scaling * (float(Y[k-1]) * xp.cos(k * xp.pi * y_coords)
                                   + float(Z[k-1]) * xp.sin(k * xp.pi * y_coords))
    window = 0.5 * (1.0 + xp.cos(xp.pi * y_coords))
    return base_val * (1.0 + amplitude * perturbation * window)
