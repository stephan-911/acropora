# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Statistical observables between two ensembles on a common grid.

All functions take stacks of shape (M, Nx, Ny) already restricted to the same
grid (see restrict.coarsen_to) and reproduce the published pipeline exactly.
"""
from __future__ import annotations
import numpy as np
import scipy.stats


def w1_percell(a, b):
    """One-point Wasserstein-1 distance, averaged over grid cells."""
    M = a.shape[0]
    af = a.reshape(M, -1); bf = b.reshape(M, -1)
    n = af.shape[1]
    s = 0.0
    for i in range(n):
        s += scipy.stats.wasserstein_distance(af[:, i], bf[:, i])
    return s / n


def w2_twopoint(a, b, n_per_dir):
    """Two-point (second correlation marginal) Wasserstein distance.

    Subsamples n_per_dir points per spatial direction, pairs all with all,
    and at each spatial point-pair computes the 2-D Wasserstein distance
    between the empirical joints via exact EMD (POT). Returns the average
    over all point-pairs. Cost ~ n_per_dir^4 EMD solves.
    """
    import ot
    Ma, Nx, Ny = a.shape
    Mb = b.shape[0]
    xs = np.linspace(0, Nx - 1, n_per_dir).astype(int)
    ys = np.linspace(0, Ny - 1, n_per_dir).astype(int)
    pts = [(i, j) for i in xs for j in ys]
    wa = np.ones(Ma) / Ma
    wb = np.ones(Mb) / Mb
    total = 0.0
    npairs = 0
    xs_cloud = np.zeros((Ma, 2))
    xt_cloud = np.zeros((Mb, 2))
    for (i1, j1) in pts:
        a1 = a[:, i1, j1]; b1 = b[:, i1, j1]
        for (i2, j2) in pts:
            xs_cloud[:, 0] = a1
            xs_cloud[:, 1] = a[:, i2, j2]
            xt_cloud[:, 0] = b1
            xt_cloud[:, 1] = b[:, i2, j2]
            Mcost = ot.dist(xs_cloud, xt_cloud, metric='euclidean')
            G = ot.emd(wa, wb, Mcost)
            total += float(np.sum(G * Mcost))
            npairs += 1
    return total / npairs


def strong_l1(a, b):
    """Per-sample relative L1 error (relative to b), averaged over samples."""
    M = a.shape[0]
    errs = np.empty(M)
    for j in range(M):
        denom = np.sum(np.abs(b[j]))
        errs[j] = np.sum(np.abs(a[j] - b[j])) / denom if denom > 0 else 0.0
    return float(np.mean(errs))


def moment_l1(a, b):
    """Per-cell-averaged L1 errors of the ensemble mean and population std."""
    ma = a.mean(axis=0); mb = b.mean(axis=0)
    sa = a.std(axis=0);  sb = b.std(axis=0)
    ncell = ma.size
    mean_err = float(np.sum(np.abs(ma - mb)) / ncell)
    std_err = float(np.sum(np.abs(sa - sb)) / ncell)
    return mean_err, std_err
