# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Centerline (y=0) ensemble analysis: percentile bands and one-point PDFs.

Produces the quantities behind the spatially resolved figures: per-x ensemble
mean, population std, 16/84 and 5/95 percentile bands of the centerline
density, and the per-x histogram of the density ensemble (the discretized
one-point probability density f(x, rho)). Streaming access touches only the
centerline rows, so peak memory is O(M*Nx).
"""
from __future__ import annotations
import numpy as np


def read_centerline(fp, hoff, s, t, T, Nx, Ny, rows):
    """(Nx,) centerline profile of sample s at checkpoint t; `rows` is a slice
    over the transverse axis, averaged if it spans more than one row."""
    if t < 0:
        t = T + t
    fb = Nx * Ny * 8
    off = hoff + (s * T + t) * fb
    mm = np.memmap(fp, dtype=np.float64, mode="r", offset=off, shape=(Nx, Ny))
    band = np.array(mm[:, rows])
    del mm
    return band.mean(axis=1) if band.ndim == 2 and band.shape[1] > 1 else band[:, 0]


def centerline_stats(fp, hoff, healthy, ck, T, Nx, Ny, rows,
                     pct_lo1=16.0, pct_hi1=84.0, pct_lo2=5.0, pct_hi2=95.0,
                     return_stack=False):
    """Ensemble mean (Welford), population std, and two percentile bands of
    the centerline density over the healthy ensemble.

    The 16/84 band is the distribution-free analogue of mean +/- 1 std (~68%
    central coverage) and 5/95 gives ~90% coverage; both remain meaningful
    under the strong right-skew at the turbulent tip, where a symmetric
    mean +/- std envelope would run below zero.
    Returns (mean, std, n, p16, p84, p05, p95[, stack]).
    """
    n = 0
    mean = np.zeros(Nx)
    M2 = np.zeros(Nx)
    stack = np.empty((len(healthy), Nx), dtype=np.float64)
    for k, s in enumerate(healthy):
        prof = read_centerline(fp, hoff, s, ck, T, Nx, Ny, rows)
        stack[k] = prof
        n += 1
        delta = prof - mean
        mean += delta / n
        M2 += delta * (prof - mean)
    var = M2 / n if n > 0 else np.zeros(Nx)
    std = np.sqrt(var)
    p_lo1, p_hi1, p_lo2, p_hi2 = np.percentile(
        stack, [pct_lo1, pct_hi1, pct_lo2, pct_hi2], axis=0)
    if return_stack:
        return mean, std, n, p_lo1, p_hi1, p_lo2, p_hi2, stack
    return mean, std, n, p_lo1, p_hi1, p_lo2, p_hi2


def centerline_pdf_grid(stack, nbins, rho_min=None, rho_max=None):
    """Per-x histogram of the centerline ensemble: the discretized one-point
    density f(x, rho). Returns (rho_centers, counts, pdf) with each x-slice
    of pdf integrating to one."""
    M, Nx = stack.shape
    if rho_min is None:
        rho_min = float(np.min(stack))
    if rho_max is None:
        rho_max = float(np.max(stack))
    if rho_max <= rho_min:
        rho_max = rho_min + 1.0
    edges = np.linspace(rho_min, rho_max, nbins + 1)
    d_rho = edges[1] - edges[0]
    rho_centers = 0.5 * (edges[:-1] + edges[1:])
    counts = np.empty((Nx, nbins), dtype=np.float64)
    for i in range(Nx):
        counts[i], _ = np.histogram(stack[:, i], bins=edges)
    pdf = counts / (M * d_rho)
    return rho_centers, counts, pdf


def write_centerline_dat(path, x, sample, mean, std, p16, p84, p05, p95,
                         header=""):
    """Eight-column .dat consumed by the TikZ figures:
    x  rho_sample  rho_mean  rho_std  p16  p84  p05  p95."""
    cols = np.column_stack([x, sample, mean, std, p16, p84, p05, p95])
    np.savetxt(path, cols, fmt="%.8e",
               header=header + "x  rho_sample  rho_mean  rho_std  p16  p84  p05  p95")


def write_pdf_grid_dat(path, x, rho_centers, pdf, counts, header=""):
    """Long/tidy grid for pgfplots: one row per (x, rho) cell, columns
    `x rho pdf count`, with a blank line between x-slices (the mesh/surf
    scan-line format)."""
    nx = len(x); nb = len(rho_centers)
    with open(path, "w") as fh:
        if header:
            for line in header.strip().splitlines():
                fh.write(f"# {line}\n")
        fh.write("# full x-axis; blank line separates x-slices "
                 "(pgfplots mesh/surf scanline format)\n")
        fh.write("# x  rho  pdf  count\n")
        for i in range(nx):
            for j in range(nb):
                fh.write(f"{x[i]:.8e} {rho_centers[j]:.8e} "
                         f"{pdf[i, j]:.8e} {counts[i, j]:.1f}\n")
            fh.write("\n")


def coarsen_pdf_grid(x, pdf, counts, n_xslices):
    """Re-bin the (x, rho) grid onto ~n_xslices streamwise stations by
    averaging the density (and summing counts) over contiguous x-blocks.
    Full-resolution surfaces exceed pgfplots' memory; the coarse grid is
    both compilable and more legible."""
    nx = len(x)
    nxs = min(n_xslices, nx)
    edges = np.linspace(0, nx, nxs + 1).astype(int)
    x_c = np.array([x[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
    pdf_c = np.array([pdf[a:b].mean(axis=0)
                      for a, b in zip(edges[:-1], edges[1:])])
    cnt_c = np.array([counts[a:b].sum(axis=0)
                      for a, b in zip(edges[:-1], edges[1:])])
    return x_c, pdf_c, cnt_c
