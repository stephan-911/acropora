# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Comparison protocols and rate fitting.

cauchy    : consecutive resolution pairs (r_i, r_{i+1})
reference : each coarser resolution vs the finest

Fit abscissa: dx = 2.5/coarse_nx, the mesh width of the coarser member of
each pair (the paper's Delta x).
"""
from __future__ import annotations
import numpy as np

from ..grid import DOMAIN, grid_from_nx
from ..io.npy_store import npy_header_offset, read_frame


def steps_for(protocol, nx_list):
    """Comparison steps (coarse_nx, fine_nx, dx_fit) for one protocol; dx_fit
    is the mesh width of the coarser member."""
    nx_list = sorted(nx_list)
    scale = DOMAIN[0]
    if protocol == "cauchy":
        return [(nx_list[i], nx_list[i + 1], scale / nx_list[i])
                for i in range(len(nx_list) - 1)]
    if protocol == "reference":
        fin = nx_list[-1]
        return [(nx, fin, scale / nx) for nx in nx_list[:-1]]
    raise ValueError(f"unknown protocol {protocol!r}")


def fit_slope(dxs, vals):
    """Least-squares slope of log(val) vs log(dx), ignoring nonpositive values."""
    dxs = np.array(dxs); vals = np.array(vals)
    good = (vals > 0) & np.isfinite(vals)
    if good.sum() >= 2:
        return float(np.polyfit(np.log(dxs[good]), np.log(vals[good]), 1)[0])
    return np.nan


def scan_healthy(files, headers, dims, samples, num_checkpoints):
    """Samples whose final checkpoint is written and finite at EVERY resolution."""
    finite = np.ones(samples, dtype=bool)
    for nx, fp in files.items():
        Nx, Ny = dims[nx]
        for i in range(samples):
            a = np.asarray(read_frame(fp, headers[nx], i, num_checkpoints - 1,
                                      num_checkpoints, Nx, Ny))
            if np.all(a == 0.0) or not np.isfinite(a).all():
                finite[i] = False
    return np.where(finite)[0]


def load_stack(fp, hoff, idx, ck, T, Nx, Ny):
    out = np.empty((len(idx), Nx, Ny), dtype=np.float64)
    for k, s in enumerate(idx):
        out[k] = np.asarray(read_frame(fp, hoff, s, ck, T, Nx, Ny))
    return out


def open_dataset(data_dir, resolutions, samples, num_checkpoints,
                 restricted_on=None):
    """Locate the ensemble files; return (files, headers, dims) keyed by the
    x-resolution Nx the data were computed at.

    restricted_on=None expects native files samples_{Nx}.npy. Setting it to a
    target resolution expects pre-restricted files
    samples_{Nx}_on{target}.npy (see restrict_dataset.py), all of which live
    on the target grid, so dims are the target's for every entry.
    """
    import os
    files, headers, dims = {}, {}, {}
    for nx in sorted(resolutions):
        if restricted_on is None:
            Nx, Ny, _ = grid_from_nx(nx)
            fp = os.path.join(data_dir, f"samples_{Nx}.npy")
        else:
            Nx, Ny, _ = grid_from_nx(restricted_on)
            fp = os.path.join(data_dir, f"samples_{nx}_on{restricted_on}.npy")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"missing {fp}")
        files[nx] = fp
        headers[nx] = npy_header_offset(fp)
        dims[nx] = (Nx, Ny)
    return files, headers, dims
