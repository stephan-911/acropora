# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Convergence study driver: the quantities behind the paper's tables.

For each protocol (cauchy/reference), checkpoint, and subsample draw, computes
over the resolution steps: the one-point per-cell Wasserstein distance W1, the
per-sample strong L1 error, the ensemble mean/std L1 errors, and optionally
the two-point Wasserstein distance W2; then the least-squares log-log rate of
each observable against dx. CSV schemas match the production pipeline, so
existing table/TikZ tooling consumes the output unchanged.

Sample selection uses one numpy default_rng(seed); each (protocol, draw)
consumes one permutation of the healthy indices, in protocol-major order,
which fixes the subsampling sequence exactly.
"""
from __future__ import annotations
import csv
import gc
import os
import warnings

import numpy as np

from .metrics import w1_percell, w2_twopoint, strong_l1, moment_l1
from .protocols import steps_for, fit_slope, scan_healthy, load_stack, open_dataset
from .restrict import coarsen_to


def run_convergence(*, data_dir, out_dir, samples=1000,
                    resolutions=(500, 1000, 2000, 4000), num_checkpoints=8,
                    max_t=0.0035, fixedM=1000, draws=1, protocols="both",
                    marginal="1", n_per_dir=10, no_moments=False,
                    restriction="coarsest",
                    restricted_on=None, seed=12345, log=print):
    """Convergence study over both comparison protocols.

    restriction='coarsest' (default, and the published methodology): every
    field of every comparison is block-averaged to the coarsest grid in
    `resolutions`, so all comparisons are evaluated on that single grid.
    restriction='pairwise': the coarser member keeps its native grid and only
    the finer member is block-averaged down to it, so comparisons happen on
    200/400/800 respectively.

    Resolutions are x-resolutions Nx, matching the data files
    (samples_500.npy, ...); the transverse count and mesh width follow from
    the domain. The compare_nx column records the grid each comparison was
    evaluated on.

    restricted_on=<Nx> reads a pre-restricted dataset (restrict_dataset.py),
    whose fields already live on that grid; the restriction step is then a
    no-op and `restriction` is ignored. The fit abscissa dx_fit is the mesh
    width 2.5/coarse_nx of the coarser member of each pair.
    """
    os.makedirs(out_dir, exist_ok=True)
    target_times = np.linspace(0, max_t, num_checkpoints)
    all_ckpts = list(range(num_checkpoints))
    rng = np.random.default_rng(seed)
    nx_list = sorted(resolutions)
    proto_list = ["cauchy", "reference"] if protocols == "both" else [protocols]

    files, headers, dims = open_dataset(data_dir, nx_list, samples,
                                        num_checkpoints, restricted_on)
    log("Scanning finite/written samples...")
    healthy = scan_healthy(files, headers, dims, samples, num_checkpoints)
    log(f"  usable samples: {len(healthy)} of {samples}")

    do_w1 = marginal in ("1", "both")
    do_w2 = marginal in ("2", "both")
    do_mom = (marginal == "moments") or (marginal in ("1", "both") and not no_moments)

    w1_rows, strong_rows, w2_rows, mom_rows, slope_rows = [], [], [], [], []

    for proto in proto_list:
        steps = steps_for(proto, nx_list)
        for draw in range(draws):
            master = rng.permutation(healthy)
            for ck in all_ckpts:
                dxs, w1s, sts, w2s, mes, ses = [], [], [], [], [], []
                for (cr, fr, dx) in steps:
                    Mc = min(fixedM, len(healthy))
                    Mf = min(fixedM, len(healthy))
                    idx_c = master[:Mc]; idx_f = master[:Mf]
                    Nx_c, Ny_c = dims[cr]; Nx_f, Ny_f = dims[fr]
                    c_raw = load_stack(files[cr], headers[cr], idx_c, ck,
                                       num_checkpoints, Nx_c, Ny_c)
                    f_raw = load_stack(files[fr], headers[fr], idx_f, ck,
                                       num_checkpoints, Nx_f, Ny_f)
                    if restricted_on is not None:
                        cgrid = restricted_on
                        c_cg = c_raw          # already on the common grid
                        f_cg = f_raw
                    elif restriction == "pairwise":
                        cgrid = cr
                        c_cg = c_raw
                        f_cg = coarsen_to(f_raw, fr, cr)
                    elif restriction == "coarsest":
                        cgrid = nx_list[0]
                        c_cg = coarsen_to(c_raw, cr, cgrid)
                        f_cg = coarsen_to(f_raw, fr, cgrid)
                    else:
                        raise ValueError(f"unknown restriction {restriction!r}")
                    dxs.append(dx)
                    if do_w1:
                        w1 = w1_percell(c_cg, f_cg)
                        Mo = min(Mc, Mf)
                        err = strong_l1(c_cg[:Mo], f_cg[:Mo])
                        w1s.append(w1); sts.append(err)
                        w1_rows.append(dict(proto=proto, ck=ck, draw=draw, cr=cr,
                                            fr=fr, dx=dx, Mc=Mc, Mf=Mf,
                                            cg=cgrid, w1=w1))
                        strong_rows.append(dict(proto=proto, ck=ck, draw=draw, cr=cr,
                                                fr=fr, dx=dx, Mo=Mo,
                                                cg=cgrid, err=err))
                    else:
                        w1s.append(np.nan); sts.append(np.nan)
                    if do_w2:
                        w2 = w2_twopoint(c_cg, f_cg, n_per_dir)
                        w2s.append(w2)
                        w2_rows.append(dict(proto=proto, ck=ck, draw=draw, cr=cr,
                                            fr=fr, dx=dx, Mc=Mc, Mf=Mf,
                                            cg=cgrid, w2=w2))
                    else:
                        w2s.append(np.nan)
                    if do_mom:
                        me, se = moment_l1(c_cg, f_cg)
                        mes.append(me); ses.append(se)
                        mom_rows.append(dict(proto=proto, ck=ck, draw=draw, cr=cr,
                                             fr=fr, dx=dx, Mc=Mc, Mf=Mf,
                                             cg=cgrid, me=me, se=se))
                    else:
                        mes.append(np.nan); ses.append(np.nan)
                    del c_raw, f_raw, c_cg, f_cg
                    gc.collect()
                slope_rows.append(dict(proto=proto, ck=ck, draw=draw,
                                       w1s=fit_slope(dxs, w1s),
                                       sts=fit_slope(dxs, sts),
                                       w2s=fit_slope(dxs, w2s),
                                       mes=fit_slope(dxs, mes),
                                       ses=fit_slope(dxs, ses)))
            log(f"  {proto} draw {draw+1}/{draws} done")
        write_csvs(out_dir, target_times, n_per_dir,
                   w1_rows, strong_rows, w2_rows, mom_rows, slope_rows)

    write_csvs(out_dir, target_times, n_per_dir,
               w1_rows, strong_rows, w2_rows, mom_rows, slope_rows)
    return dict(w1=w1_rows, strong=strong_rows, w2=w2_rows,
                moments=mom_rows, slopes=slope_rows, healthy=healthy)


def write_csvs(out_dir, target_times, n_per_dir,
               w1_rows, strong_rows, w2_rows, mom_rows, slope_rows):
    warnings.filterwarnings("ignore", category=RuntimeWarning,
                            message="Mean of empty slice")
    warnings.filterwarnings("ignore", category=RuntimeWarning,
                            message="Degrees of freedom <= 0 for slice.")
    with open(os.path.join(out_dir, "w1_values.csv"), "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["protocol", "checkpoint_idx", "t", "draw", "coarse_nx",
                       "fine_nx", "dx_fit", "M_coarse", "M_fine", "compare_nx", "W1"])
        for r in w1_rows:
            wcsv.writerow([r["proto"], r["ck"], f"{target_times[r['ck']]:.6f}", r["draw"],
                           r["cr"], r["fr"], f"{r['dx']:.6e}",
                           r["Mc"], r["Mf"], r["cg"], f"{r['w1']:.6e}"])
    with open(os.path.join(out_dir, "strong_values.csv"), "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["protocol", "checkpoint_idx", "t", "draw", "coarse_nx",
                       "fine_nx", "dx_fit", "M_overlap", "compare_nx", "strong_L1"])
        for r in strong_rows:
            wcsv.writerow([r["proto"], r["ck"], f"{target_times[r['ck']]:.6f}", r["draw"],
                           r["cr"], r["fr"], f"{r['dx']:.6e}",
                           r["Mo"], r["cg"], f"{r['err']:.6e}"])
    with open(os.path.join(out_dir, "w2_values.csv"), "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["protocol", "checkpoint_idx", "t", "draw", "coarse_nx",
                       "fine_nx", "dx_fit", "M_coarse", "M_fine", "compare_nx",
                       "n_per_dir", "W2"])
        for r in w2_rows:
            wcsv.writerow([r["proto"], r["ck"], f"{target_times[r['ck']]:.6f}", r["draw"],
                           r["cr"], r["fr"], f"{r['dx']:.6e}",
                           r["Mc"], r["Mf"], r["cg"], n_per_dir,
                           f"{r['w2']:.6e}"])
    with open(os.path.join(out_dir, "moments_values.csv"), "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["protocol", "checkpoint_idx", "t", "draw", "coarse_nx",
                       "fine_nx", "dx_fit", "M_coarse", "M_fine", "compare_nx",
                       "mean_L1", "std_L1"])
        for r in mom_rows:
            wcsv.writerow([r["proto"], r["ck"], f"{target_times[r['ck']]:.6f}", r["draw"],
                           r["cr"], r["fr"], f"{r['dx']:.6e}",
                           r["Mc"], r["Mf"], r["cg"], f"{r['me']:.6e}",
                           f"{r['se']:.6e}"])
    with open(os.path.join(out_dir, "slopes.csv"), "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["protocol", "checkpoint_idx", "t", "draw",
                       "W1_slope", "strong_slope", "W2_slope",
                       "mean_slope", "std_slope"])
        for r in slope_rows:
            wcsv.writerow([r["proto"], r["ck"], f"{target_times[r['ck']]:.6f}", r["draw"],
                           f"{r['w1s']:.4f}", f"{r['sts']:.4f}", f"{r['w2s']:.4f}",
                           f"{r['mes']:.4f}", f"{r['ses']:.4f}"])
        wcsv.writerow([]); wcsv.writerow(["# SUMMARY (mean/std over draws)"])
        wcsv.writerow(["protocol", "checkpoint_idx", "t",
                       "W1_slope_mean", "W1_slope_std",
                       "strong_slope_mean", "strong_slope_std",
                       "W2_slope_mean", "W2_slope_std",
                       "mean_slope_mean", "mean_slope_std",
                       "std_slope_mean", "std_slope_std", "n_draws"])
        keys = sorted({(r["proto"], r["ck"]) for r in slope_rows})
        for (p, ck) in keys:
            sel = [r for r in slope_rows if r["proto"] == p and r["ck"] == ck]
            w1v = [r["w1s"] for r in sel]; stv = [r["sts"] for r in sel]
            w2v = [r["w2s"] for r in sel]; mev = [r["mes"] for r in sel]
            sev = [r["ses"] for r in sel]
            wcsv.writerow([p, ck, f"{target_times[ck]:.6f}",
                           f"{np.nanmean(w1v):.4f}", f"{np.nanstd(w1v):.4f}",
                           f"{np.nanmean(stv):.4f}", f"{np.nanstd(stv):.4f}",
                           f"{np.nanmean(w2v):.4f}", f"{np.nanstd(w2v):.4f}",
                           f"{np.nanmean(mev):.4f}", f"{np.nanstd(mev):.4f}",
                           f"{np.nanmean(sev):.4f}", f"{np.nanstd(sev):.4f}", len(w1v)])
