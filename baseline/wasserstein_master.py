#!/usr/bin/env python3
# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Convergence of the statistical solution under two comparison protocols.

Reference implementation of the published analysis, kept as a standalone
script so that the library can be regression-tested against it.

Analysis choices
  * full domain, all checkpoints including t = 0
  * fixed sample count M at every resolution
  * restriction to a common grid: every field of every comparison is
    block-averaged (volume mean) to the coarsest resolution, so all
    comparisons are evaluated on that single grid. Passing
    --restriction pairwise instead leaves the coarser member on its native
    grid and averages only the finer member down to it.

Protocols
  * cauchy    : consecutive resolution pairs (Nx_i, Nx_{i+1})
  * reference : every coarser resolution against the finest

Quantities, per protocol, checkpoint and resolution step
  * W1        one-point per-cell Wasserstein distance, averaged over cells
  * strong    per-sample relative L1 error, averaged over samples
  * mean/std  L1 errors of the ensemble mean and population standard deviation
  * W2        two-point Wasserstein distance (exact EMD, --marginal 2/both)
and the least-squares log-log slope of each against dx. The slope excludes
t = 0, where the fields coincide and the distances vanish.

Fit abscissa: dx = 2.5/Nx, the mesh width of the coarser member. A constant
rescaling of the abscissa leaves the log-log slopes unchanged.

Resolutions are x-resolutions Nx, matching the data files
data/samples_{Nx}.npy of shape (S, T, Nx, Ny) with Ny = 0.8*Nx.

Outputs (long format)
  * w1_values.csv, strong_values.csv, w2_values.csv, moments_values.csv
  * slopes.csv : one fitted slope per protocol, checkpoint and observable
  * plots: slope vs time per observable; each observable vs dx at the end time

CPU only; results are written after each protocol completes.
"""
import argparse
import csv
import gc
import os
import time

import numpy as np
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DOMAIN = (2.5, 2.0)          # (length in x, height in y)


# ---------------------------------------------------------------- data access
def grid_from_nx(nx):
    """(Nx, Ny, dx) for x-resolution nx on the jet domain."""
    dx = DOMAIN[0] / nx
    return int(nx), int(round(DOMAIN[1] / dx)), dx


def get_npy_header_offset(filepath):
    with open(filepath, 'rb') as f:
        version = np.lib.format.read_magic(f)
        major, minor = version
        getattr(np.lib.format, f"read_array_header_{major}_{minor}")(f)
        return f.tell()


def read_single_frame(fp, hoff, s, t, T, Nx, Ny):
    if t < 0:
        t = T + t
    fb = Nx * Ny * 8
    off = hoff + (s * T + t) * fb
    return np.memmap(fp, dtype=np.float64, mode='r', offset=off, shape=(Nx, Ny))


def load_stack(fp, hoff, idx, ck, T, Nx, Ny):
    out = np.empty((len(idx), Nx, Ny), dtype=np.float64)
    for k, s in enumerate(idx):
        out[k] = np.asarray(read_single_frame(fp, hoff, s, ck, T, Nx, Ny))
    return out


def coarsen_to(field, nx_src, nx_tgt):
    """Block-average an (M, Nx, Ny) stack from x-resolution nx_src to nx_tgt."""
    if nx_src == nx_tgt:
        return field
    _, Ny_t, _ = grid_from_nx(nx_tgt)
    sx = nx_src // nx_tgt
    sy = grid_from_nx(nx_src)[1] // Ny_t
    return field[:, :nx_tgt * sx, :Ny_t * sy].reshape(
        -1, nx_tgt, sx, Ny_t, sy).mean(axis=(2, 4))


# ------------------------------------------------------------------- measures
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
    between the empirical joints via exact EMD. Cost ~ n_per_dir^4 EMD solves.
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
    """Per-cell-averaged L1 errors of the ensemble mean and population std.

    The standard deviation is reported rather than the variance, because it is
    the quantity plotted; for a difference of two fields the two slopes agree
    up to the fit accuracy.
    """
    ma = a.mean(axis=0); mb = b.mean(axis=0)
    sa = a.std(axis=0);  sb = b.std(axis=0)          # population std (ddof=0)
    ncell = ma.size
    return (float(np.sum(np.abs(ma - mb)) / ncell),
            float(np.sum(np.abs(sa - sb)) / ncell))


def fit_slope(dxs, vals):
    """Least-squares slope of log(val) vs log(dx), skipping nonpositive values."""
    dxs = np.array(dxs); vals = np.array(vals)
    good = (vals > 0) & np.isfinite(vals)
    if good.sum() >= 2:
        return float(np.polyfit(np.log(dxs[good]), np.log(vals[good]), 1)[0])
    return np.nan


def steps_for(protocol, nx_list):
    """Comparison steps (coarse_nx, fine_nx, dx_fit) for one protocol."""
    if protocol == "cauchy":
        return [(nx_list[i], nx_list[i + 1], DOMAIN[0] / nx_list[i])
                for i in range(len(nx_list) - 1)]
    fin = nx_list[-1]
    return [(nx, fin, DOMAIN[0] / nx) for nx in nx_list[:-1]]


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--resolutions", type=int, nargs='+',
                    default=[500, 1000, 2000, 4000],
                    help="x-resolutions Nx, matching the data files")
    ap.add_argument("--num_checkpoints", type=int, default=8)
    ap.add_argument("--max_t", type=float, default=0.0035)
    ap.add_argument("--fixedM", type=int, default=1000,
                    help="sample count used at every resolution")
    ap.add_argument("--restriction", choices=["coarsest", "pairwise"],
                    default="coarsest",
                    help="coarsest: every field averaged to the coarsest grid; "
                         "pairwise: only the finer member is averaged, onto "
                         "the coarser member's native grid")
    ap.add_argument("--protocols", choices=["cauchy", "reference", "both"],
                    default="both")
    ap.add_argument("--marginal", choices=["1", "2", "both", "moments"],
                    default="1",
                    help="1: one-point W1 with strong L1 and mean/std; "
                         "2: two-point W1; both: 1 and 2; "
                         "moments: mean/std only")
    ap.add_argument("--n_per_dir", type=int, default=10,
                    help="spatial points per direction for the two-point marginal")
    ap.add_argument("--no_moments", action="store_true",
                    help="disable the mean/std errors that otherwise ride "
                         "along with marginal 1/both")
    ap.add_argument("--seed", type=int, default=12345,
                    help="seed for the sample selection")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="stats/out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    target_times = np.linspace(0, args.max_t, args.num_checkpoints)
    all_ckpts = list(range(args.num_checkpoints))
    rng = np.random.default_rng(args.seed)
    nx_list = sorted(args.resolutions)
    protocols = (["cauchy", "reference"] if args.protocols == "both"
                 else [args.protocols])

    files, headers, dims = {}, {}, {}
    for nx in nx_list:
        Nx, Ny, _ = grid_from_nx(nx)
        fp = os.path.join(args.data_dir, f"samples_{Nx}.npy")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"missing {fp}")
        files[nx] = fp
        headers[nx] = get_npy_header_offset(fp)
        dims[nx] = (Nx, Ny)

    print("Scanning finite/written samples...", flush=True)
    finite = np.ones(args.samples, dtype=bool)
    for nx in nx_list:
        Nx, Ny = dims[nx]
        for i in range(args.samples):
            a = np.asarray(read_single_frame(files[nx], headers[nx], i,
                                             args.num_checkpoints - 1,
                                             args.num_checkpoints, Nx, Ny))
            if np.all(a == 0.0) or not np.isfinite(a).all():
                finite[i] = False
    healthy = np.where(finite)[0]
    print(f"  usable samples: {len(healthy)} of {args.samples}", flush=True)

    do_w1 = args.marginal in ("1", "both")
    do_w2 = args.marginal in ("2", "both")
    do_mom = (args.marginal == "moments") or \
             (args.marginal in ("1", "both") and not args.no_moments)

    w1_rows, strong_rows, w2_rows, mom_rows, slope_rows = [], [], [], [], []

    def write_outputs(partial=""):
        specs = [
            ("w1_values.csv", w1_rows,
             ["protocol", "checkpoint_idx", "t", "coarse_nx", "fine_nx",
              "dx_fit", "M_coarse", "M_fine", "compare_nx", "W1"],
             lambda r: [r["Mc"], r["Mf"], r["cg"], f"{r['w1']:.6e}"]),
            ("strong_values.csv", strong_rows,
             ["protocol", "checkpoint_idx", "t", "coarse_nx", "fine_nx",
              "dx_fit", "M", "compare_nx", "strong_L1"],
             lambda r: [r["Mo"], r["cg"], f"{r['err']:.6e}"]),
            ("w2_values.csv", w2_rows,
             ["protocol", "checkpoint_idx", "t", "coarse_nx", "fine_nx",
              "dx_fit", "M_coarse", "M_fine", "compare_nx", "n_per_dir", "W2"],
             lambda r: [r["Mc"], r["Mf"], r["cg"], args.n_per_dir,
                        f"{r['w2']:.6e}"]),
            ("moments_values.csv", mom_rows,
             ["protocol", "checkpoint_idx", "t", "coarse_nx", "fine_nx",
              "dx_fit", "M_coarse", "M_fine", "compare_nx",
              "mean_L1", "std_L1"],
             lambda r: [r["Mc"], r["Mf"], r["cg"], f"{r['me']:.6e}",
                        f"{r['se']:.6e}"]),
        ]
        for name, rows, header, tail in specs:
            with open(os.path.join(args.out_dir, name), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                for r in rows:
                    w.writerow([r["proto"], r["ck"],
                                f"{target_times[r['ck']]:.6f}",
                                r["cr"], r["fr"], f"{r['dx']:.6e}"] + tail(r))

        with open(os.path.join(args.out_dir, "slopes.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["protocol", "checkpoint_idx", "t", "W1_slope",
                        "strong_slope", "W2_slope", "mean_slope", "std_slope"])
            for r in slope_rows:
                w.writerow([r["proto"], r["ck"], f"{target_times[r['ck']]:.6f}",
                            f"{r['w1s']:.4f}", f"{r['sts']:.4f}",
                            f"{r['w2s']:.4f}", f"{r['mes']:.4f}",
                            f"{r['ses']:.4f}"])
        print(f"  [checkpoint] wrote w1/strong/w2/moments/slopes CSVs{partial}",
              flush=True)

    t0 = time.time()
    for proto in protocols:
        steps = steps_for(proto, nx_list)
        idx = rng.permutation(healthy)[:min(args.fixedM, len(healthy))]
        for ck in all_ckpts:
            dxs, w1s, sts, w2s, mes, ses = [], [], [], [], [], []
            for (cr, fr, dx) in steps:
                Nx_c, Ny_c = dims[cr]; Nx_f, Ny_f = dims[fr]
                c_raw = load_stack(files[cr], headers[cr], idx, ck,
                                   args.num_checkpoints, Nx_c, Ny_c)
                f_raw = load_stack(files[fr], headers[fr], idx, ck,
                                   args.num_checkpoints, Nx_f, Ny_f)
                if args.restriction == "coarsest":
                    cg = nx_list[0]
                    c_cg = coarsen_to(c_raw, cr, cg)
                    f_cg = coarsen_to(f_raw, fr, cg)
                else:
                    cg = cr
                    c_cg = c_raw
                    f_cg = coarsen_to(f_raw, fr, cr)
                M = len(idx)
                dxs.append(dx)

                if do_w1:
                    w1 = w1_percell(c_cg, f_cg)
                    err = strong_l1(c_cg, f_cg)
                    w1s.append(w1); sts.append(err)
                    w1_rows.append(dict(proto=proto, ck=ck, cr=cr, fr=fr, dx=dx,
                                        Mc=M, Mf=M, cg=cg, w1=w1))
                    strong_rows.append(dict(proto=proto, ck=ck, cr=cr, fr=fr,
                                            dx=dx, Mo=M, cg=cg, err=err))
                else:
                    w1s.append(np.nan); sts.append(np.nan)

                if do_w2:
                    w2 = w2_twopoint(c_cg, f_cg, args.n_per_dir)
                    w2s.append(w2)
                    w2_rows.append(dict(proto=proto, ck=ck, cr=cr, fr=fr, dx=dx,
                                        Mc=M, Mf=M, cg=cg, w2=w2))
                else:
                    w2s.append(np.nan)

                if do_mom:
                    me, se = moment_l1(c_cg, f_cg)
                    mes.append(me); ses.append(se)
                    mom_rows.append(dict(proto=proto, ck=ck, cr=cr, fr=fr, dx=dx,
                                         Mc=M, Mf=M, cg=cg, me=me, se=se))
                else:
                    mes.append(np.nan); ses.append(np.nan)

                del c_raw, f_raw, c_cg, f_cg
                gc.collect()

            slope_rows.append(dict(proto=proto, ck=ck,
                                   w1s=fit_slope(dxs, w1s),
                                   sts=fit_slope(dxs, sts),
                                   w2s=fit_slope(dxs, w2s),
                                   mes=fit_slope(dxs, mes),
                                   ses=fit_slope(dxs, ses)))
        print(f"  {proto} done (elapsed {time.time() - t0:.0f}s)", flush=True)
        write_outputs(partial=f"  (through {proto})")
    write_outputs()

    # ------------------------------------------------------------- plots
    protos_done = sorted({r["proto"] for r in slope_rows})
    plabel = {"cauchy": "cauchy (consecutive)",
              "reference": "reference (vs finest)"}
    for metric, field, lab in [("W1", "w1s", "Wasserstein 1-pt"),
                               ("strong", "sts", "strong L1"),
                               ("W2", "w2s", "Wasserstein 2-pt"),
                               ("mean", "mes", "mean"),
                               ("std", "ses", "standard deviation")]:
        plt.figure(figsize=(10, 6))
        drawn = False
        for p in protos_done:
            ts, ys = [], []
            for ck in all_ckpts:
                v = [r[field] for r in slope_rows
                     if r["proto"] == p and r["ck"] == ck and np.isfinite(r[field])]
                if v:
                    ts.append(target_times[ck]); ys.append(v[0])
            if ts:
                plt.plot(ts, ys, marker='o', label=plabel.get(p, p)); drawn = True
        plt.xlabel("time"); plt.ylabel(f"{lab} convergence slope")
        plt.title(f"{lab} slope vs time ({args.restriction} restriction)")
        plt.grid(True, alpha=0.3)
        if drawn:
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"{metric}_slope_vs_t.png"), dpi=150)
        plt.close()

    ck_last = all_ckpts[-1]
    for rows, name, ylab, key in [(w1_rows, "w1", "per-cell W1", "w1"),
                                  (strong_rows, "strong", "strong L1", "err"),
                                  (w2_rows, "w2", "2-point W1", "w2"),
                                  (mom_rows, "mean", "mean L1 error", "me"),
                                  (mom_rows, "std", "std L1 error", "se")]:
        if not rows:
            continue
        plt.figure(figsize=(8, 6))
        labelled = set()
        for p in protos_done:
            for (cr, fr, dx) in steps_for(p, nx_list):
                v = [r[key] for r in rows if r["proto"] == p
                     and r["ck"] == ck_last and r["cr"] == cr and r["fr"] == fr]
                if v:
                    plt.scatter([dx], [v[0]], s=40,
                                label=plabel.get(p, p) if p not in labelled else None)
                    labelled.add(p)
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel("dx (fit abscissa)"); plt.ylabel(ylab)
        plt.title(f"{ylab} vs dx at t={target_times[ck_last]:.4f}")
        plt.grid(True, which='both', alpha=0.3)
        if labelled:
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"{name}_vs_dx.png"), dpi=150)
        plt.close()

    # --------------------------------------------------- console summary
    print("\n================ PROTOCOL SLOPE SUMMARY ================")
    for p in protos_done:
        head = {"cauchy": "CAUCHY (consecutive pairs)",
                "reference": "REFERENCE (vs finest)"}.get(p, p)
        print(f"\n[{head}]  full domain, {args.restriction} restriction")
        print(f"  {'t':>8} {'W1_slope':>10} {'strong':>10} {'W2_slope':>10}"
              f" {'mean':>10} {'std':>10}")
        for ck in all_ckpts:
            def g(fld):
                v = [r[fld] for r in slope_rows
                     if r["proto"] == p and r["ck"] == ck and np.isfinite(r[fld])]
                return f"{v[0]:.3f}" if v else "--"
            print(f"  {target_times[ck]:8.4f} {g('w1s'):>10} {g('sts'):>10} "
                  f"{g('w2s'):>10} {g('mes'):>10} {g('ses'):>10}")
    print(f"\nM = {min(args.fixedM, len(healthy))} samples at every resolution.")
    if args.restriction == "coarsest":
        print(f"Restriction: every field block-averaged to Nx = {nx_list[0]}; "
              f"all comparisons evaluated on that grid.")
    else:
        print("Restriction: the finer member block-averaged to the coarser "
              "member's native grid.")
    print("The t = 0 row is empty because the fields coincide there.")


if __name__ == "__main__":
    main()
