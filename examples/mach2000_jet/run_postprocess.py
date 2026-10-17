# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

#!/usr/bin/env python3
"""Mach 2000 jet: post-processing of the generated ensemble.

Stages (run in order; each is optional):
  --merge        merge per-node part files into samples_{Nx}.npy
  --convergence  convergence study (W1, strong L1, mean/std, and
                 with --marginal 2 the two-point W2 via POT)
  --centerline   centerline percentile bands (+ optional one-point PDF grids)
  --heatmap      rasterize the PDF grids to PNG and emit the pgfplots code
  --check-bands  quantile-band diagnostics for the centerline results

Resolutions are x-resolutions Nx throughout, matching the data files
(samples_500.npy ...); outputs are named nx500/nx1000/nx2000/nx4000.

For example, do three convergence invocations, all with
the default --restriction coarsest (every field block-averaged to the 500x400
grid), first one includes W1, strong, moments:
  1) --convergence --protocols both --marginal 1
  2) --convergence --protocols reference --marginal 2 --n-per-dir 10
  3) --convergence --protocols cauchy    --marginal 2 --n-per-dir 10
with samples=1000, resolutions 500 1000 2000 4000, 8 checkpoints,
max_t=0.0035.
Passing --restriction pairwise does Cauchy downsampling (to lower resolution).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))
from acropora import grid_from_nx, node_slice  # noqa: E402
from acropora.post import (merge_parts, run_convergence, scan_healthy,  # noqa: E402
                          open_dataset, centerline_stats, centerline_pdf_grid,
                          write_centerline_dat, write_pdf_grid_dat,
                          coarsen_pdf_grid, band_diagnostics,
                          format_diagnostics, render_pdf_png, global_scale,
                          panel_tex, colorbar_tex)
from acropora.io.npy_store import npy_header_offset  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--resolutions", type=int, nargs="+",
                    default=[500, 1000, 2000, 4000],
                    help="x-resolutions Nx, matching the data files")
    ap.add_argument("--num-checkpoints", type=int, default=8)
    ap.add_argument("--max-t", type=float, default=0.0035)
    ap.add_argument("--total-nodes", type=int, default=20)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="stats/out")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--allow-gaps", action="store_true",
                    help="zero-fill missing parts instead of failing")
    ap.add_argument("--convergence", action="store_true")
    ap.add_argument("--protocols", choices=["cauchy", "reference", "both"],
                    default="both")
    ap.add_argument("--marginal", choices=["1", "2", "both", "moments"],
                    default="1")
    ap.add_argument("--n-per-dir", type=int, default=10)
    ap.add_argument("--fixedM", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--restriction", choices=["coarsest", "pairwise"],
                    default="coarsest",
                    help="coarsest (default, published methodology): every "
                         "field averaged to the coarsest grid, so all "
                         "comparisons are evaluated there. pairwise: only the "
                         "finer member is averaged, onto the coarser member's "
                         "native grid.")
    ap.add_argument("--restricted-on", type=int, default=None,
                    help="read a pre-restricted dataset produced by "
                         "restrict_dataset.py, whose fields already live on "
                         "this grid (restriction becomes a no-op)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--centerline", action="store_true")
    ap.add_argument("--centerline-out", default="centerline")
    ap.add_argument("--sample-index", type=int, default=0,
                    help="which realization to export alongside the bands")
    ap.add_argument("--pdf-grid", action="store_true",
                    help="also write the per-x one-point density grids")
    ap.add_argument("--pdf-nbins", type=int, default=80)
    ap.add_argument("--pdf-xslices", type=int, default=60,
                    help="streamwise stations in the coarse TikZ grid")
    ap.add_argument("--pdf-rho-min", type=float, default=0.0,
                    help="lower edge of the shared density axis")
    ap.add_argument("--pdf-rho-max", type=float, default=None,
                    help="upper edge of the shared density axis (default: "
                         "ensemble max over all checkpoints t > 0)")
    ap.add_argument("--heatmap", action="store_true",
                    help="rasterize PDF grids to PNG + emit pgfplots code")
    ap.add_argument("--heatmap-tags", nargs="+",
                    default=["t0p0015", "t0p0020", "t0p0025",
                             "t0p0030", "t0p0035"])
    ap.add_argument("--heatmap-nx", type=int, default=4000,
                    help="which grid (Nx) to render panels for")
    ap.add_argument("--check-bands", action="store_true",
                    help="quantile-band diagnostics on the centerline .dat")
    ap.add_argument("--head-region", type=float, nargs=2, default=[1.4, 2.3])
    args = ap.parse_args()

    if args.merge:
        for nx in args.resolutions:
            Nx, Ny, _ = grid_from_nx(nx)
            sizes = [node_slice(args.samples, args.total_nodes, i)
                     for i in range(args.total_nodes)]
            parts_sizes = [b - a for a, b in sizes]
            parts = [os.path.join(args.data_dir, f"samples_{Nx}_part{i}.npy")
                     for i in range(args.total_nodes)]
            out = os.path.join(args.data_dir, f"samples_{Nx}.npy")
            print(f"[merge] {out}")
            merge_parts(out, parts, args.samples, args.num_checkpoints, Nx, Ny,
                        parts_sizes=parts_sizes, allow_gaps=args.allow_gaps)

    if args.convergence:
        run_convergence(data_dir=args.data_dir, out_dir=args.out_dir,
                        samples=args.samples, resolutions=args.resolutions,
                        num_checkpoints=args.num_checkpoints, max_t=args.max_t,
                        fixedM=args.fixedM, draws=args.draws,
                        protocols=args.protocols, marginal=args.marginal,
                        n_per_dir=args.n_per_dir,
                        restriction=args.restriction,
                        restricted_on=args.restricted_on, seed=args.seed)

    if args.centerline:
        os.makedirs(args.centerline_out, exist_ok=True)
        target_times = np.linspace(0, args.max_t, args.num_checkpoints)
        files, headers, dims = open_dataset(args.data_dir, args.resolutions,
                                            args.samples, args.num_checkpoints)
        healthy = scan_healthy(files, headers, dims, args.samples,
                               args.num_checkpoints)
        for nx in args.resolutions:
            Nx, Ny = dims[nx]
            _, _, dx = grid_from_nx(nx)
            rows = slice(Ny // 2, Ny // 2 + 1)
            x = (np.arange(Nx) + 0.5) * dx
            stats = {}
            for ck in range(args.num_checkpoints):
                stats[ck] = centerline_stats(
                    files[nx], headers[nx], healthy, ck,
                    args.num_checkpoints, Nx, Ny, rows, return_stack=True)
            rho_lo = args.pdf_rho_min
            rho_hi = (args.pdf_rho_max if args.pdf_rho_max is not None else
                      max(float(np.max(stats[ck][7]))
                          for ck in range(1, args.num_checkpoints)))
            for ck in range(args.num_checkpoints):
                mean, std, n, p16, p84, p05, p95, stack = stats.pop(ck)
                samp = stack[list(healthy).index(args.sample_index)] \
                    if args.sample_index in healthy else stack[0]
                tlab = f"{target_times[ck]:.4f}".replace(".", "p")
                write_centerline_dat(
                    os.path.join(args.centerline_out,
                                 f"centerline_nx{Nx}_ck{ck}_t{tlab}.dat"),
                    x, samp, mean, std, p16, p84, p05, p95,
                    header=(f"Centerline density (y=0) for Nx={Nx}, "
                            f"checkpoint={ck}, t={target_times[ck]:.6f}\n"
                            f"M={len(healthy)} samples for statistics\n"))
                if args.pdf_grid and ck > 0:
                    rc, counts, pdf = centerline_pdf_grid(
                        stack, args.pdf_nbins, rho_min=rho_lo, rho_max=rho_hi)
                    write_pdf_grid_dat(
                        os.path.join(args.centerline_out,
                                     f"centerline_pdf_nx{Nx}_ck{ck}_t{tlab}.dat"),
                        x, rc, pdf, counts,
                        header=(f"Per-x centerline density grid, Nx={Nx}, "
                                f"ck={ck}, t={target_times[ck]:.6f}, "
                                f"M={len(healthy)}, nbins={args.pdf_nbins}"))
                    xc, pdfc, cntc = coarsen_pdf_grid(x, pdf, counts,
                                                      args.pdf_xslices)
                    write_pdf_grid_dat(
                        os.path.join(args.centerline_out,
                                     f"centerline_pdf_coarse_nx{Nx}_ck{ck}"
                                     f"_t{tlab}.dat"),
                        xc, rc, pdfc, cntc,
                        header=(f"Coarse ({len(xc)} stations) density grid "
                                f"for pgfplots, Nx={Nx}, ck={ck}"))
            print(f"[centerline] Nx {Nx}: {args.num_checkpoints} checkpoints, "
                  f"M={len(healthy)}")

    if args.heatmap:
        _heatmap_stage(args)

    if args.check_bands:
        _check_bands_stage(args)


def _heatmap_stage(args):
    import glob
    import re
    pat = os.path.join(args.centerline_out,
                       f"centerline_pdf_nx{args.heatmap_nx}_ck*_t*.dat")
    srcs = sorted(glob.glob(pat))
    if not srcs:
        print(f"[heatmap] no grids matching {pat}", file=sys.stderr)
        return
    vmin, vmax = global_scale(srcs)
    print(f"[heatmap] shared log scale: [{vmin}, {vmax}] over {len(srcs)} grids",
          file=sys.stderr)
    panels = {}
    for src in srcs:
        m = re.search(r"nx(\d+)_ck(\d+)_(t[\dp]+)\.dat$", os.path.basename(src))
        nx_lab, ck, tag = int(m.group(1)), int(m.group(2)), m.group(3)
        dst = os.path.join(args.centerline_out,
                           f"pdf_nx{nx_lab}_ck{ck}_{tag}.png")
        extents, pix, cells, warn = render_pdf_png(src, dst, vmin, vmax)
        for wmsg in warn:
            print(f"  note: {wmsg}", file=sys.stderr)
        print(f"  {os.path.basename(dst)}: {cells[0]} x {cells[1]} cells -> "
              f"{pix[0]} x {pix[1]} px", file=sys.stderr)
        panels[tag] = (dst, extents)
    blocks = []
    for tag in args.heatmap_tags:
        if tag not in panels:
            print(f"% no data for {tag} -- panel skipped")
            continue
        dst, extents = panels[tag]
        cap = r"\(t=" + tag[1:].replace("p", ".") + r"\)"
        blocks.append(panel_tex(os.path.basename(dst), extents, vmin, vmax, cap))
    print("\n% ---- paste into your figure environment ----\n")
    print("\n    \\\\\n".join(blocks))
    print("    \\\\")
    print(colorbar_tex(vmin, vmax))


def _check_bands_stage(args):
    import glob
    pat = os.path.join(args.centerline_out, "centerline_nx*_ck*_t*.dat")
    files = sorted(glob.glob(pat))
    if not files:
        print(f"[check-bands] no files matching {pat}", file=sys.stderr)
        return
    for f in files:
        print(format_diagnostics(
            band_diagnostics(f, head_region=tuple(args.head_region))))
        print()


if __name__ == "__main__":
    main()
