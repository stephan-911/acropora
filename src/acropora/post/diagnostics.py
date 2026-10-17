# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Diagnostics for the quantile-band statements made about the centerline
figures.

For a window of the jet head this reports
  1. the band-width ratio (p95-p05)/(p84-p16), whose Gaussian value is
     1.645, together with the largest ratios and whether they come from a
     genuinely wide outer band (intermittent fringe) or from two small
     numbers dividing (numerical artifact);
  2. a skewness proxy (p95-mean)/(mean-p05), 1 being symmetric;
  3. whether the plotted single realization locally exceeds p95.

Column layout of the .dat files:
0:x 1:rho_sample 2:rho_mean 3:rho_std 4:p16 5:p84 6:p05 7:p95
"""
from __future__ import annotations
import os

import numpy as np

GAUSS_RATIO = 1.6449   # (p95-p05)/(p84-p16) for a normal distribution
EPS = 1e-12


def band_diagnostics(path, head_region=(1.4, 2.3), wide_threshold=1.0,
                     n_report=5):
    """Return a dict of quantile-band diagnostics for one centerline file."""
    d = np.loadtxt(path)
    x = d[:, 0]
    sample, mean = d[:, 1], d[:, 2]
    p16, p84, p05, p95 = d[:, 4], d[:, 5], d[:, 6], d[:, 7]

    i = slice(*np.searchsorted(x, head_region))
    if i.start >= i.stop:
        return {"path": path, "n_points": 0,
                "note": f"head region {head_region} outside data range"}

    outer = (p95 - p05)[i]
    inner = np.maximum((p84 - p16)[i], EPS)
    ratio = outer / inner
    xs = x[i]

    order = np.argsort(ratio)[::-1][:n_report]
    largest = [dict(x=float(xs[j]), ratio=float(ratio[j]),
                    outer=float(outer[j]), inner=float(inner[j]),
                    genuine=bool(outer[j] > wide_threshold)) for j in order]

    up = (p95 - mean)[i]
    lo = np.maximum((mean - p05)[i], EPS)

    exceed = sample[i] > p95[i]
    exc = None
    if exceed.any():
        j = int(np.argmax((sample - p95)[i]))
        exc = dict(count=int(exceed.sum()), n_points=int(xs.size),
                   x=float(xs[j]), sample=float(sample[i][j]),
                   p95=float(p95[i][j]), mean=float(mean[i][j]))

    return {"path": path, "head_region": head_region,
            "n_points": int(xs.size),
            "ratio_median": float(np.median(ratio)),
            "ratio_max": float(ratio.max()),
            "gauss_ratio": GAUSS_RATIO,
            "heavy_tailed": bool(np.median(ratio) > GAUSS_RATIO),
            "largest_ratios": largest,
            "skew_proxy_median": float(np.median(up / lo)),
            "sample_exceeds_p95": exc}


def format_diagnostics(d):
    """Human-readable report for one band_diagnostics() result."""
    name = os.path.basename(d["path"])      # no directories in the report
    if d.get("n_points", 0) == 0:
        return f"=== {name} ===\n  {d.get('note', 'no data')}"
    L = [f"=== {name} ===",
         f"head region x in [{d['head_region'][0]}, {d['head_region'][1]}], "
         f"{d['n_points']} points",
         "width ratio (p95-p05)/(p84-p16):",
         f"  median = {d['ratio_median']:.4f}   "
         f"(Gaussian: {d['gauss_ratio']:.3f}; heavy tails if well above)",
         f"  max    = {d['ratio_max']:.4g}",
         f"  {'x':>8}  {'ratio':>10}  {'p95-p05':>10}  {'p84-p16':>10}  verdict"]
    for r in d["largest_ratios"]:
        verdict = ("intermittent fringe (inner collapsed, outer wide)"
                   if r["genuine"] else
                   "artifact? outer band also small -- do not quote")
        L.append(f"  {r['x']:8.4f}  {r['ratio']:10.4g}  {r['outer']:10.4g}  "
                 f"{r['inner']:10.4g}  {verdict}")
    L.append(f"skewness proxy median (p95-mean)/(mean-p05) = "
             f"{d['skew_proxy_median']:.3f}   "
             f"(1 = symmetric, >1 = positively skewed)")
    e = d["sample_exceeds_p95"]
    if e:
        L.append(f"single sample exceeds p95 at {e['count']} of "
                 f"{e['n_points']} points; largest excursion at "
                 f"x = {e['x']:.4f}: sample = {e['sample']:.2f}, "
                 f"p95 = {e['p95']:.2f}, mean = {e['mean']:.2f}")
    else:
        L.append("single sample never exceeds p95 in the head region")
    return "\n".join(L)
