# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Cluster benchmark: run on a GPU node.

Three checks, printed as a report:

(1) PAPER REPRODUCTION (bit-exact): library GPU + rng=legacy-cupy vs a slice
    of the published dataset. Point --paper-npy at samples_500_part0.npy (or
    the merged file); it recomputes samples and diffs the stored frames.
    Expected: max |diff| == 0.0 (same GPU model/CuPy version; across CUDA/
    CuPy versions ULP differences may appear -- the report will show them).

(2) GPU vs CPU cross-backend agreement: identical initial data (portable
    RNG), reports max |diff| growth per checkpoint. Expected: ~1e-13
    relative at early times, growing with t (chaotic amplification).
    Bit-equality at t=0.0035 is NOT expected and NOT claimed.

(3) Throughput: steps/s per grid on GPU vs the H200 production numbers
    (Nx=500 ~866 s, Nx=1000 ~1807 s, Nx=2000 ~3995 s, Nx=4000 ~16976 s
    per 50-sample node) and vs this machine's CPU.
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from acropora import (Euler2D, VectorialD2Q5, Mach2000JetBC, run_sample,
                     get_xp, grid_from_nx)
from acropora.io import npy_header_offset, read_frame


def build(nx, seed, backend, rng):
    xp = get_xp(backend)
    Nx, Ny, dx = grid_from_nx(nx)
    bc = Mach2000JetBC.make('ghost', Ny, seed=seed, rng_mode=rng)
    s = VectorialD2Q5(Euler2D(5/3), Nx, Ny, dx, xp=xp, boundary=bc)
    bc.initial_condition(s)
    return s


def check_paper(paper_npy, nx, n_samples, T, seed_base):
    Nx, Ny, _ = grid_from_nx(nx)
    tt = np.linspace(0, 0.0035, T)
    off = npy_header_offset(paper_npy)
    print(f"\n(1) paper reproduction, {paper_npy}, first {n_samples} samples")
    for s_id in range(n_samples):
        solver = build(nx, seed_base + s_id, "gpu", "legacy-cupy")
        frames = {}
        run_sample(solver, tt, lambda ck, fr: frames.__setitem__(ck, fr))
        worst = 0.0
        for ck in range(T):
            stored = np.array(read_frame(paper_npy, off, s_id, ck, T, Nx, Ny))
            worst = max(worst, float(np.abs(stored - frames[ck]).max()))
        print(f"  sample {s_id}: max|diff| over {T} checkpoints = {worst:.3e}"
              + ("  BIT-EXACT" if worst == 0.0 else ""))


def check_cross(nx, seed, T):
    tt = np.linspace(0, 0.0035, T)
    print(f"\n(2) GPU vs CPU (portable RNG), Nx={nx}")
    out = {}
    for backend in ("cpu", "gpu"):
        solver = build(nx, seed, backend, "portable")
        fr = {}
        run_sample(solver, tt, lambda ck, a: fr.__setitem__(ck, a))
        out[backend] = fr
    for ck in range(T):
        d = np.abs(out["cpu"][ck] - out["gpu"][ck])
        r = d.max() / max(np.abs(out["gpu"][ck]).max(), 1e-300)
        print(f"  ck{ck} t={tt[ck]:.4f}: max|diff|={d.max():.3e} rel={r:.3e}")


def check_speed(nx, n_steps):
    print(f"\n(3) throughput, Nx={nx}, {n_steps} steps")
    for backend in ("gpu", "cpu"):
        try:
            s = build(nx, 42, backend, "portable")
            if backend == "gpu":
                import cupy; s.step(); cupy.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_steps):
                s.step()
            if backend == "gpu":
                import cupy; cupy.cuda.Stream.null.synchronize()
            dtw = time.perf_counter() - t0
            print(f"  {backend}: {n_steps/dtw:8.2f} steps/s  ({dtw:.2f}s)")
        except ImportError as e:
            print(f"  {backend}: skipped ({e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-npy", default=None,
                    help="path to samples_500_part0.npy for check (1)")
    ap.add_argument("--paper-nx", type=int, default=500)
    ap.add_argument("--paper-samples", type=int, default=3)
    ap.add_argument("--num-checkpoints", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--cross-nx", type=int, default=250)
    ap.add_argument("--speed-nx", type=int, default=1000)
    ap.add_argument("--speed-steps", type=int, default=200)
    a = ap.parse_args()
    if a.paper_npy:
        check_paper(a.paper_npy, a.paper_nx, a.paper_samples,
                    a.num_checkpoints, a.seed_base)
    check_cross(a.cross_nx, a.seed_base, a.num_checkpoints)
    check_speed(a.speed_nx, a.speed_steps)
