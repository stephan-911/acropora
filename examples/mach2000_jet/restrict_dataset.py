# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

#!/usr/bin/env python3
"""Block-average a full-scale ensemble onto the coarsest grid, once.

The published convergence study evaluates every comparison on the coarsest
grid (500 x 400 for the jet ladder), so each frame of every finer dataset is
block-averaged on the fly, over and over, in every run. Doing the restriction
once and storing the result turns a ~1 TB analysis into a ~51 GB one and makes
repeated studies cheap; the coarsest dataset is copied through unchanged so
the target folder holds the complete, self-contained ensemble.

Frames are streamed one at a time, so peak memory stays at one fine frame
(~100 MB at Nx=4000) regardless of ensemble size. The run is resumable: an
existing output whose final checkpoint is written is skipped.

Output files are named samples_{Nx}_on{Nx_target}.npy, preserving which
resolution each ensemble came from while recording the grid it now lives on;
a manifest records the provenance. Feed the folder to run_postprocess.py with
--restricted-on {Nx_target}.

Example:
    python3 restrict_dataset.py \
        --data-dir /work/data --out-dir /work/data_on500 \
        --resolutions 500 1000 2000 4000 --target 500 \
        --samples 1000 --num-checkpoints 8
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))
from acropora import grid_from_nx  # noqa: E402
from acropora.io.npy_store import (create_store, open_store, npy_header_offset,  # noqa: E402
                                 read_frame, write_manifest)
from acropora.post import coarsen_to  # noqa: E402


def restrict_one(src_file, dst_file, nx_src, nx_tgt, samples, T, *,
                 resume=True, log=print):
    """Stream every (sample, checkpoint) frame from src to dst, block-
    averaging from nx_src onto nx_tgt."""
    Nx_s, Ny_s, _ = grid_from_nx(nx_src)
    Nx_t, Ny_t, _ = grid_from_nx(nx_tgt)
    if nx_src % nx_tgt or Ny_s % Ny_t:
        raise ValueError(f"{nx_src} is not an integer refinement of {nx_tgt}")

    src_off = npy_header_offset(src_file)
    if os.path.exists(dst_file) and resume:
        dst = open_store(dst_file, samples, T, Nx_t, Ny_t)
    else:
        dst = create_store(dst_file, samples, T, Nx_t, Ny_t)

    t0 = time.time()
    done = 0
    for s in range(samples):
        if resume and np.any(np.asarray(dst[s, T - 1]) != 0.0):
            done += 1
            continue
        for ck in range(T):
            frame = np.asarray(read_frame(src_file, src_off, s, ck, T,
                                          Nx_s, Ny_s))
            dst[s, ck] = coarsen_to(frame[None, ...], nx_src, nx_tgt)[0]
        done += 1
        if done % 25 == 0 or done == samples:
            el = time.time() - t0
            log(f"    {done}/{samples} samples  ({el:6.1f} s elapsed)")
    dst.flush()
    del dst
    return done


def verify(src_file, dst_file, nx_src, nx_tgt, samples, T, n_checks, rng,
           log=print):
    """Recompute a few random frames straight from the source and compare."""
    Nx_s, Ny_s, _ = grid_from_nx(nx_src)
    Nx_t, Ny_t, _ = grid_from_nx(nx_tgt)
    src_off = npy_header_offset(src_file)
    dst_off = npy_header_offset(dst_file)
    worst = 0.0
    for _ in range(n_checks):
        s = int(rng.integers(samples))
        ck = int(rng.integers(T))
        frame = np.asarray(read_frame(src_file, src_off, s, ck, T, Nx_s, Ny_s))
        ref = coarsen_to(frame[None, ...], nx_src, nx_tgt)[0]
        got = np.asarray(read_frame(dst_file, dst_off, s, ck, T, Nx_t, Ny_t))
        worst = max(worst, float(np.abs(ref - got).max()))
    log(f"    verification over {n_checks} random frames: "
        f"max |diff| = {worst:.3e}" + ("  EXACT" if worst == 0.0 else ""))
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True,
                    help="folder holding the merged samples_{Nx}.npy files")
    ap.add_argument("--out-dir", required=True,
                    help="destination folder for the restricted ensembles")
    ap.add_argument("--resolutions", type=int, nargs="+",
                    default=[500, 1000, 2000, 4000],
                    help="x-resolutions to restrict (the target is copied)")
    ap.add_argument("--target", type=int, default=None,
                    help="target x-resolution (default: coarsest given)")
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--num-checkpoints", type=int, default=8)
    ap.add_argument("--max-t", type=float, default=0.0035)
    ap.add_argument("--no-resume", action="store_true",
                    help="recompute samples even if already written")
    ap.add_argument("--verify", type=int, default=5,
                    help="random frames to re-check per resolution (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    nx_list = sorted(args.resolutions)
    target = args.target if args.target is not None else nx_list[0]
    Nx_t, Ny_t, dx_t = grid_from_nx(target)
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    frame_mb = Nx_t * Ny_t * 8 / 1e6
    total_gb = len(nx_list) * args.samples * args.num_checkpoints * frame_mb / 1e3
    print(f"target grid: {Nx_t} x {Ny_t} (dx = {dx_t:g})")
    print(f"output size: {len(nx_list)} x {args.samples} x "
          f"{args.num_checkpoints} frames x {frame_mb:.2f} MB "
          f"= {total_gb:.1f} GB\n")

    written = {}
    for nx in nx_list:
        Nx_s, Ny_s, _ = grid_from_nx(nx)
        src = os.path.join(args.data_dir, f"samples_{Nx_s}.npy")
        dst = os.path.join(args.out_dir, f"samples_{nx}_on{target}.npy")
        if not os.path.exists(src):
            raise FileNotFoundError(f"missing {src}")
        factor = nx // target
        print(f"[{Nx_s} x {Ny_s}] -> [{Nx_t} x {Ny_t}]  "
              f"(block {factor} x {Ny_s // Ny_t})"
              + ("  [copy, already on target grid]" if nx == target else ""))
        n = restrict_one(src, dst, nx, target, args.samples,
                         args.num_checkpoints, resume=not args.no_resume,
                         log=lambda m: print(m, flush=True))
        if args.verify:
            verify(src, dst, nx, target, args.samples, args.num_checkpoints,
                   args.verify, rng, log=lambda m: print(m, flush=True))
        written[nx] = dict(file=os.path.basename(dst), samples=n,
                           source=os.path.basename(src),
                           source_grid=[Nx_s, Ny_s])
        print(f"    wrote {dst}\n", flush=True)

    write_manifest(os.path.join(args.out_dir, "restriction_manifest.json"),
                   target_nx=target, target_grid=[Nx_t, Ny_t],
                   target_dx=dx_t, resolutions=nx_list,
                   samples=args.samples, num_checkpoints=args.num_checkpoints,
                   target_times=list(map(float, np.linspace(0, args.max_t,
                                                            args.num_checkpoints))),
                   operator="volume (block) mean",
                   datasets=written,
                   created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    print("done. Analyze with:")
    print(f"  python3 run_postprocess.py --convergence "
          f"--data-dir {args.out_dir} --restricted-on {target} \\")
    print(f"      --resolutions {' '.join(map(str, nx_list))} "
          f"--samples {args.samples} --num-checkpoints {args.num_checkpoints}")


if __name__ == "__main__":
    main()
