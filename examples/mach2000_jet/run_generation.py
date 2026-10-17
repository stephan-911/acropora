# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

#!/usr/bin/env python3
"""Mach 2000 astrophysical jet: Monte Carlo ensemble generation.

Example (single node, CPU):
    python3 run_generation.py --samples 8 --resolutions 125 --backend cpu \
        --total-nodes 1 --node-id 0 --workers 4
On a SLURM array (see slurm/generate_data.sh) each task runs one node id.

Reproducing the published H200 dataset requires --rng legacy-cupy on the GPU
backend (CuPy XORWOW seed stream); the default 'portable' mode yields
identical initial data on CPU and GPU instead.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))
from acropora import (generate_node, node_slice, draw_coefficients,  # noqa: E402
                    get_xp)
from acropora.io import write_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--resolutions", type=int, nargs="+",
                    default=[500, 1000, 2000, 4000],
                    help="x-resolutions Nx; the transverse count and mesh "
                         "width follow from the domain (Nx=500 -> 500x400)")
    ap.add_argument("--max-t", type=float, default=0.0035)
    ap.add_argument("--num-checkpoints", type=int, default=8)
    ap.add_argument("--boundary-mode", choices=["ghost", "sponge"],
                    default="ghost")
    ap.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel sample workers (GPU: one per device)")
    ap.add_argument("--num-devices", type=int, default=4,
                    help="visible GPUs (ignored on cpu)")
    ap.add_argument("--node-id", type=int, default=0)
    ap.add_argument("--total-nodes", type=int, default=1)
    ap.add_argument("--rng", choices=["portable", "legacy-cupy",
                                      "legacy-numpy"], default="portable")
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    if args.rng == "legacy-cupy" and args.backend != "gpu":
        ap.error("--rng legacy-cupy requires --backend gpu "
                 "(it replays the CuPy device RNG stream)")

    target_times = np.linspace(0, args.max_t, args.num_checkpoints)
    os.makedirs(args.data_dir, exist_ok=True)

    start, end = node_slice(args.samples, args.total_nodes, args.node_id)

    # record coefficients and a provenance manifest, so post-processing
    # reads the generation parameters from disk instead of CLI flags
    coeff_file = os.path.join(args.data_dir,
                              f"fourier_coeffs_part{args.node_id}.csv")
    if not os.path.exists(coeff_file):
        with open(coeff_file, "w") as cf:
            cf.write("sample_id,mode_k,Y_k,Z_k\n")
            xp_coeff = get_xp(args.backend) if args.rng == "legacy-cupy" else np
            for s in range(start, end):
                Y, Z = draw_coefficients(args.seed_base + s, 10, args.rng,
                                         xp_coeff)
                for k in range(10):
                    cf.write(f"{s},{k+1},{Y[k]:.17g},{Z[k]:.17g}\n")

    write_manifest(os.path.join(args.data_dir,
                                f"manifest_part{args.node_id}.json"),
                   samples=args.samples, total_nodes=args.total_nodes,
                   node_id=args.node_id, sample_range=[start, end],
                   resolutions=args.resolutions,
                   target_times=list(map(float, target_times)),
                   num_checkpoints=args.num_checkpoints,
                   boundary_mode=args.boundary_mode, backend=args.backend,
                   rng=args.rng, seed_base=args.seed_base,
                   created=time.strftime("%Y-%m-%dT%H:%M:%S"))

    t0 = time.time()
    generate_node(samples=args.samples, total_nodes=args.total_nodes,
                  node_id=args.node_id, resolutions=args.resolutions,
                  target_times=target_times, data_dir=args.data_dir,
                  boundary_mode=args.boundary_mode, backend=args.backend,
                  num_workers=args.workers, num_devices=args.num_devices,
                  rng_mode=args.rng, seed_base=args.seed_base)
    print(f"node {args.node_id} finished in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
