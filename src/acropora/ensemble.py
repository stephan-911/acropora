# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Monte Carlo ensemble orchestration.

- Node partitioning distributes any remainder over the first nodes, so the
  full sample count is always computed.
- Per-sample restart: samples whose final checkpoint is already written are
  skipped individually; interrupted runs resume where they stopped.
- A persistent worker pool with a per-worker device queue keeps one CUDA
  context per worker for the whole run.
"""
from __future__ import annotations
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from .backend import get_xp, use_device, free_pool
from .grid import grid_from_nx
from .boundaries import Mach2000JetBC
from .models import Euler2D
from .schemes.vectorial_d2q5 import VectorialD2Q5
from .runner import run_sample
from .io.npy_store import open_store, completed_samples

_WORKER = {}


# Workers are started with the "spawn" method: a forked child cannot use a
# CUDA context that the parent has already initialized, and the parent does
# initialize one when it records the legacy-cupy coefficients.
_CTX = mp.get_context("spawn")


def _device_queue(backend, num_devices, num_workers):
    """Queue of device ids; workers pop one each (round-robin coverage)."""
    if backend != "gpu" or num_devices <= 0:
        return None
    q = _CTX.Manager().Queue()
    for wkr in range(num_workers):
        q.put(wkr % num_devices)
    return q


def node_slice(total_samples, total_nodes, node_id):
    """Contiguous sample range for a node; remainder spread over first nodes."""
    base, rem = divmod(total_samples, total_nodes)
    start = node_id * base + min(node_id, rem)
    count = base + (1 if node_id < rem else 0)
    return start, start + count


def _init_worker(backend, device_queue):
    _WORKER["backend"] = backend
    if backend == "gpu" and device_queue is not None:
        ident = device_queue.get()  # each worker pops a unique device id
        use_device(backend, ident)
        _WORKER["device"] = ident
    else:
        _WORKER["device"] = None


def _run_one(job):
    (sample_id, local_idx, Nx, Ny, dx, target_times, boundary_mode,
     store_file, S_local, rng_mode, gamma, seed_base) = job
    backend = _WORKER["backend"]
    xp = get_xp(backend)

    store = open_store(store_file, S_local, len(target_times), Nx, Ny)

    bc = Mach2000JetBC.make(boundary_mode, Ny, seed=seed_base + sample_id,
                            rng_mode=rng_mode)
    solver = VectorialD2Q5(Euler2D(gamma=gamma), Nx, Ny, dx, xp=xp, boundary=bc)
    bc.initial_condition(solver)

    def write(ck, frame):
        store[local_idx, ck] = frame

    stalled = {}
    steps = run_sample(solver, target_times, write,
                       on_stall=lambda dt: stalled.setdefault("dt", dt))
    store.flush()
    del solver, store
    free_pool(backend)
    return sample_id, _WORKER["device"], steps, stalled.get("dt")


def generate_node(*, samples, total_nodes, node_id, resolutions, target_times,
                  data_dir="data", boundary_mode="ghost", backend="cpu",
                  num_workers=1, num_devices=0, rng_mode="portable",
                  gamma=5.0/3.0, seed_base=42, log=print):
    """Run this node's share of the ensemble for every resolution.

    `resolutions` are x-resolutions Nx (500, 1000, 2000, 4000 for the jet
    ladder); the transverse count and mesh width follow from the domain.
    """
    start, end = node_slice(samples, total_nodes, node_id)
    S_local = end - start
    os.makedirs(data_dir, exist_ok=True)
    log(f"--- Node {node_id}: samples {start}..{end-1} "
        f"({backend}, {num_workers} workers, rng={rng_mode}) ---")

    for nx in resolutions:
        Nx, Ny, dx = grid_from_nx(nx)
        store_file = os.path.join(data_dir, f"samples_{Nx}_part{node_id}.npy")

        if os.path.exists(store_file):
            done = completed_samples(store_file, S_local, len(target_times), Nx, Ny)
        else:
            open_store(store_file, S_local, len(target_times), Nx, Ny,
                       mode='w+').flush()
            done = np.zeros(S_local, dtype=bool)

        todo = [s for s in range(start, end) if not done[s - start]]
        if not todo:
            log(f"[grid {Nx}x{Ny}] all {S_local} samples already complete; skipping")
            continue
        log(f"[grid {Nx}x{Ny}] running {len(todo)}/{S_local} samples")

        jobs = [(s, s - start, Nx, Ny, dx, target_times, boundary_mode,
                 store_file, S_local, rng_mode, gamma, seed_base) for s in todo]

        if num_workers <= 1:
            _init_worker(backend, _device_queue(backend, num_devices, 1))
            for job in jobs:
                sid, dev, steps, stall = _run_one(job)
                log(f"  sample {sid} done ({steps} steps"
                    + (f", STALLED dt={stall}" if stall else "") + ")")
        else:
            dq = _device_queue(backend, num_devices, num_workers)
            with ProcessPoolExecutor(max_workers=num_workers,
                                     mp_context=_CTX,
                                     initializer=_init_worker,
                                     initargs=(backend, dq)) as pool:
                futs = {pool.submit(_run_one, j): j for j in jobs}
                for fut in as_completed(futs):
                    sid, dev, steps, stall = fut.result()
                    log(f"  sample {sid} done on device {dev} ({steps} steps"
                        + (f", STALLED dt={stall}" if stall else "") + ")")
