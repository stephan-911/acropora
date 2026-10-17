# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""CPU speed: baseline port vs restructured library (same machine, same grid).

Both run identical numerics; this checks the restructuring added no overhead.
"""
import importlib.util, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "baseline"))
from acropora import Euler2D, VectorialD2Q5, Mach2000JetBC, grid_from_nx
spec = importlib.util.spec_from_file_location("b", os.path.join(ROOT, "baseline", "lbm_baseline_numpy.py"))
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)

def bench(nx, n_steps, reps=3):
    Nx, Ny, dx = grid_from_nx(nx)
    tb = tl = 1e99
    for _ in range(reps):
        s = B.VectorialLBM_D2Q5(Nx, Ny, dx, boundary_mode='ghost', gamma=5/3)
        s.init_mach_2000_jet(seed=42)
        t0=time.perf_counter(); [s.step() for _ in range(n_steps)]
        tb=min(tb, time.perf_counter()-t0)
        bc = Mach2000JetBC.make('ghost', Ny, seed=42, rng_mode='legacy-numpy')
        s2 = VectorialD2Q5(Euler2D(5/3), Nx, Ny, dx, xp=np, boundary=bc)
        bc.initial_condition(s2)
        t0=time.perf_counter(); [s2.step() for _ in range(n_steps)]
        tl=min(tl, time.perf_counter()-t0)
    return tb, tl

if __name__ == "__main__":
    print(f"{'grid':>10} {'steps':>6} {'baseline[s]':>12} {'library[s]':>12} {'ratio':>7}")
    for nx, n in [(100, 60), (200, 30), (400, 12)]:
        Nx, Ny, _ = grid_from_nx(nx)
        tb, tl = bench(nx, n)
        print(f"{Nx}x{Ny:<5} {n:>6} {tb:>12.3f} {tl:>12.3f} {tl/tb:>7.3f}")
