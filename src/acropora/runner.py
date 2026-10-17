# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Single-sample time integration with checkpoint capture.

Checkpoint semantics (identical to production): the field written for target
time T is the state at the FIRST discrete time t >= T (up to one dt late, no
interpolation). If the solver stalls (dt ~ 0 or NaN), all remaining
checkpoints receive the last valid field; sample-health checks detect this
via equality of the trailing checkpoints.
"""
from __future__ import annotations
import numpy as np
from .backend import to_numpy


def run_sample(solver, target_times, write_frame, *, on_stall=None):
    """Advance `solver` through `target_times`, calling
    `write_frame(checkpoint_index, density_interior)` at each checkpoint.

    Returns the number of steps taken. `write_frame` receives a NumPy array
    (device transfer handled here).
    """
    sx, sy = solver.interior()
    t = 0.0
    target_idx = 0
    steps = 0
    n_ck = len(target_times)

    while t <= target_times[-1] and target_idx < n_ck:
        while target_idx < n_ck and t >= target_times[target_idx]:
            write_frame(target_idx, to_numpy(solver.U[0, sx, sy]))
            target_idx += 1

        if target_idx < n_ck:
            dt = solver.step()
            dt_f = float(dt)
            if dt_f < 1e-16 or np.isnan(dt_f):
                if on_stall is not None:
                    on_stall(dt_f)
                break
            t += dt_f
            steps += 1

    while target_idx < n_ck:
        write_frame(target_idx, to_numpy(solver.U[0, sx, sy]))
        target_idx += 1
    return steps
