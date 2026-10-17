# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Array-backend dispatch (NumPy CPU / CuPy GPU).

Every numerical module receives the array namespace `xp` explicitly; nothing
imports cupy at module level, so CPU-only installations never touch it.
"""
from __future__ import annotations
import numpy as np

_CUPY = None


def _load_cupy():
    global _CUPY
    if _CUPY is None:
        try:
            import cupy
            _CUPY = cupy
        except ImportError as e:
            raise ImportError(
                "backend='gpu' requires CuPy (pip install cupy-cuda12x)."
            ) from e
    return _CUPY


def get_xp(backend: str):
    """Return the array namespace for 'cpu' or 'gpu'."""
    if backend == "cpu":
        return np
    if backend == "gpu":
        return _load_cupy()
    raise ValueError(f"backend must be 'cpu' or 'gpu', got {backend!r}")


def to_numpy(a):
    """Device-to-host copy if needed; identity for NumPy arrays."""
    if isinstance(a, np.ndarray):
        return a
    return a.get()


def use_device(backend: str, device_id: int):
    """Pin the CUDA device for this process (no-op on CPU)."""
    if backend == "gpu":
        _load_cupy().cuda.Device(device_id).use()


def free_pool(backend: str):
    """Release cached GPU memory (no-op on CPU)."""
    if backend == "gpu":
        _load_cupy().get_default_memory_pool().free_all_blocks()
