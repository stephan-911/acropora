# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Memmapped .npy ensemble storage: creation and frame access (generation side).

Layout: shape (S, T, Nx, Ny) float64, one file per (grid, node-part).
Part merging and sample-health checks belong to the post-processing module.
"""
from __future__ import annotations
import json
import numpy as np


def npy_header_offset(filepath):
    with open(filepath, 'rb') as f:
        version = np.lib.format.read_magic(f)
        # use the public header readers (the private variant was removed
        # in numpy >= 2)
        reader = {(1, 0): np.lib.format.read_array_header_1_0,
                  (2, 0): np.lib.format.read_array_header_2_0}[version]
        reader(f)
        return f.tell()


def create_store(filepath, S, T, Nx, Ny):
    return np.lib.format.open_memmap(
        filepath, mode='w+', dtype=np.float64, shape=(S, T, Nx, Ny))


def open_store(filepath, S, T, Nx, Ny, mode='r+'):
    return np.lib.format.open_memmap(
        filepath, mode=mode, dtype=np.float64, shape=(S, T, Nx, Ny))


def read_frame(filepath, header_offset, s_idx, t_idx, total_T, Nx, Ny):
    if t_idx < 0:
        t_idx = total_T + t_idx
    frame_bytes = Nx * Ny * 8
    byte_offset = header_offset + (s_idx * total_T + t_idx) * frame_bytes
    return np.memmap(filepath, dtype=np.float64, mode='r',
                     offset=byte_offset, shape=(Nx, Ny))


def completed_samples(filepath, S, T, Nx, Ny):
    """Boolean mask of samples whose FINAL checkpoint is written (nonzero).

    Used for per-sample restart: only missing samples are recomputed when
    a partially written part file is found.
    """
    off = npy_header_offset(filepath)
    done = np.zeros(S, dtype=bool)
    for s in range(S):
        frame = read_frame(filepath, off, s, -1, T, Nx, Ny)
        done[s] = bool(np.any(frame != 0.0))
    return done


def write_manifest(path, **fields):
    with open(path, 'w') as f:
        json.dump(fields, f, indent=2, default=str)


def read_manifest(path):
    with open(path) as f:
        return json.load(f)
