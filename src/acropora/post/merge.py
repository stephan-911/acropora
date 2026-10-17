# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Merge per-node part files into the analysis store samples_{Nx}.npy.

A missing part is a hard error so that sample indices always stay aligned
with seeds and recorded coefficients; passing allow_gaps=True keeps the
alignment by writing zeros in place of the missing part (those samples are
then dropped by the health scan).
"""
from __future__ import annotations
import os

from ..io.npy_store import npy_header_offset, create_store


def merge_parts(final_file, part_files, S_total, T, Nx, Ny, *,
                parts_sizes, allow_gaps=False, chunk=1 << 30):
    full = create_store(final_file, S_total, T, Nx, Ny)
    final_offset = full.offset
    del full
    part_bytes = [sz * T * Nx * Ny * 8 for sz in parts_sizes]

    with open(final_file, 'r+b') as f_out:
        pos = final_offset
        for i, part in enumerate(part_files):
            if not os.path.exists(part):
                if not allow_gaps:
                    raise FileNotFoundError(
                        f"part file missing: {part} (use allow_gaps=True to "
                        f"keep index alignment with a zero-filled gap)")
                pos += part_bytes[i]
                continue
            f_out.seek(pos)
            src_off = npy_header_offset(part)
            with open(part, 'rb') as f_in:
                f_in.seek(src_off)
                while True:
                    data = f_in.read(chunk)
                    if not data:
                        break
                    f_out.write(data)
            pos += part_bytes[i]
