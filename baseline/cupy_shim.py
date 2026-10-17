# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Minimal CuPy-API shim over NumPy.

Purpose: run the reference generation script on CPU as a golden baseline
for regression testing. Only the APIs actually used are provided; everything
else falls through to numpy.
"""
import numpy as _np


class _Device:
    def __init__(self, dev_id):
        self.id = dev_id
    def use(self):
        return self


class _MemPool:
    def free_all_blocks(self):
        pass


class _Cuda:
    Device = _Device


class _Random:
    """Legacy cupy.random surface. NOTE: values follow numpy's legacy MT19937
    stream, NOT CuPy's XORWOW; the shim provides a self-consistent CPU
    reference stream."""
    @staticmethod
    def seed(s):
        _np.random.seed(s)
    @staticmethod
    def uniform(lo, hi):
        return _np.float64(_np.random.uniform(lo, hi))


def asnumpy(a):
    return _np.asarray(a)


cuda = _Cuda()
random = _Random()


def get_default_memory_pool():
    return _MemPool()


def __getattr__(name):
    return getattr(_np, name)
