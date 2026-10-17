# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""acropora: computing statistical solutions of nonlinear partial differential equations.

acropora evolves probability measures rather than individual states,
approximating statistical solutions of nonlinear PDEs by propagating Monte
Carlo ensembles through deterministic solvers, on CPU (NumPy) or GPU (CuPy).
Reference application: statistical solutions of a Mach 2000 astrophysical jet.
"""
from .backend import get_xp, to_numpy
from .grid import grid_from_nx
from .models import Euler2D
from .schemes.vectorial_d2q5 import VectorialD2Q5
from .boundaries import Mach2000JetBC
from .random_fields import fourier_perturbation, draw_coefficients
from .runner import run_sample
from .ensemble import generate_node, node_slice

__version__ = "1.0.0"
__all__ = ["get_xp", "to_numpy", "grid_from_nx", "Euler2D", "VectorialD2Q5", "Mach2000JetBC",
           "fourier_perturbation", "draw_coefficients", "run_sample",
           "generate_node", "node_slice"]
