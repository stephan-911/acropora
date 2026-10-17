# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""2-D compressible Euler equations (ideal gas) as a VLBM model.

Provides the conserved-variable fluxes f (x), g (y) and the Bouchut-type
kinetic speed bound a_n = 4 * max(|v| + c). The evaluation order is fixed
so that results are bit-reproducible.
"""
from __future__ import annotations


class Euler2D:
    n_fields = 4  # (rho, rho*u, rho*v, E)

    def __init__(self, gamma: float = 5.0 / 3.0):
        self.gamma = gamma

    def fluxes_and_speed(self, U, xp):
        rho = U[0]
        rho_u, rho_v, E = U[1], U[2], U[3]
        vel_x = rho_u / rho
        vel_y = rho_v / rho
        p = (self.gamma - 1) * (E - 0.5 * rho * (vel_x**2 + vel_y**2))
        f = xp.array([rho_u, rho_u*vel_x + p, rho_u*vel_y, (E + p)*vel_x])
        g = xp.array([rho_v, rho_v*vel_x, rho_v*vel_y + p, (E + p)*vel_y])
        lambda_max = xp.max(xp.sqrt(vel_x**2 + vel_y**2) + xp.sqrt(self.gamma * p / rho))
        a_n = 4.0 * lambda_max
        return a_n, f, g
