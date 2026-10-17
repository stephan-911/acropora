# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Vectorial D2Q5 lattice Boltzmann scheme for 2-D conservation laws.

One distribution vector per lattice direction (+x, -x, +y, -y) plus an
implicit rest population; exact-relaxation collision with a positivity
limiter theta in [0,1] per direction (theta=1: second order, theta=0:
first-order kinetic). Adaptive kinetic speed a_n re-evaluated every step
(CFL: dt = dx / a_n).

The update keeps a fixed evaluation order; regression tests assert
bit-identical fields against the reference implementation in `baseline/`.
"""
from __future__ import annotations
from ..limiters import density_bounds, directional_theta


class VectorialD2Q5:
    def __init__(self, model, Nx, Ny, dx, *, xp, boundary, alpha=0.5):
        self.model = model
        self.xp = xp
        self.Nx = Nx
        self.Ny_physical = Ny
        self.dx = dx
        self.dy = dx
        self.alpha = alpha
        self.boundary = boundary

        self.pad_x = boundary.pad_x
        self.pad_y = boundary.pad_y
        self.Nx_total = Nx + 2 * self.pad_x
        self.Ny_total = Ny + 2 * self.pad_y

        q = model.n_fields
        self.U = xp.zeros((q, self.Nx_total, self.Ny_total), dtype=xp.float64)
        self.u_k = xp.zeros((4, q, self.Nx_total, self.Ny_total), dtype=xp.float64)

        x = (xp.arange(self.Nx_total) - self.pad_x) * self.dx
        y = (xp.arange(self.Ny_total) - self.pad_y) * self.dy - 1.0
        self.X, self.Y = xp.meshgrid(x, y, indexing='ij')

        boundary.bind(self)

    # ------------------------------------------------------------------
    def interior(self):
        """Slices selecting the physical (unpadded) domain."""
        sx = slice(self.pad_x, -self.pad_x) if self.pad_x > 0 else slice(None)
        sy = slice(self.pad_y, -self.pad_y) if self.pad_y > 0 else slice(None)
        return sx, sy

    def compute_equilibrium(self, U, f, g, a_n):
        coeff = (1.0 - self.alpha) / 4.0
        base = coeff * U
        return (base + f/(2.0*a_n), base - f/(2.0*a_n),
                base + g/(2.0*a_n), base - g/(2.0*a_n))

    def initialize_equilibrium(self):
        """Set distributions to equilibrium of the current field (after BCs)."""
        self.boundary.apply(self)
        a_n, f, g = self.model.fluxes_and_speed(self.U, self.xp)
        M1, M2, M3, M4 = self.compute_equilibrium(self.U, f, g, a_n)
        self.u_k[0] = M1; self.u_k[1] = M2; self.u_k[2] = M3; self.u_k[3] = M4

    # ------------------------------------------------------------------
    def step(self):
        xp = self.xp
        a_n, f, g = self.model.fluxes_and_speed(self.U, xp)

        dt = self.dx / a_n
        kappa = a_n * (1.0 - self.alpha) / 2.0

        M1, M2, M3, M4 = self.compute_equilibrium(self.U, f, g, a_n)
        A1 = M1 - self.u_k[0]
        A2 = M2 - self.u_k[1]
        A3 = M3 - self.u_k[2]
        A4 = M4 - self.u_k[3]

        A2_plus_x = xp.roll(A2, -1, axis=1)
        A4_plus_y = xp.roll(A4, -1, axis=2)
        Delta_F = a_n * (A1 - A2_plus_x)
        Delta_G = a_n * (A3 - A4_plus_y)

        U_plus_x = xp.roll(self.U, -1, axis=1)
        f_plus_x = xp.roll(f, -1, axis=1)
        U_star_x = 0.5 * (U_plus_x + self.U) - (f_plus_x - f) / (2.0 * kappa)

        U_plus_y = xp.roll(self.U, -1, axis=2)
        g_plus_y = xp.roll(g, -1, axis=2)
        U_star_y = 0.5 * (U_plus_y + self.U) - (g_plus_y - g) / (2.0 * kappa)

        rho = self.U[0]
        mu, nu = density_bounds(rho, U_star_x[0], U_star_y[0], xp)

        theta_x = directional_theta(rho, U_star_x, Delta_F, mu, nu, kappa, 0, xp)
        theta_y = directional_theta(rho, U_star_y, Delta_G, mu, nu, kappa, 1, xp)

        theta_minus_x = xp.roll(theta_x, 1, axis=0)
        theta_minus_y = xp.roll(theta_y, 1, axis=1)

        self.u_k[0] = xp.roll(M1 + theta_x * A1, shift=1, axis=1)
        self.u_k[1] = xp.roll(M2 + theta_minus_x * A2, shift=-1, axis=1)
        self.u_k[2] = xp.roll(M3 + theta_y * A3, shift=1, axis=2)
        self.u_k[3] = xp.roll(M4 + theta_minus_y * A4, shift=-1, axis=2)

        self.U = self.u_k[0] + self.u_k[1] + self.u_k[2] + self.u_k[3] + self.alpha * self.U \
                 - theta_minus_x * A2 - theta_x * A1 - theta_minus_y * A4 - theta_y * A3

        self.boundary.post_step(self)
        self.boundary.apply(self)
        return dt
