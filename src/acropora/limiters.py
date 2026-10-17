# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Positivity limiter for the vectorial D2Q5 scheme.

theta in [0,1] blends first-order kinetic (theta=0) and second-order
(theta=1) updates so that density remains within local bounds (mu, nu) and
the pressure-positivity quadratic stays admissible. Guards and evaluation
order are fixed for bit-reproducibility.
"""
from __future__ import annotations


def density_bounds(rho, rho_star_x, rho_star_y, xp):
    rho_star_minus_x = xp.roll(rho_star_x, 1, axis=0)
    rho_star_minus_y = xp.roll(rho_star_y, 1, axis=1)
    mu = 0.999 * xp.minimum(rho, xp.minimum(rho_star_x, xp.minimum(
        rho_star_minus_x, xp.minimum(rho_star_y, rho_star_minus_y))))
    nu = 1.001 * xp.maximum(rho, xp.maximum(rho_star_x, xp.maximum(
        rho_star_minus_x, xp.maximum(rho_star_y, xp.maximum(rho_star_minus_y, 1e-12)))))
    return mu, nu


def directional_theta(rho, U_star, Delta, mu, nu, kappa, axis, xp):
    """theta for one direction (axis=0: x, axis=1: y)."""
    theta_rho = xp.ones_like(rho)
    mu_plus = xp.roll(mu, -1, axis=axis)
    nu_plus = xp.roll(nu, -1, axis=axis)
    rho_star = U_star[0]

    mask_pos = Delta[0] > 1e-12
    mask_neg = Delta[0] < -1e-12
    theta_rho[mask_pos] = xp.minimum(
        1.0, kappa * xp.minimum(nu_plus - rho_star, rho_star - mu)[mask_pos] / Delta[0][mask_pos])
    theta_rho[mask_neg] = xp.minimum(
        1.0, -kappa * xp.minimum(rho_star - mu_plus, nu - rho_star)[mask_neg] / Delta[0][mask_neg])

    gamma_0 = (U_star[1]**2 + U_star[2]**2 - 2 * U_star[0] * U_star[3]) / xp.maximum(U_star[0], 1e-12)
    gamma_1 = 2 * (U_star[0]*Delta[3] + U_star[3]*Delta[0]
                   - U_star[1]*Delta[1] - U_star[2]*Delta[2]) / xp.maximum(U_star[0], 1e-12)
    det = xp.maximum(gamma_1**2 - 4 * gamma_0 * (Delta[1]**2 + Delta[2]**2
                     - 2 * Delta[0]*Delta[3]) / xp.maximum(U_star[0], 1e-12), 0.0)
    eigen1 = xp.abs(Delta[0]) / xp.maximum(U_star[0], 1e-12)
    eigen2 = (xp.abs(gamma_1) + xp.sqrt(det)) / xp.maximum(xp.abs(2 * gamma_0), 1e-14)
    theta_p = xp.maximum(0.0, kappa / xp.maximum(xp.maximum(eigen1, eigen2), 1e-14) - 1e-16)

    return xp.clip(xp.minimum(theta_rho, theta_p), 0.0, 1.0)
