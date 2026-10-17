# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""Boundary handling for the Mach 2000 jet setup.

`Mach2000JetBC` owns: ghost-pad sizes, the ambient state, the (stochastically
perturbed) inlet state, zero-gradient transverse/outlet copies, forced inlet
columns, re-equilibration of pad distributions, and (optionally) a sponge
layer. The evaluation order is fixed; regression tests assert bit-identical
fields against the reference implementation in `baseline/`.

Other flow setups implement the same interface
(`bind`, `initial_condition`, `apply`, `post_step`).
"""
from __future__ import annotations
from .random_fields import fourier_perturbation


class Mach2000JetBC:
    def __init__(self, mode="ghost", *, seed, rng_mode="portable",
                 jet_half_width=0.05, rho_jet=5.0, u_jet=800.0,
                 p_inlet=0.4127, rho_amb=0.5, p_amb=0.4127,
                 num_modes=10, amplitude=0.1, p_decay=2.0):
        if mode not in ("ghost", "sponge"):
            raise ValueError(f"boundary mode must be 'ghost' or 'sponge', got {mode!r}")
        self.mode = mode
        self.seed = seed
        self.rng_mode = rng_mode
        self.jet_half_width = jet_half_width
        self.rho_jet, self.u_jet = rho_jet, u_jet
        self.p_inlet, self.rho_amb, self.p_amb = p_inlet, rho_amb, p_amb
        self.num_modes, self.amplitude, self.p_decay = num_modes, amplitude, p_decay

        if mode == "sponge":
            self.pad_x = 0
            self.pad_y = None  # set via make() (depends on Ny)
        else:
            self.pad_x = 3
            self.pad_y = 3

    # ------------------------------------------------------------------
    @classmethod
    def make(cls, mode, Ny, **kw):
        """Factory fixing sponge pad_y = max(10, Ny//20) before construction."""
        bc = cls(mode, **kw)
        if mode == "sponge":
            bc.pad_y = max(10, Ny // 20)
        return bc

    def bind(self, solver):
        xp = solver.xp
        gamma = solver.model.gamma
        if self.mode == "sponge" and self.pad_y is None:
            raise RuntimeError("sponge pad_y must be fixed before solver construction")

        solver.U_ambient_target = xp.array(
            [self.rho_amb, 0.0, 0.0, self.p_amb / (gamma - 1.0)]).reshape(4, 1, 1)

        if self.mode == "sponge":
            sigma_y = xp.zeros(solver.Ny_total)
            sigma_y[:self.pad_y] = xp.sin(xp.linspace(xp.pi/2, 0, self.pad_y))**2
            sigma_y[-self.pad_y:] = xp.sin(xp.linspace(0, xp.pi/2, self.pad_y))**2
            solver.sponge_mask = sigma_y.reshape(1, 1, solver.Ny_total)

    # ------------------------------------------------------------------
    def initial_condition(self, solver):
        xp = solver.xp
        gamma = solver.model.gamma
        solver.U_ambient = xp.array(
            [self.rho_amb, 0.0, 0.0, self.p_amb / (gamma - 1.0)]).reshape(4, 1)
        solver.U[:] = solver.U_ambient[:, :, xp.newaxis]

        jet_mask = xp.abs(solver.Y[0, :]) < self.jet_half_width
        y_jet_normalized = solver.Y[0, jet_mask] / self.jet_half_width

        rho_inlet = xp.full(solver.Ny_total, self.rho_jet, dtype=xp.float64)
        rho_inlet[jet_mask] = fourier_perturbation(
            y_jet_normalized, self.seed, self.rho_jet, xp=xp,
            num_modes=self.num_modes, amplitude=self.amplitude,
            p=self.p_decay, rng_mode=self.rng_mode)
        solver.rho_inlet = rho_inlet

        E_inlet = self.p_inlet / (gamma - 1) + 0.5 * rho_inlet * (self.u_jet**2)
        solver.U_inlet = xp.zeros((4, solver.Ny_total))
        solver.U_inlet[0, :] = rho_inlet
        solver.U_inlet[1, :] = rho_inlet * self.u_jet
        solver.U_inlet[2, :] = 0.0
        solver.U_inlet[3, :] = E_inlet

        solver.initialize_equilibrium()

    # ------------------------------------------------------------------
    def apply(self, solver):
        xp = solver.xp
        pad_x, pad_y = solver.pad_x, solver.pad_y
        Ny_total, Nx_total = solver.Ny_total, solver.Nx_total

        if pad_y > 0:
            solver.U[:, :, :pad_y] = solver.U[:, :, pad_y:pad_y+1]
            solver.U[:, :, Ny_total - pad_y:] = solver.U[:, :, Ny_total - pad_y - 1:Ny_total - pad_y]

        left_limit = max(1, pad_x)
        for i in range(left_limit):
            jet_mask_i = xp.abs(solver.Y[i, :]) < self.jet_half_width
            solver.U[:, i, jet_mask_i] = solver.U_inlet[:, jet_mask_i]
            solver.U[:, i, ~jet_mask_i] = solver.U_ambient[:, 0].reshape(4, 1)

        if pad_x > 0:
            solver.U[:, Nx_total - pad_x:, :] = solver.U[:, Nx_total - pad_x - 1:Nx_total - pad_x, :]
        else:
            solver.U[:, -1, :] = solver.U[:, -2, :]

        a_n, f, g = solver.model.fluxes_and_speed(solver.U, xp)
        M1, M2, M3, M4 = solver.compute_equilibrium(solver.U, f, g, a_n)

        for M, u_dist in zip([M1, M2, M3, M4],
                             [solver.u_k[0], solver.u_k[1], solver.u_k[2], solver.u_k[3]]):
            if pad_x > 0:
                u_dist[:, :pad_x, :] = M[:, :pad_x, :]
                u_dist[:, -pad_x:, :] = M[:, -pad_x:, :]
            else:
                u_dist[:, 0, :] = M[:, 0, :]
                u_dist[:, -1, :] = M[:, -1, :]

            if pad_y > 0:
                u_dist[:, :, :pad_y] = M[:, :, :pad_y]
                u_dist[:, :, -pad_y:] = M[:, :, -pad_y:]
            else:
                u_dist[:, :, 0] = M[:, :, 0]
                u_dist[:, :, -1] = M[:, :, -1]

    def post_step(self, solver):
        if self.mode == "sponge":
            solver.U = (1.0 - solver.sponge_mask) * solver.U \
                       + solver.sponge_mask * solver.U_ambient_target
