# This file is part of acropora: computing statistical solutions of
# nonlinear partial differential equations.
#
# Copyright 2026 Stephan Simonis
# SPDX-License-Identifier: BSD-2-Clause-Patent

import argparse
import time
import numpy as np
import cupy_shim as cp  # numpy shim provides the cupy API on CPU
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import gc

# ==========================================
# 1. Fourier Inlet Perturbation (Lye/Mishra)
# ==========================================
def fourier_perturbation(y_coords, num_modes, seed, base_val, amplitude=0.1, p=2.0):
    cp.random.seed(seed)
    perturbation = cp.zeros_like(y_coords)
    for k in range(1, num_modes + 1):
        Y_k = cp.random.uniform(-1.0, 1.0)
        Z_k = cp.random.uniform(-1.0, 1.0)
        scaling = 1.0 / (k ** p)
        perturbation += scaling * (Y_k * cp.cos(k * cp.pi * y_coords) +
                                   Z_k * cp.sin(k * cp.pi * y_coords))

    # The amplitude is directly applied without normalizing by the sample maximum.
    # This perfectly preserves the statistical independence and natural variance
    # of the random variables, which is required for Wasserstein convergence.

    window = 0.5 * (1.0 + cp.cos(cp.pi * y_coords))
    return base_val * (1.0 + amplitude * perturbation * window)

# ======================================================
# 2. Vectorial D2Q5 LBM Euler Core (Wissocq/Liu/Abgrall)
# ======================================================
class VectorialLBM_D2Q5:
    def __init__(self, Nx, Ny, dx, boundary_mode='ghost', gamma=1.4, alpha=0.5):
        self.Nx = Nx
        self.Ny_physical = Ny
        self.boundary_mode = boundary_mode
        self.dx = dx
        self.dy = dx
        self.gamma = gamma
        self.alpha = alpha

        if self.boundary_mode == 'sponge':
            self.pad_x = 0
            self.pad_y = max(10, Ny // 20)
        elif self.boundary_mode == 'ghost':
            self.pad_x = 3
            self.pad_y = 3

        self.Nx_total = self.Nx + 2 * self.pad_x
        self.Ny_total = self.Ny_physical + 2 * self.pad_y

        self.U = cp.zeros((4, self.Nx_total, self.Ny_total), dtype=cp.float64)
        self.u_k = cp.zeros((4, 4, self.Nx_total, self.Ny_total), dtype=cp.float64)

        x = (cp.arange(self.Nx_total) - self.pad_x) * self.dx
        y = (cp.arange(self.Ny_total) - self.pad_y) * self.dy - 1.0
        self.X, self.Y = cp.meshgrid(x, y, indexing='ij')

        self.U_ambient_target = cp.array([0.5, 0.0, 0.0, 0.4127 / (self.gamma - 1.0)]).reshape(4, 1, 1)

        if self.boundary_mode == 'sponge':
            sigma_y = cp.zeros(self.Ny_total)
            sigma_y[:self.pad_y] = cp.sin(cp.linspace(cp.pi/2, 0, self.pad_y))**2
            sigma_y[-self.pad_y:] = cp.sin(cp.linspace(0, cp.pi/2, self.pad_y))**2
            self.sponge_mask = sigma_y.reshape(1, 1, self.Ny_total)

    def init_mach_2000_jet(self, seed):
        self.U_ambient = cp.array([0.5, 0.0, 0.0, 0.4127 / (self.gamma - 1.0)]).reshape(4, 1)
        self.U[:] = self.U_ambient[:, :, cp.newaxis]
        jet_mask = cp.abs(self.Y[0, :]) < 0.05
        y_jet_normalized = self.Y[0, jet_mask] / 0.05

        self.rho_inlet = cp.full(self.Ny_total, 5.0, dtype=cp.float64)
        self.rho_inlet[jet_mask] = fourier_perturbation(y_coords=y_jet_normalized, num_modes=10, seed=seed, base_val=5.0, amplitude=0.1)

        u_inlet = 800.0
        p_inlet = 0.4127
        E_inlet = p_inlet / (self.gamma - 1) + 0.5 * self.rho_inlet * (u_inlet**2)

        self.U_inlet = cp.zeros((4, self.Ny_total))
        self.U_inlet[0, :] = self.rho_inlet
        self.U_inlet[1, :] = self.rho_inlet * u_inlet
        self.U_inlet[2, :] = 0.0
        self.U_inlet[3, :] = E_inlet

        self.apply_boundaries()
        a_n, f, g = self.compute_fluxes_and_speed(self.U)
        M1, M2, M3, M4 = self.compute_equilibrium(self.U, f, g, a_n)
        self.u_k[0] = M1; self.u_k[1] = M2; self.u_k[2] = M3; self.u_k[3] = M4

    def apply_boundaries(self):
        if self.pad_y > 0:
            self.U[:, :, :self.pad_y] = self.U[:, :, self.pad_y:self.pad_y+1]
            self.U[:, :, self.Ny_total - self.pad_y:] = self.U[:, :, self.Ny_total - self.pad_y - 1:self.Ny_total - self.pad_y]

        left_limit = max(1, self.pad_x)
        for i in range(left_limit):
            jet_mask_i = cp.abs(self.Y[i, :]) < 0.05
            self.U[:, i, jet_mask_i] = self.U_inlet[:, jet_mask_i]
            self.U[:, i, ~jet_mask_i] = self.U_ambient[:, 0].reshape(4, 1)

        if self.pad_x > 0:
            self.U[:, self.Nx_total - self.pad_x:, :] = self.U[:, self.Nx_total - self.pad_x - 1:self.Nx_total - self.pad_x, :]
        else:
            self.U[:, -1, :] = self.U[:, -2, :]

        a_n, f, g = self.compute_fluxes_and_speed(self.U)
        M1, M2, M3, M4 = self.compute_equilibrium(self.U, f, g, a_n)

        for M, u_dist in zip([M1, M2, M3, M4], [self.u_k[0], self.u_k[1], self.u_k[2], self.u_k[3]]):
            if self.pad_x > 0:
                u_dist[:, :self.pad_x, :] = M[:, :self.pad_x, :]
                u_dist[:, -self.pad_x:, :] = M[:, -self.pad_x:, :]
            else:
                u_dist[:, 0, :] = M[:, 0, :]
                u_dist[:, -1, :] = M[:, -1, :]

            if self.pad_y > 0:
                u_dist[:, :, :self.pad_y] = M[:, :, :self.pad_y]
                u_dist[:, :, -self.pad_y:] = M[:, :, -self.pad_y:]
            else:
                u_dist[:, :, 0] = M[:, :, 0]
                u_dist[:, :, -1] = M[:, :, -1]

    def compute_fluxes_and_speed(self, U):
        rho = U[0]
        rho_u, rho_v, E = U[1], U[2], U[3]
        vel_x = rho_u / rho
        vel_y = rho_v / rho
        p = (self.gamma - 1) * (E - 0.5 * rho * (vel_x**2 + vel_y**2))
        f = cp.array([rho_u, rho_u*vel_x + p, rho_u*vel_y, (E + p)*vel_x])
        g = cp.array([rho_v, rho_v*vel_x, rho_v*vel_y + p, (E + p)*vel_y])
        lambda_max = cp.max(cp.sqrt(vel_x**2 + vel_y**2) + cp.sqrt(self.gamma * p / rho))
        a_n = 4.0 * lambda_max
        return a_n, f, g

    def compute_equilibrium(self, U, f, g, a_n):
        coeff = (1.0 - self.alpha) / 4.0
        base = coeff * U
        return base + f/(2.0*a_n), base - f/(2.0*a_n), base + g/(2.0*a_n), base - g/(2.0*a_n)

    def step(self):
        a_n, f, g = self.compute_fluxes_and_speed(self.U)

        dt = self.dx / a_n
        kappa = a_n * (1.0 - self.alpha) / 2.0

        M1, M2, M3, M4 = self.compute_equilibrium(self.U, f, g, a_n)
        A1 = M1 - self.u_k[0]
        A2 = M2 - self.u_k[1]
        A3 = M3 - self.u_k[2]
        A4 = M4 - self.u_k[3]

        A2_plus_x = cp.roll(A2, -1, axis=1)
        A4_plus_y = cp.roll(A4, -1, axis=2)
        Delta_F = a_n * (A1 - A2_plus_x)
        Delta_G = a_n * (A3 - A4_plus_y)

        U_plus_x = cp.roll(self.U, -1, axis=1)
        f_plus_x = cp.roll(f, -1, axis=1)

        U_star_x = 0.5 * (U_plus_x + self.U) - (f_plus_x - f) / (2.0 * kappa)

        U_plus_y = cp.roll(self.U, -1, axis=2)
        g_plus_y = cp.roll(g, -1, axis=2)

        U_star_y = 0.5 * (U_plus_y + self.U) - (g_plus_y - g) / (2.0 * kappa)

        rho = self.U[0]
        rho_star_x, rho_star_y = U_star_x[0], U_star_y[0]
        rho_star_minus_x = cp.roll(rho_star_x, 1, axis=0)
        rho_star_minus_y = cp.roll(rho_star_y, 1, axis=1)

        mu = 0.999 * cp.minimum(rho, cp.minimum(rho_star_x, cp.minimum(rho_star_minus_x, cp.minimum(rho_star_y, rho_star_minus_y))))
        nu = 1.001 * cp.maximum(rho, cp.maximum(rho_star_x, cp.maximum(rho_star_minus_x, cp.maximum(rho_star_y, cp.maximum(rho_star_minus_y, 1e-12)))))

        theta_rho_x = cp.ones_like(rho)
        mu_plus_x = cp.roll(mu, -1, axis=0)
        nu_plus_x = cp.roll(nu, -1, axis=0)

        mask_pos_x = Delta_F[0] > 1e-12
        mask_neg_x = Delta_F[0] < -1e-12
        theta_rho_x[mask_pos_x] = cp.minimum(1.0, kappa * cp.minimum(nu_plus_x - rho_star_x, rho_star_x - mu)[mask_pos_x] / Delta_F[0][mask_pos_x])
        theta_rho_x[mask_neg_x] = cp.minimum(1.0, -kappa * cp.minimum(rho_star_x - mu_plus_x, nu - rho_star_x)[mask_neg_x] / Delta_F[0][mask_neg_x])

        gamma_0_x = (U_star_x[1]**2 + U_star_x[2]**2 - 2 * U_star_x[0] * U_star_x[3]) / cp.maximum(U_star_x[0], 1e-12)
        gamma_1_x = 2 * (U_star_x[0]*Delta_F[3] + U_star_x[3]*Delta_F[0] - U_star_x[1]*Delta_F[1] - U_star_x[2]*Delta_F[2]) / cp.maximum(U_star_x[0], 1e-12)
        det_x = cp.maximum(gamma_1_x**2 - 4 * gamma_0_x * (Delta_F[1]**2 + Delta_F[2]**2 - 2 * Delta_F[0]*Delta_F[3]) / cp.maximum(U_star_x[0], 1e-12), 0.0)
        eigen1_x = cp.abs(Delta_F[0]) / cp.maximum(U_star_x[0], 1e-12)
        eigen2_x = (cp.abs(gamma_1_x) + cp.sqrt(det_x)) / cp.maximum(cp.abs(2 * gamma_0_x), 1e-14)
        theta_p_x = cp.maximum(0.0, kappa / cp.maximum(cp.maximum(eigen1_x, eigen2_x), 1e-14) - 1e-16)

        theta_x = cp.clip(cp.minimum(theta_rho_x, theta_p_x), 0.0, 1.0)

        theta_rho_y = cp.ones_like(rho)
        mu_plus_y = cp.roll(mu, -1, axis=1)
        nu_plus_y = cp.roll(nu, -1, axis=1)

        mask_pos_y = Delta_G[0] > 1e-12
        mask_neg_y = Delta_G[0] < -1e-12
        theta_rho_y[mask_pos_y] = cp.minimum(1.0, kappa * cp.minimum(nu_plus_y - rho_star_y, rho_star_y - mu)[mask_pos_y] / Delta_G[0][mask_pos_y])
        theta_rho_y[mask_neg_y] = cp.minimum(1.0, -kappa * cp.minimum(rho_star_y - mu_plus_y, nu - rho_star_y)[mask_neg_y] / Delta_G[0][mask_neg_y])

        gamma_0_y = (U_star_y[1]**2 + U_star_y[2]**2 - 2 * U_star_y[0] * U_star_y[3]) / cp.maximum(U_star_y[0], 1e-12)
        gamma_1_y = 2 * (U_star_y[0]*Delta_G[3] + U_star_y[3]*Delta_G[0] - U_star_y[1]*Delta_G[1] - U_star_y[2]*Delta_G[2]) / cp.maximum(U_star_y[0], 1e-12)
        det_y = cp.maximum(gamma_1_y**2 - 4 * gamma_0_y * (Delta_G[1]**2 + Delta_G[2]**2 - 2 * Delta_G[0]*Delta_G[3]) / cp.maximum(U_star_y[0], 1e-12), 0.0)
        eigen1_y = cp.abs(Delta_G[0]) / cp.maximum(U_star_y[0], 1e-12)
        eigen2_y = (cp.abs(gamma_1_y) + cp.sqrt(det_y)) / cp.maximum(cp.abs(2 * gamma_0_y), 1e-14)
        theta_p_y = cp.maximum(0.0, kappa / cp.maximum(cp.maximum(eigen1_y, eigen2_y), 1e-14) - 1e-16)

        theta_y = cp.clip(cp.minimum(theta_rho_y, theta_p_y), 0.0, 1.0)

        theta_minus_x = cp.roll(theta_x, 1, axis=0)
        theta_minus_y = cp.roll(theta_y, 1, axis=1)

        self.u_k[0] = cp.roll(M1 + theta_x * A1, shift=1, axis=1)
        self.u_k[1] = cp.roll(M2 + theta_minus_x * A2, shift=-1, axis=1)
        self.u_k[2] = cp.roll(M3 + theta_y * A3, shift=1, axis=2)
        self.u_k[3] = cp.roll(M4 + theta_minus_y * A4, shift=-1, axis=2)

        self.U = self.u_k[0] + self.u_k[1] + self.u_k[2] + self.u_k[3] + self.alpha * self.U \
                 - theta_minus_x * A2 - theta_x * A1 - theta_minus_y * A4 - theta_y * A3

        if self.boundary_mode == 'sponge':
            self.U = (1.0 - self.sponge_mask) * self.U + self.sponge_mask * self.U_ambient_target

        self.apply_boundaries()
        return dt

# ==========================================
# 3. GPU Worker (Simulation Task)
# ==========================================
def run_single_sample(params):
    sample_id, start_sample, Nx, Ny, dx, target_times, num_gpus, boundary_mode, curr_file, samples_per_node = params
    gpu_id = (sample_id - start_sample) % num_gpus
    cp.cuda.Device(gpu_id).use()

    num_checkpoints = len(target_times)
    mmap_out = np.lib.format.open_memmap(curr_file, mode='r+', dtype=np.float64, shape=(samples_per_node, num_checkpoints, Nx, Ny))
    local_s_id = sample_id - start_sample

    solver = VectorialLBM_D2Q5(Nx, Ny, dx, boundary_mode=boundary_mode, gamma=5.0/3.0)
    solver.init_mach_2000_jet(seed=42 + sample_id)

    t = 0.0
    target_idx = 0

    if solver.pad_x > 0:
        slice_x = slice(solver.pad_x, -solver.pad_x)
    else:
        slice_x = slice(None)

    if solver.pad_y > 0:
        slice_y = slice(solver.pad_y, -solver.pad_y)
    else:
        slice_y = slice(None)

    while t <= target_times[-1] and target_idx < num_checkpoints:
        while target_idx < num_checkpoints and t >= target_times[target_idx]:
            mmap_out[local_s_id, target_idx] = cp.asnumpy(solver.U[0, slice_x, slice_y])
            target_idx += 1
            if target_idx % 50 == 0:
                mmap_out.flush()

        if target_idx < num_checkpoints:
            dt = solver.step()
            if dt < 1e-16 or np.isnan(dt):
                print(f" [CRITICAL] GPU {gpu_id} Sample {sample_id} stalled numerically (dt={dt}). Aborting worker loop.", flush=True)
                break
            t += dt

    while target_idx < num_checkpoints:
        mmap_out[local_s_id, target_idx] = cp.asnumpy(solver.U[0, slice_x, slice_y])
        target_idx += 1
        if target_idx % 50 == 0:
            mmap_out.flush()

    mmap_out.flush()
    del solver
    cp.get_default_memory_pool().free_all_blocks()

    return sample_id, gpu_id

# ==========================================
# 5. Master Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Restartable MC Vectorial LBM")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--resolutions", type=int, nargs='+', default=[200, 400, 800, 1600])
    parser.add_argument("--max_t", type=float, default=0.003)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--num_cpus", type=int, default=96)

    parser.add_argument("--num_checkpoints", type=int, default=5)
    parser.add_argument("--boundary_mode", type=str, choices=['ghost', 'sponge'], default='ghost')


    parser.add_argument("--array_id", type=int, default=0)
    parser.add_argument("--total_nodes", type=int, default=10) # Default from script 2

    args = parser.parse_args()

    target_times = np.linspace(0, args.max_t, args.num_checkpoints)

    # ---------------------------------------------------------
    # MODE 1: DISTRIBUTED SLURM ARRAY GENERATION
    # ---------------------------------------------------------
    if True:  # generation only
        mp.set_start_method('spawn', force=True)
        samples_per_node = args.samples // args.total_nodes
        start_sample = args.array_id * samples_per_node
        end_sample = start_sample + samples_per_node

        print(f"--- Node {args.array_id} Starting Generation ---", flush=True)
        print(f"Assigned Samples: {start_sample} to {end_sample-1}", flush=True)
        print(f"Boundary Mode: {args.boundary_mode}", flush=True)
        print(f"Target Times: {target_times}\n", flush=True)

        for res in args.resolutions:
            Nx = int(2.5 * res)
            Ny = res * 2
            dx = 1.0 / res

            curr_file = f"data/samples_{Nx}_part{args.array_id}.npy"

            if os.path.exists(curr_file) and os.path.getsize(curr_file) > 1024:
                print(f"\n[SKIP] Node {args.array_id} found existing data for Grid {Nx}x{Ny}.", flush=True)
                continue

            print(f"\n=============================================", flush=True)
            print(f"--- Computing Grid {Nx}x{Ny} ---", flush=True)
            print(f"=============================================\n", flush=True)

            coeff_file = f"data/fourier_coeffs_part{args.array_id}.dat"
            if not os.path.exists(coeff_file):
                print("Recording random Fourier coefficients to disk...", flush=True)
                with open(coeff_file, "w") as cf:
                    cf.write("Sample_ID,Mode_K,Y_k,Z_k\n")
                    for s in range(start_sample, end_sample):
                        cp.random.seed(42 + s)
                        for k in range(1, 11):
                            y_k = float(cp.random.uniform(-1.0, 1.0))
                            z_k = float(cp.random.uniform(-1.0, 1.0))
                            cf.write(f"{s},{k},{y_k:.6f},{z_k:.6f}\n")

            grid_densities = np.lib.format.open_memmap(
                curr_file, mode='w+', dtype=np.float64,
                shape=(samples_per_node, args.num_checkpoints, Nx, Ny)
            )

            tasks = [(s, start_sample, Nx, Ny, dx, target_times, args.num_gpus, args.boundary_mode, curr_file, samples_per_node) for s in range(start_sample, end_sample)]
            start_time = time.time()

            with ProcessPoolExecutor(max_workers=args.num_gpus) as executor:
                futures = {executor.submit(run_single_sample, t): t for t in tasks}
                for future in as_completed(futures):
                    s_id, gpu_id = future.result()
                    print(f"  > Completed Sample {s_id+1}/{args.samples} on GPU {gpu_id}", flush=True)

            grid_densities.flush()
            print(f"\n  [Grid {Nx}x{Ny} chunk completed in {time.time()-start_time:.2f} seconds]\n", flush=True)

if __name__ == "__main__":
    main()
