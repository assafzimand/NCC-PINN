"""
Kuramoto-Sivashinsky (KS) Equation Solver using Pseudo-Spectral + ETDRK4.

Solves: u_t + alpha * u * u_x + beta * u_xx + gamma * u_xxxx = 0
Domain: x in [0, 2*pi], t in [0, 1]
Initial Condition: u(x, 0) = cos(x) * (1 + sin(x))
Boundary Conditions: Periodic
Parameters: alpha = 100/16, beta = 100/16^2, gamma = 100/16^4
            (PirateNet / jaxpi benchmark; Wang et al., JMLR 2024)

Uses Fourier pseudo-spectral method for spatial discretization and
ETDRK4 (Exponential Time Differencing RK4) for time integration.
The linear part L(k) = beta*k^2 - gamma*k^4 is treated exactly,
while the nonlinear convection is stepped explicitly.
Reference: Kassam & Trefethen, SIAM J. Sci. Comput. 26(4), 2005.
"""

import numpy as np
import torch
from typing import Tuple, Dict
from scipy.interpolate import RegularGridInterpolator


def solve_ks(
    x_min: float = 0.0,
    x_max: float = 2.0 * np.pi,
    t_min: float = 0.0,
    t_max: float = 1.0,
    nx: int = 256,
    nt: int = 201,
    alpha: float = 100.0 / 16.0,
    beta: float = 100.0 / 16.0 ** 2,
    gamma: float = 100.0 / 16.0 ** 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the KS equation using Fourier pseudo-spectral + ETDRK4.

    Equation: u_t + alpha*u*u_x + beta*u_xx + gamma*u_xxxx = 0
    Rewritten: u_t = -alpha*u*u_x - beta*u_xx - gamma*u_xxxx
    In Fourier space: dv/dt = Lk*v + N_hat(v)
      where Lk = beta*k^2 - gamma*k^4 (linear, treated exactly)
      and N_hat = FFT(-alpha * u * u_x) (nonlinear, stepped explicitly)
    """
    domain_len = x_max - x_min
    dx = domain_len / nx
    x_grid = np.linspace(x_min, x_max - dx, nx, dtype=np.float64)
    t_grid = np.linspace(t_min, t_max, nt, dtype=np.float64)
    dt_save = t_grid[1] - t_grid[0] if nt > 1 else (t_max - t_min)

    k = np.fft.fftfreq(nx, d=dx) * 2.0 * np.pi

    u0 = np.cos(x_grid) * (1.0 + np.sin(x_grid))
    v = np.fft.fft(u0)

    u_solution = np.zeros((nt, nx), dtype=np.float64)
    u_solution[0, :] = u0.copy()

    # Linear operator: from u_t = ... - beta*u_xx - gamma*u_xxxx
    # F[u_xx] = (ik)^2 * v = -k^2 * v  =>  -beta * F[u_xx] = beta*k^2 * v
    # F[u_xxxx] = (ik)^4 * v = k^4 * v  =>  -gamma * F[u_xxxx] = -gamma*k^4 * v
    Lk = beta * k ** 2 - gamma * k ** 4

    # Adaptive sub-stepping based on the stiffest linear mode
    Lk_max = np.max(np.abs(Lk))
    dt_linear = 1.0 / (Lk_max + 1e-10)
    k_max = np.max(np.abs(k))
    dt_nonlinear = 0.5 / (alpha * k_max + 1e-10)
    dt_safe = min(dt_linear, dt_nonlinear)
    n_sub = max(int(np.ceil(dt_save / dt_safe)), 1)
    dt = dt_save / n_sub

    # ETDRK4 coefficients via contour integrals (Kassam & Trefethen 2005)
    E = np.exp(Lk * dt)
    E2 = np.exp(Lk * dt / 2.0)

    M = 64
    r = np.exp(2j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * Lk[:, np.newaxis] + r[np.newaxis, :]

    Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean(
        (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR ** 2)) / LR ** 3, axis=1))
    f2 = dt * np.real(np.mean(
        (2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR ** 3, axis=1))
    f3 = dt * np.real(np.mean(
        (-4.0 - 3.0 * LR - LR ** 2 + np.exp(LR) * (4.0 - LR)) / LR ** 3, axis=1))

    def N_hat(v_hat):
        """Nonlinear term in Fourier space: FFT(-alpha * u * u_x)."""
        u_phys = np.real(np.fft.ifft(v_hat))
        u_x = np.real(np.fft.ifft(1j * k * v_hat))
        return np.fft.fft(-alpha * u_phys * u_x)

    print(f"  Solving KS with ETDRK4 ({nx} modes, {nt} save points, {n_sub} sub-steps/save)...")
    for save_idx in range(1, nt):
        for _ in range(n_sub):
            Nv = N_hat(v)
            a = E2 * v + Q * Nv
            Na = N_hat(a)
            b = E2 * v + Q * Na
            Nb = N_hat(b)
            c = E2 * a + Q * (2.0 * Nb - Nv)
            Nc = N_hat(c)
            v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3

        u_solution[save_idx, :] = np.real(np.fft.ifft(v))

    return x_grid, t_grid, u_solution


class KSInterpolator:
    """Interpolator for Kuramoto-Sivashinsky equation solution with periodic boundary conditions."""

    def __init__(self, x_grid, t_grid, h_solution):
        # Store domain info for periodic wrapping
        self.x_min, self.x_max = x_grid.min(), x_grid.max()
        self.domain_length = self.x_max - self.x_min
        
        # bounds_error=True since we handle wrapping explicitly
        self.interpolator = RegularGridInterpolator(
            (t_grid, x_grid), h_solution,
            method='cubic', bounds_error=True, fill_value=None,
        )

    def __call__(self, x_points, t_points):
        """Interpolate with periodic x-wrapping."""
        x_flat = np.asarray(x_points).flatten()
        t_flat = np.asarray(t_points).flatten()
        
        # Wrap x coordinates to [x_min, x_max) for periodic BCs
        x_wrapped = self.x_min + np.mod(x_flat - self.x_min, self.domain_length)
        
        points = np.column_stack([t_flat, x_wrapped])
        return self.interpolator(points)


_cached_interpolator = None
_cached_config_hash = None


def _get_interpolator(config: Dict) -> KSInterpolator:
    problem = config.get('problem', 'ks')
    pc = config[problem]
    x_min, x_max = pc['spatial_domain'][0]
    t_min, t_max = pc['temporal_domain']
    alpha = pc.get('alpha', 100.0 / 16.0)
    beta = pc.get('beta', 100.0 / 16.0 ** 2)
    gamma_val = pc.get('gamma', 100.0 / 16.0 ** 4)

    x_grid, t_grid, h_sol = solve_ks(
        x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max,
        nx=512, nt=500, alpha=alpha, beta=beta, gamma=gamma_val,
    )
    return KSInterpolator(x_grid, t_grid, h_sol)


def _get_interpolator_cached(config: Dict) -> KSInterpolator:
    global _cached_interpolator, _cached_config_hash
    problem = config.get('problem', 'ks')
    pc = config[problem]
    config_tuple = (
        tuple(pc['spatial_domain'][0]),
        tuple(pc['temporal_domain']),
        pc.get('alpha', 100.0 / 16.0),
        pc.get('beta', 100.0 / 16.0 ** 2),
        pc.get('gamma', 100.0 / 16.0 ** 4),
    )
    if _cached_interpolator is None or _cached_config_hash != config_tuple:
        print("  Generating KS solution (256x201 grid, ETDRK4)...")
        _cached_interpolator = _get_interpolator(config)
        _cached_config_hash = config_tuple
        print("  Solution computed.")
    return _cached_interpolator


def generate_dataset(
    n_residual: int, n_ic: int, n_bc: int,
    device: torch.device, config: Dict,
) -> Dict[str, torch.Tensor]:
    """Generate dataset with KS ground truth via interpolation.

    BC points are sampled at x=x_min and x=x_max for periodic enforcement.
    """
    seed = config['seed']
    problem = config.get('problem', 'ks')
    pc = config[problem]
    spatial_dim = pc['spatial_dim']
    x_min, x_max = pc['spatial_domain'][0]
    t_min, t_max = pc['temporal_domain']

    interpolator = _get_interpolator_cached(config)
    torch.manual_seed(seed)
    np.random.seed(seed)

    N = n_residual + n_ic + n_bc
    x = torch.zeros(N, spatial_dim, device=device)
    t = torch.zeros(N, 1, device=device)
    idx = 0

    print(f"  Sampling {n_residual} residual points...")
    x[idx:idx + n_residual, 0] = torch.rand(n_residual, device=device) * (x_max - x_min) + x_min
    t[idx:idx + n_residual, 0] = torch.rand(n_residual, device=device) * (t_max - t_min) + t_min
    idx += n_residual

    print(f"  Sampling {n_ic} initial condition points...")
    x[idx:idx + n_ic, 0] = torch.rand(n_ic, device=device) * (x_max - x_min) + x_min
    t[idx:idx + n_ic, 0] = t_min
    idx += n_ic

    print(f"  Sampling {n_bc} boundary condition points...")
    n_bc_left = n_bc // 2
    n_bc_right = n_bc - n_bc_left
    t_bc = torch.rand(max(n_bc_left, n_bc_right), device=device) * (t_max - t_min) + t_min

    x[idx:idx + n_bc_left, 0] = x_min
    t[idx:idx + n_bc_left, 0] = t_bc[:n_bc_left]
    idx += n_bc_left
    x[idx:idx + n_bc_right, 0] = x_max
    t[idx:idx + n_bc_right, 0] = t_bc[:n_bc_right]

    mask_res = torch.zeros(N, dtype=torch.bool, device=device)
    mask_res[:n_residual] = True
    mask_ic = torch.zeros(N, dtype=torch.bool, device=device)
    mask_ic[n_residual:n_residual + n_ic] = True
    mask_bc = torch.zeros(N, dtype=torch.bool, device=device)
    mask_bc[n_residual + n_ic:] = True

    print("  Interpolating ground truth values...")
    x_np = x.cpu().numpy()[:, 0]
    t_np = t.cpu().numpy()[:, 0]
    h_interp = interpolator(x_np, t_np)

    h_gt = torch.zeros(N, 1, device=device, dtype=torch.float32)
    h_gt[:, 0] = torch.from_numpy(h_interp.astype(np.float32)).to(device)

    print("  Dataset generated successfully")
    return {
        "x": x, "t": t, "h_gt": h_gt,
        "mask": {"residual": mask_res, "IC": mask_ic, "BC": mask_bc},
    }


def evaluate_on_grid(x_grid: torch.Tensor, config: Dict) -> torch.Tensor:
    """Evaluate ground truth on a regular grid for frequency analysis."""
    interpolator = _get_interpolator(config)
    x_np = x_grid.cpu().numpy()
    h = interpolator(x_np[:, 0], x_np[:, 1])
    return torch.from_numpy(h.reshape(-1, 1).astype(np.float32))
