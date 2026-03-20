"""
Fisher-KPP Equation Solver using Method of Lines + BDF.

Solves: h_t = D * h_xx + kappa * h * (1 - h)
Domain: x in [0, 1], t in [0, 1]
Initial Condition: h(x, 0) = 1 / (1 + exp(sqrt(kappa/6) * (x - 0.25)))
Boundary Conditions: h(0, t) = 1, h(1, t) = 0 (Dirichlet)
Parameters: D = 1.0, kappa = 25.0
"""

import numpy as np
import torch
from typing import Tuple, Dict
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp


def solve_fisher_kpp(
    x_min: float = 0.0,
    x_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    nx: int = 512,
    nt: int = 201,
    D: float = 1.0,
    kappa: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the Fisher-KPP equation using method of lines with BDF integrator.

    Equation: h_t = D * h_xx + kappa * h * (1 - h)
    """
    x_grid = np.linspace(x_min, x_max, nx + 2, dtype=np.float64)
    dx = x_grid[1] - x_grid[0]
    t_grid = np.linspace(t_min, t_max, nt, dtype=np.float64)

    x_int = x_grid[1:-1]

    width = np.sqrt(kappa / 6.0)
    u0 = 1.0 / (1.0 + np.exp(width * (x_int - 0.25)))

    bc_left = 1.0
    bc_right = 0.0

    coeff = D / dx ** 2

    def rhs(t_val, u):
        du = np.empty_like(u)
        du[0] = coeff * (bc_left - 2 * u[0] + u[1])
        du[1:-1] = coeff * (u[:-2] - 2 * u[1:-1] + u[2:])
        du[-1] = coeff * (u[-2] - 2 * u[-1] + bc_right)
        du += kappa * u * (1.0 - u)
        return du

    print("  Solving Fisher-KPP with BDF integrator...")
    sol = solve_ivp(
        rhs,
        (t_min, t_max),
        u0,
        method='BDF',
        t_eval=t_grid,
        rtol=1e-8,
        atol=1e-10,
        max_step=0.01,
    )

    if not sol.success:
        print(f"  WARNING: solver message: {sol.message}")

    h_int = sol.y.T  # (nt, n_int)

    h_solution = np.zeros((nt, nx + 2), dtype=np.float64)
    h_solution[:, 0] = bc_left
    h_solution[:, -1] = bc_right
    h_solution[:, 1:-1] = h_int

    return x_grid, t_grid, h_solution


class FisherKPPInterpolator:
    """Interpolator for Fisher-KPP equation solution."""

    def __init__(self, x_grid, t_grid, h_solution):
        self.interpolator = RegularGridInterpolator(
            (t_grid, x_grid), h_solution,
            method='cubic', bounds_error=False, fill_value=0.0,
        )

    def __call__(self, x_points, t_points):
        x_flat = np.asarray(x_points).flatten()
        t_flat = np.asarray(t_points).flatten()
        points = np.column_stack([t_flat, x_flat])
        return self.interpolator(points)


_cached_interpolator = None
_cached_config_hash = None


def _get_interpolator(config: Dict) -> FisherKPPInterpolator:
    problem = config.get('problem', 'fisher_kpp')
    pc = config[problem]
    x_min, x_max = pc['spatial_domain'][0]
    t_min, t_max = pc['temporal_domain']
    D = pc.get('D', 1.0)
    kappa = pc.get('kappa', 25.0)

    x_grid, t_grid, h_sol = solve_fisher_kpp(
        x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max,
        nx=512, nt=201, D=D, kappa=kappa,
    )
    return FisherKPPInterpolator(x_grid, t_grid, h_sol)


def _get_interpolator_cached(config: Dict) -> FisherKPPInterpolator:
    global _cached_interpolator, _cached_config_hash
    problem = config.get('problem', 'fisher_kpp')
    pc = config[problem]
    config_tuple = (
        tuple(pc['spatial_domain'][0]),
        tuple(pc['temporal_domain']),
        pc.get('D', 1.0),
        pc.get('kappa', 25.0),
    )
    if _cached_interpolator is None or _cached_config_hash != config_tuple:
        print("  Generating Fisher-KPP solution (514x201 grid)...")
        _cached_interpolator = _get_interpolator(config)
        _cached_config_hash = config_tuple
        print("  Solution computed.")
    return _cached_interpolator


def generate_dataset(
    n_residual: int, n_ic: int, n_bc: int,
    device: torch.device, config: Dict,
) -> Dict[str, torch.Tensor]:
    """Generate dataset with Fisher-KPP ground truth via interpolation."""
    seed = config['seed']
    problem = config.get('problem', 'fisher_kpp')
    pc = config[problem]
    spatial_dim = pc['spatial_dim']
    x_min, x_max = pc['spatial_domain'][0]
    t_min, t_max = pc['temporal_domain']
    kappa = pc.get('kappa', 25.0)

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
