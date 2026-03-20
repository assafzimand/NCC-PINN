"""
Convection-Diffusion Equation Solver using Method of Lines + BDF.

Solves: h_t + beta * h_x = epsilon * h_xx
Domain: x in [-1, 1], t in [0, 1]
Initial Condition: h(x, 0) = -sin(pi * x)
Boundary Conditions: h(-1, t) = 0, h(1, t) = 0 (Dirichlet)
Parameters: beta = 1.0, epsilon = 0.01
"""

import numpy as np
import torch
from typing import Tuple, Dict
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp


def solve_conv_diff(
    x_min: float = -1.0,
    x_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    nx: int = 512,
    nt: int = 201,
    beta: float = 1.0,
    epsilon: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the convection-diffusion equation using method of lines with BDF.

    Equation: h_t + beta * h_x = epsilon * h_xx
    Rewritten: h_t = epsilon * h_xx - beta * h_x
    """
    x_grid = np.linspace(x_min, x_max, nx + 2, dtype=np.float64)
    dx = x_grid[1] - x_grid[0]
    t_grid = np.linspace(t_min, t_max, nt, dtype=np.float64)

    x_int = x_grid[1:-1]

    u0 = -np.sin(np.pi * x_int)

    bc_left = 0.0
    bc_right = 0.0

    diff_coeff = epsilon / dx ** 2
    adv_coeff = beta / (2.0 * dx)

    def rhs(t_val, u):
        du = np.empty_like(u)
        # Diffusion (central differences)
        du[0] = diff_coeff * (bc_left - 2 * u[0] + u[1])
        du[1:-1] = diff_coeff * (u[:-2] - 2 * u[1:-1] + u[2:])
        du[-1] = diff_coeff * (u[-2] - 2 * u[-1] + bc_right)
        # Advection (central differences)
        adv = np.empty_like(u)
        adv[0] = adv_coeff * (u[1] - bc_left)
        adv[1:-1] = adv_coeff * (u[2:] - u[:-2])
        adv[-1] = adv_coeff * (bc_right - u[-2])
        du -= adv
        return du

    print("  Solving Convection-Diffusion with BDF integrator...")
    sol = solve_ivp(
        rhs,
        (t_min, t_max),
        u0,
        method='BDF',
        t_eval=t_grid,
        rtol=1e-8,
        atol=1e-10,
        max_step=0.005,
    )

    if not sol.success:
        print(f"  WARNING: solver message: {sol.message}")

    h_int = sol.y.T

    h_solution = np.zeros((nt, nx + 2), dtype=np.float64)
    h_solution[:, 0] = bc_left
    h_solution[:, -1] = bc_right
    h_solution[:, 1:-1] = h_int

    return x_grid, t_grid, h_solution


class ConvDiffInterpolator:
    """Interpolator for convection-diffusion equation solution."""

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


def _get_interpolator(config: Dict) -> ConvDiffInterpolator:
    problem = config.get('problem', 'conv_diff')
    pc = config[problem]
    x_min, x_max = pc['spatial_domain'][0]
    t_min, t_max = pc['temporal_domain']
    beta = pc.get('beta', 1.0)
    epsilon = pc.get('epsilon', 0.01)

    x_grid, t_grid, h_sol = solve_conv_diff(
        x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max,
        nx=512, nt=201, beta=beta, epsilon=epsilon,
    )
    return ConvDiffInterpolator(x_grid, t_grid, h_sol)


def _get_interpolator_cached(config: Dict) -> ConvDiffInterpolator:
    global _cached_interpolator, _cached_config_hash
    problem = config.get('problem', 'conv_diff')
    pc = config[problem]
    config_tuple = (
        tuple(pc['spatial_domain'][0]),
        tuple(pc['temporal_domain']),
        pc.get('beta', 1.0),
        pc.get('epsilon', 0.01),
    )
    if _cached_interpolator is None or _cached_config_hash != config_tuple:
        print("  Generating Convection-Diffusion solution (514x201 grid)...")
        _cached_interpolator = _get_interpolator(config)
        _cached_config_hash = config_tuple
        print("  Solution computed.")
    return _cached_interpolator


def generate_dataset(
    n_residual: int, n_ic: int, n_bc: int,
    device: torch.device, config: Dict,
) -> Dict[str, torch.Tensor]:
    """Generate dataset with convection-diffusion ground truth."""
    seed = config['seed']
    problem = config.get('problem', 'conv_diff')
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
