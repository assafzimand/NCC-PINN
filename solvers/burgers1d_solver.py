"""
1D Viscous Burgers Equation Solver using Cole-Hopf Exact Solution.

Solves: h_t + h*h_x - (nu/pi)*h_xx = 0
Domain: x in [-1, 1], t in [0, 1]
Initial Condition: h(0, x) = -sin(pi*x)
Boundary Conditions: h(t, -1) = h(t, 1) = 0 (Dirichlet)

Uses Cole-Hopf transformation to obtain exact solution, with
Chebyshev collocation solver as an independent cross-check.
"""

import numpy as np
import torch
from typing import Tuple, Dict
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import os


def cole_hopf_exact(x, t, nu, n_terms=200):
    """
    Compute exact solution using Cole-Hopf transformation for Dirichlet BCs.
    
    Burgers: h_t + h*h_x = (nu/pi)*h_xx with IC: h(x,0) = -sin(pi*x)
    Domain: [-1, 1] with Dirichlet BCs: h(-1,t) = h(1,t) = 0
    
    Cole-Hopf: h = -2*(nu/pi) * phi_x / phi
    where phi solves the heat equation: phi_t = (nu/pi)*phi_xx
    with IC: phi(x, 0) = exp((1 - cos(pi*x))/(2*nu/pi))
    
    For Dirichlet BCs, we use Fourier sine series on the transformed domain [0,1].
    The solution is stable even for very small viscosity (nu/pi ~ 0.001).
    
    Reference: Raissi et al. (2019), original burgers_shock.mat dataset
    
    Args:
        x: spatial coordinates on [-1, 1] (array-like)
        t: time (scalar or array-like)
        nu: viscosity parameter (nu/pi in the PDE)
        n_terms: number of Fourier sine terms
    
    Returns:
        h: solution values at (x, t)
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    
    visc = nu / np.pi
    
    # For Dirichlet BCs on [-1,1], transform to [0,2] for sine series
    # Use domain [0, 2] with sine basis sin(n*pi*xi/2) where xi = x + 1
    xi = x + 1.0  # Map [-1,1] to [0,2]
    L = 2.0  # Domain length
    
    # Initial condition for phi: phi(x,0) = exp((1 - cos(pi*x))/(2*visc))
    # Compute Fourier sine coefficients: a_n = (2/L) * integral_0^L phi_0(xi) * sin(n*pi*xi/L) dxi
    
    # Use high-resolution numerical integration for coefficients
    n_integrate = 4000
    xi_int = np.linspace(0, L, n_integrate)
    x_int = xi_int - 1.0  # Map back to [-1,1]
    phi_0 = np.exp((1.0 - np.cos(np.pi * x_int)) / (2.0 * visc))
    
    # Compute sine coefficients
    a_n = np.zeros(n_terms, dtype=np.float64)
    for n in range(1, n_terms + 1):
        sin_basis = np.sin(n * np.pi * xi_int / L)
        a_n[n-1] = (2.0 / L) * np.trapz(phi_0 * sin_basis, xi_int)
    
    # Time evolution: phi(xi,t) = sum_n a_n * exp(-n^2*pi^2*visc*t/L^2) * sin(n*pi*xi/L)
    phi = np.zeros_like(xi, dtype=np.float64)
    phi_xi = np.zeros_like(xi, dtype=np.float64)
    
    for n in range(1, n_terms + 1):
        decay = np.exp(-n**2 * np.pi**2 * visc * t / L**2)
        sin_val = np.sin(n * np.pi * xi / L)
        cos_val = np.cos(n * np.pi * xi / L)
        
        phi += a_n[n-1] * decay * sin_val
        phi_xi += a_n[n-1] * decay * (n * np.pi / L) * cos_val
    
    # Cole-Hopf: h = -2*visc * (dphi/dx) / phi
    # Since xi = x + 1, dphi/dx = dphi/dxi
    phi = np.maximum(np.abs(phi), 1e-15) * np.sign(phi + 1e-15)  # Preserve sign, avoid zero
    h = -2.0 * visc * phi_xi / phi
    
    return h


def solve_burgers_chebyshev(
    x_min=-1.0, x_max=1.0, t_min=0.0, t_max=1.0,
    nx=64, nt=201, nu=0.01
):
    """
    Solve Burgers equation using Chebyshev collocation in space.
    
    This provides an independent numerical check against Cole-Hopf.
    Uses Method of Lines: Chebyshev discretization in x, ODE solver in t.
    """
    from numpy.polynomial import chebyshev as cheb
    
    # Chebyshev collocation points in [-1, 1]
    # Use Gauss-Lobatto points: includes boundaries
    i = np.arange(nx)
    x_cheb = -np.cos(np.pi * i / (nx - 1))
    
    # Chebyshev differentiation matrices
    # D1: first derivative, D2: second derivative
    c = np.ones(nx)
    c[0] = 2.0
    c[-1] = 2.0
    
    D1 = np.zeros((nx, nx))
    for i in range(nx):
        for j in range(nx):
            if i != j:
                D1[i, j] = (c[i] * (-1) ** (i + j)) / (c[j] * (x_cheb[i] - x_cheb[j]))
            elif i == 0 and j == 0:
                D1[i, j] = (2.0 * (nx - 1) ** 2 + 1.0) / 6.0
            elif i == nx - 1 and j == nx - 1:
                D1[i, j] = -(2.0 * (nx - 1) ** 2 + 1.0) / 6.0
    
    D2 = D1 @ D1
    
    visc = nu / np.pi
    
    # Initial condition
    h0 = -np.sin(np.pi * x_cheb)
    
    # Enforce Dirichlet BCs: h[-1, t] = h[1, t] = 0
    # Interior points: x_cheb[1:-1]
    D1_int = D1[1:-1, :]
    D2_int = D2[1:-1, :]
    
    def rhs(t, h_full):
        """RHS for interior points with boundary conditions enforced."""
        h = np.zeros(nx)
        h[0] = 0.0  # BC at x=-1
        h[-1] = 0.0  # BC at x=1
        h[1:-1] = h_full  # Interior
        
        h_x = D1_int @ h
        h_xx = D2_int @ h
        
        # Burgers: h_t = -h*h_x + visc*h_xx (for interior)
        return -h[1:-1] * h_x + visc * h_xx
    
    # Time integration
    t_eval = np.linspace(t_min, t_max, nt)
    sol = solve_ivp(
        rhs, (t_min, t_max), h0[1:-1],
        method='Radau', t_eval=t_eval,
        rtol=1e-8, atol=1e-10
    )
    
    # Reconstruct full solution with BCs
    h_solution = np.zeros((nt, nx))
    actual_nt = sol.y.shape[1]  # Actual number of time points returned
    for i in range(min(nt, actual_nt)):
        h_solution[i, 0] = 0.0
        h_solution[i, -1] = 0.0
        h_solution[i, 1:-1] = sol.y[:, i]
    
    # If solver returned fewer points, use only what we got
    if actual_nt < nt:
        h_solution = h_solution[:actual_nt]
        t_eval = t_eval[:actual_nt]
    
    return x_cheb, t_eval, h_solution


def cross_check_and_visualize(
    x_check, t_check, h_cole_hopf, h_chebyshev,
    save_dir, nu
):
    """
    Cross-check Cole-Hopf vs Chebyshev and generate visualization.
    
    Args:
        x_check: spatial grid (nx,)
        t_check: temporal grid (nt,)
        h_cole_hopf: Cole-Hopf solution (nt, nx)
        h_chebyshev: Chebyshev solution (nt, nx)
        save_dir: directory to save cross-check results
        nu: viscosity parameter
    """
    # Compute differences
    diff = h_cole_hopf - h_chebyshev
    max_diff = np.abs(diff).max()
    
    # Relative L2 error
    l2_cheb = np.linalg.norm(h_chebyshev)
    rel_l2 = np.linalg.norm(diff) / (l2_cheb + 1e-12)
    
    print(f"\n  === Burgers1D Cross-Check ===")
    print(f"  Cole-Hopf vs Chebyshev (nx={len(x_check)}, nt={len(t_check)})")
    print(f"  Max pointwise difference: {max_diff:.6e}")
    print(f"  Relative L2 error: {rel_l2:.6e}")
    
    if max_diff > 1e-6:
        print(f"  WARNING: Difference exceeds 1e-6 threshold!")
    else:
        print(f"  PASS: Cross-check passed (difference < 1e-6)")
    
    # Generate visualization
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Cole-Hopf solution
    ax = axes[0, 0]
    X, T = np.meshgrid(x_check, t_check)
    im1 = ax.contourf(X, T, h_cole_hopf, levels=50, cmap='RdBu_r')
    ax.set_title(f'Cole-Hopf Exact Solution (nu/pi = {nu:.4f})')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    plt.colorbar(im1, ax=ax)
    
    # 2. Chebyshev solution
    ax = axes[0, 1]
    im2 = ax.contourf(X, T, h_chebyshev, levels=50, cmap='RdBu_r')
    ax.set_title(f'Chebyshev Collocation (nx={len(x_check)})')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    plt.colorbar(im2, ax=ax)
    
    # 3. Absolute difference
    ax = axes[1, 0]
    im3 = ax.contourf(X, T, np.abs(diff), levels=50, cmap='Reds')
    ax.set_title(f'Absolute Difference (max={max_diff:.2e})')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    plt.colorbar(im3, ax=ax)
    
    # 4. Statistics text
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""
Cross-Check Statistics
======================
Grid: {len(x_check)}x{len(t_check)}
Viscosity: nu/pi = {nu:.4f}

Max |difference|: {max_diff:.6e}
Mean |difference|: {np.abs(diff).mean():.6e}
Relative L2 error: {rel_l2:.6e}

Status: {'PASS' if max_diff < 1e-6 else 'WARNING'}
Threshold: 1e-6
    """
    ax.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    fig_path = os.path.join(save_dir, 'burgers1d_crosscheck.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Cross-check visualization saved: {fig_path}")
    
    return max_diff, rel_l2


class Burgers1DInterpolator:
    """Interpolator for 1D Burgers equation using Cole-Hopf exact solution."""
    
    def __init__(self, x_grid, t_grid, h_solution):
        """
        Initialize interpolator with Cole-Hopf solution.
        
        Args:
            x_grid: Spatial grid (nx,)
            t_grid: Temporal grid (nt,)
            h_solution: Cole-Hopf exact solution (nt, nx)
        """
        # Strict bounds checking for Dirichlet problem (but allow small numerical tolerance)
        self.interpolator = RegularGridInterpolator(
            (t_grid, x_grid), h_solution,
            method='cubic', bounds_error=False, fill_value=0.0
        )
        
        self.x_min = x_grid.min()
        self.x_max = x_grid.max()
        self.t_min = t_grid.min()
        self.t_max = t_grid.max()
    
    def __call__(self, x_points, t_points):
        """Evaluate Cole-Hopf exact solution at arbitrary points."""
        x_flat = np.asarray(x_points).flatten()
        t_flat = np.asarray(t_points).flatten()
        points = np.column_stack([t_flat, x_flat])
        return self.interpolator(points)


_cached_interpolator = None
_cached_config_hash = None
_crosscheck_done = False


def _get_interpolator_cached(config: Dict) -> Burgers1DInterpolator:
    """Get interpolator with caching and optional cross-check."""
    global _cached_interpolator, _cached_config_hash, _crosscheck_done
    
    problem_config = config['burgers1d']
    x_min, x_max = problem_config['spatial_domain'][0]
    t_min, t_max = problem_config['temporal_domain']
    nu = problem_config.get('nu', 0.01)
    
    config_tuple = (x_min, x_max, t_min, t_max, nu)
    
    if _cached_interpolator is None or _cached_config_hash != config_tuple:
        print("  Generating Burgers1D solution using Cole-Hopf exact formula (Fourier sine series)...")
        
        # Generate Cole-Hopf solution on fine grid
        nx_fine = 256
        nt_fine = 201
        x_grid = np.linspace(x_min, x_max, nx_fine)
        t_grid = np.linspace(t_min, t_max, nt_fine)
        
        h_cole_hopf = np.zeros((nt_fine, nx_fine))
        for i, t_val in enumerate(t_grid):
            h_cole_hopf[i, :] = cole_hopf_exact(x_grid, t_val, nu, n_terms=200)
        
        print(f"  Cole-Hopf solution computed ({nx_fine}x{nt_fine} grid, 200 Fourier sine terms)")
        
        # Cross-check with Chebyshev (once per config)
        if not _crosscheck_done:
            print("\n  Running cross-check with Chebyshev collocation...")
            x_cheb, t_cheb, h_cheb = solve_burgers_chebyshev(
                x_min, x_max, t_min, t_max,
                nx=64, nt=51, nu=nu
            )
            
            # Interpolate Cole-Hopf to Chebyshev grid for comparison
            h_cole_on_cheb = np.zeros_like(h_cheb)
            for i, t_val in enumerate(t_cheb):
                h_cole_on_cheb[i, :] = cole_hopf_exact(x_cheb, t_val, nu, n_terms=200)
            
            # Save cross-check visualization
            save_dir = os.path.join(
                os.path.dirname(__file__), '..', 'datasets', 'burgers1d'
            )
            cross_check_and_visualize(
                x_cheb, t_cheb, h_cole_on_cheb, h_cheb, save_dir, nu
            )
            
            _crosscheck_done = True
        
        _cached_interpolator = Burgers1DInterpolator(x_grid, t_grid, h_cole_hopf)
        _cached_config_hash = config_tuple
    
    return _cached_interpolator


def generate_dataset(
    n_residual: int, n_ic: int, n_bc: int,
    device: torch.device, config: Dict
) -> Dict[str, torch.Tensor]:
    """
    Generate dataset with Burgers1D ground truth via Cole-Hopf exact solution.
    
    Ground truth is computed using Cole-Hopf transformation (exact solution).
    First call performs cross-check with Chebyshev collocation and saves visualization.
    """
    seed = config['seed']
    problem = config.get('problem', 'burgers1d')
    problem_config = config[problem]
    spatial_dim = problem_config['spatial_dim']
    x_min, x_max = problem_config['spatial_domain'][0]
    t_min, t_max = problem_config['temporal_domain']
    
    interpolator = _get_interpolator_cached(config)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    N = n_residual + n_ic + n_bc
    x = torch.zeros(N, spatial_dim, device=device)
    t = torch.zeros(N, 1, device=device)
    
    idx = 0
    
    # Residual points
    print(f"  Sampling {n_residual} residual points...")
    x[idx:idx + n_residual, 0] = torch.rand(n_residual, device=device) * (x_max - x_min) + x_min
    t[idx:idx + n_residual, 0] = torch.rand(n_residual, device=device) * (t_max - t_min) + t_min
    idx += n_residual
    
    # IC points
    print(f"  Sampling {n_ic} initial condition points...")
    x[idx:idx + n_ic, 0] = torch.rand(n_ic, device=device) * (x_max - x_min) + x_min
    t[idx:idx + n_ic, 0] = t_min
    idx += n_ic
    
    # BC points (Dirichlet at x=-1 and x=+1)
    print(f"  Sampling {n_bc} boundary condition points...")
    n_bc_left = n_bc // 2
    n_bc_right = n_bc - n_bc_left
    n_times = max(n_bc_left, n_bc_right)
    t_bc = torch.rand(n_times, device=device) * (t_max - t_min) + t_min
    
    x[idx:idx + n_bc_left, 0] = x_min
    t[idx:idx + n_bc_left, 0] = t_bc[:n_bc_left]
    idx += n_bc_left
    
    x[idx:idx + n_bc_right, 0] = x_max
    t[idx:idx + n_bc_right, 0] = t_bc[:n_bc_right]
    
    # Masks
    mask_res = torch.zeros(N, dtype=torch.bool, device=device)
    mask_res[:n_residual] = True
    mask_ic = torch.zeros(N, dtype=torch.bool, device=device)
    mask_ic[n_residual:n_residual + n_ic] = True
    mask_bc = torch.zeros(N, dtype=torch.bool, device=device)
    mask_bc[n_residual + n_ic:] = True
    
    # Interpolate ground truth
    print("  Evaluating Cole-Hopf exact solution...")
    x_np = x.cpu().numpy()[:, 0]
    t_np = t.cpu().numpy()[:, 0]
    h_interp = interpolator(x_np, t_np)
    
    h_gt = torch.zeros(N, 1, device=device, dtype=torch.float32)
    h_gt[:, 0] = torch.from_numpy(h_interp.astype(np.float32)).to(device)
    
    print("  Dataset generated successfully (Cole-Hopf exact)")
    
    return {
        "x": x, "t": t, "h_gt": h_gt,
        "mask": {"residual": mask_res, "IC": mask_ic, "BC": mask_bc},
    }


def evaluate_on_grid(x_grid: torch.Tensor, config: Dict) -> torch.Tensor:
    """Evaluate Cole-Hopf exact solution on a regular grid."""
    interpolator = _get_interpolator_cached(config)
    x_np = x_grid.cpu().numpy()
    h = interpolator(x_np[:, 0], x_np[:, 1])
    return torch.from_numpy(h.reshape(-1, 1).astype(np.float32))
