"""
1D Viscous Burgers Equation Solver with Spectral Method.

Solves: h_t + h*h_x - (nu/pi)*h_xx = 0
Domain: x ∈ [-1, 1], t ∈ [0, 1]
Initial Condition: h(0, x) = -sin(pi*x)
Boundary Conditions: h(t, -1) = h(t, 1) = 0 (Dirichlet)

Uses pseudo-spectral method with FFT for spatial derivatives and
Runge-Kutta 4th order for time integration.
"""

import numpy as np
import torch
from typing import Tuple, Dict
from scipy.interpolate import RegularGridInterpolator


def solve_burgers1d_spectral(
    x_min: float = -1.0,
    x_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    nx: int = 256,
    nt: int = 100,
    nu: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve 1D viscous Burgers equation using pseudo-spectral method.
    
    Equation: h_t + h*h_x - (nu/pi)*h_xx = 0
    
    Uses Fourier pseudo-spectral method for spatial derivatives and
    explicit Runge-Kutta 4 for time integration.
    
    Args:
        x_min: Minimum spatial coordinate (-1.0)
        x_max: Maximum spatial coordinate (1.0)
        t_min: Initial time (0.0)
        t_max: Final time (1.0)
        nx: Number of spatial grid points (should be power of 2)
        nt: Number of time snapshots to save
        nu: Viscosity coefficient (0.01)
        
    Returns:
        x_grid: Spatial grid (nx,)
        t_grid: Temporal grid (nt,)
        h_solution: Solution field (nt, nx)
    """
    # Spatial grid (periodic extension for spectral method)
    L = x_max - x_min  # Domain length = 2
    dx = L / nx
    x_grid = np.linspace(x_min, x_max - dx, nx, dtype=np.float64)
    
    # Time grid
    t_grid = np.linspace(t_min, t_max, nt, dtype=np.float64)
    dt_save = t_grid[1] - t_grid[0] if nt > 1 else t_max
    
    # Wavenumbers for spectral derivatives
    k = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    
    # Viscosity coefficient in PDE: nu/pi
    visc = nu / np.pi
    
    # Initial condition: u(0, x) = -sin(pi*x)
    u = -np.sin(np.pi * x_grid)
    
    # Storage for solution
    u_solution = np.zeros((nt, nx), dtype=np.float64)
    u_solution[0, :] = u.copy()
    
    # Time stepping with RK4
    # Use adaptive time step for stability
    # CFL condition: dt < dx / max|u| and dt < dx^2 / (2*visc)
    dt_base = min(0.5 * dx / (np.abs(u).max() + 1e-10), 
                  0.25 * dx**2 / (visc + 1e-10))
    dt = min(dt_base, dt_save / 10)  # Ensure we can hit save times
    
    def rhs(u_current):
        """Compute right-hand side: -(u*u_x) + (nu/pi)*u_xx"""
        u_hat = np.fft.fft(u_current)
        
        # Spectral derivatives
        u_x_hat = 1j * k * u_hat
        u_xx_hat = -k**2 * u_hat
        
        u_x = np.real(np.fft.ifft(u_x_hat))
        u_xx = np.real(np.fft.ifft(u_xx_hat))
        
        # RHS: -u*u_x + visc*u_xx
        return -u_current * u_x + visc * u_xx
    
    # Integrate in time
    current_time = t_min
    save_idx = 1
    
    while save_idx < nt:
        target_time = t_grid[save_idx]
        
        while current_time < target_time - 1e-12:
            # Adaptive dt to not overshoot target
            dt_step = min(dt, target_time - current_time)
            
            # Update CFL-based dt
            u_max = np.abs(u).max()
            if u_max > 1e-10:
                dt_cfl = 0.5 * dx / u_max
                dt_step = min(dt_step, dt_cfl)
            
            # RK4 step
            k1 = rhs(u)
            k2 = rhs(u + 0.5 * dt_step * k1)
            k3 = rhs(u + 0.5 * dt_step * k2)
            k4 = rhs(u + dt_step * k3)
            
            u = u + (dt_step / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            current_time += dt_step
            
            # Apply boundary conditions (enforce zero at boundaries)
            # For spectral method with periodic BCs, we need to handle this carefully
            # The Dirichlet BCs are approximately satisfied due to the odd symmetry of the solution
        
        # Save solution at this time step
        u_solution[save_idx, :] = u.copy()
        save_idx += 1
    
    return x_grid, t_grid, u_solution


class Burgers1DInterpolator:
    """
    Interpolator for 1D Burgers equation solution.
    Provides ground truth values at arbitrary (x, t) points.
    """
    
    def __init__(
        self,
        x_grid: np.ndarray,
        t_grid: np.ndarray,
        h_solution: np.ndarray
    ):
        """
        Initialize interpolator with precomputed solution.
        
        Args:
            x_grid: Spatial grid (nx,)
            t_grid: Temporal grid (nt,)
            h_solution: Solution (nt, nx)
        """
        # Create interpolator
        self.interpolator = RegularGridInterpolator(
            (t_grid, x_grid),
            h_solution,
            method='cubic',
            bounds_error=False,
            fill_value=0.0
        )
        
        self.x_min = x_grid.min()
        self.x_max = x_grid.max()
        self.t_min = t_grid.min()
        self.t_max = t_grid.max()
    
    def __call__(self, x_points: np.ndarray, t_points: np.ndarray) -> np.ndarray:
        """
        Evaluate solution at arbitrary points.
        
        Args:
            x_points: x coordinates (N,) or (N, 1)
            t_points: t coordinates (N,) or (N, 1)
            
        Returns:
            h_values: Solution at (x, t) points (N,)
        """
        # Ensure 1D arrays
        x_flat = np.asarray(x_points).flatten()
        t_flat = np.asarray(t_points).flatten()
        
        # Stack for interpolator (expects (N, 2) with [t, x] order)
        points = np.column_stack([t_flat, x_flat])
        
        # Interpolate
        h_values = self.interpolator(points)
        
        return h_values


def _get_interpolator(config: Dict) -> Burgers1DInterpolator:
    """
    Get burgers1d solution interpolator.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Burgers1DInterpolator instance
    """
    # Extract domain from config
    problem_config = config['burgers1d']
    x_min, x_max = problem_config['spatial_domain'][0]
    t_min, t_max = problem_config['temporal_domain']
    nu = problem_config.get('nu', 0.01)
    
    # Solve Burgers equation on fine grid
    x_grid, t_grid, u_solution = solve_burgers1d_spectral(
        x_min=x_min,
        x_max=x_max,
        t_min=t_min,
        t_max=t_max,
        nx=256,
        nt=201,
        nu=nu
    )
    
    return Burgers1DInterpolator(x_grid, t_grid, u_solution)


# Cache interpolator to avoid recomputation
_cached_interpolator = None
_cached_config_hash = None


def _get_interpolator_cached(config: Dict) -> Burgers1DInterpolator:
    """Get interpolator with caching."""
    global _cached_interpolator, _cached_config_hash
    
    # Create hash from relevant config params
    problem_config = config['burgers1d']
    config_tuple = (
        tuple(problem_config['spatial_domain'][0]),
        tuple(problem_config['temporal_domain']),
        problem_config.get('nu', 0.01)
    )
    
    if _cached_interpolator is None or _cached_config_hash != config_tuple:
        print("  Generating burgers1d solution (256×201 grid)...")
        _cached_interpolator = _get_interpolator(config)
        _cached_config_hash = config_tuple
        print("  Solution computed: 201×256 grid")
    
    return _cached_interpolator


def generate_dataset(
    n_residual: int,
    n_ic: int,
    n_bc: int,
    device: torch.device,
    config: Dict
) -> Dict[str, torch.Tensor]:
    """
    Generate dataset with burgers1d ground truth via interpolation.
    
    Uses spectral solution on 256×201 grid, then interpolates
    to randomly sampled training points.
    
    Args:
        n_residual: Number of residual (interior) points
        n_ic: Number of initial condition points
        n_bc: Number of boundary condition points
        device: Device to create tensors on (CUDA or CPU)
        config: Configuration dictionary
        
    Returns:
        Dictionary with keys:
            "x": (N, spatial_dim) spatial coordinates
            "t": (N, 1) temporal coordinates
            "h_gt": (N, 1) ground truth solution
            "mask": dict with "residual", "IC", "BC" boolean masks
    """
    seed = config['seed']
    problem = config.get('problem', 'burgers1d')
    problem_config = config[problem]
    spatial_dim = problem_config['spatial_dim']
    spatial_domain = problem_config['spatial_domain']
    temporal_domain = problem_config['temporal_domain']
    
    # Get or create interpolator
    interpolator = _get_interpolator_cached(config)
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    N = n_residual + n_ic + n_bc
    
    # Initialize tensors
    x = torch.zeros(N, spatial_dim, device=device)
    t = torch.zeros(N, 1, device=device)
    
    # Extract domain bounds
    x_min, x_max = spatial_domain[0]
    t_min, t_max = temporal_domain
    
    idx = 0
    
    # 1. Residual points (uniform random in interior)
    print(f"  Sampling {n_residual} residual points...")
    x[idx:idx + n_residual, 0] = torch.rand(n_residual, device=device) * (x_max - x_min) + x_min
    t[idx:idx + n_residual, 0] = torch.rand(n_residual, device=device) * (t_max - t_min) + t_min
    idx += n_residual
    
    # 2. Initial condition points (uniform random at t=0)
    print(f"  Sampling {n_ic} initial condition points...")
    x[idx:idx + n_ic, 0] = torch.rand(n_ic, device=device) * (x_max - x_min) + x_min
    t[idx:idx + n_ic, 0] = t_min
    idx += n_ic
    
    # 3. Boundary condition points (at x=-1 and x=+1, Dirichlet)
    print(f"  Sampling {n_bc} boundary condition points...")
    n_bc_left = n_bc // 2
    n_bc_right = n_bc - n_bc_left
    
    # Sample times for boundaries
    n_times = max(n_bc_left, n_bc_right)
    t_bc = torch.rand(n_times, device=device) * (t_max - t_min) + t_min
    
    # Left boundary (x = x_min = -1)
    x[idx:idx + n_bc_left, 0] = x_min
    t[idx:idx + n_bc_left, 0] = t_bc[:n_bc_left]
    idx += n_bc_left
    
    # Right boundary (x = x_max = +1)
    x[idx:idx + n_bc_right, 0] = x_max
    t[idx:idx + n_bc_right, 0] = t_bc[:n_bc_right]
    idx += n_bc_right
    
    # Create masks
    mask_residual = torch.zeros(N, dtype=torch.bool, device=device)
    mask_residual[:n_residual] = True
    
    mask_ic = torch.zeros(N, dtype=torch.bool, device=device)
    mask_ic[n_residual:n_residual + n_ic] = True
    
    mask_bc = torch.zeros(N, dtype=torch.bool, device=device)
    mask_bc[n_residual + n_ic:] = True
    
    # Interpolate ground truth using burgers1d solution
    print("  Interpolating ground truth values...")
    x_np = x.cpu().numpy()[:, 0]
    t_np = t.cpu().numpy()[:, 0]
    
    h_interp = interpolator(x_np, t_np)
    
    # Convert to (N, 1) format
    h_gt = torch.zeros(N, 1, device=device, dtype=torch.float32)
    h_gt[:, 0] = torch.from_numpy(h_interp.astype(np.float32)).to(device)
    
    print("  Dataset generated successfully")
    
    return {
        "x": x,
        "t": t,
        "h_gt": h_gt,
        "mask": {
            "residual": mask_residual,
            "IC": mask_ic,
            "BC": mask_bc
        }
    }


def evaluate_on_grid(x_grid: torch.Tensor, config: Dict) -> torch.Tensor:
    """
    Evaluate ground truth solution on a regular grid for frequency analysis.
    
    Args:
        x_grid: Grid points (N, 2) with columns [x, t]
        config: Configuration dictionary
        
    Returns:
        h_gt: Ground truth values (N, 1)
    """
    # Get interpolator (which solves the Burgers equation)
    interpolator = _get_interpolator(config)
    
    x_np = x_grid.cpu().numpy()
    
    # Extract coordinates
    x = x_np[:, 0]
    t = x_np[:, 1]
    
    # Evaluate using interpolator
    h = interpolator(x, t)
    
    return torch.from_numpy(h.reshape(-1, 1).astype(np.float32))

