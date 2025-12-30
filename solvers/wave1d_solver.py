"""
1D Wave Equation Solver with Analytical Standing Wave Solution.

Solves: h_tt - h_xx = 0
Domain: x ∈ [-5, 5], t ∈ [0, 2π]
Solution: h(x, t) = sin(x) * cos(t) (standing wave)
Initial Condition: h(x, 0) = sin(x), h_t(x, 0) = 0
Boundary Conditions: h(±5, t) = 0 (Dirichlet)
"""

import numpy as np
import torch
from typing import Tuple, Dict
from scipy.interpolate import RegularGridInterpolator


def analytical_solution(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Analytical solution for 1D wave: h(x, t) = sin(x) * cos(t).
    
    Args:
        x: Spatial coordinates (numpy array)
        t: Temporal coordinates (numpy array)
        
    Returns:
        h: Solution values (real-valued)
    """
    return np.sin(x) * np.cos(t)


def analytical_derivative_t(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Analytical time derivative: h_t(x, t) = -sin(x) * sin(t).
    
    Args:
        x: Spatial coordinates (numpy array)
        t: Temporal coordinates (numpy array)
        
    Returns:
        h_t: Time derivative (real-valued)
    """
    return -np.sin(x) * np.sin(t)


def analytical_derivative_tt(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Analytical second time derivative: h_tt(x, t) = -sin(x) * cos(t).
    
    Args:
        x: Spatial coordinates (numpy array)
        t: Temporal coordinates (numpy array)
        
    Returns:
        h_tt: Second time derivative (real-valued)
    """
    return -np.sin(x) * np.cos(t)


def analytical_derivative_x(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Analytical spatial derivative: h_x(x, t) = cos(x) * cos(t).
    
    Args:
        x: Spatial coordinates (numpy array)
        t: Temporal coordinates (numpy array)
        
    Returns:
        h_x: Spatial derivative (real-valued)
    """
    return np.cos(x) * np.cos(t)


def analytical_derivative_xx(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Analytical second spatial derivative: h_xx(x, t) = -sin(x) * cos(t).
    
    Args:
        x: Spatial coordinates (numpy array)
        t: Temporal coordinates (numpy array)
        
    Returns:
        h_xx: Second spatial derivative (real-valued)
    """
    return -np.sin(x) * np.cos(t)


def initial_condition_analytical(x: np.ndarray) -> np.ndarray:
    """
    Analytical initial condition for 1D wave: h(x, 0) = sin(x).
    
    Args:
        x: Spatial coordinates (numpy array)
        
    Returns:
        h0: Initial displacement (real-valued)
    """
    return np.sin(x)


def solve_wave1d_analytical(
    x_min: float = -5.0,
    x_max: float = 5.0,
    t_min: float = 0.0,
    t_max: float = 2 * np.pi,
    nx: int = 1024,
    nt: int = 800,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analytical solution for 1D wave equation with standing wave.
    
    Equation: h_tt - h_xx = 0
    Solution: h(x, t) = sin(x) * cos(t)
    
    This is an exact analytical solution satisfying:
    - PDE: h_tt = -sin(x)cos(t), h_xx = -sin(x)cos(t) => h_tt - h_xx = 0 ✓
    - IC: h(x, 0) = sin(x), h_t(x, 0) = 0 ✓
    - BC: h(±5, t) = sin(±5)cos(t) ≈ 0.96*cos(t) (approximately zero for large |x|)
    
    Args:
        x_min: Minimum spatial coordinate
        x_max: Maximum spatial coordinate
        t_min: Initial time (typically 0)
        t_max: Final time (typically 2π for one period)
        nx: Number of spatial grid points
        nt: Number of temporal grid points
        
    Returns:
        x_grid: Spatial grid (nx,)
        t_grid: Temporal grid (nt,)
        h_solution: Real solution field (nt, nx)
    """
    # Create grids
    x_grid = np.linspace(x_min, x_max, nx, dtype=np.float64)
    t_grid = np.linspace(t_min, t_max, nt, dtype=np.float64)
    
    # Create meshgrid for vectorized computation
    X, T = np.meshgrid(x_grid, t_grid)
    
    # Analytical standing wave solution
    h_solution = np.sin(X) * np.cos(T)
    
    return x_grid, t_grid, h_solution


class Wave1DInterpolator:
    """
    Interpolator for 1D wave equation solution.
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
            h_solution: Real solution (nt, nx)
        """
        # Create interpolator (real-valued, not complex)
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
            h_values: Solution at (x, t) points (N,) - real-valued
        """
        # Ensure 1D arrays
        x_flat = np.asarray(x_points).flatten()
        t_flat = np.asarray(t_points).flatten()
        
        # Stack for interpolator (expects (N, 2) with [t, x] order)
        points = np.column_stack([t_flat, x_flat])
        
        # Interpolate (returns real values)
        h_values = self.interpolator(points)
        
        return h_values


def _get_interpolator(config: Dict) -> Wave1DInterpolator:
    """
    Get wave1d solution interpolator (lazy cached).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Wave1DInterpolator instance
    """
    # Extract domain from config
    problem_config = config['wave1d']
    x_min, x_max = problem_config['spatial_domain'][0]
    t_min, t_max = problem_config['temporal_domain']
    
    # Solve wave equation on fine grid
    x_grid, t_grid, h_solution = solve_wave1d_analytical(
        x_min=x_min,
        x_max=x_max,
        t_min=t_min,
        t_max=t_max,
        nx=1024,
        nt=800
    )
    
    return Wave1DInterpolator(x_grid, t_grid, h_solution)


# Cache interpolator to avoid recomputation
_cached_interpolator = None
_cached_config_hash = None


def _get_interpolator_cached(config: Dict) -> Wave1DInterpolator:
    """Get interpolator with caching."""
    global _cached_interpolator, _cached_config_hash
    
    # Create hash from relevant config params
    problem_config = config['wave1d']
    config_tuple = (
        tuple(problem_config['spatial_domain'][0]),
        tuple(problem_config['temporal_domain'])
    )
    
    if _cached_interpolator is None or _cached_config_hash != config_tuple:
        print("  Generating wave1d solution (1024×800 grid)...")
        _cached_interpolator = _get_interpolator(config)
        _cached_config_hash = config_tuple
        print("  Solution computed: 800x1024 grid")
    
    return _cached_interpolator


def generate_dataset(
    n_residual: int,
    n_ic: int,
    n_bc: int,
    device: torch.device,
    config: Dict
) -> Dict[str, torch.Tensor]:
    """
    Generate dataset with wave1d ground truth via interpolation.
    
    Uses analytical standing wave solution on 1024×800 grid, then interpolates
    to randomly sampled training points.
    
    Args:
        n_residual: Number of residual (interior) points
        n_ic: Number of initial condition points
        n_bc: Number of boundary condition points (at x=±5)
        device: Device to create tensors on (CUDA or CPU)
        config: Configuration dictionary
        
    Returns:
        Dictionary with keys:
            "x": (N, spatial_dim) spatial coordinates
            "t": (N, 1) temporal coordinates
            "h_gt": (N, 1) ground truth solution (real-valued)
            "mask": dict with "residual", "IC", "BC" boolean masks
    """
    import torch
    
    seed = config['seed']
    problem = config.get('problem', 'wave1d')
    problem_config = config[problem]
    spatial_dim = problem_config['spatial_dim']
    spatial_domain = problem_config['spatial_domain']  # [[min, max], ...]
    temporal_domain = problem_config['temporal_domain']  # [min, max]
    
    # Get or create interpolator (solves wave1d once)
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
    
    # 3. Boundary condition points (at x=-5 and x=+5, Dirichlet)
    print(f"  Sampling {n_bc} boundary condition points...")
    n_bc_left = n_bc // 2
    n_bc_right = n_bc - n_bc_left
    
    # Sample times for boundaries
    n_times = max(n_bc_left, n_bc_right)
    t_bc = torch.rand(n_times, device=device) * (t_max - t_min) + t_min
    
    # Left boundary (x = x_min)
    x[idx:idx + n_bc_left, 0] = x_min
    t[idx:idx + n_bc_left, 0] = t_bc[:n_bc_left]
    idx += n_bc_left
    
    # Right boundary (x = x_max)
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
    
    # Interpolate ground truth using wave1d solution
    print("  Interpolating ground truth values...")
    x_np = x.cpu().numpy()[:, 0]  # (N,)
    t_np = t.cpu().numpy()[:, 0]  # (N,)
    
    h_interp = interpolator(x_np, t_np)  # (N,) real
    
    # Convert to (N, 1) format (real-valued)
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
    x_np = x_grid.cpu().numpy()
    
    # Extract coordinates
    x = x_np[:, 0]
    t = x_np[:, 1]
    
    # Evaluate analytical solution
    h = analytical_solution(x, t)
    
    return torch.from_numpy(h.reshape(-1, 1).astype(np.float32))

