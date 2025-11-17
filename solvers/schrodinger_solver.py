"""
Schrödinger Equation Solver using Split-Step Fourier Method.

Solves: i*h_t + 0.5*h_xx + |h|^2*h = 0
Domain: x ∈ [-5, 5], t ∈ [0, π/2]
Initial Condition: h(x, 0) = 2*sech(x)
Boundary Conditions: Periodic
"""

import numpy as np
import torch
from typing import Dict, Tuple
from scipy.interpolate import RegularGridInterpolator


def initial_condition_analytical(x: np.ndarray) -> np.ndarray:
    """
    Analytical initial condition for NLSE: h(x, 0) = 2*sech(x).
    
    Args:
        x: Spatial coordinates (numpy array)
        
    Returns:
        h0: Complex initial condition (real-valued soliton)
    """
    h0 = 2.0 / np.cosh(x)
    return h0.astype(np.complex128)


def solve_nlse_splitstep(
    x_min: float = -5.0,
    x_max: float = 5.0,
    t_min: float = 0.0,
    t_max: float = np.pi / 2,
    nx: int = 1024,
    nt: int = 800,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the Nonlinear Schrödinger Equation using split-step Fourier method.
    
    Equation: i*h_t + 0.5*h_xx + |h|^2*h = 0
    
    The split-step method alternates between:
    - Linear step (dispersion): i*h_t + 0.5*h_xx = 0 (solved in Fourier space)
    - Nonlinear step: i*h_t + |h|^2*h = 0 (solved in real space)
    
    Args:
        x_min: Minimum spatial coordinate
        x_max: Maximum spatial coordinate
        t_min: Initial time (typically 0)
        t_max: Final time (typically π/2)
        nx: Number of spatial grid points (use power of 2 for FFT efficiency)
        nt: Number of temporal grid points
        
    Returns:
        x_grid: Spatial grid (nx,)
        t_grid: Temporal grid (nt,)
        h_solution: Complex solution field (nt, nx)
    """
    # Create spatial grid
    x_grid = np.linspace(x_min, x_max, nx, dtype=np.float64)
    dx = x_grid[1] - x_grid[0]
    
    # Create temporal grid
    t_grid = np.linspace(t_min, t_max, nt, dtype=np.float64)
    dt = t_grid[1] - t_grid[0]
    
    # Create wavenumber grid for Fourier space (periodic BC)
    k = 2.0 * np.pi * np.fft.fftfreq(nx, dx)
    
    # Initialize solution array
    h_solution = np.zeros((nt, nx), dtype=np.complex128)
    
    # Set initial condition
    h_solution[0, :] = initial_condition_analytical(x_grid)
    
    # Precompute linear evolution operator in Fourier space
    # Linear step: exp(-i * 0.5 * k^2 * dt)
    # From: i*h_t + 0.5*h_xx = 0 => h_t = i*0.5*h_xx
    # In Fourier space: h_t = -i*0.5*k^2*h_k
    # Solution: h_k(t+dt) = h_k(t) * exp(-i*0.5*k^2*dt)
    linear_operator = np.exp(-0.5j * k**2 * dt)
    
    # Time integration using split-step method
    h = h_solution[0, :].copy()
    
    for n in range(nt - 1):
        # Half-step nonlinear evolution in real space
        # From: i*h_t + |h|^2*h = 0 => h_t = i*|h|^2*h
        # Solution: h(t+dt/2) = h(t) * exp(i*|h|^2*dt/2)
        nonlinear_phase = 1j * np.abs(h)**2 * (dt / 2)
        h = h * np.exp(nonlinear_phase)
        
        # Full-step linear evolution in Fourier space
        h_k = np.fft.fft(h)
        h_k = h_k * linear_operator
        h = np.fft.ifft(h_k)
        
        # Half-step nonlinear evolution in real space
        nonlinear_phase = 1j * np.abs(h)**2 * (dt / 2)
        h = h * np.exp(nonlinear_phase)
        
        # Store solution
        h_solution[n + 1, :] = h
    
    return x_grid, t_grid, h_solution


class NLSEInterpolator:
    """
    Interpolator for NLSE solution on fine grid.
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
            h_solution: Complex solution (nt, nx)
        """
        # Create interpolators for real and imaginary parts
        self.real_interp = RegularGridInterpolator(
            (t_grid, x_grid),
            h_solution.real,
            method='cubic',
            bounds_error=False,
            fill_value=None
        )
        
        self.imag_interp = RegularGridInterpolator(
            (t_grid, x_grid),
            h_solution.imag,
            method='cubic',
            bounds_error=False,
            fill_value=None
        )
    
    def __call__(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Interpolate solution at given (x, t) points.
        
        Args:
            x: Spatial coordinates (N,)
            t: Temporal coordinates (N,)
            
        Returns:
            h: Complex solution values (N,)
        """
        points = np.column_stack([t, x])  # Note: (t, x) order for interpolator
        u = self.real_interp(points)
        v = self.imag_interp(points)
        return u + 1j * v


# Global interpolator (initialized on first dataset generation)
_interpolator = None


def _get_interpolator(config: Dict) -> NLSEInterpolator:
    """
    Get or create the NLSE interpolator (singleton pattern).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Interpolator instance
    """
    global _interpolator
    
    if _interpolator is None:
        print("  Generating NLSE ground truth solution (1024×800 grid)...")
        problem = config.get('problem', 'problem1')
        problem_config = config[problem]
        
        # Get domain from config
        spatial_domain = problem_config['spatial_domain'][0]  # [[x_min, x_max]]
        temporal_domain = problem_config['temporal_domain']  # [t_min, t_max]
        
        x_min, x_max = spatial_domain
        t_min, t_max = temporal_domain
        
        # Solve NLSE on fine grid
        x_grid, t_grid, h_solution = solve_nlse_splitstep(
            x_min=x_min,
            x_max=x_max,
            t_min=t_min,
            t_max=t_max,
            nx=1024,
            nt=800
        )
        
        print(f"  Solution computed: {h_solution.shape[0]}x{h_solution.shape[1]} grid")
        
        # Create interpolator
        _interpolator = NLSEInterpolator(x_grid, t_grid, h_solution)
        print("  Interpolator ready")
    
    return _interpolator


def solve_ground_truth(x: torch.Tensor, t: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """
    Ground truth solver using interpolated NLSE solution.
    
    Args:
        x: Spatial coordinates (N, spatial_dim) on CUDA
        t: Temporal coordinates (N, 1) on CUDA
        seed: Random seed (unused, kept for compatibility)
        
    Returns:
        u: Solution tensor (N, 2) where u[:, 0]=real, u[:, 1]=imag
    """
    # This function is called during dataset generation
    # Need to get config from somewhere - will be passed via generate_dataset
    # For now, return zeros as placeholder (actual interpolation in generate_dataset)
    N = x.shape[0]
    device = x.device
    return torch.zeros(N, 2, device=device)


def initial_condition(x: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """
    Initial condition: h(x, 0) = 2*sech(x).
    
    Args:
        x: Spatial coordinates (N, spatial_dim) on CUDA
        seed: Random seed (unused, kept for compatibility)
        
    Returns:
        u: Initial values (N, 2) where u[:, 0]=real, u[:, 1]=imag
    """
    # Move to CPU for numpy computation
    x_np = x.cpu().numpy()
    if x_np.ndim == 2:
        x_np = x_np[:, 0]  # Extract 1D coordinate
    
    # Compute h0 = 2*sech(x)
    h0 = initial_condition_analytical(x_np)
    
    # Convert to (real, imag) format
    u = np.zeros((len(x_np), 2), dtype=np.float32)
    u[:, 0] = h0.real
    u[:, 1] = h0.imag
    
    # Move back to original device
    return torch.from_numpy(u).to(x.device)


def boundary_condition(x: torch.Tensor, t: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """
    Boundary condition using interpolated solution.
    
    Args:
        x: Spatial coordinates at boundary (N, spatial_dim) on CUDA
        t: Temporal coordinates (N, 1) on CUDA
        seed: Random seed (unused, kept for compatibility)
        
    Returns:
        u: Boundary values (N, 2) where u[:, 0]=real, u[:, 1]=imag
    """
    # Placeholder - actual interpolation happens in generate_dataset
    N = x.shape[0]
    device = x.device
    return torch.zeros(N, 2, device=device)


def generate_dataset(
    n_residual: int,
    n_ic: int,
    n_bc: int,
    device: torch.device,
    config: Dict
) -> Dict[str, torch.Tensor]:
    """
    Generate dataset with NLSE ground truth via interpolation.
    
    Uses split-step Fourier solver on 1024×800 grid, then interpolates
    to randomly sampled training points.
    
    Args:
        n_residual: Number of residual (interior) points
        n_ic: Number of initial condition points
        n_bc: Number of boundary condition points (total, split between boundaries)
        device: Device to create tensors on (CUDA or CPU)
        config: Configuration dictionary
        
    Returns:
        Dictionary with keys:
            "x": (N, spatial_dim) spatial coordinates
            "t": (N, 1) temporal coordinates
            "u_gt": (N, 2) ground truth solution (real, imag)
            "mask": dict with "residual", "IC", "BC" boolean masks
    """
    seed = config['seed']
    problem = config.get('problem', 'problem1')
    problem_config = config[problem]
    spatial_dim = problem_config['spatial_dim']
    spatial_domain = problem_config['spatial_domain']  # [[min, max], ...]
    temporal_domain = problem_config['temporal_domain']  # [min, max]
    
    # Get or create interpolator (solves NLSE once)
    interpolator = _get_interpolator(config)
    
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
    
    # 3. Boundary condition points (paired at x=-5 and x=+5)
    print(f"  Sampling {n_bc} boundary condition points...")
    n_bc_left = n_bc // 2
    n_bc_right = n_bc - n_bc_left
    
    # Sample times (enough for both boundaries to enforce pairing)
    # Use max to handle odd n_bc
    n_times = max(n_bc_left, n_bc_right)
    t_bc = torch.rand(n_times, device=device) * (t_max - t_min) + t_min
    
    # Left boundary (x = x_min)
    x[idx:idx + n_bc_left, 0] = x_min
    t[idx:idx + n_bc_left, 0] = t_bc[:n_bc_left]
    idx += n_bc_left
    
    # Right boundary (x = x_max)
    x[idx:idx + n_bc_right, 0] = x_max
    t[idx:idx + n_bc_right, 0] = t_bc[:n_bc_right]  # Use same times
    idx += n_bc_right
    
    # Create masks
    mask_residual = torch.zeros(N, dtype=torch.bool, device=device)
    mask_residual[:n_residual] = True
    
    mask_ic = torch.zeros(N, dtype=torch.bool, device=device)
    mask_ic[n_residual:n_residual + n_ic] = True
    
    mask_bc = torch.zeros(N, dtype=torch.bool, device=device)
    mask_bc[n_residual + n_ic:] = True
    
    # Interpolate ground truth using NLSE solution
    print("  Interpolating ground truth values...")
    x_np = x.cpu().numpy()[:, 0]  # (N,)
    t_np = t.cpu().numpy()[:, 0]  # (N,)
    
    h_interp = interpolator(x_np, t_np)  # (N,) complex
    
    # Convert to (real, imag) format
    u_gt = torch.zeros(N, 2, device=device, dtype=torch.float32)
    u_gt[:, 0] = torch.from_numpy(h_interp.real.astype(np.float32)).to(device)
    u_gt[:, 1] = torch.from_numpy(h_interp.imag.astype(np.float32)).to(device)
    
    print("  Dataset generated successfully")
    
    return {
        "x": x,
        "t": t,
        "u_gt": u_gt,
        "mask": {
            "residual": mask_residual,
            "IC": mask_ic,
            "BC": mask_bc
        }
    }
