"""
Placeholder solver for problem1.
Uses deterministic random CUDA tensors as ground truth.
"""

import torch
from typing import Dict


def solve_ground_truth(x: torch.Tensor, t: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """
    Placeholder ground truth solver using deterministic random values.
    
    Args:
        x: Spatial coordinates (N, spatial_dim) on CUDA
        t: Temporal coordinates (N, 1) on CUDA
        seed: Random seed for deterministic output
        
    Returns:
        u: Solution tensor (N, out_dim=2) on CUDA
    """
    N = x.shape[0]
    device = x.device
    
    # Use deterministic random based on seed
    # Create a generator for reproducibility
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    
    # Placeholder: deterministic random output with 2 components
    u = torch.randn(N, 2, generator=generator, device=device)
    
    return u


def initial_condition(x: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """
    Placeholder initial condition: u(x, t=0).
    
    Args:
        x: Spatial coordinates (N, spatial_dim) on CUDA
        seed: Random seed for deterministic output
        
    Returns:
        u: Initial values (N, out_dim=2) on CUDA
    """
    N = x.shape[0]
    device = x.device
    
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 100)  # Different seed for IC
    
    # Placeholder: deterministic random IC
    u_ic = torch.randn(N, 2, generator=generator, device=device)
    
    return u_ic


def boundary_condition(x: torch.Tensor, t: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """
    Placeholder boundary condition: u(x_boundary, t).
    
    Args:
        x: Spatial coordinates at boundary (N, spatial_dim) on CUDA
        t: Temporal coordinates (N, 1) on CUDA
        seed: Random seed for deterministic output
        
    Returns:
        u: Boundary values (N, out_dim=2) on CUDA
    """
    N = x.shape[0]
    device = x.device
    
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 200)  # Different seed for BC
    
    # Placeholder: deterministic random BC
    u_bc = torch.randn(N, 2, generator=generator, device=device)
    
    return u_bc


def generate_dataset(
    n_residual: int,
    n_ic: int,
    n_bc: int,
    device: torch.device,
    config: Dict
) -> Dict[str, torch.Tensor]:
    """
    Generate dataset with residual, IC, and BC points.
    
    Args:
        n_residual: Number of residual (interior) points
        n_ic: Number of initial condition points
        n_bc: Number of boundary condition points
        device: Device to create tensors on (CUDA or CPU)
        config: Configuration dictionary with spatial_domain, temporal_domain, etc.
        
    Returns:
        Dictionary with keys:
            "x": (N, spatial_dim) spatial coordinates
            "t": (N, 1) temporal coordinates
            "u_gt": (N, out_dim) ground truth solution
            "mask": dict with "residual", "IC", "BC" boolean masks (N,)
        where N = n_residual + n_ic + n_bc
    """
    seed = config['seed']
    problem = config.get('problem', 'problem1')
    problem_config = config[problem]
    spatial_dim = problem_config['spatial_dim']
    spatial_domain = problem_config['spatial_domain']  # [[min, max], ...]
    temporal_domain = problem_config['temporal_domain']  # [min, max]
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    N = n_residual + n_ic + n_bc
    
    # Initialize tensors
    x = torch.zeros(N, spatial_dim, device=device)
    t = torch.zeros(N, 1, device=device)
    
    idx = 0
    
    # 1. Residual points (interior domain)
    for d in range(spatial_dim):
        x_min, x_max = spatial_domain[d]
        x[idx:idx + n_residual, d] = torch.rand(n_residual, device=device) * (x_max - x_min) + x_min
    t_min, t_max = temporal_domain
    t[idx:idx + n_residual, 0] = torch.rand(n_residual, device=device) * (t_max - t_min) + t_min
    idx += n_residual
    
    # 2. Initial condition points (t = t_min)
    for d in range(spatial_dim):
        x_min, x_max = spatial_domain[d]
        x[idx:idx + n_ic, d] = torch.rand(n_ic, device=device) * (x_max - x_min) + x_min
    t[idx:idx + n_ic, 0] = t_min
    idx += n_ic
    
    # 3. Boundary condition points (x at boundaries, t sampled)
    # For 1D: split n_bc between left and right boundaries
    # For higher D: this is a placeholder (needs problem-specific logic)
    if spatial_dim == 1:
        n_bc_left = n_bc // 2
        n_bc_right = n_bc - n_bc_left
        
        # Left boundary (x = x_min)
        x[idx:idx + n_bc_left, 0] = spatial_domain[0][0]
        t[idx:idx + n_bc_left, 0] = torch.rand(n_bc_left, device=device) * (t_max - t_min) + t_min
        idx += n_bc_left
        
        # Right boundary (x = x_max)
        x[idx:idx + n_bc_right, 0] = spatial_domain[0][1]
        t[idx:idx + n_bc_right, 0] = torch.rand(n_bc_right, device=device) * (t_max - t_min) + t_min
        idx += n_bc_right
    else:
        # For higher dimensions, uniform random on boundary (placeholder)
        for d in range(spatial_dim):
            x_min, x_max = spatial_domain[d]
            x[idx:idx + n_bc, d] = torch.rand(n_bc, device=device) * (x_max - x_min) + x_min
        t[idx:idx + n_bc, 0] = torch.rand(n_bc, device=device) * (t_max - t_min) + t_min
    
    # Create masks
    mask_residual = torch.zeros(N, dtype=torch.bool, device=device)
    mask_residual[:n_residual] = True
    
    mask_ic = torch.zeros(N, dtype=torch.bool, device=device)
    mask_ic[n_residual:n_residual + n_ic] = True
    
    mask_bc = torch.zeros(N, dtype=torch.bool, device=device)
    mask_bc[n_residual + n_ic:] = True
    
    # Generate ground truth using appropriate functions
    u_gt = torch.zeros(N, 2, device=device)
    
    # Residual points: use general solution
    u_gt[mask_residual] = solve_ground_truth(
        x[mask_residual], t[mask_residual], seed=seed
    )
    
    # Initial condition points: use IC function
    u_gt[mask_ic] = initial_condition(x[mask_ic], seed=seed)
    
    # Boundary condition points: use BC function
    u_gt[mask_bc] = boundary_condition(x[mask_bc], t[mask_bc], seed=seed)
    
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

