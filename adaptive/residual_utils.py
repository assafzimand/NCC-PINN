"""Utility functions for computing point-wise PDE residuals.

Used by the adaptive region detector to weight wavelet norms by error.
"""

import torch
import torch.nn as nn
from typing import Dict


def compute_pde_residuals(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    problem: str,
    config: Dict
) -> torch.Tensor:
    """
    Compute point-wise PDE residual magnitudes |L[u] - f| for region weighting.
    
    Args:
        model: The PINN model
        x: (N, spatial_dim) spatial coordinates
        t: (N, 1) temporal coordinates
        problem: Problem name ('schrodinger', 'burgers1d', 'burgers2d', 'wave1d')
        config: Configuration dictionary with problem-specific parameters
        
    Returns:
        (N,) tensor of residual magnitudes (always positive)
    """
    device = x.device
    
    # Enable gradients for computing derivatives
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    
    # Forward pass
    inputs = torch.cat([x, t], dim=1)
    output = model(inputs)
    
    if problem == 'schrodinger':
        return _schrodinger_residual(output, x, t)
    elif problem == 'burgers1d':
        nu = config.get('burgers1d', {}).get('nu', 0.01)
        return _burgers1d_residual(output, x, t, nu)
    elif problem == 'burgers2d':
        nu = config.get('burgers2d', {}).get('nu', 0.1)
        return _burgers2d_residual(output, x, t, nu)
    elif problem == 'wave1d':
        return _wave1d_residual(output, x, t)
    else:
        # Fallback: return ones (uniform weighting)
        return torch.ones(x.shape[0], device=device)


def _schrodinger_residual(output: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Schrödinger: i*h_t + 0.5*h_xx + |h|²*h = 0"""
    u = output[:, 0:1]  # Real part
    v = output[:, 1:2]  # Imaginary part
    
    # First derivatives
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v),
                              create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v),
                              create_graph=True, retain_graph=True)[0]
    
    # Second derivatives
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=False, retain_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x),
                               create_graph=False, retain_graph=False)[0]
    
    # |h|² = u² + v²
    h_sq = u**2 + v**2
    
    # PDE: i*h_t + 0.5*h_xx + |h|²*h = 0
    # Real part: -v_t + 0.5*u_xx + |h|²*u = 0
    # Imag part:  u_t + 0.5*v_xx + |h|²*v = 0
    residual_real = -v_t + 0.5 * u_xx + h_sq * u
    residual_imag = u_t + 0.5 * v_xx + h_sq * v
    
    # Magnitude: sqrt(real² + imag²)
    residual_mag = torch.sqrt(residual_real**2 + residual_imag**2 + 1e-8)
    
    return residual_mag.squeeze()


def _burgers1d_residual(output: torch.Tensor, x: torch.Tensor, t: torch.Tensor, nu: float) -> torch.Tensor:
    """Burgers 1D: u_t + u*u_x - nu*u_xx = 0"""
    u = output
    
    # First derivatives
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    
    # Second derivative
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=False, retain_graph=False)[0]
    
    # PDE: u_t + u*u_x - nu*u_xx = 0
    residual = u_t + u * u_x - nu * u_xx
    
    return torch.abs(residual).squeeze()


def _burgers2d_residual(output: torch.Tensor, x: torch.Tensor, t: torch.Tensor, nu: float) -> torch.Tensor:
    """Burgers 2D: u_t + u*(u_x + u_y) - nu*(u_xx + u_yy) = 0"""
    u = output
    
    # x has shape (N, 2) for 2D spatial domain
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    
    # First derivatives
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    
    # Spatial derivatives (need to handle 2D)
    grads = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                create_graph=True, retain_graph=True)[0]
    u_x1 = grads[:, 0:1]
    u_x2 = grads[:, 1:2]
    
    # Second derivatives
    u_x1x1 = torch.autograd.grad(u_x1, x, grad_outputs=torch.ones_like(u_x1),
                                  create_graph=False, retain_graph=True)[0][:, 0:1]
    u_x2x2 = torch.autograd.grad(u_x2, x, grad_outputs=torch.ones_like(u_x2),
                                  create_graph=False, retain_graph=False)[0][:, 1:2]
    
    # PDE: u_t + u*(u_x1 + u_x2) - nu*(u_x1x1 + u_x2x2) = 0
    residual = u_t + u * (u_x1 + u_x2) - nu * (u_x1x1 + u_x2x2)
    
    return torch.abs(residual).squeeze()


def _wave1d_residual(output: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Wave 1D: u_tt - u_xx = 0"""
    u = output
    
    # First derivatives
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              create_graph=True, retain_graph=True)[0]
    
    # Second derivatives
    u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t),
                               create_graph=False, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=False, retain_graph=False)[0]
    
    # PDE: u_tt - u_xx = 0
    residual = u_tt - u_xx
    
    return torch.abs(residual).squeeze()
