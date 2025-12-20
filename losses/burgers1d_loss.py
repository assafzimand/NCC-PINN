"""
Physics-Informed Loss Function for the 1D Viscous Burgers Equation.

Implements the three-component loss:
    L = w_res*MSE_f + w_ic*MSE_0 + w_bc*MSE_b

where:
- MSE_f: PDE residual loss (h_t + h*h_x - (nu/pi)*h_xx = 0)
- MSE_0: Initial condition loss (h(0,x) = -sin(pi*x))
- MSE_b: Boundary condition loss (h(t,-1) = h(t,1) = 0)
"""

import torch
import torch.nn as nn
from typing import Dict, Callable, Tuple
import numpy as np


def compute_derivatives(
    h: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute derivatives of scalar field h using vectorized autograd.
    
    Computes dh/dt, dh/dx, and d²h/dx² using PyTorch autograd.
    All operations stay on the same device (GPU if available).
    
    Args:
        h: Scalar field, shape (batch_size,)
        x: Spatial coordinates, shape (batch_size, 1), requires_grad=True
        t: Temporal coordinates, shape (batch_size, 1), requires_grad=True
        
    Returns:
        Tuple of (h_t, h_x, h_xx):
        - h_t: dh/dt
        - h_x: dh/dx
        - h_xx: d²h/dx²
    """
    # Create grad_outputs once
    ones_h = torch.ones_like(h)
    
    # Compute h derivatives w.r.t. BOTH x and t in one call
    h_grads = torch.autograd.grad(
        outputs=h,
        inputs=[x, t],
        grad_outputs=ones_h,
        create_graph=True,
        retain_graph=True,
    )
    h_x = h_grads[0]
    h_t = h_grads[1]
    
    # Second derivative w.r.t. x
    ones_hx = torch.ones_like(h_x)
    
    h_xx = torch.autograd.grad(
        outputs=h_x,
        inputs=x,
        grad_outputs=ones_hx,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Squeeze to remove trailing dimensions
    h_t = h_t.squeeze(-1)
    h_x = h_x.squeeze(-1)
    h_xx = h_xx.squeeze(-1)
    
    return h_t, h_x, h_xx


def pde_residual(
    h: torch.Tensor,
    h_t: torch.Tensor,
    h_x: torch.Tensor,
    h_xx: torch.Tensor,
    nu: float = 0.01,
) -> torch.Tensor:
    """
    Compute the PDE residual: h_t + h*h_x - (nu/pi)*h_xx.
    
    For the viscous Burgers equation: h_t + h*h_x - (nu/pi)*h_xx = 0
    
    Args:
        h: Solution field h
        h_t: Time derivative dh/dt
        h_x: Spatial derivative dh/dx
        h_xx: Second spatial derivative d²h/dx²
        nu: Viscosity coefficient (default 0.01)
        
    Returns:
        Residual tensor
    """
    visc = nu / np.pi
    residual = h_t + h * h_x - visc * h_xx
    return residual


def build_loss(**cfg) -> Callable:
    """
    Build physics-informed loss function for the 1D viscous Burgers equation.
    
    Args:
        **cfg: Configuration dictionary containing:
            - problem: problem name (e.g., 'burgers1d')
            - burgers1d: dict with 'loss_weights' (residual, ic, bc) and 'nu'
            
    Returns:
        Callable loss function that takes (model, batch) and returns
        scalar tensor
    """
    # Extract loss weights and parameters
    problem = cfg.get('problem', 'burgers1d')
    problem_config = cfg.get(problem, {})
    loss_weights = problem_config.get('loss_weights', {})
    
    weight_residual = loss_weights.get('residual', 1.0)
    weight_ic = loss_weights.get('ic', 1.0)
    weight_bc = loss_weights.get('bc', 1.0)
    
    # Get viscosity parameter
    nu = problem_config.get('nu', 0.01)
    
    def loss_fn(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute physics-informed loss for 1D viscous Burgers equation.
        
        Args:
            model: Neural network model (output_dim=1 for real-valued h)
            batch: Dictionary with keys:
                - 'x': (N, spatial_dim) spatial coordinates
                - 't': (N, 1) temporal coordinates
                - 'h_gt': (N, 1) ground truth (for IC)
                - 'mask': dict with 'residual', 'IC', 'BC' boolean masks
                
        Returns:
            Scalar loss tensor
        """
        x = batch['x']  # (N, spatial_dim)
        t = batch['t']  # (N, 1)
        h_gt = batch['h_gt']  # (N, 1)
        masks = batch['mask']  # dict with boolean masks
        
        device = x.device
        
        # ============================================================
        # MSE_f: PDE Residual Loss
        # ============================================================
        if masks['residual'].sum() > 0:
            # Boolean indexing + .contiguous() for GPU efficiency
            x_f = x[masks['residual']].contiguous()
            t_f = t[masks['residual']].contiguous()
            
            # Enable gradients for autograd
            x_f = x_f.clone().detach().requires_grad_(True)
            t_f = t_f.clone().detach().requires_grad_(True)
            
            # Model prediction: concatenate x,t -> predict h
            xt_f = torch.cat([x_f, t_f], dim=1)
            h_pred = model(xt_f)  # (N_f, 1)
            
            # Extract h (squeeze output dimension for derivative computation)
            h_f = h_pred[:, 0]
            
            # Compute derivatives
            h_t, h_x, h_xx = compute_derivatives(h_f, x_f, t_f)
            
            # Compute PDE residual: h_t + h*h_x - (nu/pi)*h_xx
            residual = pde_residual(h_f, h_t, h_x, h_xx, nu=nu)
            
            # MSE of residual
            mse_residual = torch.mean(residual ** 2)
        else:
            mse_residual = torch.tensor(0.0, device=device)
        
        # ============================================================
        # MSE_0: Initial Condition Loss
        # h(0, x) = -sin(pi*x)
        # ============================================================
        if masks['IC'].sum() > 0:
            x_0 = x[masks['IC']].contiguous()
            t_0 = t[masks['IC']].contiguous()
            h_gt_0 = h_gt[masks['IC']].contiguous()  # (N_0, 1)
            
            # Model prediction
            xt_0 = torch.cat([x_0, t_0], dim=1)
            h_pred_0 = model(xt_0)  # (N_0, 1)
            
            # IC: h(0, x) = -sin(pi*x)
            # Ground truth is stored in h_gt_0
            mse_ic = torch.mean((h_pred_0 - h_gt_0) ** 2)
        else:
            mse_ic = torch.tensor(0.0, device=device)
        
        # ============================================================
        # MSE_b: Boundary Condition Loss
        # h(t, -1) = h(t, 1) = 0 (Dirichlet)
        # ============================================================
        if masks['BC'].sum() > 0:
            x_b = x[masks['BC']].contiguous()
            t_b = t[masks['BC']].contiguous()
            
            # Model prediction
            xt_b = torch.cat([x_b, t_b], dim=1)
            h_pred_b = model(xt_b)  # (N_b, 1)
            
            # BC: h should be 0 at boundaries
            mse_bc = torch.mean(h_pred_b ** 2)
        else:
            mse_bc = torch.tensor(0.0, device=device)
        
        # ============================================================
        # Total Weighted Loss
        # ============================================================
        total_loss = (
            weight_residual * mse_residual +
            weight_ic * mse_ic +
            weight_bc * mse_bc
        )
        
        return total_loss
    
    return loss_fn

