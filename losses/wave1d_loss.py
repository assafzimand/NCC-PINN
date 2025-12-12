"""
Physics-Informed Loss Function for the 1D Wave Equation.

Implements the three-component loss:
    L = w_res*MSE_f + w_ic*MSE_0 + w_bc*MSE_b

where:
- MSE_f: PDE residual loss (h_tt - h_xx = 0)
- MSE_0: Initial condition loss (h(x,0) = sin(x), h_t(x,0) = 0)
- MSE_b: Boundary condition loss (h(±5,t) = 0, Dirichlet)
"""

import torch
import torch.nn as nn
from typing import Dict, Callable, Tuple


def compute_derivatives(
    h: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute derivatives of scalar field h using vectorized autograd.
    
    Computes ∂h/∂t, ∂²h/∂t², ∂h/∂x, and ∂²h/∂x² using PyTorch autograd.
    All operations stay on the same device (GPU if available).
    
    Args:
        h: Scalar field, shape (batch_size,)
        x: Spatial coordinates, shape (batch_size, 1), requires_grad=True
        t: Temporal coordinates, shape (batch_size, 1), requires_grad=True
        
    Returns:
        Tuple of (h_t, h_tt, h_x, h_xx):
        - h_t: ∂h/∂t
        - h_tt: ∂²h/∂t²
        - h_x: ∂h/∂x
        - h_xx: ∂²h/∂x²
    """
    
    # Create grad_outputs once
    ones_h = torch.ones_like(h)
    
    # Call 1: Compute h derivatives w.r.t. BOTH x and t in one call
    h_grads = torch.autograd.grad(
        outputs=h,
        inputs=[x, t],
        grad_outputs=ones_h,
        create_graph=True,
        retain_graph=True,
    )
    h_x = h_grads[0]
    h_t = h_grads[1]
    
    # Second derivatives
    ones_hx = torch.ones_like(h_x)
    ones_ht = torch.ones_like(h_t)
    
    # Call 2: h_xx
    h_xx = torch.autograd.grad(
        outputs=h_x,
        inputs=x,
        grad_outputs=ones_hx,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Call 3: h_tt
    h_tt = torch.autograd.grad(
        outputs=h_t,
        inputs=t,
        grad_outputs=ones_ht,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Squeeze to remove trailing dimensions
    h_t = h_t.squeeze(-1)
    h_tt = h_tt.squeeze(-1)
    h_x = h_x.squeeze(-1)
    h_xx = h_xx.squeeze(-1)
    
    return h_t, h_tt, h_x, h_xx


def pde_residual(
    h_tt: torch.Tensor,
    h_xx: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the PDE residual: h_tt - h_xx.
    
    For the wave equation: h_tt - h_xx = 0
    
    Args:
        h_tt: Second time derivative ∂²h/∂t²
        h_xx: Second spatial derivative ∂²h/∂x²
        
    Returns:
        Residual tensor
    """
    residual = h_tt - h_xx
    return residual


def build_loss(**cfg) -> Callable:
    """
    Build physics-informed loss function for the 1D wave equation.
    
    Args:
        **cfg: Configuration dictionary containing:
            - problem: problem name (e.g., 'wave1d')
            - wave1d: dict with 'loss_weights' (residual, ic, bc)
            
    Returns:
        Callable loss function that takes (model, batch) and returns
        scalar tensor
    """
    # Extract loss weights
    problem = cfg.get('problem', 'wave1d')
    problem_config = cfg.get(problem, {})
    loss_weights = problem_config.get('loss_weights', {})
    
    weight_residual = loss_weights.get('residual', 1.0)
    weight_ic = loss_weights.get('ic', 1.0)
    weight_bc = loss_weights.get('bc', 1.0)
    
    def loss_fn(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute physics-informed loss for 1D wave equation.
        
        Args:
            model: Neural network model (output_dim=1 for real-valued h)
            batch: Dictionary with keys:
                - 'x': (N, spatial_dim) spatial coordinates
                - 't': (N, 1) temporal coordinates
                - 'u_gt': (N, 1) ground truth (for IC/BC)
                - 'mask': dict with 'residual', 'IC', 'BC' boolean masks
                
        Returns:
            Scalar loss tensor
        """
        x = batch['x']  # (N, spatial_dim)
        t = batch['t']  # (N, 1)
        u_gt = batch['u_gt']  # (N, 1)
        masks = batch['mask']  # dict with boolean masks
        
        device = x.device
        
        # ============================================================
        # MSE_f: PDE Residual Loss
        # ============================================================
        if masks['residual'].sum() > 0:
            # Boolean indexing + .contiguous() for GPU efficiency
            x_f = x[masks['residual']].contiguous()  # (N_f, spatial_dim)
            t_f = t[masks['residual']].contiguous()  # (N_f, 1)
            
            # Enable gradients for autograd
            x_f = x_f.clone().detach().requires_grad_(True)
            t_f = t_f.clone().detach().requires_grad_(True)
            
            # Model prediction: concatenate x,t -> predict h
            xt_f = torch.cat([x_f, t_f], dim=1)
            h_pred = model(xt_f)  # (N_f, 1)
            
            # Extract h (squeeze output dimension for derivative computation)
            h_f = h_pred[:, 0]
            
            # Compute derivatives
            h_t, h_tt, h_x, h_xx = compute_derivatives(h_f, x_f, t_f)
            
            # Compute PDE residual: h_tt - h_xx
            residual = pde_residual(h_tt, h_xx)
            
            # MSE of residual
            mse_residual = torch.mean(residual ** 2)
        else:
            mse_residual = torch.tensor(0.0, device=device)
        
        # ============================================================
        # MSE_0: Initial Condition Loss
        # ============================================================
        if masks['IC'].sum() > 0:
            # Boolean indexing + .contiguous() for GPU efficiency
            x_0 = x[masks['IC']].contiguous()  # (N_0, spatial_dim)
            t_0 = t[masks['IC']].contiguous()  # (N_0, 1)
            u_gt_0 = u_gt[masks['IC']].contiguous()  # (N_0, 1)
            
            # Enable gradients for h_t computation
            x_0 = x_0.clone().detach().requires_grad_(True)
            t_0 = t_0.clone().detach().requires_grad_(True)
            
            # Model prediction
            xt_0 = torch.cat([x_0, t_0], dim=1)
            h_pred = model(xt_0)  # (N_0, 1)
            h_0 = h_pred[:, 0]
            
            # Compute h_t for velocity IC
            h_t, _, _, _ = compute_derivatives(h_0, x_0, t_0)
            
            # IC: h(x,0) = sin(x) and h_t(x,0) = 0
            # MSE for position: |h(x,0) - sin(x)|²
            mse_position = torch.mean((h_0 - u_gt_0[:, 0]) ** 2)
            
            # MSE for velocity: |h_t(x,0) - 0|²
            mse_velocity = torch.mean(h_t ** 2)
            
            mse_ic = mse_position + mse_velocity
        else:
            mse_ic = torch.tensor(0.0, device=device)
        
        # ============================================================
        # MSE_b: Boundary Condition Loss (Dirichlet: h(±5,t) = 0)
        # ============================================================
        if masks['BC'].sum() > 0:
            # Boolean indexing + .contiguous() for GPU efficiency
            x_b = x[masks['BC']].contiguous()  # (N_b, spatial_dim)
            t_b = t[masks['BC']].contiguous()  # (N_b, 1)
            
            # Model prediction
            xt_b = torch.cat([x_b, t_b], dim=1)
            h_pred = model(xt_b)  # (N_b, 1)
            
            # Dirichlet BC: h(boundary, t) = 0
            # MSE: |h(boundary, t) - 0|²
            mse_bc = torch.mean(h_pred ** 2)
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

