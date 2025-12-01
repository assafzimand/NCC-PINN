"""
Physics-Informed Loss Function for the Schrödinger Equation.

Implements the three-component loss:
    L = w_res*MSE_f + w_ic*MSE_0 + w_bc*MSE_b

where:
- MSE_f: PDE residual loss (i*h_t + 0.5*h_xx + |h|²*h = 0)
- MSE_0: Initial condition loss (h(x,0) = 2*sech(x))
- MSE_b: Boundary condition loss (periodic BC)
"""

import torch
import torch.nn as nn
from typing import Dict, Callable, Tuple
import numpy as np


def compute_derivatives(
    u: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute derivatives of complex field h = u + iv using vectorized autograd.
    
    Optimized to use 4 autograd calls (batched) for efficiency.
    Computes ∂h/∂t, ∂h/∂x, and ∂²h/∂x² using PyTorch autograd.
    All operations stay on the same device (GPU if available).
    
    Args:
        u: Real part of field, shape (batch_size,)
        v: Imaginary part of field, shape (batch_size,)
        x: Spatial coordinates, shape (batch_size, 1), requires_grad=True
        t: Temporal coordinates, shape (batch_size, 1), requires_grad=True
        
    Returns:
        Tuple of (h_t, h_x, h_xx):
        - h_t: ∂h/∂t, complex tensor
        - h_x: ∂h/∂x, complex tensor
        - h_xx: ∂²h/∂x², complex tensor
    """
    
    # Create grad_outputs once
    ones_u = torch.ones_like(u)
    ones_v = torch.ones_like(v)
    
    # Call 1: Compute u derivatives w.r.t. BOTH x and t in one call
    u_grads = torch.autograd.grad(
        outputs=u,
        inputs=[x, t],
        grad_outputs=ones_u,
        create_graph=True,
        retain_graph=True,
    )
    u_x = u_grads[0]
    u_t = u_grads[1]
    
    # Call 2: Compute v derivatives w.r.t. BOTH x and t in one call
    v_grads = torch.autograd.grad(
        outputs=v,
        inputs=[x, t],
        grad_outputs=ones_v,
        create_graph=True,
        retain_graph=True,
    )
    v_x = v_grads[0]
    v_t = v_grads[1]
    
    # Second derivatives w.r.t space
    ones_ux = torch.ones_like(u_x)
    ones_vx = torch.ones_like(v_x)
    
    # Call 3: u_xx
    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=x,
        grad_outputs=ones_ux,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Call 4: v_xx
    v_xx = torch.autograd.grad(
        outputs=v_x,
        inputs=x,
        grad_outputs=ones_vx,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Pack as complex (stays on device)
    # Use squeeze(-1) to only remove last dimension, preserve batch dimension
    h_t = torch.complex(u_t, v_t).squeeze(-1)
    h_x = torch.complex(u_x, v_x).squeeze(-1)
    h_xx = torch.complex(u_xx, v_xx).squeeze(-1)
    
    return h_t, h_x, h_xx


def pde_residual(
    h: torch.Tensor,
    h_t: torch.Tensor,
    h_xx: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the PDE residual: i*h_t + 0.5*h_xx + |h|²*h.
    
    For the Schrödinger equation: i*h_t + 0.5*h_xx + |h|²*h = 0
    
    Args:
        h: Complex field
        h_t: Time derivative ∂h/∂t
        h_xx: Second spatial derivative ∂²h/∂x²
        
    Returns:
        Complex residual tensor
    """
    # r = i*h_t + 0.5*h_xx + |h|²*h
    residual = 1j * h_t + 0.5 * h_xx + (h.abs() ** 2) * h
    return residual


def build_loss(**cfg) -> Callable:
    """
    Build physics-informed loss function for the Schrödinger equation.
    
    Args:
        **cfg: Configuration dictionary containing:
            - problem: problem name (e.g., 'problem1')
            - problem1: dict with 'loss_weights' (residual, ic, bc)
            
    Returns:
        Callable loss function that takes (model, batch) and returns
        scalar CUDA tensor
    """
    # Extract loss weights
    problem = cfg.get('problem', 'problem1')
    problem_config = cfg.get(problem, {})
    loss_weights = problem_config.get('loss_weights', {})
    
    weight_residual = loss_weights.get('residual', 1.0)
    weight_ic = loss_weights.get('ic', 1.0)
    weight_bc = loss_weights.get('bc', 1.0)
    
    def loss_fn(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute physics-informed loss for Schrödinger equation.
        
        Args:
            model: Neural network model (output_dim=2 for real, imag)
            batch: Dictionary with keys:
                - 'x': (N, spatial_dim) spatial coordinates
                - 't': (N, 1) temporal coordinates
                - 'u_gt': (N, 2) ground truth (for IC/BC)
                - 'mask': dict with 'residual', 'IC', 'BC' boolean masks
                
        Returns:
            Scalar loss tensor
        """
        x = batch['x']  # (N, spatial_dim)
        t = batch['t']  # (N, 1)
        u_gt = batch['u_gt']  # (N, 2)
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
            
            # Model prediction: concatenate x,t -> predict (u,v)
            xt_f = torch.cat([x_f, t_f], dim=1)
            uv_f = model(xt_f)  # (N_f, 2)
            
            # Extract u and v (these should have grad_fn from the model)
            u_f = uv_f[:, 0]
            v_f = uv_f[:, 1]
            
            
            # Compute derivatives
            h_t, h_x, h_xx = compute_derivatives(u_f, v_f, x_f, t_f)
            
            # Compute PDE residual: i*h_t + 0.5*h_xx + |h|²*h
            # Need h for |h|² term
            h_f = torch.complex(u_f, v_f)
            residual = pde_residual(h_f, h_t, h_xx)
            
            # MSE of residual magnitude: |r|² = r.real² + r.imag²
            mse_residual = torch.mean(residual.real ** 2 + residual.imag ** 2)
        else:
            mse_residual = torch.tensor(0.0, device=device)
        
        # ============================================================
        # MSE_0: Initial Condition Loss
        # ============================================================
        if masks['IC'].sum() > 0:
            # Boolean indexing + .contiguous() for GPU efficiency
            x_0 = x[masks['IC']].contiguous()  # (N_0, spatial_dim)
            t_0 = t[masks['IC']].contiguous()  # (N_0, 1)
            u_gt_0 = u_gt[masks['IC']].contiguous()  # (N_0, 2)
            
            # Model prediction
            xt_0 = torch.cat([x_0, t_0], dim=1)
            uv_0 = model(xt_0)  # (N_0, 2)
            
            # Convert to complex
            h_pred = torch.complex(uv_0[:, 0], uv_0[:, 1])
            h_true = torch.complex(u_gt_0[:, 0], u_gt_0[:, 1])
            
            # MSE: |h_pred - h_true|²
            diff = h_pred - h_true
            mse_ic = torch.mean(diff.real ** 2 + diff.imag ** 2)
        else:
            mse_ic = torch.tensor(0.0, device=device)
        
        # ============================================================
        # MSE_b: Boundary Condition Loss (Periodic)
        # ============================================================
        if masks['BC'].sum() > 0:
            # Boolean indexing + .contiguous() for GPU efficiency
            x_b = x[masks['BC']].contiguous()  # (N_b, spatial_dim)
            t_b = t[masks['BC']].contiguous()  # (N_b, 1)
            
            # Split into left and right boundaries
            # Assumption: first half are left (x=-5), second half are right (x=+5)
            n_b_total = masks['BC'].sum().item()
            n_b_left = n_b_total // 2
            
            x_b_left = x_b[:n_b_left]
            t_b_left = t_b[:n_b_left]
            x_b_right = x_b[n_b_left:]
            t_b_right = t_b[n_b_left:]
            
            # Enable gradients for derivative computation
            x_b_left = x_b_left.clone().detach().requires_grad_(True)
            t_b_left = t_b_left.clone().detach().requires_grad_(True)
            x_b_right = x_b_right.clone().detach().requires_grad_(True)
            t_b_right = t_b_right.clone().detach().requires_grad_(True)
            
            # Vectorized: stack left and right, then split back
            x_stacked = torch.cat([x_b_left, x_b_right], dim=0)
            t_stacked = torch.cat([t_b_left, t_b_right], dim=0)
            
            # Single forward pass for both boundaries
            xt_stacked = torch.cat([x_stacked, t_stacked], dim=1)
            uv_stacked = model(xt_stacked)
            u_stacked = uv_stacked[:, 0]
            v_stacked = uv_stacked[:, 1]
            
            # Compute spatial derivatives at boundaries
            _, h_x_stacked, _ = compute_derivatives(u_stacked, v_stacked, x_stacked, t_stacked)
            
            # Split predictions and derivatives (use actual sizes, not torch.chunk)
            u_left = u_stacked[:n_b_left]
            u_right = u_stacked[n_b_left:]
            v_left = v_stacked[:n_b_left]
            v_right = v_stacked[n_b_left:]
            h_x_left = h_x_stacked[:n_b_left]
            h_x_right = h_x_stacked[n_b_left:]
            
            # Convert to complex for comparison
            h_left = torch.complex(u_left, v_left)
            h_right = torch.complex(u_right, v_right)
            
            # Periodic BC: h(-5,t) = h(5,t) and h_x(-5,t) = h_x(5,t)
            # Only compare paired points (min of left/right counts)
            n_pairs = min(len(h_left), len(h_right))
            
            if n_pairs == 0:
                # No paired BC points in this batch (e.g., batch has only 1 BC point)
                mse_bc = torch.tensor(0.0, device=device)
            else:
                diff_value = h_left[:n_pairs] - h_right[:n_pairs]
                mse_value = torch.mean(diff_value.real ** 2 + diff_value.imag ** 2)
                
                diff_derivative = h_x_left[:n_pairs] - h_x_right[:n_pairs]
                mse_derivative = torch.mean(diff_derivative.real ** 2 + diff_derivative.imag ** 2)
                
                mse_bc = mse_value + mse_derivative
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
