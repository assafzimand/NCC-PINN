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
from torch.func import jvp
from typing import Dict, Callable, Tuple
import numpy as np


def compute_derivatives(
    model: torch.nn.Module,
    inputs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute derivatives using forward-mode AD (JVP) for efficiency.
    
    Forward-mode is optimal for low-dimensional inputs (x,t are 2D).
    Computes ∂h/∂t, ∂h/∂x, and ∂²h/∂x² using torch.func.jvp.
    Significantly faster than reverse-mode autograd.grad for this use case.
    
    Args:
        model: Neural network model that outputs (N, 2) for [u, v]
        inputs: (N, 2) tensor of [x, t] coordinates
        
    Returns:
        Tuple of (h, h_t, h_x, h_xx):
        - h: h = u + iv, complex tensor (N,)
        - h_t: ∂h/∂t, complex tensor (N,)
        - h_x: ∂h/∂x, complex tensor (N,)
        - h_xx: ∂²h/∂x², complex tensor (N,)
    """
    
    # inputs has shape (N, 2) for [x, t]
    # We need to enable gradients on inputs for the computation
    inputs = inputs.requires_grad_(True)
    
    # Define tangent vectors for directional derivatives
    # v_x: direction vector for ∂/∂x (first coordinate)
    # v_t: direction vector for ∂/∂t (second coordinate)
    v_x = torch.zeros_like(inputs)
    v_x[:, 0] = 1.0  # [1, 0]
    
    v_t = torch.zeros_like(inputs)
    v_t[:, 1] = 1.0  # [0, 1]
    
    # === First derivatives via JVP ===
    # Compute u, v and their first derivatives in one pass each
    
    # ∂(u,v)/∂x via JVP with tangent v_x
    uv, (du_dx, dv_dx) = jvp(
        lambda inp: (model(inp)[:, 0], model(inp)[:, 1]),
        (inputs,),
        (v_x,)
    )
    u, v = uv[0], uv[1]
    u_x = du_dx
    v_x = dv_dx
    
    # ∂(u,v)/∂t via JVP with tangent v_t
    _, (u_t, v_t) = jvp(
        lambda inp: (model(inp)[:, 0], model(inp)[:, 1]),
        (inputs,),
        (v_t,)
    )
    
    # === Second derivatives via reverse-mode ===
    # Use traditional autograd for u_xx, v_xx (simpler, avoids nested JVP issues)
    # The first derivatives u_x, v_x have requires_grad from JVP
    
    ones = torch.ones_like(u_x)
    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=inputs,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True
    )[0][:, 0]  # Only x-derivative (first column)
    
    v_x_rename = v_x  # Avoid name collision
    v_xx = torch.autograd.grad(
        outputs=v_x_rename,
        inputs=inputs,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True
    )[0][:, 0]  # Only x-derivative (first column)
    
    # Pack as complex tensors
    h = torch.complex(u, v)
    h_t = torch.complex(u_t, v_t)
    h_x = torch.complex(u_x, v_x_rename)
    h_xx = torch.complex(u_xx, v_xx)
    
    return h, h_t, h_x, h_xx


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
    
    def loss_fn(model: nn.Module, batch: Dict[str, torch.Tensor], 
                for_tree_spawning: bool = False):
        """
        Compute physics-informed loss for Schrödinger equation.
        
        Args:
            model: Neural network model (output_dim=2 for real, imag)
            batch: Dictionary with keys:
                - 'x': (N, spatial_dim) spatial coordinates
                - 't': (N, 1) temporal coordinates
                - 'h_gt': (N, 2) ground truth h = u + iv as (real, imag)
                - 'mask': dict with 'residual', 'IC', 'BC' boolean masks
            for_tree_spawning: If True, return per-sample loss components dict
                
        Returns:
            - If for_tree_spawning=False: Scalar total loss
            - If for_tree_spawning=True: Dict with keys 'residual', 'ic', 'bc'
              containing per-sample loss tensors (N,)
        """
        x = batch['x']  # (N, spatial_dim)
        t = batch['t']  # (N, 1)
        h_gt = batch['h_gt']  # (N, 2) as (real, imag)
        masks = batch['mask']  # dict with boolean masks
        
        N = x.shape[0]
        device = x.device
        
        # Timer (attached to model by trainer)
        _t = getattr(model, '_timer', None)
        
        # Initialize per-sample arrays if needed
        if for_tree_spawning:
            residual_per_sample = torch.zeros(N, device=device)
            ic_per_sample = torch.zeros(N, device=device)
            bc_per_sample = torch.zeros(N, device=device)
        
        # ============================================================
        # MSE_f: PDE Residual Loss
        # ============================================================
        if masks['residual'].sum() > 0:
            # Boolean indexing + .contiguous() for GPU efficiency
            x_f = x[masks['residual']].contiguous()  # (N_f, spatial_dim)
            t_f = t[masks['residual']].contiguous()  # (N_f, 1)
            
            # Concatenate x,t -> (N_f, 2) inputs for model
            xt_f = torch.cat([x_f, t_f], dim=1)
            
            # Compute derivatives using forward-mode AD (JVP)
            # This replaces: model forward + 4 autograd.grad calls
            # with: efficient forward-mode computation
            if _t: _t.start('loss.residual.derivatives')
            h_f, h_t, h_x, h_xx = compute_derivatives(model, xt_f)
            if _t: _t.stop('loss.residual.derivatives')
            
            # Compute PDE residual: i*h_t + 0.5*h_xx + |h|²*h
            residual = pde_residual(h_f, h_t, h_xx)
            
            # Per-sample squared residual
            residual_squared = residual.real ** 2 + residual.imag ** 2
            
            if for_tree_spawning:
                residual_per_sample[masks['residual']] = residual_squared
            else:
                mse_residual = torch.mean(residual_squared)
        else:
            if not for_tree_spawning:
                mse_residual = torch.tensor(0.0, device=device)
        
        # ============================================================
        # MSE_0: Initial Condition Loss
        # ============================================================
        if masks['IC'].sum() > 0:
            # Boolean indexing + .contiguous() for GPU efficiency
            x_0 = x[masks['IC']].contiguous()  # (N_0, spatial_dim)
            t_0 = t[masks['IC']].contiguous()  # (N_0, 1)
            h_gt_0 = h_gt[masks['IC']].contiguous()  # (N_0, 2)
            
            # Model prediction
            xt_0 = torch.cat([x_0, t_0], dim=1)
            if _t: _t.start('loss.ic.forward')
            uv_0 = model(xt_0)  # (N_0, 2)
            if _t: _t.stop('loss.ic.forward')
            
            # Convert to complex
            h_pred = torch.complex(uv_0[:, 0], uv_0[:, 1])
            h_true = torch.complex(h_gt_0[:, 0], h_gt_0[:, 1])
            
            # MSE: |h_pred - h_true|²
            diff = h_pred - h_true
            ic_squared = diff.real ** 2 + diff.imag ** 2
            
            if for_tree_spawning:
                ic_per_sample[masks['IC']] = ic_squared
            else:
                mse_ic = torch.mean(ic_squared)
        else:
            if not for_tree_spawning:
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
            
            # Concatenate into (N, 2) format for model
            xt_left = torch.cat([x_b_left, t_b_left], dim=1)
            xt_right = torch.cat([x_b_right, t_b_right], dim=1)
            xt_stacked = torch.cat([xt_left, xt_right], dim=0)
            
            # Compute h and h_x using forward-mode AD (JVP)
            if _t: _t.start('loss.bc.derivatives')
            h_stacked, _, h_x_stacked, _ = compute_derivatives(model, xt_stacked)
            if _t: _t.stop('loss.bc.derivatives')
            
            # Split predictions and derivatives
            h_left = h_stacked[:n_b_left]
            h_right = h_stacked[n_b_left:]
            h_x_left = h_x_stacked[:n_b_left]
            h_x_right = h_x_stacked[n_b_left:]
            
            # Periodic BC: h(-5,t) = h(5,t) and h_x(-5,t) = h_x(5,t)
            # Only compare paired points (min of left/right counts)
            n_pairs = min(len(h_left), len(h_right))
            
            if n_pairs == 0:
                # No paired BC points in this batch (e.g., batch has only 1 BC point)
                if not for_tree_spawning:
                    mse_bc = torch.tensor(0.0, device=device)
            else:
                diff_value = h_left[:n_pairs] - h_right[:n_pairs]
                bc_value_squared = diff_value.real ** 2 + diff_value.imag ** 2
                
                diff_derivative = h_x_left[:n_pairs] - h_x_right[:n_pairs]
                bc_deriv_squared = diff_derivative.real ** 2 + diff_derivative.imag ** 2
                
                bc_paired_loss = bc_value_squared + bc_deriv_squared
                
                if for_tree_spawning:
                    # Split loss equally between left and right points
                    bc_mask_indices = torch.where(masks['BC'])[0]
                    left_indices = bc_mask_indices[:n_b_left][:n_pairs]
                    right_indices = bc_mask_indices[n_b_left:][:n_pairs]
                    
                    bc_per_sample[left_indices] = bc_paired_loss / 2.0
                    bc_per_sample[right_indices] = bc_paired_loss / 2.0
                else:
                    mse_value = torch.mean(bc_value_squared)
                    mse_derivative = torch.mean(bc_deriv_squared)
                    mse_bc = mse_value + mse_derivative
        else:
            if not for_tree_spawning:
                mse_bc = torch.tensor(0.0, device=device)
        
        # ============================================================
        # Return
        # ============================================================
        if for_tree_spawning:
            return {
                'residual': residual_per_sample,  # (N,)
                'ic': ic_per_sample,              # (N,)
                'bc': bc_per_sample               # (N,)
            }
        else:
            total_loss = (
                weight_residual * mse_residual +
                weight_ic * mse_ic +
                weight_bc * mse_bc
            )
            return total_loss
    
    return loss_fn
