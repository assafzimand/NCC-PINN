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


def compute_derivatives_decomposed(
    components: list,
    x: torch.Tensor,
    t: torch.Tensor,
    need_ht: bool = True,
    need_hxx: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute derivatives of complex field h = u + iv via product rule on decomposed
    expert outputs, using BATCHED autograd for all experts simultaneously.
    
    Key optimization: each expert was evaluated with its own input copy (inputs_k),
    making their autograd graphs independent. This allows batching all K expert
    derivative computations into a SINGLE autograd.grad call per derivative type
    (4 calls total for residual, same as baseline) instead of K separate calls.
    
    For composed output u(x,t) = Σ_k ψ̃_k · u_k, the product rule gives:
        u_x  = Σ_k (ψ̃_k_x · u_k  +  ψ̃_k · u_k_x)
        u_t  = Σ_k (ψ̃_k_t · u_k  +  ψ̃_k · u_k_t)
        u_xx = Σ_k (ψ̃_k_xx · u_k  +  2·ψ̃_k_x · u_k_x  +  ψ̃_k · u_k_xx)
    
    Indicator derivatives (ψ̃ terms) are computed via autograd on the original
    shared inputs (x, t), then DETACHED since indicators have no learnable params.
    
    Args:
        components: list of dicts from model.forward_for_pde_derivatives(), each with:
            - 'u': (N, output_dim) model output, on autograd graph
            - 'inputs': (N, D) per-expert input copy (leaf, requires_grad=True)
            - 'psi_norm': (N, 1) normalized weight, on autograd graph
            - 'constant_psi': bool, True if weight is constant
        x: spatial coordinates (N, spatial_dim), requires_grad=True (original, for psi autograd)
        t: temporal coordinates (N, 1), requires_grad=True (original, for psi autograd)
        need_ht: compute ∂h/∂t (needed for PDE residual, not for BC)
        need_hxx: compute ∂²h/∂x² (needed for PDE residual, not for BC)
        
    Returns:
        Tuple of (h_t, h_x, h_xx) as complex tensors (squeeze to batch dim).
        h_t is None if need_ht=False, h_xx is None if need_hxx=False.
    """
    N = x.shape[0]
    device = x.device
    num_components = len(components)
    spatial_dim = x.shape[1]  # number of spatial columns in inputs (1 for Schrödinger)
    t_col = spatial_dim       # time column index in inputs_k
    
    # ====================================================================
    # BATCHED expert first derivatives (2 autograd calls for ALL experts)
    # ====================================================================
    # Collect per-expert inputs for batched autograd
    all_inputs = [c['inputs'] for c in components]
    
    # Sum all expert real/imag outputs into scalars
    total_real = sum(c['u'][:, 0].sum() for c in components)
    total_imag = sum(c['u'][:, 1].sum() for c in components)
    
    # Call 1: all real first derivatives in one shot
    # Returns tuple of (N, D) tensors, one per expert
    # grads[k][:, 0:spatial_dim] = du_k_real/dx, grads[k][:, t_col:] = du_k_real/dt
    real_grads = torch.autograd.grad(
        total_real, all_inputs, create_graph=True, retain_graph=True)
    
    # Call 2: all imag first derivatives in one shot
    imag_grads = torch.autograd.grad(
        total_imag, all_inputs, create_graph=True, retain_graph=True)
    
    # ====================================================================
    # BATCHED expert second derivatives (2 more calls if needed)
    # ====================================================================
    d2_real = None
    d2_imag = None
    if need_hxx:
        # Sum the spatial (x) components of first derivatives across all experts
        total_du_real_dx = sum(
            real_grads[k][:, 0:spatial_dim].sum() for k in range(num_components))
        total_du_imag_dx = sum(
            imag_grads[k][:, 0:spatial_dim].sum() for k in range(num_components))
        
        # Call 3: all real second spatial derivatives
        d2_real = torch.autograd.grad(
            total_du_real_dx, all_inputs, create_graph=True, retain_graph=True)
        
        # Call 4: all imag second spatial derivatives
        d2_imag = torch.autograd.grad(
            total_du_imag_dx, all_inputs, create_graph=True, retain_graph=True)
    
    # ====================================================================
    # Per-component: indicator autograd + product rule assembly
    # ====================================================================
    asm_x_real = torch.zeros(N, 1, device=device)
    asm_x_imag = torch.zeros(N, 1, device=device)
    asm_t_real = torch.zeros(N, 1, device=device) if need_ht else None
    asm_t_imag = torch.zeros(N, 1, device=device) if need_ht else None
    asm_xx_real = torch.zeros(N, 1, device=device) if need_hxx else None
    asm_xx_imag = torch.zeros(N, 1, device=device) if need_hxx else None
    
    for k, c in enumerate(components):
        u_k = c['u']
        psi_k = c['psi_norm']
        is_constant = c.get('constant_psi', False)
        
        u_k_real = u_k[:, 0:1]  # (N, 1)
        u_k_imag = u_k[:, 1:2]  # (N, 1)
        
        # Extract per-expert derivatives from batched results
        du_real_dx = real_grads[k][:, 0:spatial_dim]  # (N, spatial_dim)
        du_imag_dx = imag_grads[k][:, 0:spatial_dim]  # (N, spatial_dim)
        
        if need_ht:
            du_real_dt = real_grads[k][:, t_col:t_col+1]  # (N, 1)
            du_imag_dt = imag_grads[k][:, t_col:t_col+1]  # (N, 1)
        
        if need_hxx:
            d2u_real_dx2 = d2_real[k][:, 0:spatial_dim]  # (N, spatial_dim)
            d2u_imag_dx2 = d2_imag[k][:, 0:spatial_dim]  # (N, spatial_dim)
        
        # ============================================================
        # Indicator derivatives (autograd on ORIGINAL x, t — then detach)
        # ============================================================
        if is_constant:
            psi_k_d = psi_k.detach()
            dpsi_dx = torch.zeros(N, 1, device=device)
            if need_hxx:
                d2psi_dx2 = torch.zeros(N, 1, device=device)
            if need_ht:
                dpsi_dt = torch.zeros(N, 1, device=device)
        else:
            if need_hxx:
                dpsi_dx = torch.autograd.grad(
                    psi_k.sum(), x, create_graph=True, retain_graph=True)[0]
                d2psi_dx2 = torch.autograd.grad(
                    dpsi_dx.sum(), x, retain_graph=True)[0]
                dpsi_dx = dpsi_dx.detach()
                d2psi_dx2 = d2psi_dx2.detach()
            else:
                dpsi_dx = torch.autograd.grad(
                    psi_k.sum(), x, retain_graph=True)[0]
                dpsi_dx = dpsi_dx.detach()
            
            if need_ht:
                dpsi_dt = torch.autograd.grad(
                    psi_k.sum(), t, retain_graph=True)[0]
                dpsi_dt = dpsi_dt.detach()
            
            psi_k_d = psi_k.detach()
        
        # ============================================================
        # Product rule assembly
        # ============================================================
        asm_x_real = asm_x_real + dpsi_dx * u_k_real + psi_k_d * du_real_dx
        asm_x_imag = asm_x_imag + dpsi_dx * u_k_imag + psi_k_d * du_imag_dx
        
        if need_ht:
            asm_t_real = asm_t_real + dpsi_dt * u_k_real + psi_k_d * du_real_dt
            asm_t_imag = asm_t_imag + dpsi_dt * u_k_imag + psi_k_d * du_imag_dt
        
        if need_hxx:
            asm_xx_real = (asm_xx_real
                          + d2psi_dx2 * u_k_real
                          + 2 * dpsi_dx * du_real_dx
                          + psi_k_d * d2u_real_dx2)
            asm_xx_imag = (asm_xx_imag
                          + d2psi_dx2 * u_k_imag
                          + 2 * dpsi_dx * du_imag_dx
                          + psi_k_d * d2u_imag_dx2)
    
    # Pack as complex, squeeze last dim to match original compute_derivatives output
    h_x = torch.complex(asm_x_real, asm_x_imag).squeeze(-1)
    
    h_t = None
    if need_ht:
        h_t = torch.complex(asm_t_real, asm_t_imag).squeeze(-1)
    
    h_xx = None
    if need_hxx:
        h_xx = torch.complex(asm_xx_real, asm_xx_imag).squeeze(-1)
    
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
        
        # Check if model supports decomposed derivative computation (Step C)
        # Use decomposed approach when model is adaptive with active experts
        use_decomposed = (hasattr(model, 'forward_for_pde_derivatives')
                          and len(getattr(model, 'experts', [])) > 0)
        
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
            
            # Enable gradients for autograd
            x_f = x_f.clone().detach().requires_grad_(True)
            t_f = t_f.clone().detach().requires_grad_(True)
            
            # Model prediction: concatenate x,t -> predict (u,v)
            xt_f = torch.cat([x_f, t_f], dim=1)
            
            if use_decomposed:
                # === Decomposed approach (Step C): product rule on per-expert outputs ===
                if _t: _t.start('loss.residual.forward')
                decomposed = model.forward_for_pde_derivatives(xt_f)
                if _t: _t.stop('loss.residual.forward')
                
                composed = decomposed['composed']  # (N_f, 2)
                u_f = composed[:, 0]
                v_f = composed[:, 1]
                
                if _t: _t.start('loss.residual.derivatives')
                h_t, h_x, h_xx = compute_derivatives_decomposed(
                    decomposed['components'], x_f, t_f, need_ht=True, need_hxx=True)
                if _t: _t.stop('loss.residual.derivatives')
            else:
                # === Standard approach: differentiate composed output directly ===
                if _t: _t.start('loss.residual.forward')
                uv_f = model(xt_f)  # (N_f, 2)
                if _t: _t.stop('loss.residual.forward')
                
                u_f = uv_f[:, 0]
                v_f = uv_f[:, 1]
                
                if _t: _t.start('loss.residual.derivatives')
                h_t, h_x, h_xx = compute_derivatives(u_f, v_f, x_f, t_f)
                if _t: _t.stop('loss.residual.derivatives')
            
            # Compute PDE residual: i*h_t + 0.5*h_xx + |h|²*h
            # Need h for |h|² term
            h_f = torch.complex(u_f, v_f)
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
            
            if use_decomposed:
                # === Decomposed approach: product rule (only h_x needed for BC) ===
                if _t: _t.start('loss.bc.forward')
                decomposed_bc = model.forward_for_pde_derivatives(xt_stacked)
                if _t: _t.stop('loss.bc.forward')
                
                composed_bc = decomposed_bc['composed']
                u_stacked = composed_bc[:, 0]
                v_stacked = composed_bc[:, 1]
                
                if _t: _t.start('loss.bc.derivatives')
                _, h_x_stacked, _ = compute_derivatives_decomposed(
                    decomposed_bc['components'], x_stacked, t_stacked,
                    need_ht=False, need_hxx=False)
                if _t: _t.stop('loss.bc.derivatives')
            else:
                # === Standard approach ===
                if _t: _t.start('loss.bc.forward')
                uv_stacked = model(xt_stacked)
                if _t: _t.stop('loss.bc.forward')
                u_stacked = uv_stacked[:, 0]
                v_stacked = uv_stacked[:, 1]
                
                if _t: _t.start('loss.bc.derivatives')
                _, h_x_stacked, _ = compute_derivatives(u_stacked, v_stacked, x_stacked, t_stacked)
                if _t: _t.stop('loss.bc.derivatives')
            
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
