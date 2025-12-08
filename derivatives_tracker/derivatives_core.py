"""Core derivatives computation via autograd and probes."""

import torch
from typing import Dict, Tuple
import numpy as np


def compute_ground_truth_derivatives(
    x: torch.Tensor,
    t: torch.Tensor,
    config: Dict
) -> Dict[str, np.ndarray]:
    """
    Compute ground truth derivatives using analytical solution and autograd.
    
    This computes h_gt, h_t_gt, h_xx_gt from the exact NLSE solution.
    
    Args:
        x: Spatial coordinates (N, 1) - will be set to requires_grad
        t: Temporal coordinates (N, 1) - will be set to requires_grad
        config: Configuration dict (to get interpolator)
        
    Returns:
        Dictionary with h_gt, h_t_gt, h_xx_gt as numpy arrays (N, 2)
    """
    from solvers.schrodinger_solver import _get_interpolator
    
    # Get analytical solution interpolator
    interpolator = _get_interpolator(config)
    
    # Get coordinates as numpy (detach first)
    x_np = x.detach().cpu().numpy().flatten()
    t_np = t.detach().cpu().numpy().flatten()
    
    # Get complex solution from interpolator
    h_complex = interpolator(x_np, t_np)  # (N,) complex
    u_gt = torch.from_numpy(h_complex.real).float()
    v_gt = torch.from_numpy(h_complex.imag).float()
    
    # We need to make u_gt and v_gt differentiable w.r.t. x and t
    # The issue is that the interpolator is not a PyTorch function
    # We need to compute derivatives on a dense grid instead
    
    # Alternative: Use finite differences on the analytical solution
    eps_x = 1e-6
    eps_t = 1e-6
    
    # Compute h_t using finite differences
    h_t_plus = interpolator(x_np, t_np + eps_t)
    h_t_minus = interpolator(x_np, t_np - eps_t)
    h_t_complex = (h_t_plus - h_t_minus) / (2 * eps_t)
    
    # Compute h_xx using finite differences
    h_x_plus = interpolator(x_np + eps_x, t_np)
    h_x_minus = interpolator(x_np - eps_x, t_np)
    h_xx_complex = (h_x_plus - 2*h_complex + h_x_minus) / (eps_x**2)
    
    # Convert to (N, 2) format
    h_gt = np.column_stack([h_complex.real, h_complex.imag])
    h_t_gt = np.column_stack([h_t_complex.real, h_t_complex.imag])
    h_xx_gt = np.column_stack([h_xx_complex.real, h_xx_complex.imag])
    
    return {
        'h_gt': h_gt,
        'h_t_gt': h_t_gt,
        'h_xx_gt': h_xx_gt
    }


def apply_linear_probe(embeddings: torch.Tensor, probe: torch.nn.Linear) -> torch.Tensor:
    """
    Apply a trained linear probe to embeddings.
    
    Args:
        embeddings: Layer activations (N, hidden_dim) WITH gradients
        probe: Trained linear probe (torch.nn.Linear)
        
    Returns:
        Predictions (N, 2) - treated as (u, v) for complex h = u + iv
    """
    return probe(embeddings)


def compute_layer_derivatives_via_probe(
    model: torch.nn.Module,
    layer_name: str,
    probe: torch.nn.Linear,
    x: torch.Tensor,
    t: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Compute derivatives of probe output w.r.t. inputs using autograd.
    
    This computes:
    - h = P_i(layer_i(...layer_1(x,t)))  where P_i is the probe
    - h_t = ∂h/∂t
    - h_xx = ∂²h/∂x²
    
    Args:
        model: Neural network model
        layer_name: Which layer to analyze (e.g., 'layer_1')
        probe: Trained linear probe for this layer
        x: Spatial coordinates (N, 1) - MUST have requires_grad=True
        t: Temporal coordinates (N, 1) - MUST have requires_grad=True
        
    Returns:
        Dictionary with:
            'h': Probe predictions (N, 2) - (u, v)
            'h_t': Temporal derivative (N, 2)
            'h_xx': Second spatial derivative (N, 2)
    """
    # Ensure gradients can flow
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    
    # Register hooks with keep_gradients=True
    handles = model.register_ncc_hooks([layer_name], keep_gradients=True)
    
    # Forward pass through model to get layer activations (with gradients)
    inputs = torch.cat([x, t], dim=1)
    _ = model(inputs)
    
    # Get embeddings from hook (they have gradients!)
    embeddings = model.activations[layer_name]
    
    # Apply probe to get h = (u, v)
    h = apply_linear_probe(embeddings, probe)  # (N, 2)
    
    N = h.shape[0]
    
    # Compute h_t: ∂h/∂t for both u and v
    h_t = torch.zeros_like(h)
    for i in range(2):  # u (i=0) and v (i=1)
        grad_outputs = torch.ones_like(h[:, i])
        h_t[:, i:i+1] = torch.autograd.grad(
            outputs=h[:, i],
            inputs=t,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
    
    # Compute h_x: ∂h/∂x (needed for second derivative)
    h_x = torch.zeros_like(h)
    for i in range(2):
        grad_outputs = torch.ones_like(h[:, i])
        h_x[:, i:i+1] = torch.autograd.grad(
            outputs=h[:, i],
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
    
    # Compute h_xx: ∂²h/∂x²
    h_xx = torch.zeros_like(h)
    for i in range(2):
        grad_outputs = torch.ones_like(h_x[:, i])
        h_xx[:, i:i+1] = torch.autograd.grad(
            outputs=h_x[:, i],
            inputs=x,
            grad_outputs=grad_outputs,
            retain_graph=(i == 0)  # Keep graph for first iteration
        )[0]
    
    # Remove hooks and detach results
    model.remove_hooks()
    
    return {
        'h': h.detach(),
        'h_t': h_t.detach(),
        'h_x': h_x.detach(),
        'h_xx': h_xx.detach()
    }


def compute_residual_terms(
    h: torch.Tensor,
    h_t: torch.Tensor,
    h_xx: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Compute residual terms using complex arithmetic.
    
    Given h = u + iv (stored as [u, v] in columns):
    - |h|² = u² + v²
    - |h|²h = (|h|²u, |h|²v)
    - Residual f = i*h_t + 0.5*h_xx + |h|²*h
      - f_u (real part) = -v_t + 0.5*u_xx + (u²+v²)*u
      - f_v (imag part) = u_t + 0.5*v_xx + (u²+v²)*v
    
    Args:
        h: Probe predictions (N, 2) where [:,0]=u, [:,1]=v
        h_t: Temporal derivatives (N, 2)
        h_xx: Second spatial derivatives (N, 2)
        
    Returns:
        Dictionary with:
            'nonlinear': |h|²*h term (N, 2)
            'residual': Full residual f (N, 2)
            'h_magnitude_sq': |h|² (N,) for reference
    """
    # Extract u and v
    u = h[:, 0:1]  # (N, 1)
    v = h[:, 1:2]  # (N, 1)
    
    u_t = h_t[:, 0:1]
    v_t = h_t[:, 1:2]
    
    u_xx = h_xx[:, 0:1]
    v_xx = h_xx[:, 1:2]
    
    # Compute |h|² = u² + v²
    h_mag_sq = u**2 + v**2  # (N, 1)
    
    # Compute nonlinear term: |h|²*h = (|h|²*u, |h|²*v)
    nonlinear_u = h_mag_sq * u
    nonlinear_v = h_mag_sq * v
    nonlinear = torch.cat([nonlinear_u, nonlinear_v], dim=1)  # (N, 2)
    
    # Compute residual using complex arithmetic
    # f = i*h_t + 0.5*h_xx + |h|²*h
    # Real part: f_u = -v_t + 0.5*u_xx + |h|²*u
    # Imag part: f_v = u_t + 0.5*v_xx + |h|²*v
    
    residual_u = -v_t + 0.5 * u_xx + nonlinear_u
    residual_v = u_t + 0.5 * v_xx + nonlinear_v
    residual = torch.cat([residual_u, residual_v], dim=1)  # (N, 2)
    
    return {
        'nonlinear': nonlinear,
        'residual': residual,
        'h_magnitude_sq': h_mag_sq.squeeze()  # (N,)
    }


def track_all_layers(
    model: torch.nn.Module,
    probes_dict: Dict[str, torch.nn.Linear],
    x: torch.Tensor,
    t: torch.Tensor,
    device: torch.device
) -> Dict[str, Dict]:
    """
    Track derivatives and residuals for all layers.
    
    Args:
        model: Trained neural network
        probes_dict: Dictionary mapping layer_name -> trained probe
        x: Spatial coordinates (N, 1)
        t: Temporal coordinates (N, 1)
        device: Device to run on
        
    Returns:
        Dictionary mapping layer_name -> {
            'h': (N, 2),
            'h_t': (N, 2),
            'h_x': (N, 2),
            'h_xx': (N, 2),
            'nonlinear': (N, 2),
            'residual': (N, 2),
            'h_magnitude_sq': (N,),
            'norms': {
                'h_norm': float,
                'h_t_norm': float,
                'h_x_norm': float,
                'h_xx_norm': float,
                'nonlinear_norm': float,
                'residual_norm': float,
                'residual_inf_norm': float
            }
        }
    """
    model.eval()
    x = x.to(device)
    t = t.to(device)
    
    results = {}
    
    for layer_name, probe in probes_dict.items():
        print(f"  Processing {layer_name}...")
        
        # Compute derivatives for this layer
        derivatives = compute_layer_derivatives_via_probe(
            model=model,
            layer_name=layer_name,
            probe=probe,
            x=x,
            t=t
        )
        
        h = derivatives['h']
        h_t = derivatives['h_t']
        h_x = derivatives['h_x']
        h_xx = derivatives['h_xx']
        
        # Compute residual terms
        residual_terms = compute_residual_terms(h, h_t, h_xx)
        
        # Compute L2 / L_inf norms across samples (mean per-sample norms)
        # For each term, compute ||term|| per sample, then take mean
        h_norms = torch.norm(h, dim=1)  # (N,)
        h_t_norms = torch.norm(h_t, dim=1)
        h_x_norms = torch.norm(h_x, dim=1)
        h_xx_norms = torch.norm(h_xx, dim=1)
        nonlinear_norms = torch.norm(residual_terms['nonlinear'], dim=1)
        residual_norms = torch.norm(residual_terms['residual'], dim=1)
        residual_inf = torch.max(torch.abs(residual_terms['residual']), dim=1).values
        
        norms = {
            'h_norm': h_norms.mean().item(),
            'h_t_norm': h_t_norms.mean().item(),
            'h_x_norm': h_x_norms.mean().item(),
            'h_xx_norm': h_xx_norms.mean().item(),
            'nonlinear_norm': nonlinear_norms.mean().item(),
            'residual_norm': residual_norms.mean().item(),
            'residual_inf_norm': residual_inf.mean().item()
        }
        
        # Store all results for this layer
        results[layer_name] = {
            'h': h.cpu().numpy(),
            'h_t': h_t.cpu().numpy(),
            'h_x': h_x.cpu().numpy(),
            'h_xx': h_xx.cpu().numpy(),
            'nonlinear': residual_terms['nonlinear'].cpu().numpy(),
            'residual': residual_terms['residual'].cpu().numpy(),
            'h_magnitude_sq': residual_terms['h_magnitude_sq'].cpu().numpy(),
            'norms': norms
        }
        
        print(f"    Residual L2 norm: {norms['residual_norm']:.6f} | "
              f"L_inf: {norms['residual_inf_norm']:.6f}")
    
    return results

