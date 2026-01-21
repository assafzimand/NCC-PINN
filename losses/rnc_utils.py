"""
Residual Norm Control (RNC) utilities for PINN training.

This module provides the RNC penalty computation that can be used
by any problem-specific loss function.
"""

import torch
from typing import Dict, Optional, Tuple


def compute_rnc_penalty(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    probes: Dict[str, torch.nn.Linear],
    target_norms: Dict[str, Dict[str, float]],
    rnc_config: Dict,
    cfg: Dict
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute RNC penalty for all hidden layers (excluding last layer).
    
    The penalty encourages intermediate layer residual term norms to not exceed
    the last layer's target norms. Only penalizes when norm > target.
    
    Args:
        model: Neural network model
        batch: Batch dictionary with 'x', 't' keys
        probes: Dict mapping layer_name -> trained linear probe
        target_norms: Dict mapping layer_name -> {term_key -> target_norm_value}
        rnc_config: RNC configuration with keys:
            - base_weight: Weight for first layer
            - weight_growth_factor: Multiplier per layer (deeper = stronger)
        cfg: Full configuration dict (contains problem name, etc.)
        
    Returns:
        Tuple of (penalty_tensor, metrics_dict)
    """
    from derivatives_tracker.derivatives_core import compute_layer_derivatives_via_probe
    from derivatives_tracker.residuals import get_residual_module
    
    device = batch['x'].device
    penalty = torch.tensor(0.0, device=device)
    metrics = {}
    
    # Get hidden layer names (exclude the output layer which is last)
    hidden_layers = list(probes.keys())[:-1]
    
    if not hidden_layers:
        return penalty, metrics
    
    # Get RNC config parameters
    base_weight = rnc_config.get('base_weight', 0.1)
    weight_growth_factor = rnc_config.get('weight_growth_factor', 1.0)
    
    # Get problem-specific residual module
    problem_name = cfg.get('problem', 'schrodinger')
    residual_module = get_residual_module(problem_name)
    
    # Prepare inputs with gradients
    x = batch['x'].clone().detach().requires_grad_(True)
    t = batch['t'].clone().detach().requires_grad_(True)
    
    for i, layer_name in enumerate(hidden_layers):
        # Compute layer weight (deeper layers can have stronger penalty)
        layer_weight = base_weight * (weight_growth_factor ** i)
        
        # Compute derivatives for this layer (with gradients for backprop)
        derivs = compute_layer_derivatives_via_probe(
            model=model,
            layer_name=layer_name,
            probe=probes[layer_name],
            x=x,
            t=t,
            config=cfg,
            detach=False  # Keep gradients for training!
        )
        
        # Compute residual terms using problem-specific function
        residual_terms = residual_module.compute_residual_terms(**derivs)
        
        # Get target norms for this layer (use last layer's norms as targets)
        layer_targets = target_norms.get(layer_name, {})
        
        # Compute penalty for each term (excluding 'residual' itself)
        for term_key, term_tensor in residual_terms.items():
            if term_key in ['residual', 'h_magnitude_sq']:
                continue  # Skip total residual and helper terms
            
            # Compute L2 norm across samples
            if term_tensor.dim() == 1:
                norm = torch.abs(term_tensor).mean()
            else:
                norm = torch.norm(term_tensor, dim=1).mean()
            
            # Get target (default to current norm if not set yet)
            target = layer_targets.get(term_key, norm.detach().item())
            if isinstance(target, torch.Tensor):
                target = target.item()
            
            # One-sided penalty: only penalize if norm > target
            if norm > target:
                penalty = penalty + layer_weight * (norm - target) ** 2
            
            # Store metrics for logging
            metrics[f'{layer_name}_{term_key}_norm'] = norm.detach().item()
            metrics[f'{layer_name}_{term_key}_target'] = target
    
    metrics['rnc_penalty'] = penalty.detach().item()
    
    return penalty, metrics


def compute_target_norms(
    model: torch.nn.Module,
    data: Dict[str, torch.Tensor],
    probes: Dict[str, torch.nn.Linear],
    cfg: Dict
) -> Dict[str, Dict[str, float]]:
    """
    Compute target norms from the last hidden layer using all eval data.
    
    These targets are used by all hidden layers as reference norms.
    
    Args:
        model: Neural network model
        data: Eval dataset dict with 'x', 't' keys (uses all points)
        probes: Dict mapping layer_name -> trained linear probe
        cfg: Configuration dict
        
    Returns:
        Dict mapping layer_name -> {term_key -> target_norm_value}
    """
    from derivatives_tracker.derivatives_core import compute_layer_derivatives_via_probe
    from derivatives_tracker.residuals import get_residual_module
    
    device = data['x'].device
    
    # Get last hidden layer (second to last in probes, since last is output)
    layer_names = list(probes.keys())
    if len(layer_names) < 2:
        return {}
    
    last_hidden_layer = layer_names[-2]  # Second to last
    
    # Use all data points
    x = data['x']
    t = data['t']
    
    # Compute derivatives for last hidden layer (detached - just for target computation)
    with torch.no_grad():
        x_grad = x.clone().requires_grad_(True)
        t_grad = t.clone().requires_grad_(True)
    
    derivs = compute_layer_derivatives_via_probe(
        model=model,
        layer_name=last_hidden_layer,
        probe=probes[last_hidden_layer],
        x=x_grad,
        t=t_grad,
        config=cfg,
        detach=True  # Detach for target computation
    )
    
    # Compute residual terms
    problem_name = cfg.get('problem', 'schrodinger')
    residual_module = get_residual_module(problem_name)
    residual_terms = residual_module.compute_residual_terms(**derivs)
    
    # Compute norms for each term
    target_norms = {}
    term_norms = {}
    
    for term_key, term_tensor in residual_terms.items():
        if term_key in ['residual', 'h_magnitude_sq']:
            continue
        
        if term_tensor.dim() == 1:
            norm = torch.abs(term_tensor).mean().item()
        else:
            norm = torch.norm(term_tensor, dim=1).mean().item()
        
        term_norms[term_key] = norm
    
    # Set the same targets for all hidden layers (based on last hidden layer)
    for layer_name in layer_names[:-1]:  # All hidden layers
        target_norms[layer_name] = term_norms.copy()
    
    return target_norms


def train_probes_for_rnc(
    model: torch.nn.Module,
    train_data: Dict[str, torch.Tensor],
    cfg: Dict,
    device: torch.device
) -> Dict[str, torch.nn.Linear]:
    """
    Train linear probes for all hidden layers for RNC.
    
    Args:
        model: Neural network model
        train_data: Training data dict with 'x', 't', 'h_gt' keys (uses all points)
        cfg: Configuration dict
        device: Device for computation
        
    Returns:
        Dict mapping layer_name -> trained linear probe
    """
    from probes.probe_core import train_linear_probe
    
    model.eval()
    
    # Get all hidden layer names (exclude input and output layers)
    layer_names = model.get_layer_names()
    hidden_layers = layer_names[:-1]  # Exclude output layer
    
    # Use all training data
    x = train_data['x'].to(device)
    t = train_data['t'].to(device)
    h_gt = train_data['h_gt'].to(device)
    
    # Register hooks to capture all hidden layer activations
    handles = model.register_ncc_hooks(hidden_layers, keep_gradients=False)
    
    # Forward pass to get activations
    inputs = torch.cat([x, t], dim=1)
    with torch.no_grad():
        _ = model(inputs)
    
    # Train a probe for each hidden layer
    probes = {}
    for layer_name in hidden_layers:
        embeddings = model.activations[layer_name]
        probe = train_linear_probe(embeddings, h_gt)
        probes[layer_name] = probe
    
    # Clean up hooks
    model.remove_hooks()
    
    return probes
