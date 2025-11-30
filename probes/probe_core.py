"""Core linear probing logic."""

import torch
from typing import Dict, Tuple
import numpy as np


def train_linear_probe(
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    ridge_lambda: float = 1e-6
) -> torch.nn.Linear:
    """
    Train a linear probe using closed-form least squares (ridge regression).
    
    This computes: weights = (X^T X + λI)^-1 X^T y
    
    Args:
        embeddings: Layer activations (N, hidden_dim)
        targets: Ground truth outputs (N, output_dim)
        ridge_lambda: Regularization parameter for stability
        
    Returns:
        Linear layer with trained weights
    """
    N, hidden_dim = embeddings.shape
    _, output_dim = targets.shape
    device = embeddings.device
    
    # Add bias term by appending ones column to embeddings
    X = torch.cat([embeddings, torch.ones(N, 1, device=device)], dim=1)  # (N, hidden_dim+1)
    y = targets  # (N, output_dim)
    
    # Compute X^T X + λI
    XtX = X.t() @ X  # (hidden_dim+1, hidden_dim+1)
    
    # Add ridge regularization
    ridge_term = ridge_lambda * torch.eye(XtX.shape[0], device=device)
    XtX_reg = XtX + ridge_term
    
    # Compute (X^T X + λI)^-1 X^T y
    try:
        # Use torch.linalg.solve for numerical stability
        weights_with_bias = torch.linalg.solve(XtX_reg, X.t() @ y)  # (hidden_dim+1, output_dim)
    except RuntimeError:
        # Fallback to pseudo-inverse if solve fails
        XtX_inv = torch.linalg.pinv(XtX_reg)
        weights_with_bias = XtX_inv @ X.t() @ y
    
    # Extract weights and bias
    weights = weights_with_bias[:-1, :]  # (hidden_dim, output_dim)
    bias = weights_with_bias[-1, :]  # (output_dim,)
    
    # Create Linear layer and set weights
    linear_probe = torch.nn.Linear(hidden_dim, output_dim, device=device)
    linear_probe.weight.data = weights.t()  # PyTorch expects (output_dim, hidden_dim)
    linear_probe.bias.data = bias
    
    return linear_probe


def compute_probe_predictions(
    probe: torch.nn.Linear,
    embeddings: torch.Tensor
) -> torch.Tensor:
    """
    Compute predictions using trained linear probe.
    
    Args:
        probe: Trained linear layer
        embeddings: Layer activations (N, hidden_dim)
        
    Returns:
        Predictions (N, output_dim)
    """
    with torch.no_grad():
        predictions = probe(embeddings)
    return predictions


def compute_probe_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    Compute probe prediction metrics.
    
    Args:
        predictions: Probe predictions (N, output_dim)
        targets: Ground truth outputs (N, output_dim)
        
    Returns:
        Dictionary with rel_l2 and inf_norm errors
    """
    from trainer.utils import compute_relative_l2_error, compute_infinity_norm_error
    
    rel_l2 = compute_relative_l2_error(predictions, targets).item()
    inf_norm = compute_infinity_norm_error(predictions, targets).item()
    
    return {
        'rel_l2': rel_l2,
        'inf_norm': inf_norm
    }


def probe_all_layers(
    embeddings_dict: Dict[str, torch.Tensor],
    train_targets: torch.Tensor,
    eval_embeddings_dict: Dict[str, torch.Tensor],
    eval_targets: torch.Tensor,
    device: torch.device
) -> Dict[str, Dict]:
    """
    Train linear probes for all hidden layers and evaluate them.
    
    Args:
        embeddings_dict: Training embeddings {layer_name: (N_train, hidden_dim)}
        train_targets: Training ground truth (N_train, output_dim)
        eval_embeddings_dict: Evaluation embeddings {layer_name: (N_eval, hidden_dim)}
        eval_targets: Evaluation ground truth (N_eval, output_dim)
        device: Device for computation
        
    Returns:
        Dictionary: {layer_name: {'train_metrics', 'eval_metrics', 'probe'}}
    """
    results = {}
    
    print("Training linear probes for each layer...")
    
    for layer_name in embeddings_dict.keys():
        train_embeddings = embeddings_dict[layer_name]
        eval_embeddings = eval_embeddings_dict[layer_name]
        
        print(f"  Layer {layer_name}: {train_embeddings.shape[1]} dimensions")
        
        # Train probe on training data
        probe = train_linear_probe(train_embeddings, train_targets)
        
        # Evaluate on training data
        train_predictions = compute_probe_predictions(probe, train_embeddings)
        train_metrics = compute_probe_metrics(train_predictions, train_targets)
        
        # Evaluate on evaluation data
        eval_predictions = compute_probe_predictions(probe, eval_embeddings)
        eval_metrics = compute_probe_metrics(eval_predictions, eval_targets)
        
        results[layer_name] = {
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics,
            'probe': probe,
            'hidden_dim': train_embeddings.shape[1]
        }
        
        print(f"    Train: Rel-L2={train_metrics['rel_l2']:.6f}, Inf={train_metrics['inf_norm']:.6f}")
        print(f"    Eval:  Rel-L2={eval_metrics['rel_l2']:.6f}, Inf={eval_metrics['inf_norm']:.6f}")
    
    return results

