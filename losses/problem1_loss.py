"""
Placeholder loss function for problem1.
Uses weighted MSE between model predictions and ground truth.
"""

import torch
import torch.nn as nn
from typing import Dict, Callable


def build_loss(**cfg) -> Callable:
    """
    Build a loss function for problem1.

    Placeholder implementation: weighted MSE over residual/IC/BC points.

    Args:
        **cfg: Configuration dictionary containing loss weights

    Returns:
        Callable loss function that takes (model, batch) and returns
        scalar CUDA tensor
    """
    # Extract loss weights with defaults
    problem = cfg.get('problem', 'problem1')
    problem_config = cfg.get(problem, {})
    loss_weights = problem_config.get('loss_weights', {})

    weight_residual = loss_weights.get('residual', 1.0)
    weight_ic = loss_weights.get('ic', 1.0)
    weight_bc = loss_weights.get('bc', 1.0)

    def loss_fn(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted MSE loss.

        Args:
            model: Neural network model
            batch: Dictionary with keys 'x', 't', 'u_gt', 'mask'

        Returns:
            Scalar loss tensor on same device as inputs
        """
        x = batch['x']  # (N, spatial_dim)
        t = batch['t']  # (N, 1)
        u_gt = batch['u_gt']  # (N, out_dim)
        masks = batch['mask']  # dict with 'residual', 'IC', 'BC'

        # Concatenate x and t as model input
        inputs = torch.cat([x, t], dim=1)  # (N, spatial_dim + 1)

        # Forward pass
        u_pred = model(inputs)  # (N, out_dim)

        # Compute MSE for each point type
        mse = (u_pred - u_gt) ** 2  # (N, out_dim)

        # Sum over output dimensions
        mse_per_point = mse.sum(dim=1)  # (N,)

        # Weighted loss for each point type
        loss_residual = (mse_per_point * masks['residual']).sum()
        loss_ic = (mse_per_point * masks['IC']).sum()
        loss_bc = (mse_per_point * masks['BC']).sum()

        # Normalize by number of points in each category
        n_residual = masks['residual'].sum() + 1e-8
        n_ic = masks['IC'].sum() + 1e-8
        n_bc = masks['BC'].sum() + 1e-8

        loss_residual = loss_residual / n_residual
        loss_ic = loss_ic / n_ic
        loss_bc = loss_bc / n_bc

        # Weighted sum
        total_loss = (
            weight_residual * loss_residual +
            weight_ic * loss_ic +
            weight_bc * loss_bc
        )

        return total_loss

    return loss_fn

