"""Training utility functions."""

import torch
from typing import Dict


def compute_relative_l2_error(
    u_pred: torch.Tensor,
    u_gt: torch.Tensor
) -> torch.Tensor:
    """
    Compute relative L2 error.

    Args:
        u_pred: Predicted values (N, output_dim)
        u_gt: Ground truth values (N, output_dim)

    Returns:
        Scalar relative L2 error: ||u_pred - u_gt||_2 / ||u_gt||_2
    """
    diff = u_pred - u_gt
    numerator = torch.norm(diff, p=2)
    denominator = torch.norm(u_gt, p=2) + 1e-10  # Avoid division by zero

    return numerator / denominator


def compute_infinity_norm_error(
    u_pred: torch.Tensor,
    u_gt: torch.Tensor
) -> torch.Tensor:
    """
    Compute infinity norm (max absolute) error.

    Args:
        u_pred: Predicted values (N, output_dim)
        u_gt: Ground truth values (N, output_dim)

    Returns:
        Scalar infinity norm error: max(|u_pred - u_gt|)
    """
    diff = u_pred - u_gt
    return torch.max(torch.abs(diff))
