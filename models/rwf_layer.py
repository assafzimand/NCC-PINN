"""Random Weight Factorization (RWF) linear layer.

Factorizes the weight matrix as W_eff = diag(exp(s)) @ W, where s is a
trainable per-neuron log-scale vector. This parameterization ensures the
scale is always positive and multiplicative.

Reference: Wang et al. (2022) "Random Weight Factorization Improves the
Training of Continuous Neural Representations."
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RWFLinear(nn.Linear):
    """Linear layer with Random Weight Factorization.

    W_eff = diag(exp(log_scale)) @ weight
    output = x @ W_eff.T + bias

    Compared to nn.Linear:
    - Adds one extra scalar parameter per output neuron (log_scale).
    - log_scale initialized from N(0, 0.1) so exp(log_scale) ≈ 1 at start.
    - weight initialized with Kaiming uniform (same as nn.Linear default).
    - bias initialized to zero (same as nn.Linear default).

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If True, adds a learnable bias.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        self.log_scale = nn.Parameter(
            torch.empty(out_features).normal_(0.0, 0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = torch.exp(self.log_scale).unsqueeze(1) * self.weight
        return F.linear(x, W, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, rwf=True"
        )
