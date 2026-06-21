"""Random Weight Factorization (RWF) linear layer.

Factorizes the weight matrix as W_eff = diag(exp(s)) @ W, where s is a
trainable per-neuron log-scale vector. This parameterization ensures the
scale is always positive and multiplicative.

The base weight init is preserved: W is divided by exp(s) at init so that
W_eff = exp(s) * (W0/exp(s)) = W0, matching JaxPI's RWF implementation.

Reference: Wang et al. (2022) "Random Weight Factorization Improves the
Training of Continuous Neural Representations."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RWFLinear(nn.Linear):
    """Linear layer with Random Weight Factorization.

    W_eff = diag(exp(log_scale)) @ weight
    output = x @ W_eff.T + bias

    Matches JaxPI (Wang et al. 2024): log_scale ~ N(mean, std) with the base
    weight divided by exp(log_scale) at init so the effective weight equals
    the Kaiming/Glorot init at step 0.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If True, adds a learnable bias.
        mean: Mean of log_scale init distribution (default 1.0, JaxPI).
        std: Std of log_scale init distribution (default 0.1, JaxPI).
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, mean: float = 1.0, std: float = 0.1):
        super().__init__(in_features, out_features, bias=bias)
        self.log_scale = nn.Parameter(
            torch.empty(out_features).normal_(mean, std)
        )
        with torch.no_grad():
            scale = torch.exp(self.log_scale).unsqueeze(1)
            self.weight.data = self.weight.data / scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = torch.exp(self.log_scale).unsqueeze(1) * self.weight
        return F.linear(x, W, self.bias)

    def extra_repr(self) -> str:
        mean = self.log_scale.data.mean().item()
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, rwf=True, "
            f"log_scale_mean~{mean:.3f}"
        )
