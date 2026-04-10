"""Fourier Feature Embedding for PINN models.

Maps input z ∈ ℝ^{input_dim} to [cos(Bz), sin(Bz)] ∈ ℝ^{2*fourier_dim}
using a fixed random projection matrix B ~ N(0, scale²).

The projection matrix B is stored as a non-trainable buffer so it:
- Moves to the correct device automatically with .to(device)
- Is excluded from optimizer parameter groups
- Is saved/loaded with model state_dict

Reference: Tancik et al. (2020) "Fourier Features Let Networks Learn High
Frequency Functions in Low Dimensional Domains."
"""

import torch
import torch.nn as nn


class FourierFeatureEmbedding(nn.Module):
    """Random Fourier Feature embedding with frozen projection matrix.

    Args:
        input_dim: Dimension of the input (e.g., spatial_dim + 1).
        fourier_dim: Number of Fourier features. Output size is 2*fourier_dim.
        scale: Standard deviation of B initialization. Controls frequency band.
               Higher scale → captures higher frequencies.
    """

    def __init__(self, input_dim: int, fourier_dim: int, scale: float = 1.0):
        super().__init__()
        B = torch.randn(fourier_dim, input_dim) * scale
        self.register_buffer('B', B)

    @property
    def output_dim(self) -> int:
        """Output dimensionality: 2 * fourier_dim."""
        return 2 * self.B.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map input to Fourier feature space.

        Args:
            x: Input tensor of shape (N, input_dim).

        Returns:
            Fourier features of shape (N, 2 * fourier_dim).
        """
        proj = x @ self.B.T  # (N, fourier_dim)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

    def extra_repr(self) -> str:
        fourier_dim, input_dim = self.B.shape
        return (
            f"input_dim={input_dim}, fourier_dim={fourier_dim}, "
            f"output_dim={self.output_dim}, scale=frozen"
        )
