"""Fourier Feature Embedding for PINN models.

Maps input z ∈ ℝ^{input_dim} to [cos(Bz), sin(Bz)] ∈ ℝ^{2*fourier_dim}
using a fixed random projection matrix B ~ N(0, scale²).

The projection matrix B is stored as a non-trainable buffer so it:
- Moves to the correct device automatically with .to(device)
- Is excluded from optimizer parameter groups
- Is saved/loaded with model state_dict

Also provides PeriodicSpatialFourierEmbedding for problems with periodic
spatial BCs (e.g., KS). Enabled via fourier_features.periodic: true in config.

Reference: Tancik et al. (2020) "Fourier Features Let Networks Learn High
Frequency Functions in Low Dimensional Domains."
"""

import math
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


class PeriodicSpatialFourierEmbedding(nn.Module):
    """Exact-periodic Fourier embedding for problems with periodic spatial BCs.

    Spatial dimensions are encoded with integer-frequency features
    [cos(2π/L · k · x), sin(2π/L · k · x)] for k = 1..K, which are exactly
    periodic with spatial period L. This enforces periodicity in the network
    input itself — no soft BC penalty is needed.

    Temporal dimension uses standard random Fourier features (same as
    FourierFeatureEmbedding) so high-frequency temporal content is captured.

    Output dim = 4 * fourier_dim  (2K spatial + 2K temporal).

    Args:
        spatial_dim: Number of spatial input dimensions (1 for KS).
        fourier_dim: K — number of frequency components for each of spatial
                     and temporal parts.
        scale: Std of random temporal frequencies (controls temporal freq band).
        L: Spatial period (e.g., 2π for KS on [0, 2π]).
    """

    def __init__(self, spatial_dim: int, fourier_dim: int, scale: float = 1.0,
                 L: float = 2 * math.pi):
        super().__init__()
        self.spatial_dim = spatial_dim
        # Integer frequencies k=1..K scaled by 2π/L so features are periodic with period L
        k = torch.arange(1, fourier_dim + 1, dtype=torch.float32) * (2 * math.pi / L)
        self.register_buffer('k', k)            # (K,)
        # Random temporal frequencies; shape (K, 1) for 1D time
        B_t = torch.randn(fourier_dim, 1) * scale
        self.register_buffer('B_t', B_t)        # (K, 1)

    @property
    def output_dim(self) -> int:
        """Output dimensionality: 4 * fourier_dim."""
        return 4 * self.k.shape[0]

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        """Map (x, t) input to periodic-spatial Fourier features.

        Args:
            xt: Input tensor (N, spatial_dim + 1).

        Returns:
            Feature tensor (N, 4 * fourier_dim).
        """
        x = xt[:, :self.spatial_dim]   # (N, spatial_dim)
        t = xt[:, self.spatial_dim:]   # (N, 1)

        # Spatial: (N, spatial_dim) @ (spatial_dim, K) — for 1D: (N,1)·(1,K) = (N,K)
        phi_x = x @ self.k.unsqueeze(0)    # (N, K)
        # Temporal: (N, 1) @ (1, K) = (N, K)
        phi_t = t @ self.B_t.T             # (N, K)

        return torch.cat([
            torch.cos(phi_x), torch.sin(phi_x),
            torch.cos(phi_t), torch.sin(phi_t),
        ], dim=-1)  # (N, 4K)

    def extra_repr(self) -> str:
        K = self.k.shape[0]
        return (
            f"spatial_dim={self.spatial_dim}, fourier_dim={K}, "
            f"output_dim={self.output_dim}, periodic=True"
        )
