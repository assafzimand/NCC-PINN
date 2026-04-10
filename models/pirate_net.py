"""PirateNet backbone for PINN expert networks.

Architecture:
    1. Fourier Feature embedding: FF(z) = [cos(Bz), sin(Bz)]
    2. Input projection: h0 = σ(W_in * FF(z))
    3. Gating streams (fixed, computed once per forward):
           U = σ(W_U * FF(z))
           V = σ(W_V * FF(z))
    4. Residual layers with UV-gating and trainable α skip:
           h_k = σ(W_k * h_{k-1}) ⊙ (1 - U) + h_{k-1} ⊙ V + α_k * h_{k-1}
       where α_k is a per-layer scalar initialized at 0 (pure UV-gating at start)
    5. Output projection: W_out * h_last (no activation)

Optional:
    - RWF: applies W_eff = diag(exp(s)) @ W on hidden layers (enabled via config['rwf'])
    - Least-squares init of output layer (piratenet.ls_init=true, expensive)

Reference: Fang (2023) "PirateNets: Physics-informed Deep Learning with
Residual Adaptive Networks."

Constructor signature matches FCNet/ResNetModel for drop-in factory use.
Architecture: layers = [input_dim, h, h, ..., h, output_dim]
  - hidden_dim = layers[1] (uniform width required)
  - n_layers = len(layers) - 2

Fourier features are read from config['fourier_features'] (same key as FCNet/ResNet).
RWF is read from config['rwf'] (same key).
ls_init is read from config.get('piratenet', {}).get('ls_init', False).
"""

import torch
import torch.nn as nn
from typing import List, Dict
from models.rwf_layer import RWFLinear
from models.fourier_features import FourierFeatureEmbedding


def _get_activation(name: str) -> nn.Module:
    activations = {
        'tanh': nn.Tanh(),
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'gelu': nn.GELU(),
        'elu': nn.ELU(),
        'leaky_relu': nn.LeakyReLU(),
    }
    name = name.lower()
    if name not in activations:
        raise ValueError(
            f"Unknown activation: {name}. Available: {list(activations.keys())}")
    return activations[name]


class PirateNet(nn.Module):
    """PirateNet: Physics-informed Residual Adaptive Network.

    Args:
        layers: [input_dim, h, h, ..., h, output_dim]. All hidden dims must
                be equal. At least 1 hidden layer required.
        activation: Activation function name.
        config: Full project config dict (reads 'fourier_features', 'rwf',
                'piratenet' sub-keys).
        is_base: If True, validate input_dim == spatial_dim + 1.
    """

    def __init__(self, layers: List[int], activation: str, config: Dict,
                 is_base: bool = True):
        super().__init__()

        self.is_base = is_base
        self.layers = layers
        self.activation_name = activation
        self.config = config

        problem = config['problem']
        problem_config = config[problem]
        spatial_dim = problem_config['spatial_dim']
        output_dim = problem_config.get('output_dim', 1)

        if is_base:
            expected_input_dim = spatial_dim + 1
            assert layers[0] == expected_input_dim, (
                f"PirateNet input_dim {layers[0]} != expected {expected_input_dim}"
            )
        assert layers[-1] == output_dim, (
            f"PirateNet output_dim {layers[-1]} != expected {output_dim}"
        )

        hidden = layers[1:-1]
        if len(hidden) < 1:
            raise ValueError(
                f"PirateNet requires at least 1 hidden layer, got {len(hidden)}.")
        if len(set(hidden)) != 1:
            raise ValueError(
                f"PirateNet requires uniform hidden width, got {hidden}.")

        input_dim = layers[0]
        h = hidden[0]
        n_layers = len(hidden)
        self.hidden_dim = h
        self.n_layers = n_layers

        # Fourier Feature embedding
        ff_cfg = config.get('fourier_features', {})
        use_ff = ff_cfg.get('enabled', False)
        if use_ff:
            ff_dim = ff_cfg.get('dim', 64)
            ff_scale = ff_cfg.get('scale', 1.0)
            self.ff_emb = FourierFeatureEmbedding(input_dim, ff_dim, ff_scale)
            ff_out = self.ff_emb.output_dim
        else:
            # No FF: use a fallback dim so architecture still makes sense
            # (input goes directly to projections)
            self.ff_emb = None
            ff_out = input_dim

        # RWF
        use_rwf = config.get('rwf', False)
        LinearCls = RWFLinear if use_rwf else nn.Linear

        self.activation = _get_activation(activation)

        # Three projections from FF space → hidden space
        self.input_proj = LinearCls(ff_out, h)   # h0 = σ(input_proj(FF(z)))
        self.U_proj = LinearCls(ff_out, h)        # U gating stream
        self.V_proj = LinearCls(ff_out, h)        # V gating stream

        # Residual hidden layers
        self.hidden_layers = nn.ModuleList(
            [LinearCls(h, h) for _ in range(n_layers)]
        )

        # Per-layer adaptive skip scalars, initialized at 0 (pure UV-gating)
        self.alphas = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in range(n_layers)]
        )

        # Output projection — always plain nn.Linear for output scale stability
        self.output_proj = nn.Linear(h, layers[-1])

        # Least-squares init of output layer (optional)
        pirate_cfg = config.get('piratenet', {})
        if pirate_cfg.get('ls_init', False):
            self._ls_init_pending = True  # deferred until first forward with IC data
        else:
            self._ls_init_pending = False

        # Empty NCC hooks interface (not supported for PirateNet)
        self.activations: Dict[str, torch.Tensor] = {}
        self.hook_handles = []

    def forward(self, x: torch.Tensor, return_activation: bool = False):
        """Forward pass.

        Args:
            x: Input tensor (N, input_dim).
            return_activation: Ignored for PirateNet (not needed by AToE/AToELeaves).

        Returns:
            Output tensor (N, output_dim).
        """
        # Apply FF embedding if enabled
        if self.ff_emb is not None:
            z = self.ff_emb(x)
        else:
            z = x

        # Compute gating streams (fixed for this forward pass)
        U = self.activation(self.U_proj(z))   # (N, h)
        V = self.activation(self.V_proj(z))   # (N, h)

        # Input projection
        h = self.activation(self.input_proj(z))  # (N, h)

        # Residual layers with UV-gating and adaptive α
        for layer, alpha in zip(self.hidden_layers, self.alphas):
            h_new = self.activation(layer(h))
            h = h_new * (1 - U) + h * V + alpha * h

        if return_activation:
            return self.output_proj(h), h
        return self.output_proj(h)

    def get_activation_dim(self) -> int:
        """Hidden dim (used if return_activation=True is ever needed)."""
        return self.hidden_dim

    def get_layer_names(self) -> List[str]:
        """PirateNet does not support NCC hook analysis — returns empty list."""
        return []

    def register_ncc_hooks(self, layer_names, keep_gradients=False):
        """NCC hooks not supported — no-op."""
        return []

    def remove_hooks(self):
        """No-op."""
        self.activations = {}
        self.hook_handles = []

    def __repr__(self) -> str:
        ff_info = f"ff_dim={self.ff_emb.B.shape[0]}" if self.ff_emb else "no_ff"
        return (
            f"PirateNet(\n"
            f"  architecture: {self.layers}\n"
            f"  activation: {self.activation_name}\n"
            f"  hidden_dim: {self.hidden_dim}, n_layers: {self.n_layers}\n"
            f"  {ff_info}, rwf={isinstance(self.input_proj, RWFLinear)}\n"
            f")"
        )
