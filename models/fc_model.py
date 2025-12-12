"""Fully-connected neural network with named layers and NCC hooks."""

import torch
import torch.nn as nn
from typing import List, Dict
from torch.utils.hooks import RemovableHandle


class FCNet(nn.Module):
    """
    Fully-connected neural network for PINN problems.

    Features:
    - Named layers for NCC analysis (layer_1, layer_2, ...)
    - Hook registration for activation capture
    - Configurable architecture and activation function
    """

    def __init__(self, layers: List[int], activation: str, config: Dict):
        """
        Initialize FCNet.

        Args:
            layers: List of layer sizes [input_dim, hidden1, ..., output_dim]
            activation: Activation function name ('tanh', 'relu', 'sigmoid')
            config: Configuration dict for verification

        Example:
            layers = [2, 50, 100, 50, 2]
            Creates: input(2) -> hidden(50) -> hidden(100) ->
                     hidden(50) -> output(2)
        """
        super().__init__()

        # Verify architecture matches problem configuration
        problem = config['problem']
        problem_config = config[problem]
        spatial_dim = problem_config['spatial_dim']
        output_dim = problem_config.get('output_dim', 2)  # Default to 2 for legacy
        expected_input_dim = spatial_dim + 1  # x + t

        assert layers[0] == expected_input_dim, (
            f"Architecture input dimension {layers[0]} does not match "
            f"expected dimension {expected_input_dim} "
            f"(spatial_dim={spatial_dim} + 1 for time)"
        )
        
        assert layers[-1] == output_dim, (
            f"Architecture output dimension {layers[-1]} does not match "
            f"expected dimension {output_dim} (problem={problem})"
        )

        self.layers = layers
        self.activation_name = activation
        self.config = config

        # Get activation function
        self.activation = self._get_activation(activation)

        # Build network with named layers
        self.network = nn.ModuleDict()

        for i in range(len(layers) - 1):
            layer_name = f"layer_{i + 1}"
            self.network[layer_name] = nn.Linear(layers[i], layers[i + 1])

        # Storage for activations captured by hooks
        self.activations: Dict[str, torch.Tensor] = {}

        # Storage for hook handles
        self.hook_handles: List[RemovableHandle] = []

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU()
        }

        if activation.lower() not in activations:
            raise ValueError(
                f"Unknown activation: {activation}. "
                f"Available: {list(activations.keys())}"
            )

        return activations[activation.lower()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (N, input_dim) where
               input_dim = spatial_dim + 1 (concatenated [x, t])

        Returns:
            Output tensor of shape (N, output_dim)
        """
        out = x

        # Pass through all layers except the last
        layer_names = list(self.network.keys())
        for i, layer_name in enumerate(layer_names[:-1]):
            out = self.network[layer_name](out)
            out = self.activation(out)

        # Last layer (no activation)
        out = self.network[layer_names[-1]](out)

        return out

    def register_ncc_hooks(
        self,
        layer_names: List[str],
        keep_gradients: bool = False
    ) -> List[RemovableHandle]:
        """
        Register forward hooks to capture activations for NCC analysis.

        Args:
            layer_names: List of layer names to hook (e.g., ['layer_1',
                        'layer_2'])
            keep_gradients: If True, don't detach activations (for derivatives
                          tracking). Default False for NCC/probes.

        Returns:
            List of RemovableHandle objects for hook management

        Example:
            handles = model.register_ncc_hooks(['layer_1', 'layer_2'])
            # ... run forward pass ...
            activations = model.activations  # {'layer_1': tensor, ...}
            # ... cleanup ...
            for handle in handles:
                handle.remove()
        """
        # Clear previous hooks
        self.remove_hooks()
        self.activations = {}

        handles = []

        for layer_name in layer_names:
            if layer_name not in self.network:
                raise ValueError(
                    f"Layer '{layer_name}' not found. "
                    f"Available: {list(self.network.keys())}"
                )

            # Create hook function for this layer
            def make_hook(name):
                def hook(module, input, output):
                    # Store post-activation values
                    self.activations[name] = output.detach()
                return hook

            # Register hook on the layer
            # We want post-activation, so we hook after linear + activation
            # We need to hook the activation, not the linear layer
            # But since we apply activation explicitly, we hook the output
            # after activation in the forward pass

            # Actually, let's hook at the Linear layer and apply
            # activation in the hook
            def make_hook_with_activation(name):
                def hook(module, input, output):
                    # Apply activation and store
                    if name != list(self.network.keys())[-1]:
                        # Not the output layer - apply activation
                        activated = self.activation(output)
                        self.activations[name] = activated if keep_gradients else activated.detach()
                    else:
                        # Output layer - no activation
                        self.activations[name] = output if keep_gradients else output.detach()
                return hook

            handle = self.network[layer_name].register_forward_hook(
                make_hook_with_activation(layer_name)
            )
            handles.append(handle)

        self.hook_handles = handles
        return handles

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.activations = {}

    def get_layer_names(self) -> List[str]:
        """Get list of all layer names in the network."""
        return list(self.network.keys())

    def __repr__(self) -> str:
        """String representation of the model."""
        layers_str = " -> ".join(map(str, self.layers))
        return (
            f"FCNet(\n"
            f"  architecture: {layers_str}\n"
            f"  activation: {self.activation_name}\n"
            f"  layers: {self.get_layer_names()}\n"
            f")"
        )

