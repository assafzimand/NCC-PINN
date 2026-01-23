"""Adaptive Expert PINN with dynamic regional expert spawning.

Implements a composed PINN that combines:
- A global base model u_0(x,t) trained on the full domain
- Regional expert models u_i(x,t) that specialize on high-error regions

The composed solution is:
    u(x,t) = u_0(x,t) + Σ 1_Ωi(x,t) · u_i(x,t)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union
from torch.utils.hooks import RemovableHandle
from pathlib import Path

from models.fc_model import FCNet
from adaptive.indicators import (
    RegionDescriptor, 
    HardIndicator, 
    SoftIndicator,
    create_indicator
)


class AdaptiveExpertPINN(nn.Module):
    """
    Adaptive Expert PINN with dynamic regional expert spawning.
    
    Combines a base model with regional expert models using indicator functions.
    Experts are spawned during training based on wavelet-detected high-error regions.
    """
    
    def __init__(
        self,
        base_architecture: List[int],
        activation: str,
        config: Dict,
        adaptive_config: Dict
    ):
        """
        Initialize AdaptiveExpertPINN.
        
        Args:
            base_architecture: Layer sizes for base model [input_dim, ..., output_dim]
            activation: Activation function name
            config: Full configuration dictionary (for FCNet)
            adaptive_config: Adaptive PINN specific configuration
        """
        super().__init__()
        
        self.base_architecture = base_architecture
        self.activation = activation
        self.config = config
        self.adaptive_config = adaptive_config
        
        # Extract adaptive parameters
        self.max_experts = adaptive_config.get('max_experts', 5)
        self.blending_mode = adaptive_config.get('blending_mode', 'hard')
        self.blending_sigma = adaptive_config.get('blending_sigma', 0.1)
        self.base_everywhere = adaptive_config.get('base_everywhere', True)
        self.freeze_mode = adaptive_config.get('freeze_mode', 'none')
        self.expert_architectures = adaptive_config.get('expert_architectures', None)
        
        # Create base model
        self.base_model = FCNet(base_architecture, activation, config)
        
        # Expert storage
        self.experts = nn.ModuleList()
        self.regions: List[RegionDescriptor] = []
        self.indicators: List[Union[HardIndicator, SoftIndicator]] = []
        
        # Hook management
        self.activations: Dict[str, torch.Tensor] = {}
        self.hook_handles: List[RemovableHandle] = []
    
    @property
    def num_experts(self) -> int:
        """Number of spawned experts (not counting base model)."""
        return len(self.experts)
    
    def get_expert_architecture(self, expert_idx: int) -> List[int]:
        """Get architecture for a new expert."""
        if self.expert_architectures is None:
            # Use base architecture
            return self.base_architecture
        elif isinstance(self.expert_architectures, list):
            if expert_idx < len(self.expert_architectures):
                return self.expert_architectures[expert_idx]
            else:
                # Fall back to base architecture if list exhausted
                return self.base_architecture
        else:
            return self.base_architecture
    
    def spawn_expert(self, region: RegionDescriptor) -> int:
        """
        Spawn a new expert PINN for the given region.
        
        Args:
            region: RegionDescriptor defining the expert's domain
            
        Returns:
            Index of the new expert
        """
        if len(self.experts) >= self.max_experts:
            print(f"  Cannot spawn more experts: max_experts={self.max_experts} reached")
            return -1
        
        expert_idx = len(self.experts)
        architecture = self.get_expert_architecture(expert_idx)
        
        # Create new expert FCNet
        expert = FCNet(architecture, self.activation, self.config)
        
        # Move to same device as base model
        device = next(self.base_model.parameters()).device
        expert = expert.to(device)
        
        # Store expert and region
        self.experts.append(expert)
        self.regions.append(region)
        
        # Create indicator function
        indicator = create_indicator(region, self.blending_mode, self.blending_sigma)
        self.indicators.append(indicator)
        
        print(f"  Spawned Expert {expert_idx + 1}:")
        print(f"    Architecture: {architecture}")
        print(f"    Region bounds: {region.bounds_lower} -> {region.bounds_upper}")
        print(f"    Wavelet norm: {region.wavelet_norm:.4f}")
        print(f"    Spawn epoch: {region.spawn_epoch}")
        
        return expert_idx
    
    def freeze_models(self, mode: Optional[str] = None):
        """
        Apply freezing strategy to models.
        
        Args:
            mode: 'none', 'previous', or 'base_only' (uses self.freeze_mode if None)
        """
        mode = mode or self.freeze_mode
        
        if mode == 'none':
            # Unfreeze all
            for param in self.base_model.parameters():
                param.requires_grad = True
            for expert in self.experts:
                for param in expert.parameters():
                    param.requires_grad = True
                    
        elif mode == 'base_only':
            # Freeze base, train experts
            for param in self.base_model.parameters():
                param.requires_grad = False
            for expert in self.experts:
                for param in expert.parameters():
                    param.requires_grad = True
                    
        elif mode == 'previous':
            # Freeze base and all but last expert
            for param in self.base_model.parameters():
                param.requires_grad = False
            for i, expert in enumerate(self.experts):
                is_last = (i == len(self.experts) - 1)
                for param in expert.parameters():
                    param.requires_grad = is_last
        else:
            raise ValueError(f"Unknown freeze_mode: {mode}")
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Composed forward pass.
        
        u(x,t) = u_0(x,t) + Σ 1_Ωi(x,t) · u_i(x,t)
        
        Args:
            inputs: (N, n_dims) tensor of coordinates [x, t] or [x, y, t]
            
        Returns:
            u: (N, output_dim) composed solution
        """
        # Base model prediction
        u_total = self.base_model(inputs)  # (N, output_dim)
        
        # Add expert contributions
        for expert, indicator in zip(self.experts, self.indicators):
            # Get indicator mask
            mask = indicator(inputs)  # (N, 1)
            
            # Expert prediction
            u_expert = expert(inputs)  # (N, output_dim)
            
            # Masked contribution
            u_total = u_total + mask * u_expert
        
        return u_total
    
    def forward_decomposed(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning individual model contributions.
        
        Useful for analysis and debugging.
        
        Args:
            inputs: (N, n_dims) tensor of coordinates
            
        Returns:
            Dict with 'base', 'expert_0', 'expert_1', ..., 'composed', 'masks'
        """
        result = {}
        
        # Base model
        result['base'] = self.base_model(inputs)
        result['masks'] = {}
        
        # Experts
        u_total = result['base'].clone()
        for i, (expert, indicator) in enumerate(zip(self.experts, self.indicators)):
            mask = indicator(inputs)
            u_expert = expert(inputs)
            result[f'expert_{i}'] = u_expert
            result['masks'][f'expert_{i}'] = mask
            u_total = u_total + mask * u_expert
        
        result['composed'] = u_total
        return result
    
    def get_layer_names(self) -> List[str]:
        """Get layer names from base model (for tracker compatibility)."""
        return self.base_model.get_layer_names()
    
    def register_ncc_hooks(
        self,
        layer_names: List[str],
        keep_gradients: bool = False
    ) -> List[RemovableHandle]:
        """
        Register hooks on base model layers for NCC/probe analysis.
        
        Args:
            layer_names: Layers to hook (from base model)
            keep_gradients: Whether to keep gradients (for derivatives tracking)
            
        Returns:
            List of RemovableHandle
        """
        # Clear previous hooks
        self.remove_hooks()
        self.activations = {}
        
        # Register on base model
        handles = self.base_model.register_ncc_hooks(layer_names, keep_gradients)
        self.hook_handles = handles
        
        return handles
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        self.base_model.remove_hooks()
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.activations = {}
    
    @property
    def activations(self) -> Dict[str, torch.Tensor]:
        """Get activations from base model (for tracker compatibility)."""
        return self.base_model.activations
    
    @activations.setter
    def activations(self, value):
        """Set activations (for initialization)."""
        self._activations = value
    
    def get_domain_bounds(self) -> Dict[str, List[float]]:
        """Get domain bounds from problem configuration."""
        problem = self.config['problem']
        problem_config = self.config[problem]
        spatial_domain = problem_config['spatial_domain']
        temporal_domain = problem_config['temporal_domain']
        
        # Build bounds based on spatial dimension
        if len(spatial_domain) == 1:
            # 1D spatial: [x_min, t_min], [x_max, t_max]
            return {
                'lower': [spatial_domain[0][0], temporal_domain[0]],
                'upper': [spatial_domain[0][1], temporal_domain[1]]
            }
        elif len(spatial_domain) == 2:
            # 2D spatial: [x_min, y_min, t_min], [x_max, y_max, t_max]
            return {
                'lower': [spatial_domain[0][0], spatial_domain[1][0], temporal_domain[0]],
                'upper': [spatial_domain[0][1], spatial_domain[1][1], temporal_domain[1]]
            }
        else:
            raise ValueError(f"Unsupported spatial dimension: {len(spatial_domain)}")
    
    def state_dict_extended(self) -> Dict:
        """
        Get extended state dict including regions and indicators.
        
        Returns:
            Dict with model state and adaptive state
        """
        return {
            'base_model': self.base_model.state_dict(),
            'experts': [expert.state_dict() for expert in self.experts],
            'regions': [r.to_dict() for r in self.regions],
            'num_experts': len(self.experts),
            'base_architecture': self.base_architecture,
            'activation': self.activation,
            'adaptive_config': self.adaptive_config
        }
    
    def load_state_dict_extended(self, state_dict: Dict):
        """
        Load extended state dict including regions and indicators.
        
        Args:
            state_dict: Dict from state_dict_extended()
        """
        # Load base model
        self.base_model.load_state_dict(state_dict['base_model'])
        
        # Recreate experts and regions
        self.experts = nn.ModuleList()
        self.regions = []
        self.indicators = []
        
        for i, (expert_state, region_dict) in enumerate(zip(
            state_dict['experts'], state_dict['regions']
        )):
            region = RegionDescriptor.from_dict(region_dict)
            architecture = self.get_expert_architecture(i)
            
            expert = FCNet(architecture, self.activation, self.config)
            expert.load_state_dict(expert_state)
            
            # Move to same device as base
            device = next(self.base_model.parameters()).device
            expert = expert.to(device)
            
            self.experts.append(expert)
            self.regions.append(region)
            
            indicator = create_indicator(region, self.blending_mode, self.blending_sigma)
            self.indicators.append(indicator)
    
    def __repr__(self) -> str:
        """String representation."""
        base_str = " -> ".join(map(str, self.base_architecture))
        expert_archs = [
            " -> ".join(map(str, self.get_expert_architecture(i)))
            for i in range(len(self.experts))
        ]
        
        repr_str = (
            f"AdaptiveExpertPINN(\n"
            f"  base: {base_str}\n"
            f"  activation: {self.activation}\n"
            f"  blending: {self.blending_mode}\n"
            f"  num_experts: {len(self.experts)}/{self.max_experts}\n"
        )
        
        for i, (arch, region) in enumerate(zip(expert_archs, self.regions)):
            repr_str += f"  expert_{i}: {arch}, region={region.bounds_lower}->{region.bounds_upper}\n"
        
        repr_str += ")"
        return repr_str
