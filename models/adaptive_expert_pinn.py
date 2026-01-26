"""Adaptive Expert PINN with dynamic regional expert spawning.

Implements a composed PINN that combines:
- A global base model u_0(x,t) trained on the full domain
- Regional expert models u_i(x,t) that specialize on high-error regions

The composed solution is:
    u(x,t) = u_0(x,t) + Σ 1_Ωi(x,t) · u_i(x,t)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union, Tuple
from torch.utils.hooks import RemovableHandle
from pathlib import Path

from models.fc_model import FCNet
from adaptive.indicators import (
    RegionDescriptor, 
    HardIndicator, 
    SoftIndicator,
    UniformIndicator,
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
        self.max_depth = adaptive_config.get('max_depth', 5)  # Maximum depth in expert tree
        self.blending_mode = adaptive_config.get('blending_mode', 'hard')
        self.blending_sigma = adaptive_config.get('blending_sigma', 0.1)  # Legacy parameter
        self.sigma_fraction = adaptive_config.get('sigma_fraction', 0.2)  # New: fraction of region size
        self.base_weight = adaptive_config.get('base_weight', 1.0)  # Uniform weight for base model
        self.base_everywhere = adaptive_config.get('base_everywhere', True)
        self.freeze_mode = adaptive_config.get('freeze_mode', 'none')
        self.expert_architectures = adaptive_config.get('expert_architectures', None)
        
        # Create base model
        self.base_model = FCNet(base_architecture, activation, config)
        
        # Create uniform indicator for base model (used in soft blending for partition of unity)
        self.base_indicator: Optional[UniformIndicator] = None
        if self.blending_mode == 'soft':
            self.base_indicator = UniformIndicator(self.base_weight)
        
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
    
    def get_regions_at_depth(self, depth: int, before_epoch: int = None) -> List[RegionDescriptor]:
        """
        Get all regions at a specific depth level.
        
        Args:
            depth: Depth level (1 = children of base model)
            before_epoch: If provided, only include regions spawned before this epoch
                         (useful for excluding regions just spawned in current step)
            
        Returns:
            List of RegionDescriptors at that depth
        """
        result = []
        for r in self.regions:
            if r.depth == depth:
                # Filter by spawn epoch if specified
                if before_epoch is not None and r.spawn_epoch >= before_epoch:
                    continue
                result.append(r)
        return result
    
    def get_experts_at_depth(self, depth: int) -> List[tuple]:
        """
        Get all (expert, region) pairs at a specific depth level.
        
        Args:
            depth: Depth level (1 = children of base model)
            
        Returns:
            List of (expert, region) tuples at that depth
        """
        result = []
        for expert, region in zip(self.experts, self.regions):
            if region.depth == depth:
                result.append((expert, region))
        return result
    
    def get_highest_depth(self) -> int:
        """
        Get the highest depth level with at least one expert.
        
        Returns:
            Maximum depth (0 if no experts spawned yet)
        """
        if not self.regions:
            return 0
        return max(r.depth for r in self.regions)
    
    def get_union_mask_at_depth(self, inputs: torch.Tensor, depth: int, before_epoch: int = None) -> torch.Tensor:
        """
        Get a boolean mask for points inside ANY region at the given depth.
        
        Args:
            inputs: (N, n_dims) tensor of coordinates
            depth: Depth level to check
            before_epoch: If provided, only include regions spawned before this epoch
                         (useful for excluding regions just spawned in current step)
            
        Returns:
            (N,) boolean tensor - True if point is inside any depth-d region
        """
        N = inputs.shape[0]
        union_mask = torch.zeros(N, dtype=torch.bool, device=inputs.device)
        
        for region, indicator in zip(self.regions, self.indicators):
            if region.depth == depth:
                # Filter by spawn epoch if specified
                if before_epoch is not None and region.spawn_epoch >= before_epoch:
                    continue
                # Get indicator mask (0.0 or 1.0 for hard indicators)
                mask = indicator(inputs)  # (N, 1)
                inside = mask.squeeze().bool()  # (N,)
                union_mask = union_mask | inside
        
        return union_mask
    
    def count_experts_at_depth(self, depth: int) -> int:
        """Count number of experts at a specific depth."""
        return sum(1 for r in self.regions if r.depth == depth)
    
    def get_children_of_parent(self, parent_idx: int, before_epoch: int = None) -> List[RegionDescriptor]:
        """
        Get all child regions of a specific parent.
        
        Args:
            parent_idx: Index of the parent expert (-1 for base model)
            before_epoch: If provided, only include regions spawned before this epoch
            
        Returns:
            List of RegionDescriptors that are children of the specified parent
        """
        result = []
        for r in self.regions:
            if r.parent_idx == parent_idx:
                if before_epoch is not None and r.spawn_epoch >= before_epoch:
                    continue
                result.append(r)
        return result
    
    def get_mask_for_expert(self, inputs: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """
        Get boolean mask for points inside a specific expert's region.
        
        Args:
            inputs: (N, n_dims) tensor of coordinates
            expert_idx: Index of the expert
            
        Returns:
            (N,) boolean tensor - True if point is inside the expert's region
        """
        if expert_idx < 0 or expert_idx >= len(self.indicators):
            # Return all True for base model (idx=-1) or invalid index
            return torch.ones(inputs.shape[0], dtype=torch.bool, device=inputs.device)
        
        indicator = self.indicators[expert_idx]
        mask = indicator(inputs)  # (N, 1)
        return mask.squeeze().bool()  # (N,)
    
    def compute_children_coverage(
        self, 
        inputs: torch.Tensor, 
        parent_idx: int, 
        before_epoch: int = None
    ) -> float:
        """
        Compute what fraction of a parent's domain is covered by its children.
        
        Uses point sampling to estimate coverage.
        
        Args:
            inputs: (N, n_dims) tensor of coordinates (eval points)
            parent_idx: Index of the parent expert (-1 for base model)
            before_epoch: If provided, only include children spawned before this epoch
            
        Returns:
            Coverage fraction (0.0 to 1.0)
        """
        # Get parent mask
        if parent_idx == -1:
            # Base model: entire domain
            parent_mask = torch.ones(inputs.shape[0], dtype=torch.bool, device=inputs.device)
        else:
            parent_mask = self.get_mask_for_expert(inputs, parent_idx)
        
        parent_count = parent_mask.sum().item()
        if parent_count == 0:
            return 0.0
        
        # Get union of children masks
        children = self.get_children_of_parent(parent_idx, before_epoch=before_epoch)
        if not children:
            return 0.0
        
        # Find which points in parent are covered by children
        children_union = torch.zeros_like(parent_mask)
        for child_region in children:
            # Find child index
            child_idx = self.regions.index(child_region)
            child_mask = self.get_mask_for_expert(inputs, child_idx)
            children_union = children_union | child_mask
        
        # Count points that are both in parent AND covered by children
        covered = (parent_mask & children_union).sum().item()
        
        return covered / parent_count
    
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
            region: RegionDescriptor defining the expert's domain (includes depth)
            
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
        
        # Create indicator function (use sigma_fraction for soft blending)
        indicator = create_indicator(region, self.blending_mode, self.sigma_fraction)
        self.indicators.append(indicator)
        
        parent_info = f"Base Model" if region.parent_idx == -1 else f"E{region.parent_idx + 1}"
        print(f"  Spawned Expert {expert_idx + 1} (depth={region.depth}, parent={parent_info}):")
        print(f"    Architecture: {architecture}")
        print(f"    Region bounds: {region.bounds_lower} -> {region.bounds_upper}")
        print(f"    Residual-weighted wavelet norm: {region.wavelet_norm:.6f}")
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
        
        For hard blending:
            u(x,t) = u_0(x,t) + Σ 1_Ωi(x,t) · u_i(x,t)
        
        For soft blending (partition of unity):
            u(x,t) = Σ_k ψ̃_k(x,t) · u_k(x,t)
            where ψ̃_k = ψ_k / Σ_j ψ_j (normalized weights)
        
        Args:
            inputs: (N, n_dims) tensor of coordinates [x, t] or [x, y, t]
            
        Returns:
            u: (N, output_dim) composed solution
        """
        if self.blending_mode == 'hard':
            return self._forward_hard(inputs)
        else:
            return self._forward_soft(inputs)
    
    def _forward_hard(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Hard blending forward pass with efficient filtering.
        
        u(x,t) = u_0(x,t) + Σ 1_Ωi(x,t) · u_i(x,t)
        
        Only forwards points through each expert if they lie within
        that expert's domain, saving computation.
        """
        # Base model prediction (all points)
        u_total = self.base_model(inputs)  # (N, output_dim)
        
        # Add expert contributions (only inside points)
        for expert, indicator in zip(self.experts, self.indicators):
            # Get indicator mask (0.0 or 1.0 for hard indicators)
            mask = indicator(inputs)  # (N, 1)
            inside = mask.squeeze().bool()  # (N,) boolean
            
            if inside.any():
                # Only forward points inside this expert's domain
                u_expert_inside = expert(inputs[inside])  # (M, output_dim)
                
                # Add contribution at inside indices
                u_total[inside] = u_total[inside] + u_expert_inside
        
        return u_total
    
    def _forward_soft(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Soft blending forward pass with partition-of-unity normalization.
        
        u(x,t) = Σ_k ψ̃_k(x,t) · u_k(x,t)
        
        where:
            - ψ_0 = uniform constant (base_weight) for base model
            - ψ_k = sigmoid-based bump function for expert k
            - ψ̃_k = ψ_k / Σ_j ψ_j (partition of unity: Σ ψ̃_k = 1)
        
        All models contribute everywhere, weighted by their normalized soft indicators.
        """
        N = inputs.shape[0]
        
        # Step 1: Compute all unnormalized weights
        # Base model weight (uniform)
        psi_base = self.base_indicator(inputs)  # (N, 1)
        
        # Expert weights
        psi_experts = []
        for indicator in self.indicators:
            psi_k = indicator(inputs)  # (N, 1)
            psi_experts.append(psi_k)
        
        # Step 2: Compute normalization (sum of all weights)
        psi_sum = psi_base.clone()
        for psi_k in psi_experts:
            psi_sum = psi_sum + psi_k
        
        # Avoid division by zero (shouldn't happen with uniform base > 0)
        psi_sum = psi_sum.clamp(min=1e-8)
        
        # Step 3: Normalized weights (partition of unity: sum = 1)
        psi_base_norm = psi_base / psi_sum  # (N, 1)
        psi_experts_norm = [psi_k / psi_sum for psi_k in psi_experts]
        
        # Step 4: Compute weighted outputs
        # Base model contribution
        u_base = self.base_model(inputs)  # (N, output_dim)
        u_total = psi_base_norm * u_base  # (N, output_dim) - broadcasting
        
        # Expert contributions (all points, weighted by soft mask)
        for expert, psi_k_norm in zip(self.experts, psi_experts_norm):
            u_expert = expert(inputs)  # (N, output_dim)
            u_total = u_total + psi_k_norm * u_expert
        
        return u_total
    
    def forward_decomposed(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning individual model contributions.
        
        Useful for analysis and debugging.
        
        For hard blending: Uses efficient filtering (only compute for inside points)
        For soft blending: Returns unnormalized masks and normalized weights
        
        Args:
            inputs: (N, n_dims) tensor of coordinates
            
        Returns:
            Dict with:
                - 'base': base model output (N, output_dim)
                - 'expert_0', 'expert_1', ...: expert outputs (N, output_dim)
                - 'composed': final composed output (N, output_dim)
                - 'masks': dict of unnormalized masks per expert
                - 'weights_normalized': (soft only) dict of normalized weights per model
        """
        if self.blending_mode == 'hard':
            return self._forward_decomposed_hard(inputs)
        else:
            return self._forward_decomposed_soft(inputs)
    
    def _forward_decomposed_hard(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Hard blending decomposed forward pass."""
        result = {}
        N = inputs.shape[0]
        output_dim = self.base_architecture[-1]
        
        # Base model
        result['base'] = self.base_model(inputs)
        result['masks'] = {}
        
        # Experts (with efficient filtering)
        u_total = result['base'].clone()
        for i, (expert, indicator) in enumerate(zip(self.experts, self.indicators)):
            mask = indicator(inputs)  # (N, 1)
            inside = mask.squeeze().bool()  # (N,) boolean
            
            # Initialize full tensor with zeros for this expert
            u_expert_full = torch.zeros(N, output_dim, device=inputs.device, dtype=inputs.dtype)
            
            if inside.any():
                # Only compute for inside points
                u_expert_inside = expert(inputs[inside])  # (M, output_dim)
                u_expert_full[inside] = u_expert_inside
                u_total[inside] = u_total[inside] + u_expert_inside
            
            result[f'expert_{i}'] = u_expert_full
            result['masks'][f'expert_{i}'] = mask
        
        result['composed'] = u_total
        return result
    
    def _forward_decomposed_soft(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Soft blending decomposed forward pass with partition-of-unity weights."""
        result = {}
        N = inputs.shape[0]
        output_dim = self.base_architecture[-1]
        
        # Compute all unnormalized weights
        psi_base = self.base_indicator(inputs)  # (N, 1)
        psi_experts = [indicator(inputs) for indicator in self.indicators]  # list of (N, 1)
        
        # Compute normalization
        psi_sum = psi_base.clone()
        for psi_k in psi_experts:
            psi_sum = psi_sum + psi_k
        psi_sum = psi_sum.clamp(min=1e-8)
        
        # Normalized weights (partition of unity)
        psi_base_norm = psi_base / psi_sum
        psi_experts_norm = [psi_k / psi_sum for psi_k in psi_experts]
        
        # Store masks (unnormalized)
        result['masks'] = {'base': psi_base}
        for i, psi_k in enumerate(psi_experts):
            result['masks'][f'expert_{i}'] = psi_k
        
        # Store normalized weights
        result['weights_normalized'] = {'base': psi_base_norm}
        for i, psi_k_norm in enumerate(psi_experts_norm):
            result['weights_normalized'][f'expert_{i}'] = psi_k_norm
        
        # Compute outputs
        u_base = self.base_model(inputs)
        result['base'] = u_base
        
        # Composed output with normalized weights
        u_total = psi_base_norm * u_base
        
        for i, (expert, psi_k_norm) in enumerate(zip(self.experts, psi_experts_norm)):
            u_expert = expert(inputs)
            result[f'expert_{i}'] = u_expert
            u_total = u_total + psi_k_norm * u_expert
        
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
            
            indicator = create_indicator(region, self.blending_mode, self.sigma_fraction)
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
