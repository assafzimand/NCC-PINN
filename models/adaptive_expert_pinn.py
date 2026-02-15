"""Adaptive Expert PINN with dynamic regional expert spawning.

Implements a composed PINN that combines:
- A global base model u_0(x,t) with constant indicator (active everywhere)
- Regional expert models u_i(x,t) with soft sigmoid indicators

The composed solution depends on the mode:

**Hard blending:**
    u(x,t) = u_0(x,t) + Σ 1_Ωi(x,t) · u_i(x,t)

**Soft blending (partition of unity):**
    u(x,t) = Σ_k ψ̃_k(x,t) · u_k(x,t)
    where ψ̃_k = ψ_k / Σ_j ψ_j (normalized, Σ ψ̃_k = 1)
    Base has constant ψ_0 = base_weight everywhere before normalization.

"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from torch.utils.hooks import RemovableHandle
from pathlib import Path

from models.fc_model import FCNet
from adaptive.indicators import (
    RegionDescriptor,
    BatchedIndicators
)

class BatchedModels:
    """Batched forward pass: loop over [base, *experts], stack results.

    Output tensor indices: [:, 0, :] = base, [:, 1:, :] = experts.
    GPU naturally parallelizes the sequential small model calls.
    """

    def __init__(self):
        self._models: List[nn.Module] = []

    def sync_from_models(self, base_model: nn.Module, experts: nn.ModuleList) -> None:
        """Register all models for batched forward. Call after spawning experts."""
        self._models = [base_model] + list(experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward all models and stack results.

        Args:
            x: Input tensor (N, input_dim)

        Returns:
            (N, K+1, output_dim) where K = number of experts
        """
        if not self._models:
            raise RuntimeError("No models registered. Call sync_from_models first.")
        if len(self._models) == 1:
            return self._models[0](x).unsqueeze(1)  # (N, 1, out_dim)
        return torch.stack([m(x) for m in self._models], dim=1)  # (N, K+1, out_dim)


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
        self.blending_mode = adaptive_config.get('blending_mode', 'soft')
        self.sigma_fraction = adaptive_config.get('sigma_fraction', 0.2)  # For soft: sigma = fraction * region_size
        self.base_weight = adaptive_config.get('base_weight', 1.0)  # Uniform weight for base model
        self.base_everywhere = adaptive_config.get('base_everywhere', True)
        self.freeze_mode = adaptive_config.get('freeze_mode', 'none')
        self.expert_architectures = adaptive_config.get('expert_architectures', None)
        
        # Store config architectures (before any pretrained loading)
        # This is used to determine expert architecture when pretrained base is loaded
        self.config_base_architecture = base_architecture
        
        # Create base model
        self.base_model = FCNet(base_architecture, activation, config)
        
        # Expert storage
        self.experts = nn.ModuleList()
        self.regions: List[RegionDescriptor] = []

        # Batched indicators for vectorized mask computation (broadcasting)
        self.batched_indicators = BatchedIndicators(base_weight=self.base_weight)

        # Batched models: [base, *experts] with loop + stack forward
        self.batched_models = BatchedModels()
        self.batched_models.sync_from_models(self.base_model, self.experts)

        # Hook management
        self.activations: Dict[str, torch.Tensor] = {}
        self.hook_handles: List[RemovableHandle] = []
        
        # Flag to indicate if base model is frozen (pretrained)
        # When True, caller should pass precomputed u_base to forward()
        self._base_frozen = False
        
        # Optional epoch timer for performance profiling (set by trainer)
        self._timer = None
    
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
        
        Uses vectorized batched indicators for efficient GPU computation.
        
        Args:
            inputs: (N, n_dims) tensor of coordinates
            depth: Depth level to check
            before_epoch: If provided, only include regions spawned before this epoch
                         (useful for excluding regions just spawned in current step)
            
        Returns:
            (N,) boolean tensor - True if point is inside any depth-d region
        """
        N = inputs.shape[0]
        
        # Get indices at this depth
        depth_indices = []
        for i, r in enumerate(self.regions):
            if r.depth == depth:
                if before_epoch is not None and r.spawn_epoch >= before_epoch:
                    continue
                depth_indices.append(i)
        
        if not depth_indices:
            return torch.zeros(N, dtype=torch.bool, device=inputs.device)
        
        # Compute all hard masks at once using batched indicators
        all_masks = self.batched_indicators.compute_hard_masks_only(inputs)  # (N, K)
        
        if all_masks.shape[1] == 0:
            return torch.zeros(N, dtype=torch.bool, device=inputs.device)
        
        # Union (any) of masks at this depth - vectorized
        depth_masks = all_masks[:, depth_indices]  # (N, num_at_depth)
        return depth_masks.any(dim=1)  # (N,)
    
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
        
        Uses batched indicators for efficient GPU computation.
        
        Args:
            inputs: (N, n_dims) tensor of coordinates
            expert_idx: Index of the expert
            
        Returns:
            (N,) boolean tensor - True if point is inside the expert's region
        """
        if expert_idx < 0 or expert_idx >= len(self.regions):
            # Return all True for base model (idx=-1) or invalid index
            return torch.ones(inputs.shape[0], dtype=torch.bool, device=inputs.device)
        
        # Use batched indicators for efficiency
        all_masks = self.batched_indicators.compute_hard_masks_only(inputs)  # (N, K)
        
        if all_masks.shape[1] == 0 or expert_idx >= all_masks.shape[1]:
            return torch.ones(inputs.shape[0], dtype=torch.bool, device=inputs.device)
        
        return all_masks[:, expert_idx].bool()  # (N,)
    
    def compute_children_coverage(
        self, 
        inputs: torch.Tensor, 
        parent_idx: int, 
        before_epoch: int = None
    ) -> float:
        """
        Compute what fraction of a parent's domain is covered by its children.
        
        Uses vectorized batched indicators for efficient GPU computation.
        
        Args:
            inputs: (N, n_dims) tensor of coordinates (eval points)
            parent_idx: Index of the parent expert (-1 for base model)
            before_epoch: If provided, only include children spawned before this epoch
            
        Returns:
            Coverage fraction (0.0 to 1.0)
        """
        # Compute all hard masks at once using batched indicators
        all_masks = self.batched_indicators.compute_hard_masks_only(inputs)  # (N, K)
        
        # Get parent mask
        if parent_idx == -1:
            # Base model: entire domain
            parent_mask = torch.ones(inputs.shape[0], dtype=torch.bool, device=inputs.device)
        else:
            if parent_idx >= all_masks.shape[1]:
                return 0.0
            parent_mask = all_masks[:, parent_idx].bool()
        
        parent_count = parent_mask.sum().item()
        if parent_count == 0:
            return 0.0
        
        # Get children indices
        children_indices = []
        for i, r in enumerate(self.regions):
            if r.parent_idx == parent_idx:
                if before_epoch is not None and r.spawn_epoch >= before_epoch:
                    continue
                children_indices.append(i)
        
        if not children_indices:
            return 0.0
        
        # Union of all children masks - vectorized
        children_masks = all_masks[:, children_indices]  # (N, num_children)
        children_union = children_masks.any(dim=1)  # (N,)
        
        # Count points that are both in parent AND covered by children
        covered = (parent_mask & children_union).sum().item()
        
        return covered / parent_count
    
    def get_expert_architecture(self, expert_idx: int) -> List[int]:
        """
        Get architecture for a new expert.
        
        Returns the configured expert architecture from adaptive_config.
        """
        if self.expert_architectures is None:
            # Use config base architecture (not pretrained base if loaded)
            return self.config_base_architecture
        elif isinstance(self.expert_architectures, list):
            if expert_idx < len(self.expert_architectures):
                return self.expert_architectures[expert_idx]
            else:
                # Fall back to config base architecture if list exhausted
                return self.config_base_architecture
        else:
            return self.config_base_architecture
    
    def sync_batched_indicators(self) -> None:
        """
        Synchronize batched indicators with current regions.
        
        Call this after spawning experts to update the batched tensors
        used for vectorized indicator computation.
        """
        if not self.regions:
            return
        
        device = next(self.base_model.parameters()).device
        self.batched_indicators.update(
            regions=self.regions,
            device=device,
            mode=self.blending_mode,
            sigma_fraction=self.sigma_fraction
        )
    
    def sync_batched_models(self) -> None:
        """
        Synchronize batched models (base + experts) for O(1) forward pass.
        
        Call this after spawning experts or loading state dict to update
        the batched structure used for parallel computation.
        """
        self.batched_models.sync_from_models(self.base_model, self.experts)
    
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

        # Zero-initialize final layer so new expert starts with zero output,
        # preserving the current composed solution at spawn time.
        layer_names = expert.get_layer_names()
        if layer_names:
            final_layer = expert.network[layer_names[-1]]
            nn.init.zeros_(final_layer.weight)
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)

        # DIAGNOSTIC: Verify expert is on correct device
        actual_device = next(expert.parameters()).device
        print(f"    Expert created on device: {actual_device}")
        
        # Store expert and region
        self.experts.append(expert)
        self.regions.append(region)

        parent_info = f"Base Model" if region.parent_idx == -1 else f"E{region.parent_idx + 1}"
        print(f"  Spawned Expert {expert_idx + 1} (depth={region.depth}, parent={parent_info}):")
        print(f"    Architecture: {architecture}")
        print(f"    Region bounds: {region.bounds_lower} -> {region.bounds_upper}")
        print(f"    Residual-weighted wavelet norm: {region.wavelet_norm:.6f}")
        print(f"    Spawn epoch: {region.spawn_epoch}")
        
        # Sync batched structures for vectorized forward pass
        self.sync_batched_indicators()
        self.sync_batched_models()
        
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
    
    def freeze_base_model(self) -> None:
        """Freeze base model weights (convenience method)."""
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def unfreeze_experts(self) -> None:
        """Unfreeze all expert weights (for training after tree building)."""
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = True
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Composed forward pass.
        
        For hard blending:
            u(x,t) = u_0(x,t) + Σ 1_Ωi(x,t) · u_i(x,t)
        
        For soft blending (partition of unity, non-pretrained):
            u(x,t) = Σ_k ψ̃_k(x,t) · u_k(x,t)
            where ψ̃_k = ψ_k / Σ_j ψ_j (normalized weights, including base)
        
        For soft blending with pretrained base (additive mode):
            u(x,t) = u_base(x,t) + Σ_k ψ̃_k(x,t) · u_k(x,t)
            where ψ̃_k = ψ_k / Σ_j ψ_j for j=1..K (experts only, base not normalized)
            Base model contributes fully everywhere, experts are additive corrections.
        
        Args:
            inputs: (N, n_dims) tensor of coordinates [x, t] or [x, y, t]
            
        Returns:
            u: (N, output_dim) composed solution
        """
        # Check if sparse expert activation is enabled
        threshold = self.adaptive_config.get('expert_activation_threshold', None)
        
        # Convert to float if it's a string (from YAML parsing)
        if threshold is not None:
            threshold = float(threshold)
        
        if threshold is not None and len(self.experts) > 0:
            # Use sparse activation (only evaluate experts with psi > threshold)
            if self.blending_mode == 'hard':
                return self._forward_hard_sparse(inputs, threshold)
            else:
                return self._forward_soft_sparse(inputs, threshold)
        else:
            # Use standard dense evaluation (all experts)
            if self.blending_mode == 'hard':
                return self._forward_hard(inputs)
            else:
                return self._forward_soft(inputs)
    
    def _forward_hard(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Vectorized hard blending forward pass.
        
        u(x,t) = u_0(x,t) + Σ 1_Ωi(x,t) · u_i(x,t)
        
        Optimized with batched computation: All models (base + K experts) computed
        in parallel operations supporting heterogeneous architectures.
        
        Args:
            inputs: (N, n_dims) input coordinates
        """
        _t = self._timer
        
        # Compute all model outputs at once: [base, expert_0, ..., expert_K]
        if _t: _t.start('fwd.batched_models')
        u_all = self.batched_models.forward(inputs)  # (N, K+1, output_dim)
        if _t: _t.stop('fwd.batched_models')
        
        # Extract base and experts
        u_base = u_all[:, 0, :]  # (N, output_dim)
        
        if len(self.experts) == 0:
            return u_base
        
        u_experts = u_all[:, 1:, :]  # (N, K, output_dim)
        
        # Compute masks (already vectorized)
        if _t: _t.start('fwd.compute_masks')
        masks = self.batched_indicators.compute_hard_masks_only(inputs)  # (N, K)
        if _t: _t.stop('fwd.compute_masks')
        
        # Apply masks and sum - fully vectorized
        if _t: _t.start('fwd.blend')
        weighted_experts = masks.unsqueeze(-1) * u_experts  # (N, K, out_dim)
        u_total = u_base + weighted_experts.sum(dim=1)  # (N, out_dim)
        if _t: _t.stop('fwd.blend')
        
        return u_total
    
    def _forward_soft(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Vectorized soft blending forward pass with partition-of-unity normalization.
        
        u(x,t) = Σ_k ψ̃_k(x,t) · u_k(x,t)
        where:
            - ψ_0 = uniform constant (base_weight) for base model
            - ψ_k = sigmoid-based bump function for expert k
            - ψ̃_k = ψ_k / Σ_j ψ_j (partition of unity: Σ ψ̃_k = 1)
        
        Optimized with batched computation: All models (base + K experts) computed
        in parallel operations supporting heterogeneous architectures.
        
        Args:
            inputs: (N, n_dims) input coordinates
        """
        _t = self._timer
        
        # Step 1: Compute ALL masks at once using batched indicators (already vectorized)
        # psi_base: (N, 1), psi_experts: (N, K)
        if _t: _t.start('fwd.compute_masks')
        psi_base, psi_experts = self.batched_indicators(inputs)
        if _t: _t.stop('fwd.compute_masks')
        
        # Step 2: Compute all model outputs at once: [base, expert_0, ..., expert_K]
        if _t: _t.start('fwd.batched_models')
        u_all = self.batched_models.forward(inputs)  # (N, K+1, output_dim)
        if _t: _t.stop('fwd.batched_models')
        
        # Extract base and experts
        u_base = u_all[:, 0, :]  # (N, output_dim)
        
        if len(self.experts) == 0:
            # No experts - just return base model output (normalized weight = 1)
            return u_base
        
        u_experts = u_all[:, 1:, :]  # (N, K, output_dim)
        
        # Step 3: Compute normalized weights (partition of unity)
        # ψ̃_k = ψ_k / Σ_j ψ_j for j=0..K
        if _t: _t.start('fwd.blend')
        psi_sum = psi_base + psi_experts.sum(dim=1, keepdim=True)  # (N, 1)
        psi_sum = psi_sum.clamp(min=1e-8)
        psi_base_norm = psi_base / psi_sum  # (N, 1)
        psi_experts_norm = psi_experts / psi_sum  # (N, K)
        
        # Step 4: Compute final output
        # u = ψ̃_0 · u_base + Σ ψ̃_k · u_k
        weighted_experts = psi_experts_norm.unsqueeze(-1) * u_experts  # (N, K, out_dim)
        u_total = psi_base_norm * u_base + weighted_experts.sum(dim=1)  # (N, out_dim)
        if _t: _t.stop('fwd.blend')
        
        return u_total
    
    def _forward_hard_sparse(self, inputs: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Sparse hard blending: only evaluate experts with mask == 1.
        
        u(x,t) = u_0(x,t) + Σ_{k: mask_k=1} u_k(x,t)
        
        For hard blending, masks are binary (0 or 1), so threshold filtering
        is equivalent to mask > 0. This still provides speedup by avoiding
        evaluation of experts with mask == 0.
        
        Args:
            inputs: (N, n_dims) input coordinates
            threshold: Minimum mask value (typically 0 for hard masks)
        """
        _t = self._timer
        N = inputs.shape[0]
        output_dim = self.base_model.layers[-1]
        device = inputs.device
        
        # Step 1: Compute masks FIRST (cheap, no model evaluation)
        if _t: _t.start('fwd.compute_masks')
        masks = self.batched_indicators.compute_hard_masks_only(inputs)  # (N, K)
        if _t: _t.stop('fwd.compute_masks')
        
        # Step 2: Identify which experts are active for ANY point
        if _t: _t.start('fwd.sparse_selection')
        active_experts_any = (masks.sum(dim=0) > 20)  # (K,) - which experts are used anywhere
        active_expert_indices = torch.nonzero(active_experts_any, as_tuple=True)[0]  # indices of active experts
        num_active = len(active_expert_indices)
        if _t: _t.stop('fwd.sparse_selection')
        
        # Step 3: Always evaluate base model
        if _t: _t.start('fwd.sparse_eval')
        u_base = self.base_model(inputs)  # (N, output_dim)
        
        if num_active == 0:
            # No experts are active anywhere - just return base
            if _t: _t.stop('fwd.sparse_eval')
            return u_base
        
        # Step 4: Evaluate only active experts
        u_experts_sparse = torch.zeros(N, len(self.experts), output_dim, device=device)  # (N, K, output_dim)
        
        for expert_idx in active_expert_indices:
            expert_idx_item = expert_idx.item()
            u_experts_sparse[:, expert_idx_item, :] = self.experts[expert_idx_item](inputs)
        
        if _t: _t.stop('fwd.sparse_eval')
        
        # Step 5: Blend using masks
        if _t: _t.start('fwd.blend')
        weighted_experts = masks.unsqueeze(-1) * u_experts_sparse  # (N, K, out_dim)
        u_total = u_base + weighted_experts.sum(dim=1)  # (N, out_dim)
        if _t: _t.stop('fwd.blend')
        
        return u_total
    
    def _forward_soft_sparse(self, inputs: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Sparse soft blending: only evaluate experts with psi > threshold.
        
        This is where the real speedup happens for soft blending, as we filter
        experts that contribute negligibly (e.g., psi < 1e-4 means <0.01% contribution).
        
        Standard mode:
            u(x,t) = Σ_{k: ψ_k > threshold} ψ̃_k(x,t) · u_k(x,t)
            where ψ̃_k = ψ_k / Σ_j ψ_j (normalized among ALL, including filtered)
        
        Additive mode:
            u(x,t) = u_base + Σ_{k: ψ_k > threshold} ψ̃_k(x,t) · u_k(x,t)
            where ψ̃_k = ψ_k / Σ_j ψ_j (experts only)
        
        Args:
            inputs: (N, n_dims) input coordinates
            threshold: Minimum psi value to evaluate expert (e.g., 1e-4)
        """
        _t = self._timer
        N = inputs.shape[0]
        output_dim = self.base_model.layers[-1]
        device = inputs.device
        
        # Step 1: Compute ALL psi weights FIRST (cheap, no model evaluation)
        if _t: _t.start('fwd.compute_masks')
        psi_base, psi_experts = self.batched_indicators(inputs)  # psi_base: (N, 1), psi_experts: (N, K)
        if _t: _t.stop('fwd.compute_masks')
        
        # Step 2: Identify which experts have significant weight ANYWHERE
        if _t: _t.start('fwd.sparse_selection')
        active_mask = psi_experts > threshold  # (N, K) - boolean mask
        active_experts_any = active_mask.sum(dim=0) > 20  # (K,) - which experts are used anywhere
        active_expert_indices = torch.nonzero(active_experts_any, as_tuple=True)[0]  # indices
        num_active = len(active_expert_indices)
        if _t: _t.stop('fwd.sparse_selection')
        
        # Step 3: Always evaluate base model
        if _t: _t.start('fwd.sparse_eval')
        u_base = self.base_model(inputs)  # (N, output_dim)
        
        if num_active == 0:
            # No experts are active anywhere - just return base
            if _t: _t.stop('fwd.sparse_eval')
            return u_base
        
        # Step 4: Evaluate only active experts
        u_experts_sparse = torch.zeros(N, len(self.experts), output_dim, device=device)  # (N, K, output_dim)
        psi_experts_filtered = psi_experts.clone()  # Keep all psi values for normalization
        
        for expert_idx in active_expert_indices:
            expert_idx_item = expert_idx.item()
            u_experts_sparse[:, expert_idx_item, :] = self.experts[expert_idx_item](inputs)
        
        # Zero out psi weights below threshold (for proper normalization)
        psi_experts_filtered = psi_experts_filtered * active_mask.float()  # Zero out below-threshold
        
        if _t: _t.stop('fwd.sparse_eval')
        
        # Step 5: Normalize weights and blend (partition of unity)
        if _t: _t.start('fwd.blend')
        psi_sum = psi_base + psi_experts_filtered.sum(dim=1, keepdim=True)  # (N, 1)
        psi_sum = psi_sum.clamp(min=1e-8)
        psi_base_norm = psi_base / psi_sum  # (N, 1)
        psi_experts_norm = psi_experts_filtered / psi_sum  # (N, K)
        
        weighted_experts = psi_experts_norm.unsqueeze(-1) * u_experts_sparse  # (N, K, out_dim)
        u_total = psi_base_norm * u_base + weighted_experts.sum(dim=1)  # (N, out_dim)
        
        if _t: _t.stop('fwd.blend')
        
        return u_total
    
    def forward_for_pde_derivatives(self, inputs: torch.Tensor) -> dict:
        """
        Forward pass returning decomposed components for product-rule derivative computation.
        
        Instead of returning the composed scalar output, returns individual expert outputs
        and their normalized weights. The loss function uses these to compute PDE derivatives
        via the product rule, creating K small autograd graphs instead of one massive one.
        
        u(x,t) = Σ_k ψ̃_k(x,t) · u_k(x,t),  k includes base
        where ψ̃_k = ψ_k / Z, Z = Σ_j ψ_j
        
        Product rule gives: u_x = Σ_k (ψ̃_k_x · u_k + ψ̃_k · u_k_x), etc.
        
        Args:
            inputs: (N, n_dims) input coordinates. The x,t components must
                    have requires_grad=True (set by the loss function).
        
        Returns:
            dict with:
                - 'components': list of dicts, each with:
                    - 'u': (N, output_dim) model output, on autograd graph
                    - 'inputs': (N, D) per-expert input copy (for batched autograd)
                    - 'psi_norm': (N, 1) normalized weight, on autograd graph
                    - 'constant_psi': bool, True if psi_norm is constant (skip indicator derivatives)
                - 'composed': (N, output_dim) assembled output (uses detached psi for efficiency)
                - 'indicator_data': dict with bounds/sigma for analytical derivatives
        """
        _t = self._timer
        N = inputs.shape[0]
        output_dim = self.base_model.layers[-1]
        device = inputs.device
        
        threshold = self.adaptive_config.get('expert_activation_threshold', None)
        if threshold is not None:
            threshold = float(threshold)
        
        # Step 1: Compute indicators (on autograd graph since inputs require grad)
        if _t: _t.start('fwd.compute_masks')
        psi_base, psi_experts = self.batched_indicators(inputs)  # (N, 1), (N, K)
        if _t: _t.stop('fwd.compute_masks')
        
        # Step 2: Sparse selection (same logic as _forward_soft_sparse)
        if _t: _t.start('fwd.sparse_selection')
        K = psi_experts.shape[1]
        
        if threshold is not None and K > 0:
            active_mask = psi_experts > threshold  # (N, K)
            active_experts_any = active_mask.sum(dim=0) > 20  # (K,)
            active_expert_indices = torch.nonzero(active_experts_any, as_tuple=True)[0]
            # Zero out below-threshold psi values
            psi_experts_filtered = psi_experts * active_mask.float()
        elif K > 0:
            active_expert_indices = torch.arange(K, device=device)
            psi_experts_filtered = psi_experts
        else:
            active_expert_indices = []
            psi_experts_filtered = psi_experts  # (N, 0)
        
        num_active = len(active_expert_indices)
        if _t: _t.stop('fwd.sparse_selection')
        
        # Step 3: Normalize weights (partition of unity)
        Z = psi_base + psi_experts_filtered.sum(dim=1, keepdim=True)  # (N, 1)
        Z = Z.clamp(min=1e-8)
        psi_norm_base = psi_base / Z  # (N, 1)
        psi_norm_experts = psi_experts_filtered / Z  # (N, K)
        
        # Step 4: Build components list and evaluate models
        # Each expert gets its OWN copy of inputs (detached leaf with requires_grad).
        # This makes their autograd graphs independent, enabling batched autograd.grad
        # calls in the loss function (K expert derivatives in 1 call instead of K calls).
        if _t: _t.start('fwd.sparse_eval')
        components = []
        
        # Base model — own input copy
        # Base has constant unnormalized psi, but psi_norm depends on x,t via Z
        inputs_base = inputs.detach().clone().requires_grad_(True)
        u_base = self.base_model(inputs_base)  # (N, output_dim)
        components.append({
            'u': u_base,
            'inputs': inputs_base,
            'psi_norm': psi_norm_base,
            'constant_psi': False,  # psi_norm depends on x,t through Z
        })
        
        # Active experts — each gets own input copy
        for idx in active_expert_indices:
            k = idx.item() if torch.is_tensor(idx) else idx
            inputs_k = inputs.detach().clone().requires_grad_(True)
            u_k = self.experts[k](inputs_k)  # (N, output_dim)
            components.append({
                'u': u_k,
                'inputs': inputs_k,
                'psi_norm': psi_norm_experts[:, k:k+1],  # (N, 1)
                'constant_psi': False,
            })
        
        if _t: _t.stop('fwd.sparse_eval')
        
        # Step 5: Composed output (with detached psi for efficiency — correct since
        # indicators have no learnable params, backward only needs to flow through u_k)
        composed = torch.zeros(N, output_dim, device=device, dtype=inputs.dtype)
        for c in components:
            composed = composed + c['psi_norm'].detach() * c['u']
        
        # Step 6: Pack indicator metadata for analytical derivative computation
        # The loss function uses bounds/sigma to compute ψ̃ derivatives analytically
        # instead of using autograd (eliminates ~63 autograd calls at K=20).
        indicator_data = {
            'all_lower': self.batched_indicators.all_lower,   # (K, D) or None
            'all_upper': self.batched_indicators.all_upper,   # (K, D) or None
            'all_sigma': self.batched_indicators.all_sigma,   # (K, D) or None
            'psi_base': psi_base,                             # (N, 1)
            'psi_experts_filtered': psi_experts_filtered,     # (N, K)
            'active_expert_indices': active_expert_indices,   # tensor of active indices
        }
        
        return {
            'components': components,
            'composed': composed,
            'indicator_data': indicator_data,
        }
    
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
        """Vectorized hard blending decomposed forward pass with batched models."""
        result = {}
        
        # Compute all model outputs at once
        u_all = self.batched_models.forward(inputs)  # (N, K+1, output_dim)
        result['base'] = u_all[:, 0, :]  # (N, output_dim)
        
        # Compute masks (already vectorized)
        all_masks = self.batched_indicators.compute_hard_masks_only(inputs)
        
        result['masks'] = {}
        
        if len(self.experts) > 0:
            # Store individual expert outputs and masks
            for i in range(len(self.experts)):
                result[f'expert_{i}'] = u_all[:, i+1, :]  # (N, out_dim)
                result['masks'][f'expert_{i}'] = all_masks[:, i:i+1]  # (N, 1)
            
            # Compute composed output - vectorized
            u_experts = u_all[:, 1:, :]  # (N, K, output_dim)
            weighted_experts = all_masks.unsqueeze(-1) * u_experts  # (N, K, out_dim)
            u_total = result['base'] + weighted_experts.sum(dim=1)  # (N, out_dim)
        else:
            u_total = result['base'].clone()
        
        result['composed'] = u_total
        return result
    
    def _forward_decomposed_soft(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Vectorized soft blending decomposed forward pass with batched models (partition of unity)."""
        result = {}
        
        # Compute all masks at once using batched indicators (already vectorized)
        psi_base, psi_experts = self.batched_indicators(inputs)  # (N, 1), (N, K)
        
        # Store unnormalized masks
        result['masks'] = {'base': psi_base}
        for i in range(psi_experts.shape[1]):
            result['masks'][f'expert_{i}'] = psi_experts[:, i:i+1]  # (N, 1)
        
        # Compute normalized weights (partition of unity)
        psi_sum = psi_base + psi_experts.sum(dim=1, keepdim=True)  # (N, 1)
        psi_sum = psi_sum.clamp(min=1e-8)
        psi_base_norm = psi_base / psi_sum  # (N, 1)
        psi_experts_norm = psi_experts / psi_sum  # (N, K)
        
        # Store normalized weights
        result['weights_normalized'] = {'base': psi_base_norm}
        for i in range(psi_experts_norm.shape[1]):
            result['weights_normalized'][f'expert_{i}'] = psi_experts_norm[:, i:i+1]  # (N, 1)
        result['blending_mode_info'] = 'partition_of_unity'
        
        # Compute all model outputs at once
        u_all = self.batched_models.forward(inputs)  # (N, K+1, output_dim)
        u_base = u_all[:, 0, :]  # (N, output_dim)
        result['base'] = u_base
        
        if len(self.experts) > 0:
            # Extract expert outputs
            u_experts = u_all[:, 1:, :]  # (N, K, output_dim)
            
            # Store individual expert outputs
            for i in range(len(self.experts)):
                result[f'expert_{i}'] = u_experts[:, i, :]  # (N, out_dim)
            
            # Composed output (partition of unity: u = ψ̃_0 · u_base + Σ ψ̃_k · u_k)
            weighted_experts = psi_experts_norm.unsqueeze(-1) * u_experts  # (N, K, out_dim)
            u_total = psi_base_norm * u_base + weighted_experts.sum(dim=1)  # (N, out_dim)
        else:
            u_total = u_base
        
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
            'config_base_architecture': self.config_base_architecture,
            'activation': self.activation,
            'adaptive_config': self.adaptive_config,
            'base_frozen': self._base_frozen
        }
    
    def load_state_dict_extended(self, state_dict: Dict):
        """
        Load extended state dict including regions and indicators.
        
        Handles architecture mismatches by recreating models with the correct
        architecture from the checkpoint.
        
        Args:
            state_dict: Dict from state_dict_extended()
        """
        # Check if base model architecture needs to be recreated
        saved_base_arch = state_dict.get('base_architecture')
        saved_activation = state_dict.get('activation', self.activation)
        
        if saved_base_arch is None:
            # Infer from state dict if not explicitly saved
            saved_base_arch = self._infer_architecture_from_state_dict(state_dict['base_model'])
        
        # Recreate base model if architecture differs
        if saved_base_arch != self.base_architecture:
            print(f"  Recreating base model: {self.base_architecture} -> {saved_base_arch}")
            device = next(self.base_model.parameters()).device
            self.base_model = FCNet(saved_base_arch, saved_activation, self.config)
            self.base_model = self.base_model.to(device)
            self.base_architecture = saved_base_arch
            
            # If this was a pretrained base (different from config), mark as frozen
            if saved_base_arch != self.config_base_architecture:
                self._base_frozen = True
                for param in self.base_model.parameters():
                    param.requires_grad = False
                print(f"  Base model marked as frozen (pretrained architecture)")
        
        # Load base model weights
        self.base_model.load_state_dict(state_dict['base_model'])
        
        # Recreate experts and regions
        self.experts = nn.ModuleList()
        self.regions = []
        
        for i, (expert_state, region_dict) in enumerate(zip(
            state_dict['experts'], state_dict['regions']
        )):
            region = RegionDescriptor.from_dict(region_dict)
            
            # Infer expert architecture from its state dict (more reliable than config)
            expert_arch = self._infer_architecture_from_state_dict(expert_state)
            
            expert = FCNet(expert_arch, self.activation, self.config)
            expert.load_state_dict(expert_state)
            
            # Move to same device as base
            device = next(self.base_model.parameters()).device
            expert = expert.to(device)
            
            self.experts.append(expert)
            self.regions.append(region)

        # Restore base frozen state if saved
        if 'base_frozen' in state_dict:
            self._base_frozen = state_dict['base_frozen']
            if self._base_frozen:
                for param in self.base_model.parameters():
                    param.requires_grad = False
        
        # Sync batched structures
        self.sync_batched_indicators()
        self.sync_batched_models()
    
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
