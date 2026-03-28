"""Adaptive Tree of Experts (AToE) PINN with dynamic regional expert spawning.

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
from typing import List, Dict, Optional, Set
from torch.utils.hooks import RemovableHandle

from models.fc_model import FCNet
from models.network_factory import create_network
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


class AToE(nn.Module):
    """Adaptive Tree of Experts PINN with dynamic regional expert spawning.

    Combines a base model with regional expert models using indicator functions.
    Experts are spawned during training based on wavelet-detected high-error regions.
    All experts always participate in the forward pass (no leaves concept).
    """

    supports_decomposed = True

    def __init__(
        self,
        base_architecture: List[int],
        activation: str,
        config: Dict,
        adaptive_config: Dict
    ):
        """
        Initialize AToE.

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

        self.max_experts = adaptive_config.get('max_experts', 5)
        self.max_depth = adaptive_config.get('max_depth', 5)
        self.blending_mode = adaptive_config.get('blending_mode', 'soft')
        self.sigma_fraction = adaptive_config.get('sigma_fraction', 0.2)
        self.base_weight = adaptive_config.get('base_weight', 1.0)
        self.base_everywhere = adaptive_config.get('base_everywhere', True)
        self.freeze_mode = adaptive_config.get('freeze_mode', 'none')
        self.expert_type = adaptive_config.get('expert_type', 'mlp')

        self.atoe_threshold_capacity = adaptive_config.get(
            'AToE_threshold_capacity', None
        )
        problem = config['problem']
        problem_config = config[problem]
        self.input_dim = base_architecture[0]
        self.output_dim = problem_config.get('output_dim', 2)
        if self.atoe_threshold_capacity is not None:
            self.wavelet_threshold = problem_config.get(
                'wavelet_threshold', 1.0
            )

        self.config_base_architecture = base_architecture

        self.base_model = create_network(
            base_architecture, activation, config,
            is_base=True, expert_type=self.expert_type
        )

        self.experts = nn.ModuleList()
        self.regions: List[RegionDescriptor] = []

        self.leaf_indices: Set[int] = {-1}

        self.batched_indicators = BatchedIndicators(base_weight=self.base_weight)

        self.batched_models = BatchedModels()
        self.batched_models.sync_from_models(self.base_model, self.experts)

        self.activations: Dict[str, torch.Tensor] = {}
        self.hook_handles: List[RemovableHandle] = []

        self._timer = None

    @property
    def num_experts(self) -> int:
        """Number of spawned experts (not counting base model)."""
        return len(self.experts)

    def get_leaf_info(self):
        """Return (region_or_None, expert_idx) for leaf nodes the trainer should try to split."""
        result = []
        if -1 in self.leaf_indices:
            result.append((None, -1))
        for i in sorted(self.leaf_indices):
            if i >= 0:
                result.append((self.regions[i], i))
        return result

    def get_regions_at_depth(self, depth: int, before_epoch: int = None) -> List[RegionDescriptor]:
        """
        Get all regions at a specific depth level.

        Args:
            depth: Depth level (1 = children of base model)
            before_epoch: If provided, only include regions spawned before this epoch

        Returns:
            List of RegionDescriptors at that depth
        """
        result = []
        for r in self.regions:
            if r.depth == depth:
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

        Returns:
            (N,) boolean tensor - True if point is inside any depth-d region
        """
        N = inputs.shape[0]

        depth_indices = []
        for i, r in enumerate(self.regions):
            if r.depth == depth:
                if before_epoch is not None and r.spawn_epoch >= before_epoch:
                    continue
                depth_indices.append(i)

        if not depth_indices:
            return torch.zeros(N, dtype=torch.bool, device=inputs.device)

        all_masks = self.batched_indicators.compute_hard_masks_only(inputs)  # (N, K)

        if all_masks.shape[1] == 0:
            return torch.zeros(N, dtype=torch.bool, device=inputs.device)

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
            return torch.ones(inputs.shape[0], dtype=torch.bool, device=inputs.device)

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
        all_masks = self.batched_indicators.compute_hard_masks_only(inputs)  # (N, K)

        if parent_idx == -1:
            parent_mask = torch.ones(inputs.shape[0], dtype=torch.bool, device=inputs.device)
        else:
            if parent_idx >= all_masks.shape[1]:
                return 0.0
            parent_mask = all_masks[:, parent_idx].bool()

        parent_count = parent_mask.sum().item()
        if parent_count == 0:
            return 0.0

        children_indices = []
        for i, r in enumerate(self.regions):
            if r.parent_idx == parent_idx:
                if before_epoch is not None and r.spawn_epoch >= before_epoch:
                    continue
                children_indices.append(i)

        if not children_indices:
            return 0.0

        children_masks = all_masks[:, children_indices]  # (N, num_children)
        children_union = children_masks.any(dim=1)  # (N,)

        covered = (parent_mask & children_union).sum().item()

        return covered / parent_count

    def get_expert_architecture(self, region: RegionDescriptor) -> List[int]:
        """Get architecture for a new expert based on region norm."""
        if self.atoe_threshold_capacity is None:
            return self.config_base_architecture

        from models.architecture_bank import get_architecture_for_capacity
        ratio = max(region.wavelet_norm / self.wavelet_threshold, 1.0)
        target_capacity = self.atoe_threshold_capacity * ratio
        return get_architecture_for_capacity(
            target_capacity, self.input_dim, self.output_dim
        )

    def sync_batched_indicators(self) -> None:
        """Synchronize batched indicators with current regions.

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
        """Synchronize batched models (base + experts) for O(1) forward pass.

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
        expert_idx = len(self.experts)
        architecture = self.get_expert_architecture(region)
        device = next(self.base_model.parameters()).device

        expert = create_network(
            architecture, self.activation, self.config,
            is_base=True, expert_type=self.expert_type
        )
        expert = expert.to(device)
        if self.expert_type == 'resnet':
            nn.init.zeros_(expert.output_proj.weight)
            if expert.output_proj.bias is not None:
                nn.init.zeros_(expert.output_proj.bias)
        else:
            layer_names = expert.get_layer_names()
            if layer_names:
                final_layer = expert.network[layer_names[-1]]
                nn.init.zeros_(final_layer.weight)
                if final_layer.bias is not None:
                    nn.init.zeros_(final_layer.bias)

        self.experts.append(expert)
        self.regions.append(region)

        self.leaf_indices.add(expert_idx)
        self.leaf_indices.discard(region.parent_idx)

        parent_info = "Base Model" if region.parent_idx == -1 else f"E{region.parent_idx + 1}"
        print(f"  Spawned Expert {expert_idx + 1} (depth={region.depth}, parent={parent_info}):")
        print(f"    Architecture: {architecture}")
        print(f"    Region bounds: {region.bounds_lower} -> {region.bounds_upper}")
        print(f"    Residual-weighted wavelet norm: {region.wavelet_norm:.6f}")
        print(f"    Spawn epoch: {region.spawn_epoch}")

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
            for param in self.base_model.parameters():
                param.requires_grad = True
            for expert in self.experts:
                for param in expert.parameters():
                    param.requires_grad = True

        elif mode == 'base_only':
            for param in self.base_model.parameters():
                param.requires_grad = False
            for expert in self.experts:
                for param in expert.parameters():
                    param.requires_grad = True

        elif mode == 'previous':
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

        For soft blending (partition of unity):
            u(x,t) = Σ_k ψ̃_k(x,t) · u_k(x,t)
            where ψ̃_k = ψ_k / Σ_j ψ_j (normalized weights, including base)

        Args:
            inputs: (N, n_dims) tensor of coordinates [x, t] or [x, y, t]

        Returns:
            u: (N, output_dim) composed solution
        """
        threshold = self.adaptive_config.get('expert_activation_threshold', None)
        if threshold is not None:
            threshold = float(threshold)

        if threshold is not None and len(self.experts) > 0:
            if self.blending_mode == 'hard':
                return self._forward_hard_sparse(inputs, threshold)
            else:
                return self._forward_soft_sparse(inputs, threshold)
        else:
            if self.blending_mode == 'hard':
                return self._forward_hard(inputs)
            else:
                return self._forward_soft(inputs)

    def _forward_hard(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Vectorized hard blending forward pass.

        u(x,t) = u_0(x,t) + Σ 1_Ωi(x,t) · u_i(x,t)

        Args:
            inputs: (N, n_dims) input coordinates
        """
        _t = self._timer

        if _t: _t.start('fwd.batched_models')
        u_all = self.batched_models.forward(inputs)  # (N, K+1, output_dim)
        if _t: _t.stop('fwd.batched_models')

        u_base = u_all[:, 0, :]  # (N, output_dim)

        if len(self.experts) == 0:
            return u_base

        u_experts = u_all[:, 1:, :]  # (N, K, output_dim)

        if _t: _t.start('fwd.compute_masks')
        masks = self.batched_indicators.compute_hard_masks_only(inputs)  # (N, K)
        if _t: _t.stop('fwd.compute_masks')

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

        Args:
            inputs: (N, n_dims) input coordinates
        """
        _t = self._timer

        if _t: _t.start('fwd.compute_masks')
        psi_base, psi_experts = self.batched_indicators(inputs)
        if _t: _t.stop('fwd.compute_masks')

        if _t: _t.start('fwd.batched_models')
        u_all = self.batched_models.forward(inputs)  # (N, K+1, output_dim)
        if _t: _t.stop('fwd.batched_models')

        u_base = u_all[:, 0, :]  # (N, output_dim)

        if len(self.experts) == 0:
            return u_base

        u_experts = u_all[:, 1:, :]  # (N, K, output_dim)

        if _t: _t.start('fwd.blend')
        psi_sum = psi_base + psi_experts.sum(dim=1, keepdim=True)  # (N, 1)
        psi_sum = psi_sum.clamp(min=1e-8)
        psi_base_norm = psi_base / psi_sum  # (N, 1)
        psi_experts_norm = psi_experts / psi_sum  # (N, K)

        weighted_experts = psi_experts_norm.unsqueeze(-1) * u_experts  # (N, K, out_dim)
        u_total = psi_base_norm * u_base + weighted_experts.sum(dim=1)  # (N, out_dim)
        if _t: _t.stop('fwd.blend')

        return u_total

    def _forward_hard_sparse(self, inputs: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Sparse hard blending: only evaluate experts with mask == 1.

        u(x,t) = u_0(x,t) + Σ_{k: mask_k=1} u_k(x,t)

        Args:
            inputs: (N, n_dims) input coordinates
            threshold: Minimum mask value (typically 0 for hard masks)
        """
        _t = self._timer
        N = inputs.shape[0]
        output_dim = self.base_model.layers[-1]
        device = inputs.device

        if _t: _t.start('fwd.compute_masks')
        masks = self.batched_indicators.compute_hard_masks_only(inputs)  # (N, K)
        if _t: _t.stop('fwd.compute_masks')

        if _t: _t.start('fwd.sparse_selection')
        _ms = self.adaptive_config.get(
            'relevant_samples_to_activate_expert', 0)
        active_experts_any = (masks.sum(dim=0) > _ms)  # (K,)
        active_expert_indices = torch.nonzero(active_experts_any, as_tuple=True)[0]
        num_active = len(active_expert_indices)
        if _t: _t.stop('fwd.sparse_selection')

        if _t: _t.start('fwd.sparse_eval')
        u_base = self.base_model(inputs)  # (N, output_dim)

        if num_active == 0:
            if _t: _t.stop('fwd.sparse_eval')
            return u_base

        u_experts_sparse = torch.zeros(N, len(self.experts), output_dim, device=device)

        for expert_idx in active_expert_indices:
            expert_idx_item = expert_idx.item()
            u_experts_sparse[:, expert_idx_item, :] = self.experts[expert_idx_item](inputs)

        if _t: _t.stop('fwd.sparse_eval')

        if _t: _t.start('fwd.blend')
        weighted_experts = masks.unsqueeze(-1) * u_experts_sparse  # (N, K, out_dim)
        u_total = u_base + weighted_experts.sum(dim=1)  # (N, out_dim)
        if _t: _t.stop('fwd.blend')

        return u_total

    def _forward_soft_sparse(self, inputs: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Sparse soft blending: only evaluate experts with psi > threshold.

        Normalization uses the full set of experts (same as non-sparse).
        Inactive experts contribute 0 to the output (their u_k is not evaluated).

        Args:
            inputs: (N, n_dims) input coordinates
            threshold: Minimum psi value to evaluate expert (e.g., 1e-4)
        """
        _t = self._timer
        N = inputs.shape[0]
        output_dim = self.base_model.layers[-1]
        device = inputs.device

        if _t: _t.start('fwd.compute_masks')
        psi_base, psi_experts = self.batched_indicators(inputs)  # psi_base: (N, 1), psi_experts: (N, K)
        if _t: _t.stop('fwd.compute_masks')

        if _t: _t.start('fwd.sparse_selection')
        active_mask = psi_experts > threshold  # (N, K)
        _ms = self.adaptive_config.get(
            'relevant_samples_to_activate_expert', 0)
        active_experts_any = active_mask.sum(dim=0) > _ms  # (K,)
        active_expert_indices = torch.nonzero(active_experts_any, as_tuple=True)[0]
        num_active = len(active_expert_indices)
        if _t: _t.stop('fwd.sparse_selection')

        if _t: _t.start('fwd.sparse_eval')
        u_base = self.base_model(inputs)  # (N, output_dim)

        if num_active == 0:
            if _t: _t.stop('fwd.sparse_eval')
            return u_base

        u_experts_sparse = torch.zeros(N, len(self.experts), output_dim, device=device)

        for expert_idx in active_expert_indices:
            expert_idx_item = expert_idx.item()
            u_experts_sparse[:, expert_idx_item, :] = self.experts[expert_idx_item](inputs)

        if _t: _t.stop('fwd.sparse_eval')

        if _t: _t.start('fwd.blend')
        psi_sum = psi_base + psi_experts.sum(dim=1, keepdim=True)  # (N, 1)
        psi_sum = psi_sum.clamp(min=1e-8)
        psi_base_norm = psi_base / psi_sum  # (N, 1)
        psi_experts_norm = psi_experts / psi_sum  # (N, K)

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

        if _t: _t.start('fwd.compute_masks')
        psi_base, psi_experts = self.batched_indicators(inputs)  # (N, 1), (N, K)
        if _t: _t.stop('fwd.compute_masks')

        if _t: _t.start('fwd.sparse_selection')
        K = psi_experts.shape[1]

        if threshold is not None and K > 0:
            active_mask = psi_experts > threshold  # (N, K)
            _ms = self.adaptive_config.get(
                'relevant_samples_to_activate_expert', 0)
            active_experts_any = active_mask.sum(dim=0) > _ms  # (K,)
            active_expert_indices = torch.nonzero(active_experts_any, as_tuple=True)[0]
        elif K > 0:
            active_expert_indices = torch.arange(K, device=device)
        else:
            active_expert_indices = []

        if _t: _t.stop('fwd.sparse_selection')

        Z = psi_base + psi_experts.sum(dim=1, keepdim=True)  # (N, 1)
        Z = Z.clamp(min=1e-8)
        psi_norm_base = psi_base / Z  # (N, 1)
        psi_norm_experts = psi_experts / Z  # (N, K)

        if _t: _t.start('fwd.sparse_eval')
        components = []

        inputs_base = inputs.detach().clone().requires_grad_(True)
        u_base = self.base_model(inputs_base)  # (N, output_dim)
        components.append({
            'u': u_base,
            'inputs': inputs_base,
            'psi_norm': psi_norm_base,
            'constant_psi': False,
        })

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

        composed = torch.zeros(N, output_dim, device=device, dtype=inputs.dtype)
        for c in components:
            composed = composed + c['psi_norm'].detach() * c['u']

        indicator_data = {
            'all_lower': self.batched_indicators.all_lower,   # (K, D) or None
            'all_upper': self.batched_indicators.all_upper,   # (K, D) or None
            'all_sigma': self.batched_indicators.all_sigma,   # (K, D) or None
            'psi_base': psi_base,                             # (N, 1)
            'psi_experts_filtered': psi_experts,              # (N, K) full psi for normalization
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

        u_all = self.batched_models.forward(inputs)  # (N, K+1, output_dim)
        result['base'] = u_all[:, 0, :]  # (N, output_dim)

        all_masks = self.batched_indicators.compute_hard_masks_only(inputs)

        result['masks'] = {}

        if len(self.experts) > 0:
            for i in range(len(self.experts)):
                result[f'expert_{i}'] = u_all[:, i+1, :]  # (N, out_dim)
                result['masks'][f'expert_{i}'] = all_masks[:, i:i+1]  # (N, 1)

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

        psi_base, psi_experts = self.batched_indicators(inputs)  # (N, 1), (N, K)

        result['masks'] = {'base': psi_base}
        for i in range(psi_experts.shape[1]):
            result['masks'][f'expert_{i}'] = psi_experts[:, i:i+1]  # (N, 1)

        psi_sum = psi_base + psi_experts.sum(dim=1, keepdim=True)  # (N, 1)
        psi_sum = psi_sum.clamp(min=1e-8)
        psi_base_norm = psi_base / psi_sum  # (N, 1)
        psi_experts_norm = psi_experts / psi_sum  # (N, K)

        result['weights_normalized'] = {'base': psi_base_norm}
        for i in range(psi_experts_norm.shape[1]):
            result['weights_normalized'][f'expert_{i}'] = psi_experts_norm[:, i:i+1]  # (N, 1)
        result['blending_mode_info'] = 'partition_of_unity'

        u_all = self.batched_models.forward(inputs)  # (N, K+1, output_dim)
        u_base = u_all[:, 0, :]  # (N, output_dim)
        result['base'] = u_base

        if len(self.experts) > 0:
            u_experts = u_all[:, 1:, :]  # (N, K, output_dim)

            for i in range(len(self.experts)):
                result[f'expert_{i}'] = u_experts[:, i, :]  # (N, out_dim)

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
        self.remove_hooks()
        self.activations = {}

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

        if len(spatial_domain) == 1:
            return {
                'lower': [spatial_domain[0][0], temporal_domain[0]],
                'upper': [spatial_domain[0][1], temporal_domain[1]]
            }
        elif len(spatial_domain) == 2:
            return {
                'lower': [spatial_domain[0][0], spatial_domain[1][0], temporal_domain[0]],
                'upper': [spatial_domain[0][1], spatial_domain[1][1], temporal_domain[1]]
            }
        else:
            raise ValueError(f"Unsupported spatial dimension: {len(spatial_domain)}")

    def state_dict_extended(self) -> Dict:
        """Get extended state dict including regions and indicators."""
        return {
            'base_model': self.base_model.state_dict(),
            'experts': [expert.state_dict() for expert in self.experts],
            'expert_architectures': [e.layers for e in self.experts],
            'regions': [r.to_dict() for r in self.regions],
            'num_experts': len(self.experts),
            'base_architecture': self.base_architecture,
            'config_base_architecture': self.config_base_architecture,
            'activation': self.activation,
            'adaptive_config': self.adaptive_config,
        }

    @staticmethod
    def _infer_architecture_from_state_dict(state_dict: Dict) -> List[int]:
        architecture = []
        layer_idx = 1
        while f'network.layer_{layer_idx}.weight' in state_dict:
            weight = state_dict[f'network.layer_{layer_idx}.weight']
            if layer_idx == 1:
                architecture.append(weight.shape[1])
            architecture.append(weight.shape[0])
            layer_idx += 1
        if not architecture:
            raise ValueError("Could not infer architecture from state dict")
        return architecture

    def load_state_dict_extended(self, state_dict: Dict):
        """
        Load extended state dict including regions and indicators.

        Handles architecture mismatches by recreating models with the correct
        architecture from the checkpoint.

        Args:
            state_dict: Dict from state_dict_extended()
        """
        saved_base_arch = state_dict.get('base_architecture')
        saved_activation = state_dict.get('activation', self.activation)

        if saved_base_arch is None:
            saved_base_arch = self._infer_architecture_from_state_dict(state_dict['base_model'])

        saved_adaptive = state_dict.get('adaptive_config', {})
        saved_expert_type = saved_adaptive.get('expert_type', 'mlp')

        if saved_base_arch != self.base_architecture:
            print(f"  Recreating base model: {self.base_architecture} -> {saved_base_arch}")
            device = next(self.base_model.parameters()).device
            self.base_model = create_network(
                saved_base_arch, saved_activation, self.config,
                is_base=True, expert_type=saved_expert_type
            )
            self.base_model = self.base_model.to(device)
            self.base_architecture = saved_base_arch

        self.base_model.load_state_dict(state_dict['base_model'])

        self.experts = nn.ModuleList()
        self.regions = []
        saved_expert_archs = state_dict.get(
            'expert_architectures', None
        )

        for i, (expert_state, region_dict) in enumerate(zip(
            state_dict['experts'], state_dict['regions']
        )):
            region = RegionDescriptor.from_dict(region_dict)

            if saved_expert_archs is not None:
                expert_arch = saved_expert_archs[i]
            else:
                expert_arch = self._infer_architecture_from_state_dict(expert_state)

            expert = create_network(
                expert_arch, self.activation, self.config,
                is_base=True, expert_type=saved_expert_type
            )
            expert.load_state_dict(expert_state)

            device = next(self.base_model.parameters()).device
            expert = expert.to(device)

            self.experts.append(expert)
            self.regions.append(region)

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
            f"AToE(\n"
            f"  base: {base_str}\n"
            f"  activation: {self.activation}\n"
            f"  blending: {self.blending_mode}\n"
            f"  num_experts: {len(self.experts)}/{self.max_experts}\n"
        )

        for i, (arch, region) in enumerate(zip(expert_archs, self.regions)):
            repr_str += f"  expert_{i}: {arch}, region={region.bounds_lower}->{region.bounds_upper}\n"

        repr_str += ")"
        return repr_str
