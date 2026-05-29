"""Adaptive Tree-of-Experts PINN using only leaf experts.

Only leaf experts (those with no children) participate in the solution.
When a parent gets children, the parent is removed from the leaf set and
its children are added. Children copy their parent's weights at spawn time.

Soft blending (partition of unity) over leaves:
    u(x,t) = Σ_{j ∈ leaves} ψ̃_j(x,t) · u_j(x,t)
    where ψ̃_j = ψ_j / Σ_{k ∈ leaves} ψ_k
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


class AToELeaves(nn.Module):

    supports_decomposed = True

    def __init__(
        self,
        base_architecture: List[int],
        activation: str,
        config: Dict,
        adaptive_config: Dict
    ):
        super().__init__()

        self.base_architecture = base_architecture
        self.activation = activation
        self.config = config
        self.adaptive_config = adaptive_config

        self.max_experts = adaptive_config['max_experts']
        self.sigma_fraction = adaptive_config['sigma_fraction']
        self.base_weight = adaptive_config['base_weight']
        self.base_everywhere = adaptive_config['base_everywhere']
        self.freeze_mode = adaptive_config['freeze_mode']
        self.expert_type = adaptive_config['expert_type']
        self.blending_mode = 'soft'

        self.atoe_threshold_capacity = adaptive_config.get(
            'AToE_threshold_capacity', None
        )
        problem = config['problem']
        problem_config = config[problem]
        self.input_dim = base_architecture[0]
        self.output_dim = problem_config['output_dim']
        if self.atoe_threshold_capacity is not None:
            # Read variable_for_expert_size and corresponding threshold
            self.variable_for_expert_size = adaptive_config['variable_for_expert_size']
            if self.variable_for_expert_size == 'norm':
                self.expert_size_threshold = problem_config['wavelet_threshold']
            elif self.variable_for_expert_size == 'new_norm':
                self.expert_size_threshold = problem_config['new_norm_threshold']
            elif self.variable_for_expert_size == 'smoothness':
                self.expert_size_threshold = problem_config['tree_smoothness_threshold']
            else:
                self.expert_size_threshold = 1.0

        self.leaf_indices: Set[int] = {-1}

        self.config_base_architecture = base_architecture

        self.base_model = create_network(
            base_architecture, activation, config,
            is_base=True, expert_type=self.expert_type
        )

        self.experts = nn.ModuleList()
        self.regions: List[RegionDescriptor] = []

        self.batched_indicators = BatchedIndicators(base_weight=self.base_weight)

        self.activations: Dict[str, torch.Tensor] = {}
        self.hook_handles: List[RemovableHandle] = []

        self._timer = None

    @property
    def num_experts(self) -> int:
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
        result = []
        for r in self.regions:
            if r.depth == depth:
                if before_epoch is not None and r.spawn_epoch >= before_epoch:
                    continue
                result.append(r)
        return result

    def get_experts_at_depth(self, depth: int) -> List[tuple]:
        result = []
        for expert, region in zip(self.experts, self.regions):
            if region.depth == depth:
                result.append((expert, region))
        return result

    def get_highest_depth(self) -> int:
        if not self.regions:
            return 0
        return max(r.depth for r in self.regions)

    def get_union_mask_at_depth(self, inputs: torch.Tensor, depth: int, before_epoch: int = None) -> torch.Tensor:
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
        return sum(1 for r in self.regions if r.depth == depth)

    def get_children_of_parent(self, parent_idx: int, before_epoch: int = None) -> List[RegionDescriptor]:
        result = []
        for r in self.regions:
            if r.parent_idx == parent_idx:
                if before_epoch is not None and r.spawn_epoch >= before_epoch:
                    continue
                result.append(r)
        return result

    def get_mask_for_expert(self, inputs: torch.Tensor, expert_idx: int) -> torch.Tensor:
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
        if self.atoe_threshold_capacity is None:
            return self.config_base_architecture

        from models.architecture_bank import get_architecture_for_capacity
        
        # Get the metric value based on configured variable
        if self.variable_for_expert_size == 'norm':
            metric_value = region.wavelet_norm_squared
        elif self.variable_for_expert_size == 'new_norm':
            metric_value = region.new_wavelet_norm_squared
        elif self.variable_for_expert_size == 'smoothness':
            metric_value = region.smoothness_alpha if region.smoothness_alpha is not None else 0.0
        else:
            metric_value = region.wavelet_norm_squared
        
        ratio = max(metric_value / self.expert_size_threshold, 1.0)
        target_capacity = self.atoe_threshold_capacity * ratio
        return get_architecture_for_capacity(
            target_capacity, self.input_dim, self.output_dim
        )

    def sync_batched_indicators(self) -> None:
        if not self.regions:
            return

        device = next(self.base_model.parameters()).device
        self.batched_indicators.update(
            regions=self.regions,
            device=device,
            mode='soft',
            sigma_fraction=self.sigma_fraction
        )

    def spawn_expert(self, region: RegionDescriptor, copy_from_idx: Optional[int] = None) -> int:
        expert_idx = len(self.experts)
        architecture = self.get_expert_architecture(region)
        device = next(self.base_model.parameters()).device

        if copy_from_idx is not None and self.atoe_threshold_capacity is None:
            expert = create_network(
                architecture, self.activation, self.config,
                is_base=True, expert_type=self.expert_type
            )
            expert = expert.to(device)
            # Weight copy handled by apply_parent_copy_init in trainer.py
        else:
            expert = create_network(
                architecture, self.activation, self.config,
                is_base=True, expert_type=self.expert_type
            )
            expert = expert.to(device)

        self.experts.append(expert)
        self.regions.append(region)

        self.leaf_indices.add(expert_idx)
        self.leaf_indices.discard(region.parent_idx)

        parent_info = f"Base Model" if region.parent_idx == -1 else f"E{region.parent_idx + 1}"
        print(f"  Spawned Expert {expert_idx + 1} (depth={region.depth}, parent={parent_info}):")
        print(f"    Architecture: {architecture}")
        print(f"    Region bounds: {region.bounds_lower} -> {region.bounds_upper}")
        print(f"    Wavelet norm: {region.wavelet_norm_squared:.6f}")
        print(f"    Spawn epoch: {region.spawn_epoch}")

        self.sync_batched_indicators()

        return expert_idx

    def freeze_models(self, mode: Optional[str] = None):
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

        if -1 not in self.leaf_indices:
            for param in self.base_model.parameters():
                param.requires_grad = False
        for i, expert in enumerate(self.experts):
            for param in expert.parameters():
                param.requires_grad = (i in self.leaf_indices)

    def freeze_base_model(self) -> None:
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_experts(self) -> None:
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        threshold = self.adaptive_config.get('expert_activation_threshold', None)
        if threshold is not None:
            threshold = float(threshold)

        if threshold is not None and len(self.leaf_indices - {-1}) > 0:
            return self._forward_soft_sparse_only_leaves(inputs, threshold)
        else:
            return self._forward_soft_only_leaves(inputs)

    def _forward_soft_only_leaves(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Soft blending using only leaf experts.

        u(x,t) = Σ_{j ∈ leaves} ψ̃_j · u_j
        where ψ̃_j = ψ_j / Σ_{k ∈ leaves} ψ_k
        """
        if -1 in self.leaf_indices:
            return self.base_model(inputs)

        leaf_list = sorted(self.leaf_indices)
        _, psi_experts = self.batched_indicators(inputs)  # (N, K)
        psi_leaves = psi_experts[:, leaf_list]  # (N, L)
        psi_norm = psi_leaves / psi_leaves.sum(dim=1, keepdim=True)
        u_leaves = torch.stack([self.experts[i](inputs) for i in leaf_list], dim=1)
        return (psi_norm.unsqueeze(-1) * u_leaves).sum(dim=1)

    def _forward_soft_sparse_only_leaves(self, inputs: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Sparse soft blending using only leaf experts.

        Normalization uses the full set of leaves (same as non-sparse).
        Inactive leaves contribute 0 to the output (their u_k is not evaluated).
        """
        if -1 in self.leaf_indices:
            return self.base_model(inputs)

        leaf_list = sorted(self.leaf_indices)
        _, psi_experts = self.batched_indicators(inputs)  # (N, K)
        psi_leaves = psi_experts[:, leaf_list]  # (N, L)

        active_mask = psi_leaves > threshold  # (N, L)
        _ms = self.adaptive_config.get(
            'relevant_samples_to_activate_expert', 0)
        active_any = active_mask.sum(dim=0) > _ms  # (L,)
        active_local_indices = torch.nonzero(active_any, as_tuple=True)[0]

        psi_sum = psi_leaves.sum(dim=1, keepdim=True)
        psi_norm = psi_leaves / psi_sum

        if len(active_local_indices) == 0:
            u_leaves = torch.stack([self.experts[i](inputs) for i in leaf_list], dim=1)
            return (psi_norm.unsqueeze(-1) * u_leaves).sum(dim=1)

        N = inputs.shape[0]
        output_dim = self.base_model.layers[-1]
        device = inputs.device
        u_leaves = torch.zeros(N, len(leaf_list), output_dim, device=device, dtype=inputs.dtype)
        for local_idx in active_local_indices:
            expert_idx = leaf_list[local_idx.item()]
            u_leaves[:, local_idx.item(), :] = self.experts[expert_idx](inputs)

        return (psi_norm.unsqueeze(-1) * u_leaves).sum(dim=1)

    def forward_for_pde_derivatives(self, inputs: torch.Tensor) -> dict:
        """
        Forward pass returning decomposed components for product-rule derivative computation.

        Returns individual expert outputs and their normalized weights so the loss
        function can compute PDE derivatives via the product rule.

        u(x,t) = Σ_{j ∈ leaves} ψ̃_j · u_j, normalized over leaves only.
        """
        _t = self._timer
        N = inputs.shape[0]
        output_dim = self.base_model.layers[-1]
        device = inputs.device

        if -1 in self.leaf_indices:
            inputs_base = inputs.detach().clone().requires_grad_(True)
            u_base = self.base_model(inputs_base)
            components = [{
                'u': u_base,
                'inputs': inputs_base,
                'psi_norm': torch.ones(N, 1, device=device, dtype=inputs.dtype),
                'constant_psi': True,
            }]
            return {
                'components': components,
                'composed': u_base,
                'indicator_data': {
                    'all_lower': None, 'all_upper': None, 'all_sigma': None,
                    'psi_base': torch.ones(N, 1, device=device),
                    'psi_experts_filtered': torch.zeros(N, 0, device=device),
                    'active_expert_indices': [],
                },
            }

        leaf_list = sorted(self.leaf_indices)

        if _t: _t.start('fwd.compute_masks')
        _, psi_experts = self.batched_indicators(inputs)  # (N, K)
        if _t: _t.stop('fwd.compute_masks')

        psi_leaves = psi_experts[:, leaf_list]  # (N, L)

        Z = psi_leaves.sum(dim=1, keepdim=True)  # (N, 1)
        psi_norm_leaves = psi_leaves / Z  # (N, L)

        if _t: _t.start('fwd.sparse_eval')
        components = []
        for local_idx, expert_idx in enumerate(leaf_list):
            inputs_k = inputs.detach().clone().requires_grad_(True)
            u_k = self.experts[expert_idx](inputs_k)
            components.append({
                'u': u_k,
                'inputs': inputs_k,
                'psi_norm': psi_norm_leaves[:, local_idx:local_idx+1],
                'constant_psi': False,
            })
        if _t: _t.stop('fwd.sparse_eval')

        composed = torch.zeros(N, output_dim, device=device, dtype=inputs.dtype)
        for c in components:
            composed = composed + c['psi_norm'].detach() * c['u']

        active_expert_indices = torch.tensor(leaf_list, device=device)

        psi_experts_filtered = torch.zeros_like(psi_experts)
        for local_idx, expert_idx in enumerate(leaf_list):
            psi_experts_filtered[:, expert_idx] = psi_leaves[:, local_idx]

        indicator_data = {
            'all_lower': self.batched_indicators.all_lower,
            'all_upper': self.batched_indicators.all_upper,
            'all_sigma': self.batched_indicators.all_sigma,
            'psi_base': torch.zeros(N, 1, device=device),
            'psi_experts_filtered': psi_experts_filtered,
            'active_expert_indices': active_expert_indices,
        }

        return {
            'components': components,
            'composed': composed,
            'indicator_data': indicator_data,
        }

    def forward_decomposed(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning individual model contributions.

        Returns:
            Dict with per-expert outputs, composed output, masks, and normalized weights.
        """
        result = {}
        N = inputs.shape[0]
        output_dim = self.base_model.layers[-1]
        device = inputs.device

        if -1 in self.leaf_indices:
            u_base = self.base_model(inputs)
            result['base'] = u_base
            result['composed'] = u_base
            result['masks'] = {}
            result['weights_normalized'] = {'base': torch.ones(N, 1, device=device)}
            result['blending_mode_info'] = 'only_leaves'
            return result

        leaf_list = sorted(self.leaf_indices)
        _, psi_experts = self.batched_indicators(inputs)  # (N, K)
        psi_leaves = psi_experts[:, leaf_list]  # (N, L)
        psi_sum = psi_leaves.sum(dim=1, keepdim=True)
        psi_norm = psi_leaves / psi_sum

        result['masks'] = {}
        result['weights_normalized'] = {}
        result['blending_mode_info'] = 'only_leaves'

        u_total = torch.zeros(N, output_dim, device=device, dtype=inputs.dtype)
        for local_idx, expert_idx in enumerate(leaf_list):
            u_k = self.experts[expert_idx](inputs)
            result[f'expert_{expert_idx}'] = u_k
            result['masks'][f'expert_{expert_idx}'] = psi_leaves[:, local_idx:local_idx+1]
            result['weights_normalized'][f'expert_{expert_idx}'] = psi_norm[:, local_idx:local_idx+1]
            u_total = u_total + psi_norm[:, local_idx:local_idx+1] * u_k

        result['composed'] = u_total
        return result

    def get_layer_names(self) -> List[str]:
        return self.base_model.get_layer_names()

    def register_ncc_hooks(
        self,
        layer_names: List[str],
        keep_gradients: bool = False
    ) -> List[RemovableHandle]:
        self.remove_hooks()
        self.activations = {}

        handles = self.base_model.register_ncc_hooks(layer_names, keep_gradients)
        self.hook_handles = handles

        return handles

    def remove_hooks(self):
        self.base_model.remove_hooks()
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.activations = {}

    @property
    def activations(self) -> Dict[str, torch.Tensor]:
        return self.base_model.activations

    @activations.setter
    def activations(self, value):
        self._activations = value

    def get_domain_bounds(self) -> Dict[str, List[float]]:
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
            'leaf_indices': sorted(self.leaf_indices),
        }

    def load_state_dict_extended(self, state_dict: Dict):
        saved_base_arch = state_dict.get('base_architecture')
        saved_activation = state_dict.get('activation', self.activation)
        saved_adaptive = state_dict.get('adaptive_config', {})
        saved_expert_type = saved_adaptive.get('expert_type', 'mlp')

        if saved_base_arch is None:
            saved_base_arch = self._infer_architecture_from_state_dict(state_dict['base_model'])

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

        if 'leaf_indices' in state_dict:
            self.leaf_indices = set(state_dict['leaf_indices'])

        self.sync_batched_indicators()
        self.freeze_models()

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

    def __repr__(self) -> str:
        base_str = " -> ".join(map(str, self.base_architecture))
        expert_archs = [
            " -> ".join(map(str, self.get_expert_architecture(i)))
            for i in range(len(self.experts))
        ]

        repr_str = (
            f"AToELeaves(\n"
            f"  base: {base_str}\n"
            f"  activation: {self.activation}\n"
            f"  blending: soft (leaves only)\n"
            f"  num_experts: {len(self.experts)}/{self.max_experts}\n"
            f"  leaf_indices: {sorted(self.leaf_indices)}\n"
        )

        for i, (arch, region) in enumerate(zip(expert_archs, self.regions)):
            leaf_marker = " [LEAF]" if i in self.leaf_indices else ""
            repr_str += f"  expert_{i}: {arch}, region={region.bounds_lower}->{region.bounds_upper}{leaf_marker}\n"

        repr_str += ")"
        return repr_str
