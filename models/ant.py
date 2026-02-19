"""Adaptive Neural Tree (ANT) model.

Implements a tree-structured PINN where each expert takes its
parent's last hidden layer activation as input rather than the
raw (x,t) coordinates. Only leaf nodes contribute to the final
solution via a partition of unity.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from collections import defaultdict
from torch.utils.hooks import RemovableHandle

from models.fc_model import FCNet
from adaptive.indicators import (
    RegionDescriptor,
    BatchedIndicators,
)


class ANT(nn.Module):

    supports_decomposed = False

    def __init__(
        self,
        base_architecture: List[int],
        activation: str,
        config: Dict,
        adaptive_config: Dict,
    ):
        super().__init__()

        self.base_architecture = base_architecture
        self.activation = activation
        self.config = config
        self.adaptive_config = adaptive_config

        self.max_experts = adaptive_config.get(
            'max_experts', 5
        )
        self.max_depth = adaptive_config.get(
            'max_depth', 5
        )
        self.blending_mode = adaptive_config.get(
            'blending_mode', 'soft'
        )
        self.sigma_fraction = adaptive_config.get(
            'sigma_fraction', 0.2
        )
        self.base_weight = adaptive_config.get(
            'base_weight', 1.0
        )
        self.freeze_mode = adaptive_config.get(
            'freeze_mode', 'none'
        )
        ant_layer_ratio = adaptive_config.get(
            'ANT_layer_size_norm_ratio', [30]
        )
        if isinstance(ant_layer_ratio, float):
            raise NotImplementedError(
                "ANT_layer_size_norm_ratio as a float ratio is not yet "
                "implemented. Provide a list of hidden layer sizes instead."
            )
        self.expert_hidden_layers = ant_layer_ratio

        problem = config['problem']
        problem_config = config[problem]
        self.output_dim = problem_config.get(
            'output_dim', 2
        )

        self.base_model = FCNet(
            base_architecture, activation, config,
            is_base=True,
        )

        self.experts = nn.ModuleList()
        self.regions: List[RegionDescriptor] = []

        self.parent_indices: List[int] = []
        self.depths: List[int] = []
        self.leaf_status: List[bool] = []
        self.base_is_leaf: bool = True
        self.experts_by_depth: defaultdict = defaultdict(
            list
        )

        self.batched_indicators = BatchedIndicators(
            base_weight=self.base_weight
        )

        self.activations: Dict[str, torch.Tensor] = {}
        self.hook_handles: List[RemovableHandle] = []
        self._timer = None

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def get_highest_depth(self) -> int:
        if not self.depths:
            return 0
        return max(self.depths)

    def get_regions_at_depth(
        self, depth: int, before_epoch: int = None
    ) -> List[RegionDescriptor]:
        result = []
        for i, r in enumerate(self.regions):
            if self.depths[i] == depth:
                if (before_epoch is not None
                        and r.spawn_epoch >= before_epoch):
                    continue
                result.append(r)
        return result

    def get_experts_at_depth(
        self, depth: int
    ) -> List[tuple]:
        result = []
        for i, (expert, region) in enumerate(
            zip(self.experts, self.regions)
        ):
            if self.depths[i] == depth:
                result.append((expert, region))
        return result

    def count_experts_at_depth(self, depth: int) -> int:
        return sum(1 for d in self.depths if d == depth)

    def get_children_of_parent(
        self, parent_idx: int,
        before_epoch: int = None,
    ) -> List[RegionDescriptor]:
        result = []
        for i, r in enumerate(self.regions):
            if self.parent_indices[i] == parent_idx:
                if (before_epoch is not None
                        and r.spawn_epoch >= before_epoch):
                    continue
                result.append(r)
        return result

    def get_leaf_info(self):
        result = []
        if self.base_is_leaf:
            result.append((None, -1))
        for idx, is_leaf in enumerate(self.leaf_status):
            if is_leaf:
                result.append((self.regions[idx], idx))
        return result

    def sync_batched_indicators(self) -> None:
        if not self.regions:
            return
        device = next(self.base_model.parameters()).device
        self.batched_indicators.update(
            regions=self.regions,
            device=device,
            mode=self.blending_mode,
            sigma_fraction=self.sigma_fraction,
        )

    def spawn_expert(
        self, region: RegionDescriptor
    ) -> int:
        if len(self.experts) >= self.max_experts:
            print(
                f"  Cannot spawn more experts: "
                f"max_experts={self.max_experts} reached"
            )
            return -1

        expert_idx = len(self.experts)
        parent_idx = region.parent_idx
        depth = region.depth

        if parent_idx == -1:
            parent_model = self.base_model
        else:
            parent_model = self.experts[parent_idx]

        parent_act_dim = parent_model.get_activation_dim()
        architecture = (
            [parent_act_dim]
            + self.expert_hidden_layers
            + [self.output_dim]
        )

        device = next(
            self.base_model.parameters()
        ).device
        expert = FCNet(
            architecture, self.activation, self.config,
            is_base=False,
        )
        expert = expert.to(device)

        layer_names = expert.get_layer_names()
        if layer_names:
            final_layer = expert.network[layer_names[-1]]
            nn.init.zeros_(final_layer.weight)
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)

        self.experts.append(expert)
        self.regions.append(region)
        self.parent_indices.append(parent_idx)
        self.depths.append(depth)
        self.leaf_status.append(True)
        self.experts_by_depth[depth].append(expert_idx)

        if parent_idx == -1:
            self.base_is_leaf = False
        else:
            self.leaf_status[parent_idx] = False

        self.sync_batched_indicators()

        parent_info = (
            "Base Model"
            if parent_idx == -1
            else f"E{parent_idx + 1}"
        )
        print(
            f"  Spawned Expert {expert_idx + 1} "
            f"(depth={depth}, parent={parent_info}):"
        )
        print(f"    Architecture: {architecture}")
        print(
            f"    Region bounds: "
            f"{region.bounds_lower} -> "
            f"{region.bounds_upper}"
        )
        print(
            f"    Residual-weighted wavelet norm: "
            f"{region.wavelet_norm:.6f}"
        )
        print(f"    Spawn epoch: {region.spawn_epoch}")

        return expert_idx

    def _collect_leaf_info(self):
        """Build ordered lists of leaf expert indices and their regions."""
        leaf_expert_indices = []
        leaf_regions = []
        if self.base_is_leaf:
            leaf_expert_indices.append(-1)
            leaf_regions.append(None)
        for idx, is_leaf in enumerate(self.leaf_status):
            if is_leaf:
                leaf_expert_indices.append(idx)
                leaf_regions.append(self.regions[idx])
        return leaf_expert_indices, leaf_regions

    def _compute_leaf_psi(self, inputs, leaf_expert_indices, leaf_regions):
        """Compute raw and normalized psi for leaves, with sparse filtering.

        Returns:
            psi_normalized: (N, num_active_leaves) normalized weights
            active_leaf_local: list of local indices into the leaf lists
                               that passed the activation threshold
        """
        N = inputs.size(0)
        device = inputs.device
        dtype = inputs.dtype
        num_leaves = len(leaf_regions)

        threshold = self.adaptive_config.get(
            'expert_activation_threshold', None
        )
        if threshold is not None:
            threshold = float(threshold)

        psi_raw = torch.empty(
            N, num_leaves, device=device, dtype=dtype
        )

        if self.regions:
            _, psi_experts = self.batched_indicators(inputs)
        else:
            psi_experts = None

        for i, region in enumerate(leaf_regions):
            if region is None:
                psi_raw[:, i] = self.base_weight
            else:
                region_idx = self.regions.index(region)
                psi_raw[:, i] = psi_experts[:, region_idx]

        if threshold is not None and num_leaves > 1:
            active_mask = psi_raw > threshold
            active_any = active_mask.sum(dim=0) > 20
            active_leaf_local = torch.nonzero(
                active_any, as_tuple=True
            )[0].tolist()

            if len(active_leaf_local) == 0:
                active_leaf_local = list(range(num_leaves))
            else:
                psi_raw = psi_raw * active_mask.float()
        else:
            active_leaf_local = list(range(num_leaves))

        psi_sum = psi_raw.sum(
            dim=1, keepdim=True
        ).clamp(min=1e-8)
        psi_normalized = psi_raw / psi_sum

        return psi_normalized, active_leaf_local

    def _build_path(self, eidx: int) -> list:
        """Return root-to-leaf path as list of expert indices (excludes -1 base)."""
        path = []
        cur = eidx
        while cur >= 0:
            path.append(cur)
            cur = self.parent_indices[cur]
        path.reverse()
        return path

    def _eval_active_paths(self, inputs, node_outputs, node_activations,
                           leaf_expert_indices, active_leaf_local):
        """Evaluate only the expert paths leading to active leaves."""
        for local_idx in active_leaf_local:
            eidx = leaf_expert_indices[local_idx]
            if eidx == -1:
                continue
            for node_idx in self._build_path(eidx):
                if node_idx in node_outputs:
                    continue
                pidx = self.parent_indices[node_idx]
                A_parent = node_activations[pidx]
                u_i, A_i = self.experts[node_idx](
                    A_parent, return_activation=True
                )
                node_outputs[node_idx] = u_i
                node_activations[node_idx] = A_i

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if len(self.experts) == 0:
            return self.base_model(inputs)

        _t = self._timer

        leaf_expert_indices, leaf_regions = self._collect_leaf_info()

        if _t: _t.start('fwd.compute_masks')
        psi_normalized, active_leaf_local = self._compute_leaf_psi(
            inputs, leaf_expert_indices, leaf_regions
        )
        if _t: _t.stop('fwd.compute_masks')

        if _t: _t.start('fwd.sparse_eval')
        u_base, A_base = self.base_model(
            inputs, return_activation=True
        )
        node_outputs = {-1: u_base}
        node_activations = {-1: A_base}

        self._eval_active_paths(
            inputs, node_outputs, node_activations,
            leaf_expert_indices, active_leaf_local
        )
        if _t: _t.stop('fwd.sparse_eval')

        if _t: _t.start('fwd.blend')
        leaf_outputs = [
            node_outputs[leaf_expert_indices[i]]
            for i in active_leaf_local
        ]
        psi_active = psi_normalized[:, active_leaf_local]

        leaf_stack = torch.stack(leaf_outputs, dim=1)
        result = (
            psi_active.unsqueeze(-1) * leaf_stack
        ).sum(dim=1)
        if _t: _t.stop('fwd.blend')

        return result

    def forward_decomposed(
        self, inputs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        N = inputs.size(0)
        device = inputs.device
        result = {}

        if len(self.experts) == 0:
            u_base = self.base_model(inputs)
            result['base'] = u_base
            result['composed'] = u_base
            result['masks'] = {}
            result['weights_normalized'] = {
                'base': torch.ones(
                    N, 1,
                    device=device, dtype=inputs.dtype,
                )
            }
            return result

        leaf_expert_indices, leaf_regions = self._collect_leaf_info()
        psi_normalized, active_leaf_local = self._compute_leaf_psi(
            inputs, leaf_expert_indices, leaf_regions
        )

        u_base, A_base = self.base_model(
            inputs, return_activation=True
        )
        node_outputs = {-1: u_base}
        node_activations = {-1: A_base}
        result['base'] = u_base

        self._eval_active_paths(
            inputs, node_outputs, node_activations,
            leaf_expert_indices, active_leaf_local
        )
        for eidx in node_outputs:
            if eidx >= 0:
                result[f'expert_{eidx}'] = node_outputs[eidx]

        leaf_outputs = [
            node_outputs[leaf_expert_indices[i]]
            for i in active_leaf_local
        ]
        psi_active = psi_normalized[:, active_leaf_local]

        leaf_stack = torch.stack(leaf_outputs, dim=1)
        composed = (
            psi_active.unsqueeze(-1) * leaf_stack
        ).sum(dim=1)
        result['composed'] = composed

        masks = {}
        weights_normalized = {}
        if self.regions:
            _, psi_experts = self.batched_indicators(inputs)
        else:
            psi_experts = None

        for li, local_idx in enumerate(active_leaf_local):
            eidx = leaf_expert_indices[local_idx]
            if eidx == -1:
                masks['base'] = torch.full(
                    (N, 1), self.base_weight,
                    device=device, dtype=inputs.dtype,
                )
                weights_normalized['base'] = (
                    psi_active[:, li:li + 1]
                )
            else:
                masks[f'expert_{eidx}'] = (
                    psi_experts[:, eidx:eidx + 1]
                )
                weights_normalized[f'expert_{eidx}'] = (
                    psi_active[:, li:li + 1]
                )

        result['masks'] = masks
        result['weights_normalized'] = weights_normalized
        return result

    def freeze_models(self, mode: Optional[str] = None):
        mode = mode or self.freeze_mode

        if mode == 'none':
            for p in self.base_model.parameters():
                p.requires_grad = True
            for expert in self.experts:
                for p in expert.parameters():
                    p.requires_grad = True

        elif mode == 'base_only':
            for p in self.base_model.parameters():
                p.requires_grad = False
            for expert in self.experts:
                for p in expert.parameters():
                    p.requires_grad = True

        elif mode == 'previous':
            for p in self.base_model.parameters():
                p.requires_grad = False
            for i, expert in enumerate(self.experts):
                is_leaf = self.leaf_status[i]
                for p in expert.parameters():
                    p.requires_grad = is_leaf

        else:
            raise ValueError(
                f"Unknown freeze_mode: {mode}"
            )

        if not self.base_is_leaf:
            for p in self.base_model.parameters():
                p.requires_grad = False
        for i, expert in enumerate(self.experts):
            if not self.leaf_status[i]:
                for p in expert.parameters():
                    p.requires_grad = False

    def get_domain_bounds(
        self,
    ) -> Dict[str, List[float]]:
        problem = self.config['problem']
        problem_config = self.config[problem]
        spatial_domain = problem_config['spatial_domain']
        temporal_domain = problem_config[
            'temporal_domain'
        ]

        if len(spatial_domain) == 1:
            return {
                'lower': [
                    spatial_domain[0][0],
                    temporal_domain[0],
                ],
                'upper': [
                    spatial_domain[0][1],
                    temporal_domain[1],
                ],
            }
        elif len(spatial_domain) == 2:
            return {
                'lower': [
                    spatial_domain[0][0],
                    spatial_domain[1][0],
                    temporal_domain[0],
                ],
                'upper': [
                    spatial_domain[0][1],
                    spatial_domain[1][1],
                    temporal_domain[1],
                ],
            }
        else:
            raise ValueError(
                f"Unsupported spatial dimension: "
                f"{len(spatial_domain)}"
            )

    def get_layer_names(self) -> List[str]:
        return self.base_model.get_layer_names()

    def register_ncc_hooks(
        self,
        layer_names: List[str],
        keep_gradients: bool = False,
    ) -> List[RemovableHandle]:
        self.remove_hooks()
        self.activations = {}
        handles = self.base_model.register_ncc_hooks(
            layer_names, keep_gradients
        )
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

    def get_mask_for_expert(
        self,
        inputs: torch.Tensor,
        expert_idx: int,
    ) -> torch.Tensor:
        if (expert_idx < 0
                or expert_idx >= len(self.regions)):
            return torch.ones(
                inputs.shape[0],
                dtype=torch.bool,
                device=inputs.device,
            )
        all_masks = (
            self.batched_indicators
            .compute_hard_masks_only(inputs)
        )
        if (all_masks.shape[1] == 0
                or expert_idx >= all_masks.shape[1]):
            return torch.ones(
                inputs.shape[0],
                dtype=torch.bool,
                device=inputs.device,
            )
        return all_masks[:, expert_idx].bool()

    def get_union_mask_at_depth(
        self,
        inputs: torch.Tensor,
        depth: int,
        before_epoch: int = None,
    ) -> torch.Tensor:
        N = inputs.shape[0]
        depth_indices = []
        for i, d in enumerate(self.depths):
            if d == depth:
                if (before_epoch is not None
                        and self.regions[i].spawn_epoch
                        >= before_epoch):
                    continue
                depth_indices.append(i)
        if not depth_indices:
            return torch.zeros(
                N, dtype=torch.bool,
                device=inputs.device,
            )
        all_masks = (
            self.batched_indicators
            .compute_hard_masks_only(inputs)
        )
        if all_masks.shape[1] == 0:
            return torch.zeros(
                N, dtype=torch.bool,
                device=inputs.device,
            )
        depth_masks = all_masks[:, depth_indices]
        return depth_masks.any(dim=1)

    def compute_children_coverage(
        self,
        inputs: torch.Tensor,
        parent_idx: int,
        before_epoch: int = None,
    ) -> float:
        all_masks = (
            self.batched_indicators
            .compute_hard_masks_only(inputs)
        )
        if parent_idx == -1:
            parent_mask = torch.ones(
                inputs.shape[0],
                dtype=torch.bool,
                device=inputs.device,
            )
        else:
            if parent_idx >= all_masks.shape[1]:
                return 0.0
            parent_mask = (
                all_masks[:, parent_idx].bool()
            )
        parent_count = parent_mask.sum().item()
        if parent_count == 0:
            return 0.0
        children_indices = []
        for i in range(len(self.parent_indices)):
            if self.parent_indices[i] == parent_idx:
                if (before_epoch is not None
                        and self.regions[i].spawn_epoch
                        >= before_epoch):
                    continue
                children_indices.append(i)
        if not children_indices:
            return 0.0
        children_masks = all_masks[:, children_indices]
        children_union = children_masks.any(dim=1)
        covered = (
            (parent_mask & children_union).sum().item()
        )
        return covered / parent_count

    def state_dict_extended(self) -> Dict:
        return {
            'base_model': self.base_model.state_dict(),
            'experts': [
                e.state_dict() for e in self.experts
            ],
            'regions': [
                r.to_dict() for r in self.regions
            ],
            'num_experts': len(self.experts),
            'base_architecture': self.base_architecture,
            'activation': self.activation,
            'adaptive_config': self.adaptive_config,
            'parent_indices': self.parent_indices,
            'depths': self.depths,
            'leaf_status': self.leaf_status,
            'base_is_leaf': self.base_is_leaf,
            'experts_by_depth': dict(
                self.experts_by_depth
            ),
        }

    def load_state_dict_extended(
        self, state_dict: Dict
    ):
        saved_base_arch = state_dict.get(
            'base_architecture'
        )
        saved_activation = state_dict.get(
            'activation', self.activation
        )

        if saved_base_arch is None:
            saved_base_arch = (
                self._infer_architecture_from_state_dict(
                    state_dict['base_model']
                )
            )

        if saved_base_arch != self.base_architecture:
            print(
                f"  Recreating base model: "
                f"{self.base_architecture} -> "
                f"{saved_base_arch}"
            )
            device = next(
                self.base_model.parameters()
            ).device
            self.base_model = FCNet(
                saved_base_arch, saved_activation,
                self.config, is_base=True,
            )
            self.base_model = self.base_model.to(device)
            self.base_architecture = saved_base_arch

        self.base_model.load_state_dict(
            state_dict['base_model']
        )

        self.experts = nn.ModuleList()
        self.regions = []
        self.parent_indices = state_dict.get(
            'parent_indices', []
        )
        self.depths = state_dict.get('depths', [])
        self.leaf_status = state_dict.get(
            'leaf_status', []
        )
        self.base_is_leaf = state_dict.get(
            'base_is_leaf', True
        )

        saved_by_depth = state_dict.get(
            'experts_by_depth', {}
        )
        self.experts_by_depth = defaultdict(list)
        for k, v in saved_by_depth.items():
            self.experts_by_depth[int(k)] = v

        device = next(
            self.base_model.parameters()
        ).device

        for i, (expert_state, region_dict) in enumerate(
            zip(
                state_dict['experts'],
                state_dict['regions'],
            )
        ):
            region = RegionDescriptor.from_dict(
                region_dict
            )
            expert_arch = (
                self._infer_architecture_from_state_dict(
                    expert_state
                )
            )
            expert = FCNet(
                expert_arch, self.activation,
                self.config, is_base=False,
            )
            expert.load_state_dict(expert_state)
            expert = expert.to(device)
            self.experts.append(expert)
            self.regions.append(region)

        self.sync_batched_indicators()

    @staticmethod
    def _infer_architecture_from_state_dict(
        state_dict: Dict,
    ) -> List[int]:
        architecture = []
        layer_idx = 1
        key = f'network.layer_{layer_idx}.weight'
        while key in state_dict:
            weight = state_dict[key]
            if layer_idx == 1:
                architecture.append(weight.shape[1])
            architecture.append(weight.shape[0])
            layer_idx += 1
            key = f'network.layer_{layer_idx}.weight'
        if not architecture:
            raise ValueError(
                "Could not infer architecture "
                "from state dict"
            )
        return architecture

    def __repr__(self) -> str:
        base_str = " -> ".join(
            map(str, self.base_architecture)
        )
        lines = [
            "ANT(",
            f"  base: {base_str}",
            f"  activation: {self.activation}",
            f"  blending: {self.blending_mode}",
            f"  num_experts: "
            f"{len(self.experts)}/{self.max_experts}",
            f"  base_is_leaf: {self.base_is_leaf}",
        ]
        for i, region in enumerate(self.regions):
            arch_str = " -> ".join(
                map(str, self.experts[i].layers)
            )
            parent_info = (
                "Base"
                if self.parent_indices[i] == -1
                else f"E{self.parent_indices[i] + 1}"
            )
            leaf_tag = (
                " [leaf]" if self.leaf_status[i]
                else ""
            )
            lines.append(
                f"  expert_{i}: {arch_str}, "
                f"depth={self.depths[i]}, "
                f"parent={parent_info}, "
                f"region="
                f"{region.bounds_lower}->"
                f"{region.bounds_upper}"
                f"{leaf_tag}"
            )
        lines.append(")")
        return "\n".join(lines)
