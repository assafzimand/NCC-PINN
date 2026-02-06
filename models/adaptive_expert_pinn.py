"""Adaptive Expert PINN with dynamic regional expert spawning.

Implements a composed PINN that combines:
- A global base model u_0(x,t) trained on the full domain
- Regional expert models u_i(x,t) that specialize on high-error regions

The composed solution depends on the mode:

**Hard blending:**
    u(x,t) = u_0(x,t) + Σ 1_Ωi(x,t) · u_i(x,t)

**Soft blending (partition of unity, non-pretrained):**
    u(x,t) = Σ_k ψ̃_k(x,t) · u_k(x,t)
    where ψ̃_k = ψ_k / Σ_j ψ_j (normalized, Σ ψ̃_k = 1)

**Soft blending (additive mode, pretrained_base_model=True):**
    u(x,t) = u_0(x,t) + Σ_k ψ̃_k(x,t) · u_k(x,t)
    where ψ̃_k = ψ_k / Σ_j ψ_j for j=1..K (experts only)
    Base contributes with ψ_0 = 1 everywhere (not normalized).

**Architecture support:**
The implementation uses BatchedModels to support heterogeneous architectures
by grouping models with the same architecture and computing each group in parallel.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from torch.utils.hooks import RemovableHandle
from torch.func import stack_module_state, functional_call, vmap
from pathlib import Path

from models.fc_model import FCNet
from adaptive.indicators import (
    RegionDescriptor, 
    HardIndicator, 
    SoftIndicator,
    UniformIndicator,
    BatchedIndicators
)

class BatchedModels:
    """
    Batched model forward pass using vmap + functional_call.
    
    Computes all models (base + experts) in parallel, supporting HETEROGENEOUS
    architectures by grouping models with the same architecture and running
    vmap(functional_call) per group. Different architecture groups run on
    separate CUDA streams for concurrent execution.
    
    The output tensor indexes are: [0] = base, [1..K] = experts
    
    Note: stack_module_state is called every forward so gradients flow to
    the original model parameters during training.
    """
    
    def __init__(self, activation_fn: nn.Module):
        """
        Args:
            activation_fn: Activation function module (e.g., nn.Tanh())
        """
        self.activation_fn = activation_fn
        self._models: List[nn.Module] = []
        self._groups: Dict[Tuple[int, ...], Dict] = {}
        # {arch_tuple: {'indices': [int], 'template': FCNet, 'models': [FCNet]}}
    
    def sync_from_models(self, base_model: nn.Module, experts: nn.ModuleList) -> None:
        """
        Register all models (base + experts) for batched forward pass.
        
        Groups models by architecture to support heterogeneous expert architectures.
        
        Call this after spawning new experts or loading pretrained base.
        
        Args:
            base_model: Base FCNet model (index 0)
            experts: ModuleList of expert FCNets (indices 1..K)
        """
        # Build unified model list: [base, expert_0, expert_1, ...]
        self._models = [base_model] + list(experts)
        
        # Group models by architecture
        self._groups = {}
        for idx, model in enumerate(self._models):
            arch_tuple = tuple(model.layers)
            if arch_tuple not in self._groups:
                self._groups[arch_tuple] = {
                    'indices': [],
                    'template': model,  # Use first model as functional_call template
                    'models': []
                }
            self._groups[arch_tuple]['indices'].append(idx)
            self._groups[arch_tuple]['models'].append(model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Batched forward pass using vmap + functional_call.
        
        Per architecture group: stack_module_state to get batched params,
        then vmap(functional_call) to run all models in parallel.
        Different groups run on separate CUDA streams.
        
        Args:
            x: Input tensor (N, input_dim)
            
        Returns:
            Output tensor (N, K+1, output_dim) where:
                - [:, 0, :] is base model output
                - [:, 1:, :] are expert outputs
        """
        if len(self._models) == 0:
            raise RuntimeError("No models registered. Call sync_from_models first.")
        
        N = x.shape[0]
        num_models = len(self._models)
        output_dim = self._models[0].layers[-1]
        
        # Initialize output tensor
        outputs = torch.zeros((N, num_models, output_dim), device=x.device, dtype=x.dtype)
        
        use_streams = (x.is_cuda and len(self._groups) > 1)
        
        if use_streams:
            # Multiple architecture groups on CUDA: run on parallel streams
            streams = []
            results = {}
            
            for arch_tuple, group in self._groups.items():
                stream = torch.cuda.Stream(device=x.device)
                streams.append(stream)
                with torch.cuda.stream(stream):
                    results[arch_tuple] = self._forward_group(
                        x, group['template'], group['models']
                    )
            
            # Synchronize all streams
            for stream in streams:
                stream.synchronize()
            
            # Scatter results
            for arch_tuple, group in self._groups.items():
                group_out = results[arch_tuple]  # (K_group, N, output_dim)
                for i, model_idx in enumerate(group['indices']):
                    outputs[:, model_idx, :] = group_out[i]
        else:
            # Single group or CPU: no stream overhead
            for arch_tuple, group in self._groups.items():
                group_out = self._forward_group(
                    x, group['template'], group['models']
                )  # (K_group, N, output_dim)
                for i, model_idx in enumerate(group['indices']):
                    outputs[:, model_idx, :] = group_out[i]
        
        return outputs  # (N, K+1, output_dim)
    
    def _forward_group(
        self,
        x: torch.Tensor,
        template: nn.Module,
        models: List[nn.Module]
    ) -> torch.Tensor:
        """
        Forward pass for one architecture group using vmap + functional_call.
        
        Args:
            x: Input tensor (N, input_dim)
            template: Template model for functional_call
            models: List of models in this group
            
        Returns:
            (K_group, N, output_dim) tensor of outputs
        """
        if len(models) == 1:
            # Single model: direct forward, avoid vmap overhead
            return models[0](x).unsqueeze(0)  # (1, N, output_dim)
        
        # Stack parameters from all models in this group
        params, buffers = stack_module_state(models)
        
        # Define single-model forward using functional_call
        def single_forward(params, buffers, x):
            return functional_call(template, (params, buffers), (x,))
        
        # vmap over model dimension (dim 0 of params/buffers), broadcast input x
        batched_forward = vmap(single_forward, in_dims=(0, 0, None))
        
        return batched_forward(params, buffers, x)  # (K_group, N, output_dim)


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
        
        # Create uniform indicator for base model (used in soft blending for partition of unity)
        self.base_indicator: Optional[UniformIndicator] = None
        if self.blending_mode == 'soft':
            self.base_indicator = UniformIndicator(self.base_weight)
        
        # Expert storage
        self.experts = nn.ModuleList()
        self.regions: List[RegionDescriptor] = []
        
        # Indicator storage - always have both for clarity
        # hard_indicators: used for coverage/overlap checks (always created)
        # soft_indicators: used for forward pass blending (only created when blending_mode == 'soft')
        self.hard_indicators: List[HardIndicator] = []
        self.soft_indicators: List[SoftIndicator] = []
        
        # Batched indicators for vectorized computation (GPU optimization)
        # Computes all K expert masks in a single GPU operation
        self.batched_indicators = BatchedIndicators(base_weight=self.base_weight)
        
        # Batched models for O(1) forward pass (GPU optimization)
        # Computes all models (base + experts) with support for heterogeneous architectures
        self.batched_models = BatchedModels(
            activation_fn=self.base_model.activation
        )
        # Initial sync with base model only (no experts yet)
        self.batched_models.sync_from_models(self.base_model, self.experts)
        
        # Hook management
        self.activations: Dict[str, torch.Tensor] = {}
        self.hook_handles: List[RemovableHandle] = []
        
        # Flag to indicate if base model is frozen (pretrained)
        # When True, caller should pass precomputed u_base to forward()
        self._base_frozen = False
    
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
        
        When pretrained_base_model is used, experts use the config architecture,
        not the pretrained base architecture.
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
        
        # DIAGNOSTIC: Verify expert is on correct device
        actual_device = next(expert.parameters()).device
        print(f"    Expert created on device: {actual_device}")
        
        # Store expert and region
        self.experts.append(expert)
        self.regions.append(region)
        
        # Always create hard indicator (used for coverage/overlap checks)
        hard_indicator = HardIndicator(region)
        self.hard_indicators.append(hard_indicator)
        
        # Create soft indicator only if blending_mode is 'soft' (used for forward pass)
        if self.blending_mode == 'soft':
            soft_indicator = SoftIndicator(region, sigma_fraction=self.sigma_fraction)
            self.soft_indicators.append(soft_indicator)
        
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
    
    def load_pretrained_base(self, checkpoint_path: str) -> None:
        """
        Load a pretrained base model from checkpoint and freeze its weights.
        
        This is used for the pretrained_base_model workflow where:
        1. Base model is loaded from a previously trained checkpoint
        2. Base weights are frozen (never updated)
        3. Expert tree is built based on base model predictions only
        4. All experts are trained after tree building
        
        The base model architecture is determined FROM THE CHECKPOINT, not from
        the current config. This allows loading any pretrained model regardless
        of the expert architecture specified in config.
        
        Args:
            checkpoint_path: Path to checkpoint file (expects either standard 
                           checkpoint format with 'model_state_dict' or
                           adaptive format with 'adaptive_state')
                           Path separators are automatically normalized for cross-platform compatibility.
        """
        import os
        # Normalize path for cross-platform compatibility
        # Replace backslashes with forward slashes (forward slashes work on both Windows and Linux)
        # Then use normpath to clean up any double slashes, etc.
        checkpoint_path = checkpoint_path.replace('\\', '/')
        checkpoint_path = os.path.normpath(checkpoint_path)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pretrained base checkpoint not found: {checkpoint_path}")
        
        print(f"\n{'='*60}")
        print(f"Loading pretrained base model from: {checkpoint_path}")
        print(f"{'='*60}")
        
        # CRITICAL: Save device BEFORE any model recreation
        device = next(iter(self.base_model.parameters())).device
        print(f"  Current device: {device}")
        
        # Use weights_only=False to support checkpoints with numpy arrays (PyTorch 2.6+)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats and extract architecture
        pretrained_architecture = None
        pretrained_activation = None
        
        if 'adaptive_state' in checkpoint:
            # Adaptive PINN checkpoint - extract base model state and architecture
            adaptive_state = checkpoint['adaptive_state']
            base_state_dict = adaptive_state['base_model']
            pretrained_architecture = adaptive_state.get('base_architecture')
            pretrained_activation = adaptive_state.get('activation')
            print("  Loaded from adaptive PINN checkpoint (using base model only)")
        elif 'model_state_dict' in checkpoint:
            # Standard PINN checkpoint
            base_state_dict = checkpoint['model_state_dict']
            # Try to get architecture from config stored in checkpoint
            if 'config' in checkpoint:
                pretrained_architecture = checkpoint['config'].get('architecture')
                pretrained_activation = checkpoint['config'].get('activation')
            print("  Loaded from standard PINN checkpoint")
        else:
            # Try to load directly (raw state dict)
            base_state_dict = checkpoint
            print("  Loaded raw state dict")
        
        # If we couldn't find architecture in checkpoint, infer from state dict
        if pretrained_architecture is None:
            pretrained_architecture = self._infer_architecture_from_state_dict(base_state_dict)
            print(f"  Inferred architecture from weights: {pretrained_architecture}")
        
        if pretrained_activation is None:
            pretrained_activation = self.activation  # Fall back to current config
        
        # Check if we need to recreate the base model with different architecture
        if pretrained_architecture != self.base_architecture:
            print(f"  Pretrained architecture: {pretrained_architecture}")
            print(f"  Config architecture: {self.base_architecture}")
            print(f"  Recreating base model to match pretrained architecture...")
            
            # Recreate base model with pretrained architecture
            self.base_model = FCNet(pretrained_architecture, pretrained_activation, self.config)
            
            # Update stored architecture
            self.base_architecture = pretrained_architecture
        
        # Load weights into base model
        self.base_model.load_state_dict(base_state_dict)
        
        # Move to correct device AFTER loading (device was saved at start)
        self.base_model = self.base_model.to(device)
        print(f"  Moved base model to device: {device}")
        
        # Freeze base model weights
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Mark base as frozen
        self._base_frozen = True
        
        # Sync batched models after loading new base architecture
        self.sync_batched_models()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.base_model.parameters())
        print(f"  Base model architecture: {pretrained_architecture}")
        print(f"  Base model parameters: {total_params:,}")
        print(f"  Base model frozen: True (requires_grad=False)")
        
        # Note: Experts use config architecture (via get_expert_architecture)
        # BatchedModels supports heterogeneous architectures via grouping
        expert_arch = self.expert_architectures or self.config_base_architecture
        print(f"  Expert architecture: {expert_arch}")
        print(f"  (Batched models support mixed architectures)")
        
        print(f"{'='*60}\n")
    
    def _infer_architecture_from_state_dict(self, state_dict: Dict) -> List[int]:
        """
        Infer the network architecture from a state dict by examining layer shapes.
        
        Args:
            state_dict: Model state dictionary
            
        Returns:
            List of layer sizes [input_dim, hidden1, hidden2, ..., output_dim]
        """
        architecture = []
        layer_idx = 1
        
        while f'network.layer_{layer_idx}.weight' in state_dict:
            weight = state_dict[f'network.layer_{layer_idx}.weight']
            if layer_idx == 1:
                # First layer: input_dim is weight.shape[1]
                architecture.append(weight.shape[1])
            # Hidden/output size is weight.shape[0]
            architecture.append(weight.shape[0])
            layer_idx += 1
        
        if not architecture:
            raise ValueError("Could not infer architecture from state dict")
        
        return architecture
    
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
        # Compute all model outputs at once: [base, expert_0, ..., expert_K]
        u_all = self.batched_models.forward(inputs)  # (N, K+1, output_dim)
        
        # Extract base and experts
        u_base = u_all[:, 0, :]  # (N, output_dim)
        
        if len(self.experts) == 0:
            return u_base
        
        u_experts = u_all[:, 1:, :]  # (N, K, output_dim)
        
        # Compute masks (already vectorized)
        masks = self.batched_indicators.compute_hard_masks_only(inputs)  # (N, K)
        
        # Apply masks and sum - fully vectorized
        weighted_experts = masks.unsqueeze(-1) * u_experts  # (N, K, out_dim)
        u_total = u_base + weighted_experts.sum(dim=1)  # (N, out_dim)
        
        return u_total
    
    def _forward_soft(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Vectorized soft blending forward pass with partition-of-unity normalization.
        
        Standard mode (non-pretrained base):
            u(x,t) = Σ_k ψ̃_k(x,t) · u_k(x,t)
            where:
                - ψ_0 = uniform constant (base_weight) for base model
                - ψ_k = sigmoid-based bump function for expert k
                - ψ̃_k = ψ_k / Σ_j ψ_j (partition of unity: Σ ψ̃_k = 1)
        
        Additive mode (pretrained_base_model=True):
            u(x,t) = u_base + Σ_k ψ̃_k(x,t) · u_k(x,t)
            where:
                - Base contributes with weight 1 everywhere (not normalized)
                - ψ̃_k = ψ_k / Σ_j ψ_j for j=1..K (experts only, excluding base)
                - Experts provide additive corrections to the pretrained base
        
        Optimized with batched computation: All models (base + K experts) computed
        in parallel operations supporting heterogeneous architectures.
        
        Args:
            inputs: (N, n_dims) input coordinates
        """
        # Check if we should use additive mode (pretrained base)
        use_additive_mode = self.adaptive_config.get('pretrained_base_model', False)
        
        # Step 1: Compute ALL masks at once using batched indicators (already vectorized)
        # psi_base: (N, 1), psi_experts: (N, K)
        psi_base, psi_experts = self.batched_indicators(inputs)
        
        # Step 2: Compute all model outputs at once: [base, expert_0, ..., expert_K]
        u_all = self.batched_models.forward(inputs)  # (N, K+1, output_dim)
        
        # Extract base and experts
        u_base = u_all[:, 0, :]  # (N, output_dim)
        
        if len(self.experts) == 0:
            # No experts - just return base model output
            # In additive mode, this is just u_base (since there are no corrections)
            # In partition mode, this is also just u_base (normalized weight = 1)
            return u_base
        
        u_experts = u_all[:, 1:, :]  # (N, K, output_dim)
        
        # Step 3: Compute normalized expert weights
        if use_additive_mode:
            # Additive mode: normalize experts among themselves only (exclude base)
            # ψ̃_k = ψ_k / Σ_j ψ_j for j=1..K
            psi_experts_sum = psi_experts.sum(dim=1, keepdim=True)  # (N, 1)
            psi_experts_sum = psi_experts_sum.clamp(min=1e-8)
            psi_experts_norm = psi_experts / psi_experts_sum  # (N, K)
        else:
            # Standard partition of unity: normalize ALL (base + experts)
            # ψ̃_k = ψ_k / Σ_j ψ_j for j=0..K
            psi_sum = psi_base + psi_experts.sum(dim=1, keepdim=True)  # (N, 1)
            psi_sum = psi_sum.clamp(min=1e-8)
            psi_base_norm = psi_base / psi_sum  # (N, 1)
            psi_experts_norm = psi_experts / psi_sum  # (N, K)
        
        # Step 4: Compute final output
        weighted_experts = psi_experts_norm.unsqueeze(-1) * u_experts  # (N, K, out_dim)
        
        if use_additive_mode:
            # Additive: u = u_base + Σ ψ̃_k · u_k
            u_total = u_base + weighted_experts.sum(dim=1)  # (N, out_dim)
        else:
            # Partition of unity: u = ψ̃_0 · u_base + Σ ψ̃_k · u_k
            u_total = psi_base_norm * u_base + weighted_experts.sum(dim=1)  # (N, out_dim)
        
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
        """Vectorized soft blending decomposed forward pass with batched models.
        
        Supports both standard partition-of-unity mode and additive mode
        (when pretrained_base_model=True).
        """
        result = {}
        
        # Check if we should use additive mode (pretrained base)
        use_additive_mode = self.adaptive_config.get('pretrained_base_model', False)
        
        # Compute all masks at once using batched indicators (already vectorized)
        psi_base, psi_experts = self.batched_indicators(inputs)  # (N, 1), (N, K)
        
        # Store unnormalized masks
        result['masks'] = {'base': psi_base}
        for i in range(psi_experts.shape[1]):
            result['masks'][f'expert_{i}'] = psi_experts[:, i:i+1]  # (N, 1)
        
        # Compute normalized weights based on mode
        if use_additive_mode:
            # Additive mode: base weight is 1, experts normalized among themselves
            psi_experts_sum = psi_experts.sum(dim=1, keepdim=True)  # (N, 1)
            psi_experts_sum = psi_experts_sum.clamp(min=1e-8)
            psi_experts_norm = psi_experts / psi_experts_sum  # (N, K)
            
            # Store normalized weights (base gets special value of 1.0)
            result['weights_normalized'] = {'base': torch.ones_like(psi_base)}
            for i in range(psi_experts_norm.shape[1]):
                result['weights_normalized'][f'expert_{i}'] = psi_experts_norm[:, i:i+1]  # (N, 1)
            result['blending_mode_info'] = 'additive (pretrained_base_model=True)'
        else:
            # Standard partition of unity
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
            
            # Composed output based on mode
            weighted_experts = psi_experts_norm.unsqueeze(-1) * u_experts  # (N, K, out_dim)
            
            if use_additive_mode:
                # Additive: u = u_base + Σ ψ̃_k · u_k
                u_total = u_base + weighted_experts.sum(dim=1)  # (N, out_dim)
            else:
                # Partition of unity: u = ψ̃_0 · u_base + Σ ψ̃_k · u_k
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
        self.hard_indicators = []
        self.soft_indicators = []
        
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
            
            # Always create hard indicator (for coverage checks)
            hard_indicator = HardIndicator(region)
            self.hard_indicators.append(hard_indicator)
            
            # Create soft indicator only if blending_mode is 'soft'
            if self.blending_mode == 'soft':
                soft_indicator = SoftIndicator(region, sigma_fraction=self.sigma_fraction)
                self.soft_indicators.append(soft_indicator)
        
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
