"""Indicator functions for adaptive expert regions.

Provides hard (step function) and soft (smooth sigmoid) indicator functions
for defining expert regions as axis-aligned bounding boxes.

Also provides BatchedIndicators for computing all indicator masks (base + K experts)
in a single vectorized GPU operation for maximum efficiency.
"""

import torch
from torch.func import vmap
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class RegionDescriptor:
    """Describes an axis-aligned rectangular region in the domain.
    
    Attributes:
        bounds_lower: Lower bounds for each dimension [x_min, t_min] or [x_min, y_min, t_min]
        bounds_upper: Upper bounds for each dimension [x_max, t_max] or [x_max, y_max, t_max]
        wavelet_norm: L2 norm of the geometric wavelet (refinement priority)
        spawn_epoch: Epoch at which this region's expert was spawned
        depth: Depth level in the expert tree (1 = child of base model)
        parent_idx: Index of the parent expert (-1 for depth-1, base model is parent)
    """
    bounds_lower: List[float]
    bounds_upper: List[float]
    wavelet_norm: float = 0.0
    spawn_epoch: int = 0
    depth: int = 1  # Depth level (1 = child of base)
    parent_idx: int = -1  # Parent expert index (-1 = base model)
    
    @property
    def n_dims(self) -> int:
        """Number of dimensions in the region."""
        return len(self.bounds_lower)
    
    @property
    def volume(self) -> float:
        """Compute the volume (area in 2D) of the region."""
        vol = 1.0
        for lo, hi in zip(self.bounds_lower, self.bounds_upper):
            vol *= (hi - lo)
        return vol
    
    def contains(self, point: List[float]) -> bool:
        """Check if a point is inside the region."""
        for i, (lo, hi) in enumerate(zip(self.bounds_lower, self.bounds_upper)):
            if point[i] < lo or point[i] > hi:
                return False
        return True
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'bounds_lower': self.bounds_lower,
            'bounds_upper': self.bounds_upper,
            'wavelet_norm': self.wavelet_norm,
            'spawn_epoch': self.spawn_epoch,
            'depth': self.depth,
            'parent_idx': self.parent_idx
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'RegionDescriptor':
        """Create from dictionary."""
        return cls(
            bounds_lower=d['bounds_lower'],
            bounds_upper=d['bounds_upper'],
            wavelet_norm=d.get('wavelet_norm', 0.0),
            spawn_epoch=d.get('spawn_epoch', 0),
            depth=d.get('depth', 1),
            parent_idx=d.get('parent_idx', -1)
        )


class HardIndicator:
    """Hard (step function) indicator for axis-aligned box regions.
    
    Returns 1 if point is inside the region, 0 otherwise.
    Uses vectorized boolean masks for GPU efficiency.
    """
    
    def __init__(self, region: RegionDescriptor):
        """
        Args:
            region: RegionDescriptor defining the bounding box
        """
        self.region = region
        self._lower: Optional[torch.Tensor] = None
        self._upper: Optional[torch.Tensor] = None
        self._device: Optional[torch.device] = None
    
    def _ensure_tensors(self, device: torch.device):
        """Lazily create tensors on the correct device."""
        if self._device != device:
            self._lower = torch.tensor(self.region.bounds_lower, dtype=torch.float32, device=device)
            self._upper = torch.tensor(self.region.bounds_upper, dtype=torch.float32, device=device)
            self._device = device
    
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute indicator mask for batch of points.
        
        Args:
            inputs: (N, n_dims) tensor of coordinates [x, t] or [x, y, t]
            
        Returns:
            mask: (N, 1) float tensor - 1.0 if inside region, 0.0 otherwise
        """
        self._ensure_tensors(inputs.device)
        
        # Vectorized comparison: all dimensions must be within bounds
        # inputs >= lower: (N, n_dims) bool tensor
        # inputs <= upper: (N, n_dims) bool tensor
        inside = (inputs >= self._lower) & (inputs <= self._upper)  # (N, n_dims)
        
        # Point is inside if ALL dimensions are inside
        mask = inside.all(dim=1, keepdim=True)  # (N, 1)
        
        return mask.float()
    
    def get_bounds(self) -> tuple:
        """Return bounds as tuple of lists."""
        return (self.region.bounds_lower, self.region.bounds_upper)


class SoftIndicator:
    """Soft (smooth sigmoid) indicator for axis-aligned box regions.
    
    Uses smooth sigmoid transitions at boundaries. The sigma parameter is
    computed as a fraction of each dimension's region size, providing
    scale-invariant smoothness.
    
    The mask value is highest at the center of the region and smoothly
    decays towards the boundaries.
    """
    
    def __init__(self, region: RegionDescriptor, sigma_fraction: float = 0.2):
        """
        Args:
            region: RegionDescriptor defining the bounding box
            sigma_fraction: Fraction of region size to use as sigma per dimension.
                           Larger values = smoother transitions.
        """
        self.region = region
        self.sigma_fraction = sigma_fraction
        self._lower: Optional[torch.Tensor] = None
        self._upper: Optional[torch.Tensor] = None
        self._sigma: Optional[torch.Tensor] = None  # Per-dimension sigma
        self._device: Optional[torch.device] = None
    
    def _ensure_tensors(self, device: torch.device):
        """Lazily create tensors on the correct device."""
        if self._device != device:
            self._lower = torch.tensor(self.region.bounds_lower, dtype=torch.float32, device=device)
            self._upper = torch.tensor(self.region.bounds_upper, dtype=torch.float32, device=device)
            # Compute sigma per dimension as fraction of region size
            region_sizes = self._upper - self._lower
            self._sigma = self.sigma_fraction * region_sizes  # (n_dims,)
            # Ensure minimum sigma to avoid numerical issues
            self._sigma = torch.clamp(self._sigma, min=1e-6)
            self._device = device
    
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute soft indicator mask for batch of points.
        
        Args:
            inputs: (N, n_dims) tensor of coordinates [x, t] or [x, y, t]
            
        Returns:
            mask: (N, 1) float tensor - smooth value in [0, 1]
                  Highest at region center, smoothly decaying to boundaries.
        """
        self._ensure_tensors(inputs.device)
        
        # Distance from lower bound (positive = inside), scaled by per-dim sigma
        dist_lower = (inputs - self._lower) / self._sigma  # Broadcasting (N, n_dims)
        # Distance from upper bound (positive = inside), scaled by per-dim sigma
        dist_upper = (self._upper - inputs) / self._sigma  # Broadcasting (N, n_dims)
        
        # Sigmoid gives smooth 0→1 transition
        weight_lower = torch.sigmoid(dist_lower)  # (N, n_dims)
        weight_upper = torch.sigmoid(dist_upper)  # (N, n_dims)
        
        # Product over dimensions for intersection of half-spaces
        mask = (weight_lower * weight_upper).prod(dim=1, keepdim=True)  # (N, 1)
        
        return mask
    
    def get_bounds(self) -> tuple:
        """Return bounds as tuple of lists."""
        return (self.region.bounds_lower, self.region.bounds_upper)


class UniformIndicator:
    """Uniform weight indicator for the base model in soft blending.
    
    Returns a constant weight for all points in the domain.
    Used for partition-of-unity normalization where the base model
    contributes everywhere with uniform weight.
    """
    
    def __init__(self, base_weight: float = 1.0):
        """
        Args:
            base_weight: Constant weight to return for all points.
                        This is the unnormalized weight; actual contribution
                        depends on normalization with expert weights.
        """
        self.base_weight = base_weight
    
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute uniform weight for batch of points.
        
        Args:
            inputs: (N, n_dims) tensor of coordinates
            
        Returns:
            weights: (N, 1) float tensor - constant value for all points
        """
        return torch.full(
            (inputs.shape[0], 1), 
            self.base_weight,
            device=inputs.device, 
            dtype=inputs.dtype
        )


def create_indicator(region: RegionDescriptor, mode: str = 'hard', sigma_fraction: float = 0.2):
    """Factory function to create an indicator of the specified type.
    
    Args:
        region: RegionDescriptor defining the bounding box
        mode: 'hard' or 'soft'
        sigma_fraction: For soft indicator, fraction of region size to use as sigma.
                       Larger values = smoother transitions.
        
    Returns:
        HardIndicator or SoftIndicator instance
    """
    if mode == 'hard':
        return HardIndicator(region)
    elif mode == 'soft':
        return SoftIndicator(region, sigma_fraction=sigma_fraction)
    else:
        raise ValueError(f"Unknown indicator mode: {mode}. Use 'hard' or 'soft'.")


def _soft_indicator_fn(
    lower: torch.Tensor,
    upper: torch.Tensor,
    sigma: torch.Tensor,
    point: torch.Tensor
) -> torch.Tensor:
    """
    Pure function: compute soft sigmoid-box indicator for one region at one point.
    
    This function is designed to be vmapped:
    - Inner vmap over K regions (different lower, upper, sigma; same point)
    - Outer vmap over N points (same params; different point)
    
    Args:
        lower: (D,) lower bounds of the region
        upper: (D,) upper bounds of the region
        sigma: (D,) per-dimension smoothness parameter
        point: (D,) input coordinates
        
    Returns:
        Scalar tensor — smooth indicator value in [0, 1]
    """
    dist_lower = (point - lower) / sigma  # (D,)
    dist_upper = (upper - point) / sigma  # (D,)
    return (torch.sigmoid(dist_lower) * torch.sigmoid(dist_upper)).prod()


class BatchedIndicators:
    """Compute all indicator masks (base + K experts) in one vectorized GPU operation.
    
    This class provides significant speedup over calling individual indicators
    in a loop by batching all computations into single tensor operations.
    
    Supports both hard (step function) and soft (sigmoid) modes.
    """
    
    def __init__(self, base_weight: float = 1.0):
        """
        Args:
            base_weight: Constant weight for the base model (used in soft blending).
        """
        self.base_weight = base_weight
        self.all_lower: Optional[torch.Tensor] = None  # (K, D)
        self.all_upper: Optional[torch.Tensor] = None  # (K, D)
        self.all_sigma: Optional[torch.Tensor] = None  # (K, D) for soft mode
        self.mode: str = 'hard'
        self.sigma_fraction: float = 0.2
        self._num_experts: int = 0
    
    @property
    def num_experts(self) -> int:
        """Number of expert regions currently tracked."""
        return self._num_experts
    
    def update(
        self, 
        regions: List[RegionDescriptor], 
        device: torch.device, 
        mode: str = 'hard',
        sigma_fraction: float = 0.2
    ) -> None:
        """
        Update batched tensors from list of regions.
        
        Call this after spawning new experts to sync the batched state.
        
        Args:
            regions: List of RegionDescriptors for all experts
            device: Target device (cuda or cpu)
            mode: 'hard' or 'soft' blending mode
            sigma_fraction: For soft mode, fraction of region size to use as sigma
        """
        self.mode = mode
        self.sigma_fraction = sigma_fraction
        self._num_experts = len(regions)
        
        if len(regions) == 0:
            self.all_lower = None
            self.all_upper = None
            self.all_sigma = None
            return
        
        # Stack all bounds into (K, D) tensors
        self.all_lower = torch.stack([
            torch.tensor(r.bounds_lower, dtype=torch.float32, device=device)
            for r in regions
        ])  # (K, D)
        
        self.all_upper = torch.stack([
            torch.tensor(r.bounds_upper, dtype=torch.float32, device=device)
            for r in regions
        ])  # (K, D)
        
        # For soft mode, precompute sigma per region per dimension
        if mode == 'soft':
            region_sizes = self.all_upper - self.all_lower
            self.all_sigma = (sigma_fraction * region_sizes).clamp(min=1e-6)  # (K, D)
        else:
            self.all_sigma = None
    
    def __call__(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute base weight + all K expert masks in one vectorized operation.
        
        Args:
            inputs: (N, D) tensor of coordinates [x, t] or [x, y, t]
            
        Returns:
            Tuple of:
                - psi_base: (N, 1) base model weights (uniform)
                - psi_experts: (N, K) expert indicator masks
        """
        N = inputs.shape[0]
        device = inputs.device
        dtype = inputs.dtype
        
        # Base weight is uniform across all points
        psi_base = torch.full((N, 1), self.base_weight, device=device, dtype=dtype)
        
        # Handle case with no experts
        if self.all_lower is None or self._num_experts == 0:
            return psi_base, torch.empty((N, 0), device=device, dtype=dtype)
        
        # Ensure bounds are on correct device
        if self.all_lower.device != device:
            self.all_lower = self.all_lower.to(device)
            self.all_upper = self.all_upper.to(device)
            if self.all_sigma is not None:
                self.all_sigma = self.all_sigma.to(device)
        
        if self.mode == 'hard':
            psi_experts = self._compute_hard_masks(inputs)
        else:
            psi_experts = self._compute_soft_masks(inputs)
        
        return psi_base, psi_experts
    
    def _compute_hard_masks(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute all hard indicator masks in one vectorized operation.
        
        Args:
            inputs: (N, D) tensor of coordinates
            
        Returns:
            masks: (N, K) float tensor - 1.0 if inside, 0.0 if outside
        """
        # Expand for broadcasting: (N, 1, D) vs (1, K, D) -> (N, K, D)
        x = inputs.unsqueeze(1)  # (N, 1, D)
        lower = self.all_lower.unsqueeze(0)  # (1, K, D)
        upper = self.all_upper.unsqueeze(0)  # (1, K, D)
        
        # Check all dimensions at once
        inside = (x >= lower) & (x <= upper)  # (N, K, D)
        
        # Point is inside region if ALL dimensions are inside
        masks = inside.all(dim=2).float()  # (N, K)
        
        return masks
    
    def _compute_soft_masks(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute all soft indicator masks using vmap over a pure function.
        
        Uses vmap to evaluate the same sigmoid-box indicator function
        with K different parameter sets (region bounds) across N points
        in parallel.
        
        Args:
            inputs: (N, D) tensor of coordinates
            
        Returns:
            masks: (N, K) float tensor - smooth values in [0, 1]
        """
        # vmap over K indicators (different params, same point),
        # then vmap over N points (same params, different points)
        # Result: (N, K)
        vmapped_indicators = vmap(
            vmap(_soft_indicator_fn, in_dims=(0, 0, 0, None)),
            in_dims=(None, None, None, 0)
        )
        masks = vmapped_indicators(
            self.all_lower, self.all_upper, self.all_sigma, inputs
        )  # (N, K)
        
        return masks
    
    def compute_hard_masks_only(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute ONLY hard masks (no base), useful for coverage checks.
        
        Always uses hard indicator logic regardless of self.mode.
        
        Args:
            inputs: (N, D) tensor of coordinates
            
        Returns:
            masks: (N, K) float tensor - 1.0 if inside, 0.0 if outside
        """
        if self.all_lower is None or self._num_experts == 0:
            return torch.empty((inputs.shape[0], 0), device=inputs.device, dtype=inputs.dtype)
        
        # Ensure bounds are on correct device
        device = inputs.device
        if self.all_lower.device != device:
            self.all_lower = self.all_lower.to(device)
            self.all_upper = self.all_upper.to(device)
        
        return self._compute_hard_masks(inputs)
