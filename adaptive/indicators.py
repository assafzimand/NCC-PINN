"""Indicator functions for adaptive expert regions.

Provides hard (step function) and soft (smooth sigmoid) indicator functions
for defining expert regions as axis-aligned bounding boxes.
"""

import torch
from dataclasses import dataclass
from typing import List, Optional


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
    
    Uses smooth sigmoid transitions at boundaries, forming a partition of unity
    when combined with other soft indicators.
    """
    
    def __init__(self, region: RegionDescriptor, sigma: float = 0.1):
        """
        Args:
            region: RegionDescriptor defining the bounding box
            sigma: Smoothness parameter (smaller = sharper transition)
        """
        self.region = region
        self.sigma = sigma
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
        Compute soft indicator mask for batch of points.
        
        Args:
            inputs: (N, n_dims) tensor of coordinates [x, t] or [x, y, t]
            
        Returns:
            mask: (N, 1) float tensor - smooth value in [0, 1]
        """
        self._ensure_tensors(inputs.device)
        
        # Distance from lower bound (positive = inside)
        dist_lower = (inputs - self._lower) / self.sigma
        # Distance from upper bound (positive = inside)
        dist_upper = (self._upper - inputs) / self.sigma
        
        # Sigmoid gives smooth 0→1 transition
        weight_lower = torch.sigmoid(dist_lower)  # (N, n_dims)
        weight_upper = torch.sigmoid(dist_upper)  # (N, n_dims)
        
        # Product over dimensions for intersection of half-spaces
        mask = (weight_lower * weight_upper).prod(dim=1, keepdim=True)  # (N, 1)
        
        return mask
    
    def get_bounds(self) -> tuple:
        """Return bounds as tuple of lists."""
        return (self.region.bounds_lower, self.region.bounds_upper)


def create_indicator(region: RegionDescriptor, mode: str = 'hard', sigma: float = 0.1):
    """Factory function to create an indicator of the specified type.
    
    Args:
        region: RegionDescriptor defining the bounding box
        mode: 'hard' or 'soft'
        sigma: Smoothness parameter for soft indicator
        
    Returns:
        HardIndicator or SoftIndicator instance
    """
    if mode == 'hard':
        return HardIndicator(region)
    elif mode == 'soft':
        return SoftIndicator(region, sigma=sigma)
    else:
        raise ValueError(f"Unknown indicator mode: {mode}. Use 'hard' or 'soft'.")
