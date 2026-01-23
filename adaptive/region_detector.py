"""Region detection using Random Forest geometric wavelets.

Implements the algorithm to detect high-error regions for spawning expert PINNs:
1. Fit a Random Forest regressor to the current solution
2. Compute geometric wavelets at each tree node
3. Select the region with highest wavelet norm for refinement
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from adaptive.indicators import RegionDescriptor


@dataclass
class TreeNodeInfo:
    """Information about a single tree node."""
    node_id: int
    tree_idx: int
    is_leaf: bool
    n_samples: int
    bounds_lower: List[float]
    bounds_upper: List[float]
    prediction: float  # Q_Ω(x) - local regression value
    parent_prediction: Optional[float]  # Q_Ω_parent(x)
    wavelet_norm: float = 0.0


class RegionDetector:
    """Detects refinement regions using Random Forest geometric wavelets.
    
    Algorithm:
    1. Fit RF to current PINN solution: f_RF(x,t) ≈ u(x,t)
    2. For each tree node, compute geometric wavelet
    3. Compute L2 norm of wavelet
    4. Return region with highest norm (excluding existing expert regions)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_leaf: int = 50,
        regression_type: str = 'constant',
        domain_bounds: Optional[Dict[str, List[float]]] = None
    ):
        """
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of each tree
            min_samples_leaf: Minimum samples required in a leaf node
            regression_type: 'constant' (mean) or 'linear' (local linear fit)
            domain_bounds: {'lower': [x_min, t_min], 'upper': [x_max, t_max]}
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.regression_type = regression_type
        self.domain_bounds = domain_bounds
        
        self.rf: Optional[RandomForestRegressor] = None
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RegionDetector':
        """
        Fit Random Forest to the data.
        
        Args:
            X: (N, n_dims) array of coordinates [x, t] or [x, y, t]
            y: (N,) or (N, output_dim) array of solution values
            
        Returns:
            self for chaining
        """
        # Flatten y if multi-dimensional (take first component or norm)
        if y.ndim > 1 and y.shape[1] > 1:
            # For multi-output (e.g., Schrödinger with u,v), use norm
            y = np.linalg.norm(y, axis=1)
        elif y.ndim > 1:
            y = y.ravel()
        
        self._X = X
        self._y = y
        
        # Set domain bounds from data if not provided
        if self.domain_bounds is None:
            self.domain_bounds = {
                'lower': X.min(axis=0).tolist(),
                'upper': X.max(axis=0).tolist()
            }
        
        self.rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        self.rf.fit(X, y)
        
        return self
    
    def _get_parent_id(self, tree, node_id: int) -> Optional[int]:
        """Find the parent of a node by searching the tree."""
        if node_id == 0:
            return None  # Root has no parent
        
        children_left = tree.children_left
        children_right = tree.children_right
        
        for parent in range(tree.node_count):
            if children_left[parent] == node_id or children_right[parent] == node_id:
                return parent
        return None
    
    def _get_node_bounds(self, tree, node_id: int) -> Tuple[List[float], List[float]]:
        """
        Reconstruct the axis-aligned bounding box for a node.
        Trace from root to node, collecting split conditions.
        """
        n_features = tree.n_features
        bounds_lower = list(self.domain_bounds['lower'])
        bounds_upper = list(self.domain_bounds['upper'])
        
        if node_id == 0:
            return bounds_lower, bounds_upper
        
        # Build path from root to node
        path = []
        current = node_id
        while current != 0:
            parent = self._get_parent_id(tree, current)
            if parent is None:
                break
            went_left = (tree.children_left[parent] == current)
            path.append((parent, went_left))
            current = parent
        
        # Traverse path from root to node
        for parent, went_left in reversed(path):
            feature = tree.feature[parent]
            threshold = tree.threshold[parent]
            
            if went_left:
                # Left child: feature <= threshold
                bounds_upper[feature] = min(bounds_upper[feature], threshold)
            else:
                # Right child: feature > threshold
                bounds_lower[feature] = max(bounds_lower[feature], threshold)
        
        return bounds_lower, bounds_upper
    
    def _get_samples_in_node(self, tree, node_id: int) -> np.ndarray:
        """Get indices of samples that fall into this node."""
        # Use decision_path to find which samples reach this node
        decision_paths = tree.decision_path(self._X)
        # decision_paths is sparse matrix (n_samples, n_nodes)
        node_mask = np.array(decision_paths[:, node_id].todense()).ravel() > 0
        return np.where(node_mask)[0]
    
    def _compute_local_regression(self, sample_indices: np.ndarray) -> float:
        """Compute local regression value Q_Ω for samples in a region."""
        if len(sample_indices) == 0:
            return 0.0
        
        y_local = self._y[sample_indices]
        
        if self.regression_type == 'constant':
            return float(np.mean(y_local))
        elif self.regression_type == 'linear':
            X_local = self._X[sample_indices]
            if len(sample_indices) < X_local.shape[1] + 1:
                # Not enough samples for linear fit, fall back to constant
                return float(np.mean(y_local))
            lr = LinearRegression()
            lr.fit(X_local, y_local)
            return float(np.mean(lr.predict(X_local)))
        else:
            return float(np.mean(y_local))
    
    def compute_wavelet_norms(self) -> List[TreeNodeInfo]:
        """
        Compute geometric wavelet norms for all tree nodes.
        
        Returns:
            List of TreeNodeInfo with wavelet norms
        """
        if self.rf is None:
            raise RuntimeError("Must call fit() before compute_wavelet_norms()")
        
        all_nodes = []
        
        for tree_idx, estimator in enumerate(self.rf.estimators_):
            tree = estimator.tree_
            
            # Cache parent predictions for efficiency
            node_predictions = {}
            
            for node_id in range(tree.node_count):
                is_leaf = tree.children_left[node_id] == -1
                
                # Get samples in this node
                sample_indices = self._get_samples_in_node(tree, node_id)
                n_samples = len(sample_indices)
                
                if n_samples < self.min_samples_leaf:
                    continue
                
                # Get bounds
                bounds_lower, bounds_upper = self._get_node_bounds(tree, node_id)
                
                # Compute local regression Q_Ω
                if node_id in node_predictions:
                    prediction = node_predictions[node_id]
                else:
                    prediction = self._compute_local_regression(sample_indices)
                    node_predictions[node_id] = prediction
                
                # Get parent prediction Q_Ω_parent
                parent_id = self._get_parent_id(tree, node_id)
                parent_prediction = None
                if parent_id is not None:
                    if parent_id in node_predictions:
                        parent_prediction = node_predictions[parent_id]
                    else:
                        parent_samples = self._get_samples_in_node(tree, parent_id)
                        parent_prediction = self._compute_local_regression(parent_samples)
                        node_predictions[parent_id] = parent_prediction
                
                # Compute wavelet norm: ||ψ_Ω'||² = Σ (Q_Ω' - Q_Ω)² for samples in Ω'
                wavelet_norm = 0.0
                if parent_prediction is not None:
                    diff = prediction - parent_prediction
                    # L2 norm weighted by number of samples
                    wavelet_norm = (diff ** 2) * n_samples
                
                all_nodes.append(TreeNodeInfo(
                    node_id=node_id,
                    tree_idx=tree_idx,
                    is_leaf=is_leaf,
                    n_samples=n_samples,
                    bounds_lower=bounds_lower,
                    bounds_upper=bounds_upper,
                    prediction=prediction,
                    parent_prediction=parent_prediction,
                    wavelet_norm=wavelet_norm
                ))
        
        return all_nodes
    
    def select_refinement_region(
        self,
        existing_regions: Optional[List[RegionDescriptor]] = None,
        wavelet_threshold: Optional[float] = None,
        spawn_epoch: int = 0
    ) -> Optional[RegionDescriptor]:
        """
        Select the best region for refinement.
        
        Args:
            existing_regions: List of already-assigned expert regions
            wavelet_threshold: Minimum wavelet norm to spawn (None = always spawn)
            spawn_epoch: Current epoch for tracking
            
        Returns:
            RegionDescriptor for the selected region, or None if no suitable region
        """
        nodes = self.compute_wavelet_norms()
        
        if not nodes:
            return None
        
        # Filter out root nodes (no parent = no wavelet)
        candidate_nodes = [n for n in nodes if n.parent_prediction is not None]
        
        if not candidate_nodes:
            return None
        
        # Sort by wavelet norm (highest first)
        candidate_nodes.sort(key=lambda n: n.wavelet_norm, reverse=True)
        
        # Find best region (overlapping regions are allowed - they get more neural capacity)
        for node in candidate_nodes:
            # Check wavelet threshold
            if wavelet_threshold is not None and node.wavelet_norm < wavelet_threshold:
                continue
            
            # Found a valid region (overlapping is allowed for hierarchical refinement)
            return RegionDescriptor(
                bounds_lower=node.bounds_lower,
                bounds_upper=node.bounds_upper,
                wavelet_norm=node.wavelet_norm,
                spawn_epoch=spawn_epoch
            )
        
        return None
    
    def _check_overlap(self, node: TreeNodeInfo, existing_regions: List[RegionDescriptor]) -> bool:
        """Check if a node significantly overlaps with existing regions."""
        for region in existing_regions:
            # Check if boxes overlap (any dimension must be disjoint for no overlap)
            overlaps = True
            for i in range(len(node.bounds_lower)):
                if (node.bounds_upper[i] <= region.bounds_lower[i] or
                    node.bounds_lower[i] >= region.bounds_upper[i]):
                    overlaps = False
                    break
            
            if overlaps:
                # Compute overlap volume
                overlap_lower = [max(node.bounds_lower[i], region.bounds_lower[i]) 
                                for i in range(len(node.bounds_lower))]
                overlap_upper = [min(node.bounds_upper[i], region.bounds_upper[i])
                                for i in range(len(node.bounds_upper))]
                
                overlap_vol = 1.0
                for lo, hi in zip(overlap_lower, overlap_upper):
                    overlap_vol *= max(0, hi - lo)
                
                node_vol = 1.0
                for lo, hi in zip(node.bounds_lower, node.bounds_upper):
                    node_vol *= (hi - lo)
                
                # If overlap is more than 50% of node volume, consider it overlapping
                if node_vol > 0 and overlap_vol / node_vol > 0.5:
                    return True
        
        return False
    
    def detect(
        self,
        X: np.ndarray,
        y: np.ndarray,
        existing_regions: Optional[List[RegionDescriptor]] = None,
        wavelet_threshold: Optional[float] = None,
        spawn_epoch: int = 0
    ) -> Optional[RegionDescriptor]:
        """
        Convenience method to fit RF and detect refinement region in one call.
        
        Args:
            X: (N, n_dims) array of coordinates
            y: (N,) or (N, output_dim) array of solution values
            existing_regions: List of already-assigned expert regions
            wavelet_threshold: Minimum wavelet norm to spawn
            spawn_epoch: Current epoch for tracking
            
        Returns:
            RegionDescriptor for the selected region, or None
        """
        self.fit(X, y)
        return self.select_refinement_region(
            existing_regions=existing_regions,
            wavelet_threshold=wavelet_threshold,
            spawn_epoch=spawn_epoch
        )
