"""Region detection using Random Forest geometric wavelets.

Implements the algorithm to detect high-error regions for spawning expert PINNs:
1. Fit a Random Forest regressor to the current solution
2. Compute geometric wavelets at each tree node (Q_child - Q_parent for d-dim output)
3. Compute residual-weighted L2 norm of wavelet (prioritizes high-error regions)
4. Select the region with highest weighted norm for refinement
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
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
    prediction: np.ndarray  # Q_Ω(x) - local mean value, shape (d,) for d-dim output
    parent_prediction: Optional[np.ndarray]  # Q_Ω_parent(x), shape (d,) or None
    wavelet_norm: float = 0.0  # ||ψ||² = ||Q_child - Q_parent||² × total_loss(ω_i) × |ω_i|
    total_loss_omega: float = 0.0  # Total weighted loss in this region
    sum_residuals: float = 0.0  # DEPRECATED: Sum of residuals (for backward compat)


class RegionDetector:
    """Detects refinement regions using Random Forest geometric wavelets.
    
    Algorithm:
    1. Fit RF to current PINN solution: f_RF(x,t) ≈ u(x,t)
    2. For each tree node, compute geometric wavelet: ψ = Q_child - Q_parent
    3. Compute residual-weighted L2 norm: ||ψ||² × Σ residuals
    4. Return region with highest norm for refinement
    
    Supports multi-dimensional output (e.g., Schrödinger's [u, v]).
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_leaf: int = 50,
        domain_bounds: Optional[Dict[str, List[float]]] = None
    ):
        """
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of each tree
            min_samples_leaf: Minimum samples required in a leaf node
            domain_bounds: {'lower': [x_min, t_min], 'upper': [x_max, t_max]}
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.domain_bounds = domain_bounds
        
        self.rf: Optional[RandomForestRegressor] = None
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._residuals: Optional[np.ndarray] = None  # DEPRECATED: kept for backward compatibility
        self._residual_losses: Optional[np.ndarray] = None  # Per-sample residual losses
        self._ic_losses: Optional[np.ndarray] = None  # Per-sample IC losses
        self._bc_losses: Optional[np.ndarray] = None  # Per-sample BC losses
        self._loss_weights: Optional[Dict[str, float]] = None  # Loss weights dict
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        loss_components: Optional[Dict[str, np.ndarray]] = None,
        residuals: Optional[np.ndarray] = None
    ) -> 'RegionDetector':
        """
        Fit Random Forest to the data.
        
        Args:
            X: (N, n_dims) array of coordinates [x, t] or [x, y, t]
            y: (N,) or (N, output_dim) array of solution values
               Multi-output is supported - RF fits on d-dimensional y,
               and wavelet norm is computed as ||Q_child - Q_parent||²
            loss_components: Dict with 'residual', 'ic', 'bc' arrays and 'weights'
                (new tree spawning approach using total loss)
            residuals: DEPRECATED - (N,) array of PDE residuals (for backward compatibility)
            
        Returns:
            self for chaining
        """
        # Keep y in original dimension (d-dim for multi-output like Schrödinger)
        # RF supports multi-output regression: y can be (N,) or (N, d)
        # Only flatten if it's (N, 1) -> (N,)
        if y.ndim > 1 and y.shape[1] == 1:
            y = y.ravel()
        
        self._X = X
        self._y = y
        
        # Store loss components (new approach)
        if loss_components is not None:
            self._residual_losses = loss_components['residual']
            self._ic_losses = loss_components['ic']
            self._bc_losses = loss_components['bc']
            self._loss_weights = loss_components['weights']
            # For backward compatibility with diagnostic code
            self._residuals = loss_components['residual'].copy()
        elif residuals is not None:
            # Fallback to old residual-only approach
            if residuals.ndim > 1:
                residuals = residuals.ravel()
            self._residuals = np.abs(residuals)
            # Fill loss components with residuals for compatibility
            self._residual_losses = self._residuals.copy()
            self._ic_losses = np.zeros_like(self._residuals)
            self._bc_losses = np.zeros_like(self._residuals)
            self._loss_weights = {'residual': 1.0, 'ic': 0.0, 'bc': 0.0}
        else:
            # Fallback: uniform weighting (equivalent to n_samples)
            n = len(y) if y.ndim == 1 else y.shape[0]
            self._residuals = np.ones(n)
            self._residual_losses = np.ones(n)
            self._ic_losses = np.zeros(n)
            self._bc_losses = np.zeros(n)
            self._loss_weights = {'residual': 1.0, 'ic': 0.0, 'bc': 0.0}
        
        # ALWAYS update domain bounds from the data being fitted
        # This is critical when fitting on filtered subdomains - the bounds
        # must match the actual search domain, not the global domain
        self.domain_bounds = {
            'lower': X.min(axis=0).tolist(),
            'upper': X.max(axis=0).tolist()
        }
        
        self.rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            bootstrap=False,  # Use ALL data for each tree (no bootstrap sampling)
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
    
    def compute_wavelet_norms(self) -> List[TreeNodeInfo]:
        """
        Compute geometric wavelet norms for all tree nodes.
        
        Uses RF's internal node values (tree.value) as Q_Ω for each node.
        Supports multi-dimensional output (d > 1): Q is d-dimensional vector.
        Wavelet norm = ||Q_child - Q_parent||² × Σ residuals
        
        Returns:
            List of TreeNodeInfo with wavelet norms
        """
        if self.rf is None:
            raise RuntimeError("Must call fit() before compute_wavelet_norms()")
        
        all_nodes = []
        
        for tree_idx, estimator in enumerate(self.rf.estimators_):
            tree = estimator.tree_
            
            for node_id in range(tree.node_count):
                is_leaf = tree.children_left[node_id] == -1
                
                # Get samples in this node (for residual weighting)
                sample_indices = self._get_samples_in_node(tree, node_id)
                n_samples = len(sample_indices)
                
                if n_samples < self.min_samples_leaf:
                    continue
                
                # Get bounds
                bounds_lower, bounds_upper = self._get_node_bounds(tree, node_id)
                
                # Get Q_Ω from RF's internal node value
                # tree.value has shape (n_nodes, n_outputs, 1) for regressors
                # tree.value[node_id, :, 0] gives shape (n_outputs,) = (d,)
                prediction = tree.value[node_id, :, 0].copy()  # Shape (d,)
                
                # Get parent prediction Q_Ω_parent
                parent_id = self._get_parent_id(tree, node_id)
                parent_prediction = None
                if parent_id is not None:
                    parent_prediction = tree.value[parent_id, :, 0].copy()  # Shape (d,)
                
                # Compute wavelet norm: ||ψ||² = ||Q_child - Q_parent||² × total_loss(ω_i) × |ω_i|
                # Weights by total loss AND region size — larger high-error regions get priority
                wavelet_norm = 0.0
                total_loss_omega = 0.0
                sum_residuals = 0.0  # DEPRECATED: kept for backward compatibility
                
                if parent_prediction is not None and len(sample_indices) > 0:
                    diff = prediction - parent_prediction  # Shape (d,)
                    # L2 norm squared of the difference
                    l2_norm_squared = float(np.sum(diff ** 2))
                    
                    # Compute total_loss(ω_i) over this region
                    # total_loss(ω_i) = w_res·mean(res[ω]) + w_ic·mean(ic[ω]) + w_bc·mean(bc[ω])
                    mean_residual = float(self._residual_losses[sample_indices].mean())
                    mean_ic = float(self._ic_losses[sample_indices].mean())
                    mean_bc = float(self._bc_losses[sample_indices].mean())
                    
                    # Weighted sum (same as training loss)
                    total_loss_omega = (
                        self._loss_weights['residual'] * mean_residual +
                        self._loss_weights['ic'] * mean_ic +
                        self._loss_weights['bc'] * mean_bc
                    )
                    
                    # Wavelet norm = ||Q_child - Q_parent||² × total_loss(ω_i) × |ω_i|
                    wavelet_norm = l2_norm_squared * total_loss_omega * n_samples
                    
                    # For backward compatibility
                    sum_residuals = float(self._residual_losses[sample_indices].sum())
                
                all_nodes.append(TreeNodeInfo(
                    node_id=node_id,
                    tree_idx=tree_idx,
                    is_leaf=is_leaf,
                    n_samples=n_samples,
                    bounds_lower=bounds_lower,
                    bounds_upper=bounds_upper,
                    prediction=prediction,
                    parent_prediction=parent_prediction,
                    wavelet_norm=wavelet_norm,
                    total_loss_omega=total_loss_omega,
                    sum_residuals=sum_residuals
                ))
        
        return all_nodes
    
    # ============================================================
    # PRETRAINED CASE - COMMENTED OUT (ANT design uses non-pretrained only)
    # Will be removed in future cleanup
    # ============================================================
    # def extract_regions_from_tree(
    #     self,
    #     wavelet_threshold: Optional[float] = None,
    #     verbose: bool = True
    # ) -> List[Tuple[TreeNodeInfo, int]]:
    #     """
    #     Traverse a single decision tree (BFS) and extract spawnable regions.
    #
    #     Prerequisites:
    #     - fit() must be called first with n_estimators=1
    #
    #     Args:
    #         wavelet_threshold: Minimum wavelet norm to spawn (None = spawn all non-root)
    #         verbose: Print diagnostic information
    #
    #     Returns:
    #         List of (TreeNodeInfo, parent_tree_node_id) tuples in BFS order.
    #         parent_tree_node_id is the tree node ID of nearest SPAWNED ancestor,
    #         or -1 for base model.
    #     """
    #     from collections import deque
    #
    #     # Validation
    #     if self.rf is None:
    #         raise RuntimeError("Must call fit() before extract_regions_from_tree()")
    #     if self.n_estimators != 1:
    #         raise RuntimeError(f"extract_regions_from_tree requires n_estimators=1, got {self.n_estimators}")
    #
    #     # Get the single tree
    #     tree = self.rf.estimators_[0].tree_
    #
    #     # Compute wavelet norms for all nodes
    #     all_nodes = self.compute_wavelet_norms()
    #
    #     if not all_nodes:
    #         if verbose:
    #             print(f"    [Traverse] No nodes in tree")
    #         return []
    #
    #     # Build node_id -> TreeNodeInfo lookup
    #     node_lookup = {node.node_id: node for node in all_nodes}
    #
    #     # Result: list of (TreeNodeInfo, nearest_spawned_ancestor_tree_node_id)
    #     result = []
    #
    #     # Tracking structure: maps tree_node_id -> whether it was spawned
    #     spawned_nodes = {0: True}  # Root is always considered "spawned" (represents base model)
    #
    #     # BFS queue initialization
    #     queue = deque([(0, -1)])  # (node_id, nearest_spawned_ancestor_id)
    #     # -1 means "root" which maps to base model (parent_idx=-1 in RegionDescriptor)
    #
    #     visited_count = 0
    #     spawned_count = 0
    #     skipped_count = 0
    #
    #     if verbose:
    #         print(f"\n  [Traverse] Starting BFS traversal of tree with {tree.node_count} nodes")
    #
    #     # Traverse the tree (BFS)
    #     while queue:
    #         current_id, nearest_spawned_ancestor = queue.popleft()
    #         visited_count += 1
    #
    #         # Skip if node doesn't exist in lookup (too few samples)
    #         if current_id not in node_lookup:
    #             continue
    #
    #         node = node_lookup[current_id]
    #
    #         # Determine if this node should spawn
    #         should_spawn = False
    #         rejection_reason = None
    #
    #         if current_id == 0:
    #             # Root node: skip (represents base model, not a spawnable expert)
    #             should_spawn = False
    #             rejection_reason = "root node"
    #         elif node.parent_prediction is None:
    #             # No parent (shouldn't happen except for root, but defensive)
    #             should_spawn = False
    #             rejection_reason = "no parent prediction"
    #         elif wavelet_threshold is not None and node.wavelet_norm < wavelet_threshold:
    #             # Below threshold
    #             should_spawn = False
    #             rejection_reason = f"wavelet_norm {node.wavelet_norm:.6f} < threshold {wavelet_threshold}"
    #         else:
    #             # Passes all checks
    #             should_spawn = True
    #
    #         # Update tracking
    #         if should_spawn:
    #             spawned_nodes[current_id] = True
    #             result.append((node, nearest_spawned_ancestor))
    #             spawned_count += 1
    #
    #             if verbose:
    #                 parent_str = "Base" if nearest_spawned_ancestor == -1 else f"Node{nearest_spawned_ancestor}"
    #                 print(f"    [Traverse] Node {current_id}: SPAWN (parent={parent_str}, "
    #                       f"wavelet={node.wavelet_norm:.6f}, samples={node.n_samples})")
    #
    #             # This node becomes the new nearest spawned ancestor for its children
    #             next_nearest_spawned = current_id
    #         else:
    #             spawned_nodes[current_id] = False
    #             skipped_count += 1
    #
    #             if verbose and rejection_reason:
    #                 print(f"    [Traverse] Node {current_id}: SKIP ({rejection_reason})")
    #
    #             # Children inherit current nearest_spawned_ancestor (pass through)
    #             next_nearest_spawned = nearest_spawned_ancestor
    #
    #         # Add children to queue
    #         left_child = tree.children_left[current_id]
    #         right_child = tree.children_right[current_id]
    #
    #         if left_child != -1:  # Not a leaf
    #             queue.append((left_child, next_nearest_spawned))
    #
    #         if right_child != -1:  # Not a leaf
    #             queue.append((right_child, next_nearest_spawned))
    #
    #     if verbose:
    #         print(f"\n  [Traverse] Summary:")
    #         print(f"    Visited nodes: {visited_count}")
    #         print(f"    Spawnable nodes: {spawned_count}")
    #         print(f"    Skipped nodes: {skipped_count}")
    #
    #     return result
    # ============================================================

    def spawn_children_for_node(
        self,
        parent_region: RegionDescriptor,
        X: np.ndarray,
        y: np.ndarray,
        loss_components: Dict,
        wavelet_threshold: Optional[float] = None,
        verbose: bool = True
    ) -> Tuple[List[Tuple[TreeNodeInfo, int]], bool]:
        """
        Spawn children for a single parent node using all-or-nothing rule (ANT design).

        Fits a single-split tree (max_depth=1) on the parent's subdomain.
        Creates 2 children (left/right) and computes their wavelet norms.

        All-or-nothing rule:
        - If at least ONE child meets the wavelet threshold → spawn BOTH children
        - If BOTH children are below threshold → spawn NEITHER child

        Args:
            parent_region: The parent node's region descriptor
            X: (N, n_dims) coordinates (full eval_data)
            y: (N, output_dim) predictions (global solution)
            loss_components: Per-sample losses
            wavelet_threshold: Minimum wavelet norm to spawn (None = always spawn)
            verbose: Print diagnostic info

        Returns:
            Tuple of (children, should_spawn):
            - children: List of (TreeNodeInfo, parent_tree_node_id=-2) tuples for left/right children
            - should_spawn: True if both children should be spawned, False otherwise
        """
        # Filter X, y, loss_components to parent's subdomain
        # IMPORTANT: X is filtered to subdomain, but y is GLOBAL solution (all experts blended)
        mask = np.ones(len(X), dtype=bool)
        for dim in range(len(parent_region.bounds_lower)):
            mask &= (X[:, dim] >= parent_region.bounds_lower[dim])
            mask &= (X[:, dim] <= parent_region.bounds_upper[dim])

        X_sub = X[mask]              # Coordinates in subdomain
        y_sub = y[mask]              # Global predictions at those coordinates
        loss_sub = {
            'residual': loss_components['residual'][mask],
            'ic': loss_components['ic'][mask],
            'bc': loss_components['bc'][mask],
            'weights': loss_components['weights']
        }

        if verbose:
            print(f"      Subdomain: {parent_region.bounds_lower} -> {parent_region.bounds_upper}")
            print(f"      Filtered {len(X_sub)} / {len(X)} samples to subdomain")

        # Check if enough samples in subdomain
        if len(X_sub) < 2 * self.min_samples_leaf:
            if verbose:
                print(f"      Not enough samples ({len(X_sub)} < {2 * self.min_samples_leaf}), cannot split")
            return [], False

        # Fit single-split tree on subdomain (max_depth=1)
        # Temporarily store old settings
        old_max_depth = self.max_depth
        old_n_estimators = self.n_estimators

        # Set to max_depth=1 for single split
        self.max_depth = 1
        self.n_estimators = 1

        try:
            self.fit(X=X_sub, y=y_sub, loss_components=loss_sub)
        finally:
            # Restore settings
            self.max_depth = old_max_depth
            self.n_estimators = old_n_estimators

        # Get the tree
        tree = self.rf.estimators_[0].tree_

        # Root is node 0, check if it has children
        left_child_id = tree.children_left[0]
        right_child_id = tree.children_right[0]

        if left_child_id == -1:
            # No split happened (all samples identical or too few)
            if verbose:
                print(f"      Tree did not split (all samples have same target or too few samples)")
            return [], False

        # Compute all wavelet norms (includes root and both children)
        all_nodes = self.compute_wavelet_norms()
        node_lookup = {node.node_id: node for node in all_nodes}

        # Get both children
        left_child = node_lookup.get(left_child_id)
        right_child = node_lookup.get(right_child_id)

        if left_child is None or right_child is None:
            if verbose:
                print(f"      Cannot spawn children (missing nodes in tree)")
            return [], False

        # All-or-nothing rule:
        # Spawn BOTH if at least ONE meets threshold
        left_meets_threshold = (
            wavelet_threshold is None or
            left_child.wavelet_norm >= wavelet_threshold
        )
        right_meets_threshold = (
            wavelet_threshold is None or
            right_child.wavelet_norm >= wavelet_threshold
        )

        should_spawn = left_meets_threshold or right_meets_threshold

        if verbose:
            print(f"      Left child:  wavelet={left_child.wavelet_norm:.6f}, "
                  f"meets_threshold={left_meets_threshold}")
            print(f"      Right child: wavelet={right_child.wavelet_norm:.6f}, "
                  f"meets_threshold={right_meets_threshold}")
            if should_spawn:
                print(f"      ✓ At least one child meets threshold → SPAWN BOTH")
            else:
                print(f"      ✗ Both children below threshold → REJECT BOTH")

        if not should_spawn:
            return [], False

        # Spawn both children
        children = [
            (left_child, -2),   # -2 means "spawned during training"
            (right_child, -2)
        ]

        return children, True

    def _compute_outside_fraction(
        self, 
        node: TreeNodeInfo, 
        sibling_regions: List[RegionDescriptor]
    ) -> float:
        """
        Compute what fraction of a node's volume is OUTSIDE all sibling regions.
        
        Uses a simple approximation: compute total overlap with union of siblings.
        For non-overlapping siblings, this is exact. For overlapping siblings,
        this is a lower bound on the actual outside fraction.
        
        Args:
            node: The candidate node
            sibling_regions: List of same-depth regions (siblings)
            
        Returns:
            Fraction of node volume outside all siblings (0.0 to 1.0)
        """
        if not sibling_regions:
            return 1.0  # No siblings = 100% outside
        
        node_vol = 1.0
        for lo, hi in zip(node.bounds_lower, node.bounds_upper):
            node_vol *= (hi - lo)
        
        if node_vol <= 0:
            return 0.0
        
        # Compute total overlap volume with all siblings
        # Note: This may double-count if siblings overlap each other,
        # giving a conservative (lower) estimate of outside_fraction
        total_overlap = 0.0
        
        for region in sibling_regions:
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
                
                total_overlap += overlap_vol
        
        # Cap overlap at node volume (in case of double-counting)
        total_overlap = min(total_overlap, node_vol)
        
        # Return fraction outside
        outside_fraction = (node_vol - total_overlap) / node_vol
        return outside_fraction
    
    def _check_overlap(self, node: TreeNodeInfo, existing_regions: List[RegionDescriptor]) -> bool:
        """Check if a node significantly overlaps with existing regions.
        
        DEPRECATED: Use _compute_outside_fraction for more precise control.
        """
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
        loss_components: Optional[Dict[str, np.ndarray]] = None,
        residuals: Optional[np.ndarray] = None,
        sibling_regions: Optional[List[RegionDescriptor]] = None,
        overlap_threshold: float = 0.5,
        wavelet_threshold: Optional[float] = None,
        spawn_epoch: int = 0,
        depth: int = 1,
        parent_idx: int = -1,
        verbose: bool = True
    ) -> Optional[RegionDescriptor]:
        """
        Convenience method to fit RF and detect refinement region in one call.
        
        Args:
            X: (N, n_dims) array of coordinates
            y: (N,) or (N, output_dim) array of solution values
            loss_components: Dict with 'residual', 'ic', 'bc' arrays and 'weights'
                (new approach using total loss)
            residuals: DEPRECATED - (N,) array of PDE residuals (for backward compatibility)
            sibling_regions: List of same-parent regions to check overlap against
            overlap_threshold: Accept region if more than this fraction is outside siblings
            wavelet_threshold: Minimum wavelet norm to spawn
            spawn_epoch: Current epoch for tracking
            depth: Depth level for the new expert (1 = child of base)
            parent_idx: Index of the parent expert (-1 for depth-1 experts)
            verbose: Print diagnostic information about why regions were rejected
            
        Returns:
            RegionDescriptor for the selected region, or None
        """
        self.fit(X, y, loss_components=loss_components, residuals=residuals)
        return self.select_refinement_region(
            sibling_regions=sibling_regions,
            overlap_threshold=overlap_threshold,
            wavelet_threshold=wavelet_threshold,
            spawn_epoch=spawn_epoch,
            depth=depth,
            parent_idx=parent_idx,
            verbose=verbose
        )
