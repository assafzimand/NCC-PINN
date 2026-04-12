"""Region detection using Random Forest geometric wavelets.

Implements the algorithm to detect high-variation regions for spawning expert PINNs:
1. Fit a Random Forest regressor to the current solution
2. Compute geometric wavelets at each tree node (Q_child - Q_parent for d-dim output)
3. Compute wavelet norm: ||Q_child - Q_parent||^2 * n_samples
4. Select / prune regions based on norm threshold
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
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
    wavelet_norm_squared: float = 0.0  # ||Q_child - Q_parent||^2 * n_samples
    # Local tree-Besov smoothness: slope of log(||ψ_ν||₂/|ν|^½) vs log(|ν|) over descendants
    # Larger α = smoother region; None = not enough descendants for reliable estimate
    smoothness_alpha: Optional[float] = None
    smoothness_r2: Optional[float] = None   # R² of the log-log regression
    smoothness_n_desc: int = 0              # number of descendants used in fit


class RegionDetector:
    """Detects refinement regions using Random Forest geometric wavelets.
    
    Algorithm:
    1. Fit RF to current PINN solution: f_RF(x,t) ≈ u(x,t)
    2. For each tree node, compute geometric wavelet: ψ = Q_child - Q_parent
    3. Compute wavelet norm: ||ψ||^2 * n_samples
    4. Return regions that pass the threshold
    
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
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> 'RegionDetector':
        """
        Fit Random Forest to the data.
        
        Args:
            X: (N, n_dims) array of coordinates [x, t] or [x, y, t]
            y: (N,) or (N, output_dim) array of solution values.
               Multi-output is supported (RF multi-output regression).
            **kwargs: Accepted for backward compatibility (loss_components,
                      residuals) but no longer used.
            
        Returns:
            self for chaining
        """
        if y.ndim > 1 and y.shape[1] == 1:
            y = y.ravel()

        self.domain_bounds = {
            'lower': X.min(axis=0).tolist(),
            'upper': X.max(axis=0).tolist()
        }
        
        self.rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            bootstrap=False,
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

    def _compute_smoothness_indices(
        self,
        nodes_by_id: Dict[int, TreeNodeInfo],
        children_left,
        children_right,
        min_descendants: int = 10,
    ) -> None:
        """Populate smoothness_alpha/r2/n_desc in-place for each node.

        For each node Ω, collects all descendant nodes ν in its subtree and fits:
            log(||ψ_ν||₂ / |ν|^½)  ~  c + α * log(|ν|)
        where ||ψ_ν||₂ = sqrt(wavelet_norm_squared_ν) and |ν| ≈ n_samples_ν.
        The slope α is the local tree-Besov smoothness estimate.

        Larger α → smoother region; α ≈ 0 → rough/discontinuous.
        Sets smoothness_alpha = None when fewer than min_descendants points available.
        """
        def _collect_descendants(root_id: int) -> List[int]:
            result = []
            stack = [root_id]
            while stack:
                nid = stack.pop()
                for child in (children_left[nid], children_right[nid]):
                    if child != -1 and child in nodes_by_id:
                        result.append(child)
                        stack.append(child)
            return result

        for node_id, node_info in nodes_by_id.items():
            desc_ids = _collect_descendants(node_id)

            xs, ys = [], []
            for desc_id in desc_ids:
                desc = nodes_by_id[desc_id]
                n = desc.n_samples
                if n < self.min_samples_leaf or desc.wavelet_norm_squared <= 0.0:
                    continue
                normalized = np.sqrt(desc.wavelet_norm_squared) / np.sqrt(n)
                if normalized <= 0.0:
                    continue
                xs.append(np.log(float(n)))
                ys.append(np.log(float(normalized)))

            m = len(xs)
            node_info.smoothness_n_desc = m

            if m < min_descendants:
                node_info.smoothness_alpha = None
                node_info.smoothness_r2 = None
                continue

            x = np.array(xs, dtype=float)
            y = np.array(ys, dtype=float)
            A = np.vstack([x, np.ones_like(x)]).T
            alpha, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

            y_pred = alpha * x + intercept
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            node_info.smoothness_alpha = float(alpha)
            node_info.smoothness_r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else None

    def compute_wavelet_norms(self) -> List[TreeNodeInfo]:
        """
        Compute geometric wavelet norms for all tree nodes.
        
        Uses RF's internal node values (tree.value) as Q_Ω for each node.
        Supports multi-dimensional output (d > 1): Q is d-dimensional vector.
        Wavelet norm = ||Q_child - Q_parent||^2 * n_samples
        
        Returns:
            List of TreeNodeInfo with wavelet norms
        """
        if self.rf is None:
            raise RuntimeError("Must call fit() before compute_wavelet_norms()")

        all_nodes = []

        for tree_idx, estimator in enumerate(self.rf.estimators_):
            tree = estimator.tree_
            tree_nodes: Dict[int, TreeNodeInfo] = {}

            for node_id in range(tree.node_count):
                is_leaf = tree.children_left[node_id] == -1

                n_samples = int(tree.n_node_samples[node_id])

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

                wavelet_norm_squared = 0.0
                if parent_prediction is not None and n_samples > 0:
                    diff = prediction - parent_prediction
                    l2_norm_squared = float(np.sum(diff ** 2))
                    wavelet_norm_squared = l2_norm_squared * n_samples

                node = TreeNodeInfo(
                    node_id=node_id,
                    tree_idx=tree_idx,
                    is_leaf=is_leaf,
                    n_samples=n_samples,
                    bounds_lower=bounds_lower,
                    bounds_upper=bounds_upper,
                    prediction=prediction,
                    parent_prediction=parent_prediction,
                    wavelet_norm_squared=wavelet_norm_squared,
                )
                tree_nodes[node_id] = node

            self._compute_smoothness_indices(
                tree_nodes, tree.children_left, tree.children_right
            )
            all_nodes.extend(tree_nodes.values())

        return all_nodes

    def compute_new_wavelet_norm_squareds(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[TreeNodeInfo]:
        """
        Compute parent-assigned wavelet norms for all tree nodes.

        NEW formula — norm is assigned to each node based on its own children:
            norm(node) = ||Q_node - Q_left||^2 * n_left
                       + ||Q_node - Q_right||^2 * n_right

        For leaf nodes (no children in the fitted tree): a temporary
        DecisionTreeRegressor(max_depth=1, min_samples_leaf=1) is fitted on
        the samples that land in that leaf to obtain two hypothetical children.
        The same formula is applied and the temporary tree is discarded.
        If the leaf cannot be split, norm = 0.0.

        Args:
            X: (N, n_dims) sample coordinates — same array used to fit the tree
            y: (N,) or (N, output_dim) target values

        Returns:
            List of TreeNodeInfo with wavelet_norm_squared set to the new formula value.
            All other fields (bounds, n_samples, prediction, …) are identical
            to what compute_wavelet_norms() would return.
        """
        if self.rf is None:
            raise RuntimeError("Must call fit() before compute_new_wavelet_norm_squareds()")

        all_nodes = []

        for tree_idx, estimator in enumerate(self.rf.estimators_):
            tree = estimator.tree_
            tree_nodes: Dict[int, TreeNodeInfo] = {}

            # leaf assignment for every sample (needed for the leaf split trick)
            leaf_ids = estimator.apply(X)

            for node_id in range(tree.node_count):
                n_samples = int(tree.n_node_samples[node_id])
                if n_samples < self.min_samples_leaf:
                    continue

                is_leaf = tree.children_left[node_id] == -1
                bounds_lower, bounds_upper = self._get_node_bounds(tree, node_id)
                parent_id = self._get_parent_id(tree, node_id)

                prediction = tree.value[node_id, :, 0].copy()
                parent_prediction = (
                    tree.value[parent_id, :, 0].copy()
                    if parent_id is not None else None
                )

                if not is_leaf:
                    # ── Internal node: compute directly from tree values ──
                    l = tree.children_left[node_id]
                    r = tree.children_right[node_id]
                    Q_node  = tree.value[node_id, :, 0]
                    Q_left  = tree.value[l, :, 0]
                    Q_right = tree.value[r, :, 0]
                    n_left  = int(tree.n_node_samples[l])
                    n_right = int(tree.n_node_samples[r])
                    wavelet_norm_squared = (
                        float(np.sum((Q_left  - Q_node) ** 2)) * n_left
                        + float(np.sum((Q_right - Q_node) ** 2)) * n_right
                    )
                else:
                    # ── Leaf node: fit depth-1 subtree on this leaf's samples ──
                    mask   = (leaf_ids == node_id)
                    X_leaf = X[mask]
                    y_leaf = y[mask]

                    wavelet_norm_squared = 0.0
                    if len(X_leaf) >= 2:
                        dt = DecisionTreeRegressor(
                            max_depth=1, min_samples_leaf=1, random_state=42
                        )
                        try:
                            dt.fit(X_leaf, y_leaf)
                            sub = dt.tree_
                            sub_l = sub.children_left[0]
                            sub_r = sub.children_right[0]
                            if sub_l != -1:
                                Q_node  = sub.value[0,     :, 0]
                                Q_hl    = sub.value[sub_l, :, 0]
                                Q_hr    = sub.value[sub_r, :, 0]
                                n_hl    = int(sub.n_node_samples[sub_l])
                                n_hr    = int(sub.n_node_samples[sub_r])
                                wavelet_norm_squared = (
                                    float(np.sum((Q_hl - Q_node) ** 2)) * n_hl
                                    + float(np.sum((Q_hr - Q_node) ** 2)) * n_hr
                                )
                        except Exception:
                            pass

                node = TreeNodeInfo(
                    node_id=node_id,
                    tree_idx=tree_idx,
                    is_leaf=is_leaf,
                    n_samples=n_samples,
                    bounds_lower=bounds_lower,
                    bounds_upper=bounds_upper,
                    prediction=prediction,
                    parent_prediction=parent_prediction,
                    wavelet_norm_squared=wavelet_norm_squared,
                )
                tree_nodes[node_id] = node

            self._compute_smoothness_indices(
                tree_nodes, tree.children_left, tree.children_right
            )
            all_nodes.extend(tree_nodes.values())

        return all_nodes

    def extract_regions_from_tree(
        self,
        wavelet_threshold: Optional[float] = None,
        verbose: bool = True
    ) -> List[Tuple[TreeNodeInfo, int]]:
        """
        Traverse a single decision tree (BFS) and extract spawnable regions.

        Prerequisites:
        - fit() must be called first with n_estimators=1

        Args:
            wavelet_threshold: Minimum wavelet norm to spawn (None = spawn all non-root)
            verbose: Print diagnostic information

        Returns:
            List of (TreeNodeInfo, parent_tree_node_id) tuples in BFS order.
            parent_tree_node_id is the tree node ID of nearest SPAWNED ancestor,
            or -1 for base model.
        """
        from collections import deque

        # Validation
        if self.rf is None:
            raise RuntimeError("Must call fit() before extract_regions_from_tree()")
        if self.n_estimators != 1:
            raise RuntimeError(f"extract_regions_from_tree requires n_estimators=1, got {self.n_estimators}")

        # Get the single tree
        tree = self.rf.estimators_[0].tree_

        # Compute wavelet norms for all nodes
        all_nodes = self.compute_wavelet_norms()

        if not all_nodes:
            if verbose:
                print(f"    [Traverse] No nodes in tree")
            return []

        # Build node_id -> TreeNodeInfo lookup
        node_lookup = {node.node_id: node for node in all_nodes}

        # Result: list of (TreeNodeInfo, nearest_spawned_ancestor_tree_node_id)
        result = []

        # Tracking structure: maps tree_node_id -> whether it was spawned
        spawned_nodes = {0: True}  # Root is always considered "spawned" (represents base model)

        # BFS queue initialization
        queue = deque([(0, -1)])  # (node_id, nearest_spawned_ancestor_id)
        # -1 means "root" which maps to base model (parent_idx=-1 in RegionDescriptor)

        visited_count = 0
        spawned_count = 0
        skipped_count = 0

        if verbose:
            print(f"\n  [Traverse] Starting BFS traversal of tree with {tree.node_count} nodes")

        # Traverse the tree (BFS)
        while queue:
            current_id, nearest_spawned_ancestor = queue.popleft()
            visited_count += 1

            # Skip if node doesn't exist in lookup (too few samples)
            if current_id not in node_lookup:
                continue

            node = node_lookup[current_id]

            # Determine if this node should spawn
            should_spawn = False
            rejection_reason = None

            if current_id == 0:
                # Root node: skip (represents base model, not a spawnable expert)
                should_spawn = False
                rejection_reason = "root node"
            elif node.parent_prediction is None:
                # No parent (shouldn't happen except for root, but defensive)
                should_spawn = False
                rejection_reason = "no parent prediction"
            elif wavelet_threshold is not None and node.wavelet_norm_squared < wavelet_threshold:
                # Below threshold
                should_spawn = False
                rejection_reason = f"wavelet_norm_squared {node.wavelet_norm_squared:.6f} < threshold {wavelet_threshold}"
            else:
                # Passes all checks
                should_spawn = True

            # Update tracking
            if should_spawn:
                spawned_nodes[current_id] = True
                result.append((node, nearest_spawned_ancestor))
                spawned_count += 1

                if verbose:
                    parent_str = "Base" if nearest_spawned_ancestor == -1 else f"Node{nearest_spawned_ancestor}"
                    print(f"    [Traverse] Node {current_id}: SPAWN (parent={parent_str}, "
                          f"wavelet={node.wavelet_norm_squared:.6f}, samples={node.n_samples})")

                # This node becomes the new nearest spawned ancestor for its children
                next_nearest_spawned = current_id
            else:
                spawned_nodes[current_id] = False
                skipped_count += 1

                if verbose and rejection_reason:
                    print(f"    [Traverse] Node {current_id}: SKIP ({rejection_reason})")

                # Children inherit current nearest_spawned_ancestor (pass through)
                next_nearest_spawned = nearest_spawned_ancestor

            # Add children to queue
            left_child = tree.children_left[current_id]
            right_child = tree.children_right[current_id]

            if left_child != -1:  # Not a leaf
                queue.append((left_child, next_nearest_spawned))

            if right_child != -1:  # Not a leaf
                queue.append((right_child, next_nearest_spawned))

        if verbose:
            print(f"\n  [Traverse] Summary:")
            print(f"    Visited nodes: {visited_count}")
            print(f"    Spawnable nodes: {spawned_count}")
            print(f"    Skipped nodes: {skipped_count}")

        return result

    def spawn_children_for_node(
        self,
        parent_region: RegionDescriptor,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
        **kwargs,
    ) -> List[Tuple[TreeNodeInfo, int]]:
        """
        Spawn children for a single parent node.

        Fits a single-split tree (max_depth=1) on the parent's subdomain.
        Creates 2 children (left/right) and computes their wavelet norms.

        Args:
            parent_region: The parent node's region descriptor
            X: (N, n_dims) coordinates (full eval_data)
            y: (N, output_dim) predictions (global solution)
            verbose: Print diagnostic info
            **kwargs: Accepted for backward compat (loss_components) but unused.

        Returns:
            List of (TreeNodeInfo, parent_tree_node_id=-2) tuples.
        """
        mask = np.ones(len(X), dtype=bool)
        for dim in range(len(parent_region.bounds_lower)):
            mask &= (X[:, dim] >= parent_region.bounds_lower[dim])
            mask &= (X[:, dim] <= parent_region.bounds_upper[dim])

        X_sub = X[mask]
        y_sub = y[mask]

        if verbose:
            print(f"      Subdomain: {parent_region.bounds_lower} -> {parent_region.bounds_upper}")
            print(f"      Filtered {len(X_sub)} / {len(X)} samples to subdomain")

        # Check if enough samples in subdomain
        if len(X_sub) < 2 * self.min_samples_leaf:
            if verbose:
                print(f"      Not enough samples ({len(X_sub)} < {2 * self.min_samples_leaf}), cannot split")
            return []

        # Fit single-split tree on subdomain (max_depth=1)
        # Temporarily store old settings
        old_max_depth = self.max_depth
        old_n_estimators = self.n_estimators

        # Set to max_depth=1 for single split
        self.max_depth = 1
        self.n_estimators = 1

        try:
            self.fit(X=X_sub, y=y_sub)
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
            return []

        # Extract node information for both children
        children = []

        # Compute all wavelet norms (includes root and both children)
        all_nodes = self.compute_wavelet_norms()
        node_lookup = {node.node_id: node for node in all_nodes}

        for child_id in [left_child_id, right_child_id]:
            if child_id in node_lookup:
                child_node = node_lookup[child_id]
                # Use -2 as special marker for "spawned during training"
                children.append((child_node, -2))

                if verbose:
                    print(f"      Child {child_id}: bounds {child_node.bounds_lower} -> {child_node.bounds_upper}, "
                          f"wavelet={child_node.wavelet_norm_squared:.6f}, samples={child_node.n_samples}")

        return children

    def fit_full_tree_and_prune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        wavelet_threshold: float = 0.0,
        tree_smoothness_threshold: Optional[float] = None,
        verbose: bool = True,
        **kwargs,
    ) -> Tuple[List[Tuple[TreeNodeInfo, int]], Dict]:
        """
        Fit a full decision tree on the entire domain and prune via
        bottom-up sibling-pair thresholding.

        Pruning criterion (smoothness-based, active):
            A sibling pair is ACCEPTED if either node has smoothness_alpha < threshold
            (rough region). Nodes with smoothness_alpha=None (too few descendants,
            i.e. very fine leaves) are treated as smooth → pruned.
            A pair is pruned only when both siblings are confirmed smooth.

        Pruning criterion (wavelet-norm-based, commented out — kept for reference):
            Was: accept if wavelet_norm_squared >= wavelet_threshold.

        Args:
            X: (N, n_dims) coordinates
            y: (N,) or (N, output_dim) predictions
            wavelet_threshold: kept for reference / future use (currently not used
                for pruning decisions when tree_smoothness_threshold is set)
            tree_smoothness_threshold: smoothness_alpha cutoff. Nodes with
                alpha < threshold are rough → kept. None/missing alpha → pruned.
                If None, falls back to wavelet_norm_squared thresholding.
            verbose: print diagnostics
            **kwargs: ignored (backward compat)

        Returns:
            Tuple of:
            - List of (TreeNodeInfo, parent_tree_node_id) in BFS order.
              parent_tree_node_id is the tree node id of the nearest
              accepted ancestor, or -1 for children of root.
            - Dict of per-depth pruning statistics.
        """
        from collections import deque

        old_n_estimators = self.n_estimators
        self.n_estimators = 1
        try:
            self.fit(X=X, y=y)
        finally:
            self.n_estimators = old_n_estimators

        tree = self.rf.estimators_[0].tree_
        all_nodes = self.compute_wavelet_norms()

        if not all_nodes:
            if verbose:
                print("  [FullTree] No nodes in tree")
            return [], {}

        node_lookup = {node.node_id: node for node in all_nodes}

        # Build parent->children mapping and compute depth of each node
        children_left = tree.children_left
        children_right = tree.children_right
        node_depth = {}
        parent_map = {}
        queue = deque([(0, 0)])
        max_depth_seen = 0
        while queue:
            nid, depth = queue.popleft()
            node_depth[nid] = depth
            max_depth_seen = max(max_depth_seen, depth)
            l, r = children_left[nid], children_right[nid]
            if l != -1:
                parent_map[l] = nid
                queue.append((l, depth + 1))
            if r != -1:
                parent_map[r] = nid
                queue.append((r, depth + 1))

        # Group sibling pairs by depth
        depth_to_sibling_pairs = {}
        for nid in range(tree.node_count):
            l, r = children_left[nid], children_right[nid]
            if l != -1 and r != -1:
                child_depth = node_depth.get(l, 0)
                if child_depth not in depth_to_sibling_pairs:
                    depth_to_sibling_pairs[child_depth] = []
                depth_to_sibling_pairs[child_depth].append((l, r))

        use_smoothness = tree_smoothness_threshold is not None

        # Helper: is a node rough enough to keep? (smoothness-based)
        def _is_rough(nid: int) -> bool:
            if nid not in node_lookup:
                return False  # unknown node → treat as smooth → prune
            alpha = node_lookup[nid].smoothness_alpha
            if alpha is None:
                return False  # too few descendants → treat as smooth → prune
            return alpha < tree_smoothness_threshold

        accepted = set()
        depth_stats = {}

        # Bottom-up: deepest first
        for depth in range(max_depth_seen, 0, -1):
            pairs = depth_to_sibling_pairs.get(depth, [])
            n_by_threshold = 0
            n_by_child = 0
            n_rejected = 0
            alphas_at_depth = []

            for left_id, right_id in pairs:
                left_accepted = left_id in accepted
                right_accepted = right_id in accepted

                if use_smoothness:
                    # ── Smoothness-based pruning (active) ────────────────────
                    left_alpha = (node_lookup[left_id].smoothness_alpha
                                  if left_id in node_lookup else None)
                    right_alpha = (node_lookup[right_id].smoothness_alpha
                                   if right_id in node_lookup else None)
                    for a in (left_alpha, right_alpha):
                        if a is not None:
                            alphas_at_depth.append(a)

                    if left_accepted or right_accepted:
                        accept_pair = True
                        n_by_child += 1
                    elif _is_rough(left_id) or _is_rough(right_id):
                        accept_pair = True
                        n_by_threshold += 1
                    else:
                        accept_pair = False
                        n_rejected += 1
                else:
                    # ── Wavelet-norm-based pruning (fallback / kept for reference) ──
                    # left_wn = (node_lookup[left_id].wavelet_norm_squared
                    #            if left_id in node_lookup else 0.0)
                    # right_wn = (node_lookup[right_id].wavelet_norm_squared
                    #             if right_id in node_lookup else 0.0)
                    # alphas_at_depth.extend([left_wn, right_wn])
                    # if left_accepted or right_accepted:
                    #     accept_pair = True; n_by_child += 1
                    # elif left_wn >= wavelet_threshold or right_wn >= wavelet_threshold:
                    #     accept_pair = True; n_by_threshold += 1
                    # else:
                    #     accept_pair = False; n_rejected += 1
                    left_wn = (node_lookup[left_id].wavelet_norm_squared
                               if left_id in node_lookup else 0.0)
                    right_wn = (node_lookup[right_id].wavelet_norm_squared
                                if right_id in node_lookup else 0.0)
                    alphas_at_depth.extend([left_wn, right_wn])
                    if left_accepted or right_accepted:
                        accept_pair = True
                        n_by_child += 1
                    elif (left_wn >= wavelet_threshold
                          or right_wn >= wavelet_threshold):
                        accept_pair = True
                        n_by_threshold += 1
                    else:
                        accept_pair = False
                        n_rejected += 1

                if accept_pair:
                    for nid in (left_id, right_id):
                        accepted.add(nid)
                        cur = nid
                        while cur in parent_map:
                            cur = parent_map[cur]
                            if cur in accepted:
                                break
                            accepted.add(cur)

            if pairs and alphas_at_depth:
                depth_stats[depth] = {
                    'n_pairs': len(pairs),
                    'accepted_by_threshold': n_by_threshold,
                    'accepted_by_child': n_by_child,
                    'rejected': n_rejected,
                    'alpha_min': float(min(alphas_at_depth)),
                    'alpha_max': float(max(alphas_at_depth)),
                    'alpha_median': float(np.median(alphas_at_depth)),
                }

        # Root itself is not an expert (it is the base model)
        accepted.discard(0)

        thr_label = (f"smoothness<{tree_smoothness_threshold}"
                     if use_smoothness else f"norm>={wavelet_threshold}")
        if verbose:
            print(f"\n  [FullTree] Tree has {tree.node_count} nodes, "
                  f"max depth {max_depth_seen}")
            print(f"  [FullTree] Accepted {len(accepted)} nodes "
                  f"({thr_label})")
            print(f"  [FullTree] Per-depth pruning stats:")
            for d in sorted(depth_stats.keys()):
                s = depth_stats[d]
                print(
                    f"    Depth {d:2d}: "
                    f"{s['n_pairs']:3d} pairs | "
                    f"{s['accepted_by_threshold']:2d} by threshold | "
                    f"{s['accepted_by_child']:2d} by child | "
                    f"{s['rejected']:3d} rejected | "
                    f"alpha [{s['alpha_min']:.3f}, "
                    f"{s['alpha_median']:.3f}, "
                    f"{s['alpha_max']:.3f}]"
                )

        # Build result in BFS order with parent relationships
        result = []
        bfs = deque([(0, -1)])  # (node_id, nearest_accepted_ancestor)
        while bfs:
            nid, nearest_ancestor = bfs.popleft()
            if nid != 0 and nid in accepted and nid in node_lookup:
                result.append((node_lookup[nid], nearest_ancestor))
                if verbose:
                    node = node_lookup[nid]
                    anc_str = "Base" if nearest_ancestor == -1 else f"Node{nearest_ancestor}"
                    is_leaf_str = "leaf" if node.is_leaf else "internal"
                    alpha_str = (f"{node.smoothness_alpha:.4f}"
                                 if node.smoothness_alpha is not None else "None")
                    print(f"    [FullTree] Node {nid} ({is_leaf_str}): "
                          f"ACCEPT (parent={anc_str}, "
                          f"alpha={alpha_str}, "
                          f"samples={node.n_samples})")
                next_ancestor = nid
            else:
                next_ancestor = nearest_ancestor

            l, r = children_left[nid], children_right[nid]
            if l != -1:
                bfs.append((l, next_ancestor))
            if r != -1:
                bfs.append((r, next_ancestor))

        if verbose:
            print(f"  [FullTree] Result: {len(result)} accepted nodes")

        return result, depth_stats

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
