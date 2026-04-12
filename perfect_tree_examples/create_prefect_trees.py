"""Generate "perfect tree" visualizations and model-reconstruction
JSON files for every PDE problem.

For each problem, fits a full decision tree on ground-truth data
(from eval_data.pt), prunes it using wavelet norms with the
configured threshold, and produces:

  1. A 3-panel PNG image:
     - Regions BEFORE pruning on GT background
     - Regions AFTER pruning on GT background
     - Tree hierarchy diagram (dendrogram)

  2. A JSON file containing the full tree structure, accepted
     nodes in BFS order with parent relationships, and all
     metadata needed to reconstruct an AToE / AToELeaves / ANT
     model from the tree.
"""

import json
import sys
import importlib
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptive.region_detector import RegionDetector  # noqa: E402
from adaptive.visualization import prepare_ground_truth_grid  # noqa: E402
from utils.dataset_gen import calculate_dataset_sizes  # noqa: E402


class _NumpySafeEncoder(json.JSONEncoder):
    """Handle numpy types that stdlib json can't serialize."""
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def load_config(plan_path: Path) -> dict:
    """Load experiments_plan.yaml and return the base_config."""
    with open(plan_path, 'r') as f:
        plan = yaml.safe_load(f)
    return plan.get('base_config', {})


def get_problem_list(base_cfg: dict) -> list:
    """Return all problem names that have a sub-config with spatial_dim."""
    skip = {'sampling', 'adaptive_pinn'}
    problems = []
    for key, val in base_cfg.items():
        if isinstance(val, dict) and 'spatial_dim' in val and key not in skip:
            problems.append(key)
    return sorted(problems)


def ensure_eval_data(problem: str, base_cfg: dict) -> dict:
    """Load eval_data.pt, generating it if missing."""
    eval_path = Path('datasets') / problem / 'eval_data.pt'
    if eval_path.exists():
        return torch.load(eval_path, map_location='cpu')

    print(f"  Generating eval data for {problem}...")
    cfg = dict(base_cfg)
    cfg['problem'] = problem
    cfg['cuda'] = False
    sizes = calculate_dataset_sizes(cfg)
    solver = importlib.import_module(f'solvers.{problem}_solver')
    eval_data = solver.generate_dataset(
        n_residual=sizes['n_residual_eval'],
        n_ic=sizes['n_initial_eval'],
        n_bc=sizes['n_boundary_eval'],
        device=torch.device('cpu'),
        config=cfg,
    )
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(eval_data, eval_path)
    return eval_data


def build_domain_bounds(problem_cfg: dict) -> dict:
    """Build domain_bounds dict from problem config."""
    lower = [sd[0] for sd in problem_cfg['spatial_domain']]
    lower.append(problem_cfg['temporal_domain'][0])
    upper = [sd[1] for sd in problem_cfg['spatial_domain']]
    upper.append(problem_cfg['temporal_domain'][1])
    return {'lower': lower, 'upper': upper}


def extract_xy(eval_data: dict, output_dim: int):
    """Extract (X, y) arrays from eval data for tree fitting."""
    x = eval_data['x'].cpu().numpy() if isinstance(eval_data['x'], torch.Tensor) else eval_data['x']
    t = eval_data['t'].cpu().numpy() if isinstance(eval_data['t'], torch.Tensor) else eval_data['t']

    if x.ndim == 1:
        x = x[:, None]
    t = t.ravel()[:, None]
    X = np.hstack([x, t])

    for key in ('h_gt', 'h', 'u'):
        if key in eval_data:
            y = eval_data[key]
            break
    else:
        raise ValueError(f"No GT key found in eval_data. Keys: {list(eval_data.keys())}")

    y = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.ravel()
    return X, y


def fit_and_get_all_nodes(
    X, y, max_depth, min_samples_leaf, wavelet_threshold,
    tree_smoothness_threshold=None,
):
    """Fit tree, prune, return visualization + reconstruction data.

    Returns:
        node_dicts: list of dicts for ALL non-root nodes (for plots)
        accepted_ids: set of accepted tree node ids
        bfs_accepted: list of dicts for accepted nodes in BFS order,
            each with 'parent_tree_node_id' (nearest accepted
            ancestor, -1 for children of root). This is the data
            needed to reconstruct an adaptive model.
        children_left_arr: sklearn children_left array (to detect
            leaves of the pruned tree for AToELeaves)
    """
    detector = RegionDetector(
        n_estimators=1,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )

    accepted_nodes, depth_stats = detector.fit_full_tree_and_prune(
        X, y,
        wavelet_threshold=wavelet_threshold,
        tree_smoothness_threshold=tree_smoothness_threshold,
        verbose=True,
    )
    accepted_ids = {n.node_id for n, _ in accepted_nodes}

    tree = detector.rf.estimators_[0].tree_
    children_left = tree.children_left
    children_right = tree.children_right

    _node_depth = {0: 0}
    _parent_map = {}
    bfs = deque([0])
    while bfs:
        nid = bfs.popleft()
        for child in (children_left[nid], children_right[nid]):
            if child != -1:
                _parent_map[child] = nid
                _node_depth[child] = _node_depth[nid] + 1
                bfs.append(child)

    all_wn = detector.compute_wavelet_norms()

    node_dicts = []
    for nd in all_wn:
        if nd.node_id == 0:
            continue
        node_dicts.append({
            'node_id': nd.node_id,
            'parent_node_id': _parent_map.get(nd.node_id, -1),
            'wavelet_norm_squared': nd.wavelet_norm_squared,
            'smoothness_alpha': nd.smoothness_alpha,
            'smoothness_r2': nd.smoothness_r2,
            'smoothness_n_levels': nd.smoothness_n_levels,
            'n_samples': nd.n_samples,
            'is_leaf': bool(nd.is_leaf),
            'bounds_lower': nd.bounds_lower,
            'bounds_upper': nd.bounds_upper,
            'accepted': bool(nd.node_id in accepted_ids),
            'tree_depth': _node_depth.get(nd.node_id, -1),
        })

    # BFS-ordered accepted nodes with parent_tree_node_id
    # This mirrors what fit_full_tree_and_prune returns
    bfs_accepted = []
    for node_info, parent_tree_nid in accepted_nodes:
        is_pruned_leaf = (
            children_left[node_info.node_id] == -1
            or children_left[node_info.node_id] not in accepted_ids
        )
        bfs_accepted.append({
            'node_id': node_info.node_id,
            'parent_tree_node_id': parent_tree_nid,
            'bounds_lower': node_info.bounds_lower,
            'bounds_upper': node_info.bounds_upper,
            'wavelet_norm_squared': node_info.wavelet_norm_squared,
            'smoothness_alpha': node_info.smoothness_alpha,
            'n_samples': node_info.n_samples,
            'tree_depth': _node_depth.get(
                node_info.node_id, -1),
            'is_leaf_in_pruned_tree': is_pruned_leaf,
        })

    return (
        node_dicts,
        accepted_ids,
        bfs_accepted,
        children_left,
    )


def _plot_regions_panel(ax, regions_dicts, domain_bounds, gt_grid, grid_x, grid_t, title):
    """Draw GT heatmap + region outlines on a given axes."""
    x_min, t_min = domain_bounds['lower'][:2]
    x_max, t_max = domain_bounds['upper'][:2]

    if gt_grid is not None and grid_x is not None and grid_t is not None:
        if gt_grid.ndim == 3:
            display = np.linalg.norm(gt_grid, axis=2)
        else:
            display = gt_grid
        T, X = np.meshgrid(grid_t, grid_x)
        im = ax.pcolormesh(X, T, display, shading='auto', cmap='viridis', alpha=0.7, zorder=0)
        plt.colorbar(im, ax=ax, shrink=0.7)

    for nd in regions_dicts:
        bl = nd['bounds_lower']
        bu = nd['bounds_upper']
        rx_min, rt_min = bl[0], bl[-1]
        rx_max, rt_max = bu[0], bu[-1]
        rect = patches.Rectangle(
            (rx_min, rt_min), rx_max - rx_min, rt_max - rt_min,
            linewidth=1.0, edgecolor='black', facecolor='none', zorder=10,
        )
        ax.add_patch(rect)

    pad = 0.03
    xr = x_max - x_min
    tr = t_max - t_min
    ax.set_xlim(x_min - pad * xr, x_max + pad * xr)
    ax.set_ylim(t_min - pad * tr, t_max + pad * tr)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title(title, fontsize=11)
    ax.set_aspect('auto')


def _plot_hierarchy_panel(ax, all_nodes, smoothness_threshold, wavelet_threshold=None):
    """Dendrogram-style tree hierarchy colored by smoothness_alpha (Besov index).

    Nodes are colored red (rough, low α) → green (smooth, high α) via RdYlGn.
    The smoothness threshold is marked on the colorbar. Nodes with no smoothness
    estimate (α=None, too few descendants) are shown in gray.
    """
    if not all_nodes:
        ax.set_title('Tree Hierarchy (no nodes)')
        return

    children_map = {}
    node_map = {}
    for n in all_nodes:
        nid = n['node_id']
        pid = n.get('parent_node_id', -1)
        node_map[nid] = n
        children_map.setdefault(pid, []).append(n)

    root_children = children_map.get(0, [])
    if not root_children:
        root_children = [n for n in all_nodes if n.get('parent_node_id', -1) == -1]
    if not root_children:
        ax.set_title('Tree Hierarchy (no root children)')
        return

    leaf_counter = [0]
    positions = {}

    def _layout(nid):
        kids = children_map.get(nid, [])
        depth = node_map[nid]['tree_depth']
        if not kids:
            x = leaf_counter[0]
            leaf_counter[0] += 1
            positions[nid] = (x, depth)
            return x
        child_xs = [_layout(c['node_id']) for c in sorted(kids, key=lambda c: c['node_id'])]
        x = np.mean(child_xs)
        positions[nid] = (x, depth)
        return x

    rc_xs = [_layout(c['node_id']) for c in sorted(root_children, key=lambda c: c['node_id'])]
    root_x = np.mean(rc_xs)
    positions[0] = (root_x, 0)

    max_depth = max(n['tree_depth'] for n in all_nodes) if all_nodes else 1

    # Build colormap range from valid smoothness_alpha values
    alphas = [n['smoothness_alpha'] for n in all_nodes
              if n.get('smoothness_alpha') is not None]
    cmap = plt.get_cmap('RdYlGn')  # red=rough (low α), green=smooth (high α)

    if alphas and smoothness_threshold is not None:
        from matplotlib.colors import TwoSlopeNorm
        vmin = min(min(alphas), smoothness_threshold - 0.1)
        vmax = max(max(alphas), smoothness_threshold + 0.1)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=smoothness_threshold, vmax=vmax)
    elif alphas:
        norm = plt.Normalize(vmin=min(alphas), vmax=max(alphas))
    else:
        norm = plt.Normalize(vmin=0.0, vmax=1.0)

    # Draw edges
    for n in all_nodes:
        nid = n['node_id']
        pid = n.get('parent_node_id', -1)
        if pid >= 0 and pid in positions and nid in positions:
            px, py = positions[pid]
            cx, cy = positions[nid]
            edge_alpha = 0.7 if n['accepted'] else 0.15
            ax.plot([px, cx], [-py, -cy], 'k-', alpha=edge_alpha, lw=0.6, zorder=1)

    ax.scatter(root_x, 0, c='white', s=60, edgecolors='black', linewidths=1.5, zorder=6, marker='s')

    # Draw nodes — gray when α=None
    for n in all_nodes:
        nid = n['node_id']
        if nid not in positions:
            continue
        x, y = positions[nid]
        alpha_val = n.get('smoothness_alpha')
        if alpha_val is None:
            color = ['#bbbbbb'] if n['accepted'] else ['#dddddd']
            ec = 'black' if n['accepted'] else 'gray'
            node_alpha = 1.0 if n['accepted'] else 0.3
        else:
            color = [cmap(norm(alpha_val))]
            ec = 'black' if n['accepted'] else 'gray'
            node_alpha = 1.0 if n['accepted'] else 0.25

        size = 35 if n['accepted'] else 15
        lw = 0.8 if n['accepted'] else 0.3
        ax.scatter(x, -y, c=color, s=size, edgecolors=ec,
                   linewidths=lw, alpha=node_alpha,
                   zorder=5 if n['accepted'] else 4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, label='Smoothness α', shrink=0.7)

    # Mark threshold on colorbar
    if smoothness_threshold is not None and alphas:
        cb.ax.axhline(y=smoothness_threshold, color='black',
                      linewidth=1.5, linestyle='--')
        cb.ax.text(0.5, smoothness_threshold,
                   f' thr={smoothness_threshold}',
                   transform=cb.ax.get_yaxis_transform(),
                   va='bottom', ha='left', fontsize=7, color='black')

    ax.set_ylabel('Depth')
    ax.set_yticks([-d for d in range(max_depth + 1)])
    ax.set_yticklabels([str(d) for d in range(max_depth + 1)])
    ax.set_xticks([])
    n_acc = sum(1 for n in all_nodes if n['accepted'])
    n_none = sum(1 for n in all_nodes if n.get('smoothness_alpha') is None)
    thr_label = (f'α < {smoothness_threshold}'
                 if smoothness_threshold is not None
                 else f'norm ≥ {wavelet_threshold}')
    ax.set_title(
        f'Tree Hierarchy  (keep: {thr_label})\n'
        f'{n_acc} accepted / {len(all_nodes) - n_acc} rejected'
        f'  [{n_none} gray=no α]',
        fontsize=10)
    ax.grid(True, alpha=0.15, axis='y')


def build_problem_tree_data(
    problem, domain_bounds,
    max_depth, min_samples_leaf, wavelet_threshold,
    bfs_accepted, node_dicts,
    tree_smoothness_threshold=None,
):
    """Build the dict for one problem's perfect tree."""
    n_leaves = sum(
        1 for n in bfs_accepted
        if n['is_leaf_in_pruned_tree']
    )
    return {
        'domain_bounds': domain_bounds,
        'tree_params': {
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'wavelet_threshold': wavelet_threshold,
            'tree_smoothness_threshold': tree_smoothness_threshold,
        },
        'summary': {
            'total_nodes': len(node_dicts),
            'accepted_nodes': len(bfs_accepted),
            'pruned_tree_leaves': n_leaves,
        },
        'accepted_nodes_bfs': bfs_accepted,
        'all_nodes': node_dicts,
    }


def process_problem(
    problem: str, base_cfg: dict, output_dir: Path
):
    """Generate the 3-panel image + return tree data dict.

    Returns the tree data dict for this problem, or None if
    the problem was skipped.
    """
    print(f"\n{'='*60}")
    print(f"  Problem: {problem}")
    print(f"{'='*60}")

    problem_cfg = base_cfg[problem]
    adaptive_cfg = base_cfg.get('adaptive_pinn', {})

    max_depth = adaptive_cfg.get('tree_max_depth', 30)
    min_samples_leaf = adaptive_cfg.get(
        'tree_min_samples_leaf', 10)
    wavelet_threshold = problem_cfg.get(
        'wavelet_threshold',
        adaptive_cfg.get('wavelet_threshold', 5.0))
    tree_smoothness_threshold = problem_cfg.get(
        'tree_smoothness_threshold',
        adaptive_cfg.get('tree_smoothness_threshold', None))
    output_dim = problem_cfg.get('output_dim', 1)

    print(
        f"  max_depth={max_depth}, "
        f"min_samples_leaf={min_samples_leaf}, "
        f"wavelet_threshold={wavelet_threshold}, "
        f"tree_smoothness_threshold={tree_smoothness_threshold}")

    eval_data = ensure_eval_data(problem, base_cfg)
    domain_bounds = build_domain_bounds(problem_cfg)

    if len(domain_bounds['lower']) != 2:
        print(
            f"  Skipping {problem}: "
            f"only 2D (x,t) domains supported.")
        return None

    X, y = extract_xy(eval_data, output_dim)
    print(f"  Data: X={X.shape}, y="
          f"{y.shape if hasattr(y, 'shape') else '?'}")

    (node_dicts, accepted_ids,
     bfs_accepted, children_left) = fit_and_get_all_nodes(
        X, y, max_depth, min_samples_leaf, wavelet_threshold,
        tree_smoothness_threshold=tree_smoothness_threshold,
    )

    # -- Build tree data for unified JSON --
    tree_data = build_problem_tree_data(
        problem, domain_bounds,
        max_depth, min_samples_leaf, wavelet_threshold,
        bfs_accepted, node_dicts,
        tree_smoothness_threshold=tree_smoothness_threshold,
    )

    # -- Generate 3-panel plot --
    gt_grid, grid_x, grid_t = prepare_ground_truth_grid(
        eval_data, domain_bounds, resolution=150)

    all_region_dicts = list(node_dicts)
    accepted_region_dicts = [
        n for n in node_dicts if n['accepted']]

    n_total = len(all_region_dicts)
    n_accepted = len(accepted_region_dicts)
    print(f"  Nodes: {n_total} total, "
          f"{n_accepted} accepted")

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    _plot_regions_panel(
        axes[0], all_region_dicts, domain_bounds,
        gt_grid, grid_x, grid_t,
        f'{problem}: Before Pruning ({n_total} nodes)')
    thr_label = (f'smoothness α<{tree_smoothness_threshold}'
                 if tree_smoothness_threshold is not None
                 else f'norm≥{wavelet_threshold}')
    _plot_regions_panel(
        axes[1], accepted_region_dicts, domain_bounds,
        gt_grid, grid_x, grid_t,
        f'{problem}: After Pruning ({n_accepted} nodes, {thr_label})')
    _plot_hierarchy_panel(
        axes[2], node_dicts, tree_smoothness_threshold, wavelet_threshold)

    fig.suptitle(
        f'Perfect Tree \u2014 {problem}  '
        f'(depth={max_depth}, min_leaf={min_samples_leaf}, {thr_label})',
        fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    out_path = output_dir / f'{problem}_perfect_tree.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {out_path}")

    return tree_data


def main():
    plan_path = (
        Path(__file__).resolve().parent.parent
        / 'experiments_plan.yaml')
    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(plan_path)
    problems = get_problem_list(base_cfg)

    print(f"Found {len(problems)} problems: {problems}")
    print(f"Output directory: {output_dir}")

    all_trees = {}
    for problem in problems:
        try:
            tree_data = process_problem(
                problem, base_cfg, output_dir)
            if tree_data is not None:
                all_trees[problem] = tree_data
        except Exception as e:
            print(f"\n  ERROR processing {problem}: {e}")
            import traceback
            traceback.print_exc()

    # Save unified JSON with all problems
    json_path = output_dir / 'perfect_trees.json'
    with open(json_path, 'w') as f:
        json.dump(all_trees, f, indent=2, cls=_NumpySafeEncoder)
    print(f"\nSaved unified JSON: {json_path}")
    print(f"  Problems included: {list(all_trees.keys())}")
    for p, d in all_trees.items():
        s = d['summary']
        print(f"    {p}: {s['accepted_nodes']} accepted "
              f"({s['pruned_tree_leaves']} leaves) "
              f"/ {s['total_nodes']} total")

    print(f"\nDone! Images + JSON in {output_dir}")


if __name__ == '__main__':
    main()
