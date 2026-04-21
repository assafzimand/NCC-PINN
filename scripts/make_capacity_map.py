"""Build a parameter-capacity heatmap for each run in an experiment batch.

For every point (x, t) in the domain the capacity is:

    capacity(x, t) = base_params
                     + Σ_{k : point ∈ Ω_k} expert_k_params

Which experts are counted depends on the model type:
  - AToE  : all experts (each has a hard region Ω_k)
  - AToELeaves : leaf experts only (non-leaf parents are frozen)
  - ANT   : all experts (same as AToE; output layers of non-leaves
            are excluded from the count, matching trainer accounting)

The output per run is a side-by-side figure:
  Left  – expert_regions_final.png (spatial region layout)
  Right – capacity heatmap (continuous, same domain axes)

Usage:
    python make_capacity_map.py <batch_or_run_dir>
"""

import sys
import json
import re
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


_TS_RE = re.compile(r'\d{8}_\d{6}$')

# ──────────────────────────────────────────────────────────────────────────────
#  Directory discovery (mirrors plot_experts_predictions.py)
# ──────────────────────────────────────────────────────────────────────────────

def _find_run_dirs(batch_path: Path):
    """Return list of (label, ts_dir) pairs."""
    child_dirs = sorted(
        d for d in batch_path.iterdir()
        if d.is_dir() and d.name != 'checkpoints'
    )
    if not child_dirs:
        return []

    # Flat layout: batch_path contains timestamp dirs directly
    flat_ts = [
        d for d in child_dirs
        if _TS_RE.match(d.name) and (d / 'metrics.json').exists()
    ]
    if flat_ts:
        runs = []
        for ts_dir in flat_ts:
            cfg_file = ts_dir / 'config_used.yaml'
            label = ts_dir.name
            if cfg_file.exists():
                try:
                    with open(cfg_file) as f:
                        cfg = yaml.safe_load(f)
                    label = cfg.get('problem', ts_dir.name)
                except Exception:
                    pass
            runs.append((label, ts_dir))
        return runs

    # Nested layout: batch_path / model_name / timestamp(s)
    # Expand ALL timestamps per architecture (not just latest)
    runs = []
    for model_dir in child_dirs:
        ts_dirs = sorted(
            d for d in model_dir.iterdir()
            if d.is_dir() and d.name != 'checkpoints'
        )
        for ts_dir in ts_dirs:
            runs.append((f"{model_dir.name}/{ts_dir.name}", ts_dir))
    return runs


# ──────────────────────────────────────────────────────────────────────────────
#  Hard indicator (step function) — no torch dependency
# ──────────────────────────────────────────────────────────────────────────────

def _hard_mask(points: np.ndarray, lower: list, upper: list) -> np.ndarray:
    """Return boolean (N,) – True where point is inside the box [lower, upper]."""
    lo = np.array(lower)   # (D,)
    hi = np.array(upper)   # (D,)
    return np.all((points >= lo) & (points <= hi), axis=1)


# ──────────────────────────────────────────────────────────────────────────────
#  Capacity computation
# ──────────────────────────────────────────────────────────────────────────────

def _compute_capacity(points: np.ndarray, metrics: dict,
                      model_type: str) -> np.ndarray:
    """
    Compute per-point capacity (parameter count) for the model.

    Parameters
    ----------
    points     : (N, D) array of [x, t] (or [x, y, t]) coordinates
    metrics    : loaded metrics.json dict
    model_type : 'AToE' | 'AToELeaves' | 'ANT' (from config_used.yaml)

    Returns
    -------
    capacity : (N,) float array
    """
    adaptive = metrics.get('adaptive_pinn')
    if adaptive is None:
        total = metrics.get('total_params', 0)
        return np.full(len(points), float(total))

    base_params   = adaptive['base_params']
    expert_params = adaptive['expert_params']   # list[int], one per expert
    regions       = adaptive['regions']
    leaf_indices  = set(adaptive.get('leaf_expert_indices', []))

    # AToELeaves and ANT: only leaf experts contribute capacity at a point.
    # AToE: ALL experts whose region contains the point contribute.
    leaves_only = (model_type == 'AToELeaves')

    N = len(points)
    capacity = np.full(N, float(base_params))

    for i, region in enumerate(regions):
        if i >= len(expert_params):
            continue
        if leaves_only and i not in leaf_indices:
            continue
        lower = region['bounds_lower']
        upper = region['bounds_upper']
        mask  = _hard_mask(points, lower, upper)
        capacity[mask] += expert_params[i]

    return capacity


# ──────────────────────────────────────────────────────────────────────────────
#  Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_grid(x: np.ndarray, t: np.ndarray,
               values: np.ndarray, n_x=300, n_t=300):
    """Bin-average values onto a regular grid — no scipy required."""
    x_min, x_max = x.min(), x.max()
    t_min, t_max = t.min(), t.max()

    x_edges = np.linspace(x_min, x_max, n_x + 1)
    t_edges = np.linspace(t_min, t_max, n_t + 1)

    # 2D histogram of summed values and counts
    sum_grid, _, _ = np.histogram2d(
        x, t, bins=[x_edges, t_edges], weights=values)
    cnt_grid, _, _ = np.histogram2d(
        x, t, bins=[x_edges, t_edges])

    with np.errstate(invalid='ignore'):
        avg_grid = np.where(cnt_grid > 0, sum_grid / cnt_grid, np.nan)

    # Fill empty cells with nearest neighbour (from populated neighbours)
    from scipy.ndimage import generic_filter
    def _fill(a):
        # median of non-nan neighbours; returns nan if all neighbours are nan
        v = a[~np.isnan(a)]
        return np.median(v) if len(v) else np.nan
    nan_mask = np.isnan(avg_grid)
    if nan_mask.any():
        filled = generic_filter(avg_grid, _fill, size=5,
                                mode='nearest')
        avg_grid[nan_mask] = filled[nan_mask]

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
    return x_centers, t_centers, avg_grid


def _draw_regions(ax, regions, leaf_indices, leaves_only_model):
    """Overlay region boxes on an axis."""
    for i, region in enumerate(regions):
        lo = region['bounds_lower']
        hi = region['bounds_upper']
        is_leaf = (i in leaf_indices)
        if leaves_only_model and not is_leaf:
            color, lw, ls = 'grey', 1.0, '--'
        else:
            color, lw, ls = 'red', 1.5, '-'
        rect = mpatches.Rectangle(
            (lo[0], lo[1]),
            hi[0] - lo[0], hi[1] - lo[1],
            linewidth=lw, edgecolor=color,
            facecolor='none', linestyle=ls)
        ax.add_patch(rect)


# ──────────────────────────────────────────────────────────────────────────────
#  Per-run processing
# ──────────────────────────────────────────────────────────────────────────────

def process_run(label: str, ts_dir: Path):
    metrics_path = ts_dir / 'metrics.json'
    if not metrics_path.exists():
        print(f"  [{label}] No metrics.json, skipping")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    # ── Load config first (need model_type before adaptive checks) ──────────────
    cfg_path = ts_dir / 'config_used.yaml'
    if not cfg_path.exists():
        print(f"  [{label}] No config_used.yaml, skipping")
        return
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    problem    = cfg['problem']
    model_type = cfg.get('model', 'AToE')   # 'AToE' | 'AToELeaves' | 'ANT'
    eval_path  = Path('datasets') / problem / 'eval_data.pt'

    adaptive = metrics.get('adaptive_pinn')
    if adaptive is None:
        print(f"  [{label}] No adaptive_pinn section in metrics, skipping")
        return

    regions           = adaptive['regions']
    leaf_indices      = set(adaptive.get('leaf_expert_indices', []))
    leaves_only_model = (model_type == 'AToELeaves')
    if not eval_path.exists():
        print(f"  [{label}] No eval data at {eval_path}, skipping")
        return

    import torch
    eval_data = torch.load(eval_path, map_location='cpu', weights_only=False)
    x_np = eval_data['x'].numpy()   # (N, spatial_dim)
    t_np = eval_data['t'].numpy()   # (N, 1)

    spatial_dim = x_np.shape[1]
    if spatial_dim != 1:
        print(f"  [{label}] Only 1D spatial problems supported for now, skipping")
        return

    x_flat = x_np[:, 0]   # (N,)
    t_flat = t_np[:, 0]   # (N,)
    points  = np.column_stack([x_flat, t_flat])   # (N, 2)

    # ── Compute capacity per point ────────────────────────────────────────────
    capacity = _compute_capacity(points, metrics, model_type)   # (N,)

    # ── Build grid ────────────────────────────────────────────────────────────
    x_grid, t_grid, cap_grid = _make_grid(x_flat, t_flat, capacity)
    # cap_grid shape: (n_x, n_t)

    # ── Find expert_regions_final.png ────────────────────────────────────────
    regions_img_path = ts_dir / 'adaptive_plots' / 'expert_regions_final.png'

    # ── Build figure ─────────────────────────────────────────────────────────
    has_regions_img = regions_img_path.exists()
    n_cols = 2 if has_regions_img else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    model_type_label = cfg.get('model', 'AToE')
    n_exp  = len(regions)
    n_leaf = len(leaf_indices)
    title  = (
        f"{label}  |  {model_type_label}  |  "
        f"{n_exp} experts ({n_leaf} leaves)"
    )
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # Left panel: expert regions image (if present)
    if has_regions_img:
        img = plt.imread(str(regions_img_path))
        axes[0].imshow(img)
        axes[0].set_axis_off()
        axes[0].set_title('Expert Regions (final)', fontsize=11)

    # Right panel: capacity heatmap
    ax = axes[-1]
    vmin = capacity.min()
    vmax = capacity.max()
    im = ax.pcolormesh(
        x_grid, t_grid, cap_grid.T,
        shading='auto', cmap='YlOrRd',
        vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Parameters in region')

    _draw_regions(ax, regions, leaf_indices, leaves_only_model)

    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('t', fontsize=11)
    ax.set_title('Capacity Heatmap (params per point)', fontsize=11)

    # Legend
    solid = mpatches.Patch(edgecolor='red', facecolor='none',
                           linestyle='-', label='Leaf expert region')
    handles = [solid]
    if leaves_only_model:
        grey = mpatches.Patch(edgecolor='grey', facecolor='none',
                              linestyle='--', label='Non-leaf (frozen)')
        handles.append(grey)
    ax.legend(handles=handles, fontsize=8, loc='upper right')

    plt.tight_layout()

    out_path = ts_dir / 'capacity_map.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [{label}] Saved {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(batch_dir: str):
    batch_path = Path(batch_dir)
    if not batch_path.exists():
        print(f"Error: {batch_path} not found")
        return

    print("\n" + "=" * 70)
    print("Capacity Map Generator")
    print(f"Batch: {batch_path.name}")
    print("=" * 70)

    runs = _find_run_dirs(batch_path)
    if not runs:
        # Maybe the path IS a run dir (has metrics.json directly)
        if (batch_path / 'metrics.json').exists():
            label = batch_path.name
            process_run(label, batch_path)
        else:
            print("No runs found")
        return

    print(f"Found {len(runs)} run(s)\n")
    for label, ts_dir in runs:
        process_run(label, ts_dir)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python make_capacity_map.py <batch_or_run_dir>")
