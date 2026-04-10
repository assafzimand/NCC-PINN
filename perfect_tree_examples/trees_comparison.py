"""Compare two perfect_trees.json files side by side.

Usage:
    python trees_comparison.py PATH1 PATH2 [--top-n N] [--label1 STR] [--label2 STR]
    python trees_comparison.py PATH1 PATH2 [--problems p1 p2 ...]

For each problem present in both JSONs, produces two files inside
perfect_tree_examples/comparison/:

  1. {problem}_comparison.png  —  2-row figure:
       Row 1 (2 panels): accepted regions of T1  |  accepted regions of T2
                         (GT background when eval_data.pt is available)
       Row 2 (full width): comparison table — ALL nodes, sorted by norms_ratio.
         Columns: Node | N | Depth | In T1 | In T2 | Norm T1 | Norm T2 | Ratio (T2/T1)
         Row colour: green = gained in T2, red = lost in T2, yellow = in both.

  2. {problem}_ratio_map.png  —  top-N regions by |ratio − 1|,
       edges only, coloured by ratio on a diverging colormap
       (red = T2 norm smaller, green = T2 norm larger, yellow = ~equal).
       Colorbar on the right with ×-multiplier tick labels.
"""

import argparse
import json
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from adaptive.visualization import prepare_ground_truth_grid
    _HAS_VIZ = True
except ImportError:
    _HAS_VIZ = False

TOP_N_DEFAULT = 20
TABLE_MAX_ROWS = 60   # cap so the table stays readable


# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def try_load_gt(problem: str, script_dir: Path, domain_bounds: dict):
    """Try to load eval_data.pt and build GT grid. Returns (gt_grid, grid_x, grid_t) or (None, None, None)."""
    if not _HAS_VIZ:
        return None, None, None
    eval_path = script_dir.parent / 'datasets' / problem / 'eval_data.pt'
    if not eval_path.exists():
        return None, None, None
    try:
        eval_data = torch.load(eval_path, map_location='cpu')
        gt_grid, grid_x, grid_t = prepare_ground_truth_grid(
            eval_data, domain_bounds, resolution=150)
        return gt_grid, grid_x, grid_t
    except Exception:
        return None, None, None


# ─────────────────────────────────────────────────────────────────────
# Node merging
# ─────────────────────────────────────────────────────────────────────

def merge_nodes(data1: dict, data2: dict) -> list:
    """Merge all_nodes from both trees by node_id.

    Returns a list of dicts, one per unique node_id, with fields:
        node_id, n_samples, tree_depth, bounds_lower, bounds_upper,
        in_t1 (bool), in_t2 (bool),
        norm_t1 (float|None), norm_t2 (float|None),
        ratio (float|None)  — norm_t2 / norm_t1
    """
    nodes1 = {n['node_id']: n for n in data1.get('all_nodes', [])}
    nodes2 = {n['node_id']: n for n in data2.get('all_nodes', [])}

    all_ids = sorted(set(nodes1.keys()) | set(nodes2.keys()))

    merged = []
    for nid in all_ids:
        n1 = nodes1.get(nid)
        n2 = nodes2.get(nid)
        ref = n1 if n1 is not None else n2   # source for spatial metadata

        norm1 = float(n1['wavelet_norm']) if n1 is not None else None
        norm2 = float(n2['wavelet_norm']) if n2 is not None else None
        in1   = bool(n1['accepted'])      if n1 is not None else False
        in2   = bool(n2['accepted'])      if n2 is not None else False

        if norm1 is not None and norm2 is not None:
            if norm1 > 0:
                ratio = norm2 / norm1
            elif norm2 == 0:
                ratio = 1.0        # both zero → identical
            else:
                ratio = None       # norm1==0 but norm2>0 → undefined
        else:
            ratio = None

        merged.append({
            'node_id':      nid,
            'n_samples':    ref.get('n_samples', 0),
            'tree_depth':   ref.get('tree_depth', -1),
            'bounds_lower': ref.get('bounds_lower', []),
            'bounds_upper': ref.get('bounds_upper', []),
            'in_t1':        in1,
            'in_t2':        in2,
            'norm_t1':      norm1,
            'norm_t2':      norm2,
            'ratio':        ratio,
        })

    return merged


# ─────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────

def _plot_regions(ax, accepted_nodes: list, domain_bounds: dict,
                  gt_grid, grid_x, grid_t, title: str):
    """GT background + black edges for accepted region boxes."""
    x_min = domain_bounds['lower'][0];   x_max = domain_bounds['upper'][0]
    t_min = domain_bounds['lower'][-1];  t_max = domain_bounds['upper'][-1]

    if gt_grid is not None:
        display = np.linalg.norm(gt_grid, axis=2) if gt_grid.ndim == 3 else gt_grid
        T, X_mg = np.meshgrid(grid_t, grid_x)
        im = ax.pcolormesh(X_mg, T, display, shading='auto',
                           cmap='viridis', alpha=0.7, zorder=0)
        plt.colorbar(im, ax=ax, shrink=0.7)

    for nd in accepted_nodes:
        bl, bu = nd['bounds_lower'], nd['bounds_upper']
        rect = patches.Rectangle(
            (bl[0], bl[-1]), bu[0] - bl[0], bu[-1] - bl[-1],
            linewidth=1.0, edgecolor='black', facecolor='none', zorder=10,
        )
        ax.add_patch(rect)

    pad = 0.03
    ax.set_xlim(x_min - pad * (x_max - x_min), x_max + pad * (x_max - x_min))
    ax.set_ylim(t_min - pad * (t_max - t_min), t_max + pad * (t_max - t_min))
    ax.set_xlabel('x');  ax.set_ylabel('t')
    ax.set_title(title, fontsize=10)
    ax.set_aspect('auto')


def _plot_comparison_table(ax, merged_nodes: list, label1: str, label2: str,
                            max_rows: int = TABLE_MAX_ROWS):
    """Table of all nodes sorted by norms_ratio (ascending)."""
    ax.axis('off')

    # Exclude root, keep nodes with a defined ratio
    rows = [n for n in merged_nodes
            if n['node_id'] != 0 and n['ratio'] is not None]

    # Sort by ratio ascending (ratio < 1 first, then > 1)
    rows.sort(key=lambda r: r['ratio'])

    # If there are more rows than max_rows, keep the most extreme:
    # bottom half (low ratio) and top half (high ratio)
    if len(rows) > max_rows:
        half = max_rows // 2
        rows = rows[:half] + rows[-(max_rows - half):]

    if not rows:
        ax.set_title('No nodes available for comparison')
        return

    col_labels = [
        'Node', 'N', 'Depth',
        f'In {label1}', f'In {label2}',
        f'Norm {label1}', f'Norm {label2}',
        'Ratio\n(T2/T1)',
    ]
    cell_text = []
    for r in rows:
        cell_text.append([
            str(r['node_id']),
            str(r['n_samples']),
            str(r['tree_depth']),
            'V' if r['in_t1'] else 'X',
            'V' if r['in_t2'] else 'X',
            f"{r['norm_t1']:.3f}" if r['norm_t1'] is not None else '—',
            f"{r['norm_t2']:.3f}" if r['norm_t2'] is not None else '—',
            f"{r['ratio']:.3f}",
        ])

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.12)

    # Row colour by acceptance change
    for i, r in enumerate(rows):
        if r['in_t1'] and not r['in_t2']:
            colour = '#ffcccc'   # lost in T2
        elif not r['in_t1'] and r['in_t2']:
            colour = '#ccffcc'   # gained in T2
        elif r['in_t1'] and r['in_t2']:
            colour = '#ffffcc'   # in both
        else:
            colour = 'white'     # in neither
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(colour)

    shown = len(rows)
    total = sum(1 for n in merged_nodes if n['node_id'] != 0 and n['ratio'] is not None)
    ax.set_title(
        f'Node comparison — sorted by Ratio (T2/T1) ascending'
        + (f'  [showing {shown} most extreme of {total} total]'
           if total > max_rows else f'  [{shown} nodes]')
        + f'\n  green=gained in {label2},  red=lost in {label2},  yellow=in both',
        fontsize=9, pad=10,
    )


def _plot_ratio_map(ax, merged_nodes: list, domain_bounds: dict,
                    gt_grid, grid_x, grid_t,
                    top_n: int, label1: str, label2: str):
    """Edges-only plot of the top-N regions by |ratio − 1|, coloured by ratio."""
    x_min = domain_bounds['lower'][0];   x_max = domain_bounds['upper'][0]
    t_min = domain_bounds['lower'][-1];  t_max = domain_bounds['upper'][-1]

    if gt_grid is not None:
        display = np.linalg.norm(gt_grid, axis=2) if gt_grid.ndim == 3 else gt_grid
        T, X_mg = np.meshgrid(grid_t, grid_x)
        ax.pcolormesh(X_mg, T, display, shading='auto',
                      cmap='Greys', alpha=0.5, zorder=0)

    # Filter: exclude root, need positive ratio
    valid = [n for n in merged_nodes
             if n['node_id'] != 0 and n['ratio'] is not None and n['ratio'] > 0]
    if not valid:
        ax.set_title('No valid nodes for ratio map')
        return

    # Sort by |log(ratio)| descending — most extreme first
    log_ratios = np.array([np.log(n['ratio']) for n in valid])
    order = np.argsort(np.abs(log_ratios))[::-1]
    top_nodes  = [valid[i]      for i in order[:top_n]]
    top_log_r  = np.array([log_ratios[i] for i in order[:top_n]])

    # Diverging colormap centred at log_ratio = 0 (ratio = 1)
    max_abs = max(float(np.max(np.abs(top_log_r))), 0.3)
    norm_obj = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    cmap = plt.cm.RdYlGn   # red < 1 < green, yellow at centre

    for n, log_r in zip(top_nodes, top_log_r):
        bl, bu = n['bounds_lower'], n['bounds_upper']
        edge_rgba = cmap(norm_obj(float(np.clip(log_r, -max_abs, max_abs))))
        rect = patches.Rectangle(
            (bl[0], bl[-1]), bu[0] - bl[0], bu[-1] - bl[-1],
            linewidth=2.5, edgecolor=edge_rgba, facecolor='none', zorder=10,
        )
        ax.add_patch(rect)

    # Colorbar — ticks as ×-multipliers
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_obj)
    sm.set_array([])
    fig = ax.get_figure()
    cbar = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.02,
                        label=f'Norm ratio  {label2} / {label1}  (log scale)')
    ticks_lr = np.linspace(-max_abs, max_abs, 7)
    cbar.set_ticks(ticks_lr)
    cbar.set_ticklabels([f'{np.exp(v):.2f}×' for v in ticks_lr])

    pad = 0.03
    ax.set_xlim(x_min - pad * (x_max - x_min), x_max + pad * (x_max - x_min))
    ax.set_ylim(t_min - pad * (t_max - t_min), t_max + pad * (t_max - t_min))
    ax.set_xlabel('x');  ax.set_ylabel('t')
    ax.set_title(
        f'Top {len(top_nodes)} regions by |ratio − 1|  '
        f'(edges only — red: {label2} norm smaller,  '
        f'green: {label2} norm larger,  yellow ≈ 1)',
        fontsize=10,
    )
    ax.set_aspect('auto')


# ─────────────────────────────────────────────────────────────────────
# Per-problem entry point
# ─────────────────────────────────────────────────────────────────────

def process_problem(problem: str, data1: dict, data2: dict,
                    label1: str, label2: str,
                    output_dir: Path, top_n: int, script_dir: Path):
    print(f"\n  Problem: {problem}")

    domain_bounds = data1.get('domain_bounds') or data2.get('domain_bounds')
    if domain_bounds is None:
        print("    No domain_bounds, skipping.")
        return
    if len(domain_bounds['lower']) != 2:
        print("    Skipping: only 2D (x, t) domains supported.")
        return

    gt_grid, grid_x, grid_t = try_load_gt(problem, script_dir, domain_bounds)

    merged = merge_nodes(data1, data2)

    acc1 = [n for n in merged if n['in_t1'] and n['node_id'] != 0]
    acc2 = [n for n in merged if n['in_t2'] and n['node_id'] != 0]
    n1_acc = data1.get('summary', {}).get('accepted_nodes', len(acc1))
    n2_acc = data2.get('summary', {}).get('accepted_nodes', len(acc2))

    # ── Main comparison figure ────────────────────────────────────
    fig = plt.figure(figsize=(22, 22))
    gs  = GridSpec(2, 2, figure=fig,
                   height_ratios=[1, 1.5], hspace=0.45, wspace=0.3)

    ax_t1  = fig.add_subplot(gs[0, 0])
    ax_t2  = fig.add_subplot(gs[0, 1])
    ax_tbl = fig.add_subplot(gs[1, :])

    _plot_regions(ax_t1, acc1, domain_bounds, gt_grid, grid_x, grid_t,
                  f'{label1}  —  {n1_acc} accepted regions')
    _plot_regions(ax_t2, acc2, domain_bounds, gt_grid, grid_x, grid_t,
                  f'{label2}  —  {n2_acc} accepted regions')
    _plot_comparison_table(ax_tbl, merged, label1, label2)

    fig.suptitle(
        f'Tree Comparison  —  {problem}\n{label1}  vs  {label2}',
        fontsize=13, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    main_path = output_dir / f'{problem}_comparison.png'
    plt.savefig(main_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {main_path}")

    # ── Ratio map figure ─────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(13, 8))
    _plot_ratio_map(ax2, merged, domain_bounds, gt_grid, grid_x, grid_t,
                    top_n=top_n, label1=label1, label2=label2)
    fig2.suptitle(
        f'Norm Ratio Map  —  {problem}  '
        f'(top {top_n} by |ratio − 1|)',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    ratio_path = output_dir / f'{problem}_ratio_map.png'
    plt.savefig(ratio_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {ratio_path}")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compare two perfect_trees JSON files.')
    parser.add_argument('path1', type=Path,
                        help='First JSON file (tree 1)')
    parser.add_argument('path2', type=Path,
                        help='Second JSON file (tree 2)')
    parser.add_argument('--top-n', type=int, default=TOP_N_DEFAULT,
                        help=f'Top N regions shown in ratio map (default {TOP_N_DEFAULT})')
    parser.add_argument('--label1', type=str, default=None,
                        help='Label for tree 1 (default: parent folder name)')
    parser.add_argument('--label2', type=str, default=None,
                        help='Label for tree 2 (default: parent folder name)')
    parser.add_argument('--problems', nargs='+', default=None,
                        help='Limit to specific problems (default: all common problems)')
    args = parser.parse_args()

    path1 = args.path1.resolve()
    path2 = args.path2.resolve()
    label1 = args.label1 or path1.parent.name
    label2 = args.label2 or path2.parent.name

    print(f"Tree 1: {path1}  (label: {label1})")
    print(f"Tree 2: {path2}  (label: {label2})")

    trees1 = load_json(path1)
    trees2 = load_json(path2)

    common = sorted(set(trees1.keys()) & set(trees2.keys()))
    problems = args.problems or common
    if not problems:
        print("No common problems found between the two JSON files.")
        return
    missing = [p for p in problems if p not in common]
    if missing:
        print(f"Warning: problems not in both files: {missing}")
        problems = [p for p in problems if p in common]

    print(f"Problems: {problems}")

    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / 'comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    for problem in problems:
        try:
            process_problem(
                problem,
                trees1[problem], trees2[problem],
                label1, label2,
                output_dir, args.top_n, script_dir,
            )
        except Exception as e:
            print(f"  ERROR on {problem}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone! Results in {output_dir}")


if __name__ == '__main__':
    main()