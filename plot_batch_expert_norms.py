"""Visualize expert region norm distributions for all models in a batch."""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
import numpy as np
from pathlib import Path
import sys
import torch
from scipy.interpolate import griddata


def load_ground_truth(output_dir):
    """Try to load ground truth data for background visualization."""
    try:
        # Try to find eval_data.pt in the standard location
        problem = 'schrodinger'  # Default, could be read from config
        eval_data_path = Path("datasets") / problem / "eval_data.pt"

        if eval_data_path.exists():
            eval_data = torch.load(eval_data_path, map_location='cpu')

            # Extract spatial and temporal coordinates
            x_data = eval_data['x'].numpy()
            t_data = eval_data['t'].numpy()

            # Get ground truth (try different keys)
            if 'h_gt' in eval_data:
                h_data = eval_data['h_gt'].numpy()
            elif 'h' in eval_data:
                h_data = eval_data['h'].numpy()
            elif 'u_gt' in eval_data:
                h_data = eval_data['u_gt'].numpy()
            else:
                return None, None, None, None

            # For complex-valued, use magnitude
            if h_data.shape[1] == 2:  # [u, v] components
                h_magnitude = np.sqrt(h_data[:, 0]**2 + h_data[:, 1]**2)
            else:
                h_magnitude = h_data[:, 0]

            return x_data, t_data, h_magnitude, eval_data

    except Exception as e:
        print(f"    Could not load ground truth: {e}")

    return None, None, None, None


def _render_ground_truth_background(ax, x_data, t_data, h_magnitude):
    """Render ground truth as background on a spatial axis."""
    if x_data is None or t_data is None or h_magnitude is None:
        return

    x_min, x_max = x_data.min(), x_data.max()
    t_min, t_max = t_data.min(), t_data.max()

    resolution = 100
    grid_x = np.linspace(x_min, x_max, resolution)
    grid_t = np.linspace(t_min, t_max, resolution)
    X_grid, T_grid = np.meshgrid(grid_x, grid_t, indexing='ij')

    points = np.column_stack([x_data, t_data])
    H_grid = griddata(points, h_magnitude, (X_grid, T_grid), method='cubic')

    im = ax.pcolormesh(grid_x, grid_t, H_grid.T,
                       shading='auto', cmap='gray',
                       alpha=0.6, zorder=0)
    plt.colorbar(im, ax=ax, label='|h|')


def _plot_regions_on_axis(ax, regions, title_label):
    """Plot region rectangles on a spatial axis with norm-based coloring."""
    if not regions:
        ax.set_title(f'{title_label} (n=0)', fontsize=12, fontweight='bold')
        ax.text(0.5, 0.5, 'No regions', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='gray')
        ax.set_xlabel('Space (x)', fontsize=11)
        ax.set_ylabel('Time (t)', fontsize=11)
        return

    region_norms = [r['wavelet_norm'] for r in regions]
    norm_min = min(region_norms)
    norm_max = max(region_norms)
    norm_range = norm_max - norm_min if norm_max > norm_min else 1.0

    for region in regions:
        x_lower, t_lower = region['bounds_lower']
        x_upper, t_upper = region['bounds_upper']
        width = x_upper - x_lower
        height = t_upper - t_lower

        norm_normalized = (region['wavelet_norm'] - norm_min) / norm_range if norm_range > 0 else 0.5
        edge_color = plt.cm.RdYlGn(norm_normalized)
        face_color = list(edge_color[:3]) + [0.3]

        rect = patches.Rectangle((x_lower, t_lower), width, height,
                                 linewidth=3.0, edgecolor=edge_color,
                                 facecolor=face_color, linestyle='-', zorder=10)
        ax.add_patch(rect)

    ax.set_xlabel('Space (x)', fontsize=11)
    ax.set_ylabel('Time (t)', fontsize=11)
    ax.set_title(f'{title_label} (n={len(regions)})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, zorder=0)

    sm = plt.cm.ScalarMappable(cmap='RdYlGn',
                               norm=plt.Normalize(vmin=norm_min, vmax=norm_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Wavelet Norm', fontsize=10)


def _plot_histogram_on_axis(ax, norms, bins, color, title_label):
    """Plot a histogram with stats on an axis."""
    if not norms:
        ax.set_title(f'{title_label} (n=0)', fontsize=12, fontweight='bold')
        ax.text(0.5, 0.5, 'No regions', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='gray')
        ax.set_xlabel('Wavelet Norm', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        return

    ax.hist(norms, bins=bins, alpha=0.8,
            edgecolor='black', linewidth=1.2, color=color)

    mean_val = np.mean(norms)
    median_val = np.median(norms)
    ax.axvline(median_val, color='red', linestyle='--', linewidth=2,
               label=f'Median: {median_val:.4f}')
    ax.axvline(mean_val, color='orange', linestyle=':', linewidth=2,
               label=f'Mean: {mean_val:.4f}')

    # Add individual norms to legend (sorted high to low)
    sorted_norms = sorted(norms, reverse=True)
    norms_text = 'Norms (high\u2192low):\n' + '\n'.join([f'  {n:.6f}' for n in sorted_norms])
    norms_patch = Patch(color='none', label=norms_text)

    ax.set_xlabel('Wavelet Norm', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'{title_label} (n={len(norms)})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    handles, labels = ax.get_legend_handles_labels()
    handles.append(norms_patch)
    labels.append(norms_text)
    ax.legend(handles, labels, loc='upper right', fontsize=9,
              framealpha=0.95, handlelength=1.5)


def plot_depth_with_spatial(depth, spawned_regions, rejected_regions,
                             spawned_norms, rejected_norms, bins,
                             color, output_dir, x_data, t_data, h_magnitude):
    """Create a 2x2 figure for a specific depth: spawned (left) vs rejected (right)."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(f'Depth {depth} Analysis', fontsize=16, fontweight='bold', y=0.98)

    # Top-left: Spawned histogram
    _plot_histogram_on_axis(axes[0, 0], spawned_norms, bins, color,
                            f'Depth {depth} - Spawned')

    # Top-right: Rejected histogram
    rejected_color = np.array(color[:3]) * 0.6  # Darker shade for rejected
    rejected_color = np.clip(rejected_color, 0, 1)
    _plot_histogram_on_axis(axes[0, 1], rejected_norms, bins, rejected_color,
                            f'Depth {depth} - Rejected')

    # Bottom-left: Spawned spatial
    _render_ground_truth_background(axes[1, 0], x_data, t_data, h_magnitude)
    _plot_regions_on_axis(axes[1, 0], spawned_regions,
                          f'Depth {depth} - Spawned Regions')

    # Bottom-right: Rejected spatial
    _render_ground_truth_background(axes[1, 1], x_data, t_data, h_magnitude)
    _plot_regions_on_axis(axes[1, 1], rejected_regions,
                          f'Depth {depth} - Rejected Regions')

    plt.tight_layout()

    output_path = output_dir / f"depth_{depth}_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    n_spawned = len(spawned_norms) if spawned_norms else 0
    n_rejected = len(rejected_norms) if rejected_norms else 0
    print(f"      Depth {depth}: {n_spawned} spawned, {n_rejected} rejected")

    return output_path


def plot_expert_norms_for_model(json_path, output_dir):
    """Generate norm distribution plots for a single model."""
    with open(json_path) as f:
        data = json.load(f)

    regions = data['regions']

    # Split into spawned and rejected
    spawned_regions = [r for r in regions if r.get('spawned', True)]
    rejected_regions = [r for r in regions if not r.get('spawned', True)]

    # Group by depth, separately for spawned/rejected
    spawned_by_depth = {}
    rejected_by_depth = {}
    epoch_norms = {}

    for r in spawned_regions:
        d = r['depth']
        spawned_by_depth.setdefault(d, []).append(r)
        epoch_norms.setdefault(r['spawn_epoch'], []).append(r['wavelet_norm'])

    for r in rejected_regions:
        d = r['depth']
        rejected_by_depth.setdefault(d, []).append(r)

    # All depths that appear in either group
    all_depths = sorted(set(list(spawned_by_depth.keys()) + list(rejected_by_depth.keys())))
    n_depths = len(all_depths) if all_depths else 1

    # Determine bin edges based on ALL data (spawned + rejected)
    all_norms = [r['wavelet_norm'] for r in regions]
    if not all_norms:
        print("    No regions found, skipping")
        return None
    bins = np.histogram_bin_edges(all_norms, bins=50)

    # Define colors for each depth
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_depths))

    # ===== Main plot: Combined bar chart with spawned vs rejected =====
    fig = plt.figure(figsize=(16, 4 + 3 * n_depths))
    gs = fig.add_gridspec(n_depths + 1, 1, height_ratios=[2] + [1] * n_depths, hspace=0.4)

    ax_main = fig.add_subplot(gs[0])

    # Plot overlaid histograms: spawned solid, rejected hatched
    for i, depth in enumerate(all_depths):
        s_norms = [r['wavelet_norm'] for r in spawned_by_depth.get(depth, [])]
        r_norms = [r['wavelet_norm'] for r in rejected_by_depth.get(depth, [])]

        if s_norms:
            ax_main.hist(s_norms, bins=bins, alpha=0.7,
                         edgecolor='black', linewidth=1.0,
                         color=colors[i],
                         label=f'D{depth} spawned (n={len(s_norms)})')
        if r_norms:
            ax_main.hist(r_norms, bins=bins, alpha=0.5,
                         edgecolor='black', linewidth=1.0,
                         color=colors[i], hatch='///',
                         label=f'D{depth} rejected (n={len(r_norms)})')

    n_spawned_total = len(spawned_regions)
    n_rejected_total = len(rejected_regions)
    ax_main.set_xlabel('Wavelet Norm', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax_main.set_title(
        f'Combined Norm Distribution by Depth\n'
        f'({n_spawned_total} spawned, {n_rejected_total} rejected)',
        fontsize=15, fontweight='bold')
    ax_main.grid(True, alpha=0.3, axis='y')
    ax_main.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=2)

    # ===== Individual depth plots (spawned only in the combined figure) =====
    for i, depth in enumerate(all_depths):
        ax = fig.add_subplot(gs[i + 1])
        s_norms = [r['wavelet_norm'] for r in spawned_by_depth.get(depth, [])]
        r_norms = [r['wavelet_norm'] for r in rejected_by_depth.get(depth, [])]

        if s_norms:
            ax.hist(s_norms, bins=bins, alpha=0.8,
                    edgecolor='black', linewidth=1.2, color=colors[i],
                    label=f'Spawned (n={len(s_norms)})')
        if r_norms:
            ax.hist(r_norms, bins=bins, alpha=0.5,
                    edgecolor='black', linewidth=1.2, color=colors[i],
                    hatch='///', label=f'Rejected (n={len(r_norms)})')

        combined = s_norms + r_norms
        if combined:
            mean_val = np.mean(combined)
            median_val = np.median(combined)
            ax.axvline(median_val, color='red', linestyle='--', linewidth=2,
                       label=f'Median: {median_val:.4f}')
            ax.axvline(mean_val, color='orange', linestyle=':', linewidth=2,
                       label=f'Mean: {mean_val:.4f}')

        ax.set_xlabel('Wavelet Norm', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Depth {depth} Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right', fontsize=9)

    # Save combined plot
    output_path = output_dir / "expert_norm_distributions.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    # ===== Generate individual 2x2 depth analysis plots =====
    print(f"    Generating individual depth analysis plots...")

    # Load ground truth for spatial visualization
    x_data, t_data, h_magnitude, _ = load_ground_truth(output_dir)

    for i, depth in enumerate(all_depths):
        s_regions = spawned_by_depth.get(depth, [])
        r_regions = rejected_by_depth.get(depth, [])
        s_norms = [r['wavelet_norm'] for r in s_regions]
        r_norms = [r['wavelet_norm'] for r in r_regions]

        depth_output = plot_depth_with_spatial(
            depth, s_regions, r_regions,
            s_norms, r_norms,
            bins, colors[i], output_dir,
            x_data, t_data, h_magnitude
        )
        print(f"      Depth {depth} plot saved: {depth_output.name}")

    # Print statistics
    print(f"\n{'='*60}")
    print("Expert Norm Statistics")
    print(f"{'='*60}")
    print(f"Total spawned: {n_spawned_total}, Total rejected: {n_rejected_total}")

    spawned_norms_all = [r['wavelet_norm'] for r in spawned_regions]
    if spawned_norms_all:
        print(f"Spawned: mean={np.mean(spawned_norms_all):.4f}, "
              f"median={np.median(spawned_norms_all):.4f}, "
              f"std={np.std(spawned_norms_all):.4f}")
        print(f"  Min: {np.min(spawned_norms_all):.4f}, Max: {np.max(spawned_norms_all):.4f}")

    rejected_norms_all = [r['wavelet_norm'] for r in rejected_regions]
    if rejected_norms_all:
        print(f"Rejected: mean={np.mean(rejected_norms_all):.4f}, "
              f"median={np.median(rejected_norms_all):.4f}, "
              f"std={np.std(rejected_norms_all):.4f}")
        print(f"  Min: {np.min(rejected_norms_all):.4f}, Max: {np.max(rejected_norms_all):.4f}")

    print(f"\nBy Depth:")
    for depth in all_depths:
        s_n = [r['wavelet_norm'] for r in spawned_by_depth.get(depth, [])]
        r_n = [r['wavelet_norm'] for r in rejected_by_depth.get(depth, [])]
        s_str = f"mean={np.mean(s_n):.4f}" if s_n else "none"
        r_str = f"mean={np.mean(r_n):.4f}" if r_n else "none"
        print(f"  Depth {depth}: spawned={len(s_n)} ({s_str}), "
              f"rejected={len(r_n)} ({r_str})")

    sorted_epochs = sorted(epoch_norms.keys())
    if sorted_epochs:
        print(f"\nBy Spawn Epoch (spawned only):")
        for epoch in sorted_epochs:
            norms_e = epoch_norms[epoch]
            print(f"  Epoch {epoch}: n={len(norms_e)}, mean={np.mean(norms_e):.4f}, "
                  f"median={np.median(norms_e):.4f}")

    return output_path


def process_batch(batch_dir):
    """Process all models in a batch directory."""
    batch_path = Path(batch_dir)

    if not batch_path.exists():
        print(f"Error: Directory not found: {batch_path}")
        return

    print(f"\n{'='*70}")
    print(f"Processing batch: {batch_path.name}")
    print(f"{'='*70}")

    # Find all model directories
    model_dirs = [d for d in batch_path.iterdir()
                  if d.is_dir() and not d.name.endswith('.png')
                  and not d.name.endswith('.csv')
                  and not d.name.endswith('.yaml')]

    if not model_dirs:
        print(f"  No model directories found in {batch_path}")
        return

    print(f"  Found {len(model_dirs)} models")

    # Process each model
    for model_dir in sorted(model_dirs):
        print(f"\n  Processing: {model_dir.name}")

        # Find timestamp subdirectory
        timestamp_dirs = [d for d in model_dir.iterdir()
                          if d.is_dir() and d.name != 'checkpoints']

        if not timestamp_dirs:
            print(f"    No timestamp directory found, skipping")
            continue

        # Use most recent timestamp
        latest_timestamp = sorted(timestamp_dirs)[-1]

        # Look for expert_regions.json
        json_path = latest_timestamp / "adaptive_plots" / "expert_regions.json"

        if not json_path.exists():
            print(f"    No expert_regions.json found, skipping")
            continue

        # Generate plots
        output_dir = json_path.parent
        output_path = plot_expert_norms_for_model(json_path, output_dir)
        print(f"    Plot saved to: {output_path}")

    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Get batch directory from command line or use default
    if len(sys.argv) > 1:
        batch_dir = sys.argv[1]
    else:
        batch_dir = "outputs/experiments/AToE-New/schrodinger_tests_20260210_055733-non-pretrained-10k-epochs"

    process_batch(batch_dir)
