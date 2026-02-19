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


def _plot_spawn_epoch_page(page_epochs, epoch_regions, train_loss_at_epoch,
                           eval_loss_at_epoch, all_bins, all_colors,
                           color_offset, output_dir, output_path):
    """Render one page of spawn epoch analysis (max ~8 columns)."""
    n_cols = len(page_epochs)
    fig, axes = plt.subplots(3, n_cols, figsize=(5 * n_cols, 12), squeeze=False)
    fig.suptitle('Spawn Epoch Analysis: Norms, Loss & Expert Regions', fontsize=16, fontweight='bold')

    for col, epoch in enumerate(page_epochs):
        ep_regions = epoch_regions[epoch]
        spawned = [r for r in ep_regions if r.get('spawned', True)]
        rejected = [r for r in ep_regions if not r.get('spawned', True)]
        s_norms = [r['wavelet_norm'] for r in spawned]
        r_norms = [r['wavelet_norm'] for r in rejected]
        color = all_colors[color_offset + col]

        # ===== Row 1: norm distribution =====
        ax_hist = axes[0, col]
        if s_norms:
            ax_hist.hist(s_norms, bins=all_bins, alpha=0.8, color=color,
                         edgecolor='black', linewidth=0.8, label=f'Spawned ({len(s_norms)})')
        if r_norms:
            ax_hist.hist(r_norms, bins=all_bins, alpha=0.5, color=color,
                         edgecolor='black', linewidth=0.8, hatch='///',
                         label=f'Rejected ({len(r_norms)})')
        ax_hist.set_title(f'Epoch {epoch}', fontsize=12, fontweight='bold')
        ax_hist.set_xlabel('Wavelet Norm', fontsize=10)
        ax_hist.set_ylabel('Count', fontsize=10)
        ax_hist.legend(fontsize=8)
        ax_hist.grid(True, alpha=0.3, axis='y')

        combined = s_norms + r_norms
        if combined and len(combined) <= 10:
            norms_text = '\n'.join([f'{n:.4f}' for n in sorted(combined, reverse=True)])
            ax_hist.text(0.98, 0.98, norms_text, transform=ax_hist.transAxes,
                         fontsize=7, verticalalignment='top', horizontalalignment='right',
                         fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # ===== Row 2: loss context =====
        ax_loss = axes[1, col]
        train_ep_list = sorted(train_loss_at_epoch.keys())
        if train_ep_list:
            train_x = np.array(train_ep_list)
            train_y = np.array([train_loss_at_epoch[e] for e in train_ep_list])
            ax_loss.plot(train_x, train_y, 'b-', alpha=0.4, linewidth=0.5, label='Train loss')

        eval_ep_list = sorted(eval_loss_at_epoch.keys())
        if eval_ep_list:
            eval_x = np.array(eval_ep_list)
            eval_y = np.array([eval_loss_at_epoch[e] for e in eval_ep_list])
            ax_loss.plot(eval_x, eval_y, 'r-o', markersize=3, linewidth=1.2, label='Eval loss')

        ax_loss.axvline(epoch, color='green', linestyle='--', linewidth=2, label=f'Spawn @ {epoch}')

        if epoch in train_loss_at_epoch:
            loss_val = train_loss_at_epoch[epoch]
            ax_loss.plot(epoch, loss_val, 'g*', markersize=12, zorder=5)
            ax_loss.annotate(f'{loss_val:.4f}', (epoch, loss_val),
                           textcoords="offset points", xytext=(5, 10),
                           fontsize=8, fontweight='bold', color='green')

        ax_loss.set_xlabel('Epoch', fontsize=10)
        ax_loss.set_ylabel('Loss', fontsize=10)
        ax_loss.set_yscale('log')
        ax_loss.legend(fontsize=7, loc='upper right')
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_title(f'Loss at Epoch {epoch}', fontsize=11)

        # ===== Row 3: expert regions image =====
        ax_img = axes[2, col]
        img_path = output_dir / f"expert_regions_epoch_{epoch}.png"
        if img_path.exists():
            img = plt.imread(str(img_path))
            # Downsample large images to avoid memory issues
            max_dim = 800
            h, w = img.shape[:2]
            if h > max_dim or w > max_dim:
                step = max(h // max_dim, w // max_dim, 1)
                img = img[::step, ::step]
            ax_img.imshow(img)
            ax_img.set_title(f'Regions at Epoch {epoch}', fontsize=11)
        else:
            ax_img.text(0.5, 0.5, f'No image\nepoch {epoch}',
                       transform=ax_img.transAxes, ha='center', va='center',
                       fontsize=12, color='gray')
            ax_img.set_title(f'Regions at Epoch {epoch} (missing)', fontsize=11)
        ax_img.set_axis_off()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_spawn_epoch_analysis(regions, metrics_path, output_dir, max_cols_per_page=8):
    """Plot norm distributions, loss, and expert region images at each spawn epoch.

    Paginates into multiple files if there are more than max_cols_per_page spawn epochs.
    """
    # Group ALL regions (spawned + rejected) by spawn_epoch
    epoch_regions = {}
    for r in regions:
        ep = r['spawn_epoch']
        epoch_regions.setdefault(ep, []).append(r)

    sorted_epochs = sorted(epoch_regions.keys())
    n_epochs = len(sorted_epochs)
    if n_epochs == 0:
        return

    # Load loss data from metrics.json
    train_loss_at_epoch = {}
    eval_loss_at_epoch = {}
    if metrics_path and metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        train_epochs = metrics.get('train_loss_epochs', [])
        train_losses = metrics.get('train_loss', [])
        for ep, loss in zip(train_epochs, train_losses):
            train_loss_at_epoch[ep] = loss
        eval_epochs = metrics.get('epochs', [])
        eval_losses = metrics.get('eval_loss', [])
        for ep, loss in zip(eval_epochs, eval_losses):
            eval_loss_at_epoch[ep] = loss

    all_norms = [r['wavelet_norm'] for r in regions]
    bins = np.histogram_bin_edges(all_norms, bins=30) if all_norms else 10
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_epochs))

    # Split into pages
    pages = [sorted_epochs[i:i + max_cols_per_page]
             for i in range(0, n_epochs, max_cols_per_page)]

    output_paths = []
    for page_idx, page_epochs in enumerate(pages):
        if len(pages) == 1:
            out_path = output_dir / "spawn_epoch_analysis.png"
        else:
            out_path = output_dir / f"spawn_epoch_analysis_p{page_idx + 1}.png"

        color_offset = page_idx * max_cols_per_page
        _plot_spawn_epoch_page(page_epochs, epoch_regions,
                               train_loss_at_epoch, eval_loss_at_epoch,
                               bins, colors, color_offset, output_dir, out_path)
        output_paths.append(out_path)
        print(f"    Spawn epoch analysis saved: {out_path.name}")

    return output_paths[0] if output_paths else None


def plot_expert_norms_for_model(json_path, output_dir, metrics_path=None):
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

    # ===== Spawn epoch analysis (norms + loss per spawn step) =====
    plot_spawn_epoch_analysis(regions, metrics_path, output_dir)

    return output_path


def _find_run_dirs(batch_path):
    """Find all (label, timestamp_dir) pairs in a batch directory.

    Handles two layouts:
      1. Multiple architectures: batch/arch_a/timestamp/, batch/arch_b/timestamp/
         → picks latest timestamp per architecture, label = arch name
      2. Single architecture with many runs: batch/arch/ts1/, batch/arch/ts2/
         → expands each timestamp, label = timestamp name
    """
    model_dirs = sorted(
        d for d in batch_path.iterdir()
        if d.is_dir() and d.suffix not in ('.png', '.csv', '.yaml')
    )
    if not model_dirs:
        return []

    runs = []
    for model_dir in model_dirs:
        ts_dirs = sorted(
            d for d in model_dir.iterdir()
            if d.is_dir() and d.name != 'checkpoints'
        )
        if not ts_dirs:
            continue

        # Single architecture with multiple runs → expand all
        if len(model_dirs) == 1 and len(ts_dirs) > 1:
            for ts_dir in ts_dirs:
                runs.append((ts_dir.name, ts_dir))
        else:
            # Multiple architectures → latest timestamp per architecture
            runs.append((model_dir.name, ts_dirs[-1]))

    return runs


def process_batch(batch_dir):
    """Process all models in a batch directory."""
    batch_path = Path(batch_dir)

    if not batch_path.exists():
        print(f"Error: Directory not found: {batch_path}")
        return

    print(f"\n{'='*70}")
    print(f"Processing batch: {batch_path.name}")
    print(f"{'='*70}")

    runs = _find_run_dirs(batch_path)
    if not runs:
        print(f"  No model runs found in {batch_path}")
        return

    print(f"  Found {len(runs)} run(s)")

    for label, ts_dir in runs:
        print(f"\n  Processing: {label}")

        json_path = ts_dir / "adaptive_plots" / "expert_regions.json"
        if not json_path.exists():
            print(f"    No expert_regions.json found, skipping")
            continue

        metrics_path = ts_dir / "metrics.json"
        if not metrics_path.exists():
            metrics_path = None

        output_dir = json_path.parent
        output_path = plot_expert_norms_for_model(json_path, output_dir, metrics_path)
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
