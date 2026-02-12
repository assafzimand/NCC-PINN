"""Visualize expert region norm distributions for all models in a batch."""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
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


def plot_depth_with_spatial(depth, regions_at_depth, norms_at_depth, bins,
                             color, output_dir, x_data, t_data, h_magnitude):
    """Create a 2-panel figure for a specific depth with histogram and spatial plot."""
    fig, (ax_hist, ax_spatial) = plt.subplots(2, 1, figsize=(12, 10))

    # ===== Top: Histogram =====
    ax_hist.hist(norms_at_depth, bins=bins, alpha=0.8,
                 edgecolor='black', linewidth=1.2, color=color)

    mean_val = np.mean(norms_at_depth)
    median_val = np.median(norms_at_depth)
    ax_hist.axvline(median_val, color='red', linestyle='--', linewidth=2,
                    label=f'Median: {median_val:.4f}')
    ax_hist.axvline(mean_val, color='orange', linestyle=':', linewidth=2,
                    label=f'Mean: {mean_val:.4f}')

    # Add individual norms to legend (sorted high to low)
    sorted_norms = sorted(norms_at_depth, reverse=True)
    norms_text = 'Norms (high→low):\n' + '\n'.join([f'  {n:.6f}' for n in sorted_norms])

    # Create invisible handle for norms list
    from matplotlib.patches import Patch
    norms_patch = Patch(color='none', label=norms_text)

    ax_hist.set_xlabel('Wavelet Norm', fontsize=12, fontweight='bold')
    ax_hist.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax_hist.set_title(f'Depth {depth} - Norm Distribution (n={len(norms_at_depth)})',
                      fontsize=14, fontweight='bold')
    ax_hist.grid(True, alpha=0.3, axis='y')

    # Get existing handles and labels, then add norms
    handles, labels = ax_hist.get_legend_handles_labels()
    handles.append(norms_patch)
    labels.append(norms_text)
    ax_hist.legend(handles, labels, loc='upper right', fontsize=9,
                   framealpha=0.95, handlelength=1.5)

    # ===== Bottom: Spatial plot with expert regions =====
    # Debug: Verify region filtering
    print(f"      Depth {depth}: Filtering {len(regions_at_depth)} regions")
    regions_this_depth_only = [r for r in regions_at_depth if r['depth'] == depth]
    print(f"      After depth filter: {len(regions_this_depth_only)} regions with depth={depth}")

    if x_data is not None and t_data is not None and h_magnitude is not None:
        print(f"      Ground truth data: {len(x_data)} points")

        # Create smooth interpolated grid for background
        x_min, x_max = x_data.min(), x_data.max()
        t_min, t_max = t_data.min(), t_data.max()
        print(f"      Domain: x=[{x_min:.2f}, {x_max:.2f}], t=[{t_min:.2f}, {t_max:.2f}]")

        # Create regular grid for interpolation (like expert_regions_final)
        resolution = 100
        grid_x = np.linspace(x_min, x_max, resolution)
        grid_t = np.linspace(t_min, t_max, resolution)
        X_grid, T_grid = np.meshgrid(grid_x, grid_t, indexing='ij')

        # Interpolate scattered data onto regular grid
        points = np.column_stack([x_data, t_data])
        H_grid = griddata(points, h_magnitude, (X_grid, T_grid), method='cubic')

        # Use pcolormesh for smooth rendering (X=horizontal, T=vertical)
        im = ax_spatial.pcolormesh(grid_x, grid_t, H_grid.T,
                                    shading='auto', cmap='gray',
                                    alpha=0.6, zorder=0)
        plt.colorbar(im, ax=ax_spatial, label='|h|')
        print(f"      Successfully rendered smooth background with pcolormesh")
    else:
        print(f"      Warning: No ground truth data available")

    # Create colormap for norms: red (low) to green (high)
    # Use actual norms from regions to ensure correct scaling
    region_norms = [r['wavelet_norm'] for r in regions_this_depth_only]
    if len(region_norms) > 0:
        norm_min = min(region_norms)
        norm_max = max(region_norms)
        norm_range = norm_max - norm_min if norm_max > norm_min else 1.0
        print(f"      Norm range: [{norm_min:.6f}, {norm_max:.6f}]")
    else:
        norm_min, norm_max, norm_range = 0, 1, 1
        print(f"      Warning: No regions with norms found")

    # Plot expert regions as rectangles with filled color for better visibility
    for i, region in enumerate(regions_this_depth_only):
        x_lower, t_lower = region['bounds_lower']
        x_upper, t_upper = region['bounds_upper']
        # With x=horizontal and t=vertical:
        width = x_upper - x_lower  # x extent (horizontal)
        height = t_upper - t_lower  # t extent (vertical)

        # Normalize norm to [0, 1] for colormap
        norm_normalized = (region['wavelet_norm'] - norm_min) / norm_range if norm_range > 0 else 0.5

        # Color: red (0) to green (1)
        edge_color = plt.cm.RdYlGn(norm_normalized)
        # Use semi-transparent fill for better visibility
        face_color = list(edge_color[:3]) + [0.3]  # RGB + alpha

        # Rectangle: (x_lower, t_lower) is bottom-left corner
        rect = patches.Rectangle((x_lower, t_lower), width, height,
                                 linewidth=3.0, edgecolor=edge_color,
                                 facecolor=face_color, linestyle='-', zorder=10)
        ax_spatial.add_patch(rect)

        # Debug: print first and last few regions
        if i < 3 or i >= len(regions_this_depth_only) - 3:
            print(f"        Region {i}: norm={region['wavelet_norm']:.6f}, "
                  f"normalized={norm_normalized:.3f}, bounds=({x_lower:.2f},{t_lower:.2f})-({x_upper:.2f},{t_upper:.2f})")

    ax_spatial.set_xlabel('Space (x)', fontsize=12, fontweight='bold')
    ax_spatial.set_ylabel('Time (t)', fontsize=12, fontweight='bold')
    ax_spatial.set_title(f'Depth {depth} - Expert Regions (n={len(regions_this_depth_only)}, borders colored by norm)',
                         fontsize=14, fontweight='bold')
    ax_spatial.grid(True, alpha=0.3, zorder=0)

    # Add colorbar for norm-to-color mapping
    sm = plt.cm.ScalarMappable(cmap='RdYlGn',
                               norm=plt.Normalize(vmin=norm_min, vmax=norm_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_spatial, orientation='vertical', pad=0.02)
    cbar.set_label('Wavelet Norm', fontsize=10)

    plt.tight_layout()

    # Save
    output_path = output_dir / f"depth_{depth}_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def plot_expert_norms_for_model(json_path, output_dir):
    """Generate norm distribution plots for a single model."""
    with open(json_path) as f:
        data = json.load(f)

    regions = data['regions']

    # Extract data
    norms = [r['wavelet_norm'] for r in regions]
    depths = [r['depth'] for r in regions]
    spawn_epochs = [r['spawn_epoch'] for r in regions]

    # Group by depth and spawn_epoch
    depth_norms = {}
    epoch_norms = {}

    for r in regions:
        depth = r['depth']
        epoch = r['spawn_epoch']
        norm = r['wavelet_norm']

        if depth not in depth_norms:
            depth_norms[depth] = []
        depth_norms[depth].append(norm)

        if epoch not in epoch_norms:
            epoch_norms[epoch] = []
        epoch_norms[epoch].append(norm)

    # Create figure with main plot + individual depth plots
    sorted_depths = sorted(depth_norms.keys())
    n_depths = len(sorted_depths)

    # Create subplot grid: 1 main plot on top, then individual depth plots below
    fig = plt.figure(figsize=(16, 4 + 3*n_depths))
    gs = fig.add_gridspec(n_depths + 1, 1, height_ratios=[2] + [1]*n_depths, hspace=0.4)

    # Define colors for each depth
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_depths))

    # Determine bin edges based on all data - use finer bins
    all_norms = []
    for depth in sorted_depths:
        all_norms.extend(depth_norms[depth])

    # Use more bins (50) for finer resolution around 0
    bins = np.histogram_bin_edges(all_norms, bins=50)

    # ===== Main plot: All depths combined =====
    ax_main = fig.add_subplot(gs[0])

    # Plot histogram for each depth with different colors
    for i, depth in enumerate(sorted_depths):
        norms_at_depth = depth_norms[depth]
        ax_main.hist(norms_at_depth, bins=bins, alpha=0.7,
                     edgecolor='black', linewidth=1.2,
                     color=colors[i], label=f'Depth {depth} (n={len(norms_at_depth)})')

    ax_main.set_xlabel('Wavelet Norm', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax_main.set_title(f'Combined Norm Distribution by Depth\n({len(norms)} total experts)',
                      fontsize=15, fontweight='bold')
    ax_main.grid(True, alpha=0.3, axis='y')
    ax_main.legend(loc='upper right', fontsize=11, framealpha=0.9)

    # ===== Individual depth plots =====
    for i, depth in enumerate(sorted_depths):
        ax = fig.add_subplot(gs[i + 1])
        norms_at_depth = depth_norms[depth]

        # Use same bins for consistency
        ax.hist(norms_at_depth, bins=bins, alpha=0.8,
                edgecolor='black', linewidth=1.2, color=colors[i])

        # Add statistics
        mean_val = np.mean(norms_at_depth)
        median_val = np.median(norms_at_depth)
        ax.axvline(median_val, color='red', linestyle='--', linewidth=2,
                   label=f'Median: {median_val:.4f}')
        ax.axvline(mean_val, color='orange', linestyle=':', linewidth=2,
                   label=f'Mean: {mean_val:.4f}')

        ax.set_xlabel('Wavelet Norm', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Depth {depth} Distribution (n={len(norms_at_depth)})',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right', fontsize=9)

    # Save combined plot
    output_path = output_dir / "expert_norm_distributions.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    # ===== Generate individual depth analysis plots =====
    print(f"    Generating individual depth analysis plots...")

    # Load ground truth for spatial visualization
    x_data, t_data, h_magnitude, _ = load_ground_truth(output_dir)

    # Group regions by depth
    regions_by_depth = {}
    for region in regions:
        depth = region['depth']
        if depth not in regions_by_depth:
            regions_by_depth[depth] = []
        regions_by_depth[depth].append(region)

    # Generate plot for each depth
    for i, depth in enumerate(sorted_depths):
        regions_at_depth = regions_by_depth.get(depth, [])
        depth_output = plot_depth_with_spatial(
            depth, regions_at_depth, depth_norms[depth],
            bins, colors[i], output_dir,
            x_data, t_data, h_magnitude
        )
        print(f"      Depth {depth} plot saved: {depth_output.name}")

    # Print statistics
    print(f"\n{'='*60}")
    print("Expert Norm Statistics")
    print(f"{'='*60}")
    print(f"Total experts: {len(norms)}")
    print(f"Overall: mean={np.mean(norms):.4f}, median={np.median(norms):.4f}, "
          f"std={np.std(norms):.4f}")
    print(f"Min: {np.min(norms):.4f}, Max: {np.max(norms):.4f}")
    print(f"\nBy Depth:")
    for depth in sorted_depths:
        norms_d = depth_norms[depth]
        print(f"  Depth {depth}: n={len(norms_d)}, mean={np.mean(norms_d):.4f}, "
              f"median={np.median(norms_d):.4f}")

    sorted_epochs = sorted(epoch_norms.keys())
    print(f"\nBy Spawn Epoch:")
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
