"""Tree structure analysis: expert norm distributions and leaf loss history."""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
import numpy as np
from pathlib import Path
import sys
import torch
from scipy.interpolate import griddata


def _detect_problem_from_label(label: str) -> str:
    """Infer the problem name from a run directory label."""
    known = ['schrodinger', 'burgers2d', 'burgers1d', 'wave1d']
    for p in known:
        if p in label.lower():
            return p
    return 'schrodinger'


def load_ground_truth(output_dir, problem=None):
    """Try to load ground truth data for background visualization.

    Only works for 2D problems (1 spatial + time). Returns None
    for higher-dimensional problems like burgers2d.
    """
    if problem is None:
        problem = _detect_problem_from_label(str(output_dir))

    try:
        eval_data_path = Path("datasets") / problem / "eval_data.pt"

        if eval_data_path.exists():
            eval_data = torch.load(
                eval_data_path, map_location='cpu')

            x_data = eval_data['x'].numpy()
            if x_data.ndim == 2 and x_data.shape[1] > 1:
                return None, None, None, None
            t_data = eval_data['t'].numpy()

            if 'h_gt' in eval_data:
                h_data = eval_data['h_gt'].numpy()
            elif 'h' in eval_data:
                h_data = eval_data['h'].numpy()
            elif 'u_gt' in eval_data:
                h_data = eval_data['u_gt'].numpy()
            else:
                return None, None, None, None

            if h_data.shape[1] == 2:
                h_magnitude = np.sqrt(
                    h_data[:, 0]**2 + h_data[:, 1]**2)
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

    all_x_lo = []
    all_x_hi = []
    all_t_lo = []
    all_t_hi = []

    for region in regions:
        x_lower, t_lower = region['bounds_lower']
        x_upper, t_upper = region['bounds_upper']
        all_x_lo.append(x_lower)
        all_x_hi.append(x_upper)
        all_t_lo.append(t_lower)
        all_t_hi.append(t_upper)

        width = x_upper - x_lower
        height = t_upper - t_lower

        norm_val = region['wavelet_norm']
        norm_normalized = (
            (norm_val - norm_min) / norm_range
            if norm_range > 0 else 0.5)
        edge_color = plt.cm.RdYlGn(norm_normalized)
        face_color = list(edge_color[:3]) + [0.3]

        rect = patches.Rectangle(
            (x_lower, t_lower), width, height,
            linewidth=3.0, edgecolor=edge_color,
            facecolor=face_color, linestyle='-', zorder=10)
        ax.add_patch(rect)

    x_margin = (max(all_x_hi) - min(all_x_lo)) * 0.05
    t_margin = (max(all_t_hi) - min(all_t_lo)) * 0.05
    ax.set_xlim(min(all_x_lo) - x_margin,
                max(all_x_hi) + x_margin)
    ax.set_ylim(min(all_t_lo) - t_margin,
                max(all_t_hi) + t_margin)

    ax.set_xlabel('Space (x)', fontsize=11)
    ax.set_ylabel('Time (t)', fontsize=11)
    ax.set_title(
        f'{title_label} (n={len(regions)})',
        fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, zorder=0)

    sm = plt.cm.ScalarMappable(
        cmap='RdYlGn',
        norm=plt.Normalize(vmin=norm_min, vmax=norm_max))
    sm.set_array([])
    cbar = plt.colorbar(
        sm, ax=ax, orientation='vertical', pad=0.02)
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


def _plot_regions_3d(ax, regions, title_label):
    """Plot 3D wireframe boxes colored by wavelet norm (RdYlGn)."""
    if not regions:
        ax.set_title(
            f'{title_label} (n=0)',
            fontsize=12, fontweight='bold')
        ax.text2D(0.5, 0.5, 'No regions',
                  transform=ax.transAxes,
                  ha='center', va='center',
                  fontsize=14, color='gray')
        return

    norms = [r['wavelet_norm'] for r in regions]
    norm_min, norm_max = min(norms), max(norms)
    norm_range = norm_max - norm_min if norm_max > norm_min else 1.0

    edges = [
        (0, 1), (0, 2), (0, 4), (1, 3),
        (1, 5), (2, 3), (2, 6), (3, 7),
        (4, 5), (4, 6), (5, 7), (6, 7),
    ]

    for region in regions:
        lo = region['bounds_lower']
        hi = region['bounds_upper']
        corners = np.array([
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [lo[0], hi[1], hi[2]],
            [hi[0], hi[1], hi[2]],
        ])
        nv = (region['wavelet_norm'] - norm_min) / norm_range \
            if norm_range > 0 else 0.5
        c = plt.cm.RdYlGn(nv)
        for i, j in edges:
            ax.plot3D(
                *zip(corners[i], corners[j]),
                color=c, linewidth=1.5, alpha=0.8)
        ax.scatter(
            *corners.T, color=c, s=12, alpha=0.9)

    ax.set_xlabel('x0', fontsize=9)
    ax.set_ylabel('x1', fontsize=9)
    ax.set_zlabel('t', fontsize=9)
    ax.set_title(
        f'{title_label} (n={len(regions)})',
        fontsize=12, fontweight='bold')

    sm = plt.cm.ScalarMappable(
        cmap='RdYlGn',
        norm=plt.Normalize(vmin=norm_min, vmax=norm_max))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1,
                 label='Wavelet Norm')


def plot_depth_with_spatial(depth, spawned_regions, rejected_regions,
                             spawned_norms, rejected_norms, bins,
                             color, output_dir, x_data, t_data, h_magnitude):
    """Create a 2x2 figure for a specific depth: spawned (left) vs rejected (right)."""
    all_regions = spawned_regions + rejected_regions
    input_dim = (len(all_regions[0]['bounds_lower'])
                 if all_regions else 2)
    is_3d = input_dim > 2

    if is_3d:
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(
            f'Depth {depth} Analysis',
            fontsize=16, fontweight='bold', y=0.98)
        ax_h0 = fig.add_subplot(2, 2, 1)
        ax_h1 = fig.add_subplot(2, 2, 2)
        ax_s0 = fig.add_subplot(2, 2, 3, projection='3d')
        ax_s1 = fig.add_subplot(2, 2, 4, projection='3d')
    else:
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle(
            f'Depth {depth} Analysis',
            fontsize=16, fontweight='bold', y=0.98)
        ax_h0, ax_h1 = axes[0, 0], axes[0, 1]
        ax_s0, ax_s1 = axes[1, 0], axes[1, 1]

    _plot_histogram_on_axis(ax_h0, spawned_norms, bins, color,
                            f'Depth {depth} - Spawned')

    rejected_color = np.array(color[:3]) * 0.6
    rejected_color = np.clip(rejected_color, 0, 1)
    _plot_histogram_on_axis(ax_h1, rejected_norms, bins,
                            rejected_color,
                            f'Depth {depth} - Rejected')

    if is_3d:
        _plot_regions_3d(ax_s0, spawned_regions,
                         f'Depth {depth} - Spawned Regions')
        _plot_regions_3d(ax_s1, rejected_regions,
                         f'Depth {depth} - Rejected Regions')
    else:
        _render_ground_truth_background(
            ax_s0, x_data, t_data, h_magnitude)
        _plot_regions_on_axis(ax_s0, spawned_regions,
                              f'Depth {depth} - Spawned Regions')
        _render_ground_truth_background(
            ax_s1, x_data, t_data, h_magnitude)
        _plot_regions_on_axis(ax_s1, rejected_regions,
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


def plot_leaf_loss_history(leaf_loss_history, output_dir):
    """Plot leaf mean-loss distributions at each spawn epoch.

    For each spawn epoch a grouped bar chart shows the mean loss of every leaf.
    The worst (tallest) bar is highlighted in red. A second figure shows all
    epochs overlaid so the loss evolution is visible at a glance.
    """
    if not leaf_loss_history:
        return None

    n_epochs = len(leaf_loss_history)
    colors_epoch = plt.cm.viridis(np.linspace(0.2, 0.9, n_epochs))

    # --- Per-epoch bar charts (paginated, max 6 per page) ---
    max_per_page = 6
    pages = [leaf_loss_history[i:i + max_per_page]
             for i in range(0, n_epochs, max_per_page)]

    saved_paths = []
    for page_idx, page in enumerate(pages):
        n_cols = len(page)
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5),
                                 squeeze=False)
        fig.suptitle('Leaf Mean Loss at Each Spawn Step',
                     fontsize=15, fontweight='bold')

        for col, entry in enumerate(page):
            ax = axes[0, col]
            epoch = entry['epoch']
            leaves = entry['leaves']
            if not leaves:
                ax.set_title(f'Epoch {epoch} (no leaves)')
                continue

            labels = [f"E{l['leaf_idx']+1}" if l['leaf_idx'] >= 0
                      else 'Base' for l in leaves]
            losses = [l['mean_loss'] for l in leaves]
            worst_idx = int(np.argmax(losses))

            bar_colors = [colors_epoch[page_idx * max_per_page + col]] * len(losses)
            bar_colors[worst_idx] = 'crimson'

            bars = ax.bar(labels, losses, color=bar_colors, edgecolor='black',
                          linewidth=0.8)
            for i_b, bar in enumerate(bars):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{losses[i_b]:.4f}', ha='center', va='bottom',
                        fontsize=7, rotation=45)

            ax.set_title(f'Epoch {epoch}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Mean Loss', fontsize=10)
            ax.set_xlabel('Leaf', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        suffix = f'_p{page_idx + 1}' if len(pages) > 1 else ''
        out_path = output_dir / f'leaf_loss_bars{suffix}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        saved_paths.append(out_path)
        print(f"    Leaf loss bars saved: {out_path.name}")

    # --- Summary: all epochs overlaid as grouped bars ---
    fig, ax = plt.subplots(figsize=(max(10, 2 * n_epochs), 6))
    fig.suptitle('Leaf Mean Loss Across Spawn Steps',
                 fontsize=15, fontweight='bold')

    all_leaf_ids = sorted({
        l['leaf_idx'] for entry in leaf_loss_history for l in entry['leaves']
    })
    id_to_label = {lid: (f'E{lid+1}' if lid >= 0 else 'Base')
                   for lid in all_leaf_ids}
    x_ticks = [f"Ep {e['epoch']}" for e in leaf_loss_history]
    x = np.arange(n_epochs)
    n_leaves_max = max(len(e['leaves']) for e in leaf_loss_history)
    bar_width = 0.8 / max(n_leaves_max, 1)

    legend_handles = {}
    for entry_idx, entry in enumerate(leaf_loss_history):
        leaves = entry['leaves']
        worst_idx = int(np.argmax([l['mean_loss'] for l in leaves])) if leaves else -1
        for li, leaf in enumerate(leaves):
            lid = leaf['leaf_idx']
            label = id_to_label[lid]
            offset = (li - len(leaves) / 2 + 0.5) * bar_width
            is_worst = (li == worst_idx)
            c = 'crimson' if is_worst else colors_epoch[entry_idx]
            bar = ax.bar(x[entry_idx] + offset, leaf['mean_loss'],
                         bar_width, color=c, edgecolor='black',
                         linewidth=0.5)
            if label not in legend_handles and not is_worst:
                legend_handles[label] = bar[0]

    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks, fontsize=9)
    ax.set_ylabel('Mean Loss', fontsize=12)
    ax.set_xlabel('Spawn Step', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    from matplotlib.patches import Patch as LegPatch
    handles = list(legend_handles.values()) + [LegPatch(facecolor='crimson', edgecolor='black', label='Worst (split)')]
    labels_leg = list(legend_handles.keys()) + ['Worst (split)']
    ax.legend(handles, labels_leg, fontsize=8, loc='upper right')

    plt.tight_layout()
    summary_path = output_dir / 'leaf_loss_summary.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Leaf loss summary saved: {summary_path.name}")

    return saved_paths[0] if saved_paths else summary_path


def plot_expert_norms_for_model(json_path, output_dir, metrics_path=None):
    """Generate tree structure analysis plots for a single model.

    Produces norm distribution plots if non-trivial wavelet norms exist,
    and leaf loss distribution plots if leaf_loss_history is present.
    """
    with open(json_path) as f:
        data = json.load(f)

    regions = data['regions']
    leaf_loss_history = data.get('leaf_loss_history', None)

    if not regions and not leaf_loss_history:
        print("    No regions or loss history found, skipping")
        return None

    # --- Leaf loss distribution plots ---
    if leaf_loss_history:
        print("    Generating leaf loss distribution plots...")
        plot_leaf_loss_history(leaf_loss_history, output_dir)

    # --- Norm distribution plots (only if non-trivial norms exist) ---
    all_norms = [r['wavelet_norm'] for r in regions]
    has_norms = any(n > 0 for n in all_norms)
    if not has_norms:
        print("    No non-zero wavelet norms, skipping norm plots")
        return None

    # Split into spawned and rejected
    spawned_regions = [r for r in regions if r.get('spawned', True)]
    rejected_regions = [r for r in regions if not r.get('spawned', True)]

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

    all_depths = sorted(set(list(spawned_by_depth.keys()) + list(rejected_by_depth.keys())))
    n_depths = len(all_depths) if all_depths else 1

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

    Handles three layouts:
      1. Flat / multi-PDE: batch/YYYYMMDD_HHMMSS/ dirs with metrics.json
         → each timestamp is a run, label from config's model field
      2. Multiple architectures: batch/arch_a/timestamp/, batch/arch_b/timestamp/
         → picks latest timestamp per architecture, label = arch name
      3. Single architecture with many runs: batch/arch/ts1/, batch/arch/ts2/
         → expands each timestamp, label = timestamp name
    """
    import re
    import yaml
    _TS_RE = re.compile(r'\d{8}_\d{6}$')

    child_dirs = sorted(
        d for d in batch_path.iterdir()
        if d.is_dir() and d.name != 'checkpoints'
    )
    if not child_dirs:
        return []

    # Detect flat structure: children are timestamp dirs with metrics.json
    flat_ts = [d for d in child_dirs
               if _TS_RE.match(d.name) and (d / 'metrics.json').exists()]
    if flat_ts:
        runs = []
        for ts_dir in flat_ts:
            cfg_file = ts_dir / 'config_used.yaml'
            if cfg_file.exists():
                try:
                    with open(cfg_file) as f:
                        cfg = yaml.safe_load(f)
                    label = cfg.get('model', ts_dir.name)
                except Exception:
                    label = ts_dir.name
            else:
                label = ts_dir.name
            runs.append((label, ts_dir))
        return runs

    # Nested structure
    runs = []
    for model_dir in child_dirs:
        ts_dirs = sorted(
            d for d in model_dir.iterdir()
            if d.is_dir() and d.name != 'checkpoints'
        )
        if not ts_dirs:
            continue

        if len(child_dirs) == 1 and len(ts_dirs) > 1:
            for ts_dir in ts_dirs:
                runs.append((ts_dir.name, ts_dir))
        else:
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
