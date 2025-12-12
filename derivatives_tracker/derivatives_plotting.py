"""Plotting utilities for derivatives tracking."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List
from scipy.interpolate import griddata


def plot_residual_summary(
    train_results: Dict[str, Dict],
    eval_results: Dict[str, Dict],
    save_dir: Path
) -> None:
    """
    Plot residual L2 / L_inf evolution for train & eval in a 2x2 grid.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    layer_names = sorted(eval_results.keys())
    layer_indices = list(range(1, len(layer_names) + 1))
    
    train_l2 = [train_results[ln]['norms']['residual_norm'] for ln in layer_names]
    eval_l2 = [eval_results[ln]['norms']['residual_norm'] for ln in layer_names]
    train_inf = [train_results[ln]['norms']['residual_inf_norm'] for ln in layer_names]
    eval_inf = [eval_results[ln]['norms']['residual_inf_norm'] for ln in layer_names]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panels = [
        (axes[0, 0], train_l2, 'Train Residual L2'),
        (axes[0, 1], eval_l2, 'Eval Residual L2'),
        (axes[1, 0], train_inf, 'Train Residual L∞'),
        (axes[1, 1], eval_inf, 'Eval Residual L∞'),
    ]
    
    for ax, values, title in panels:
        ax.plot(layer_indices, values, marker='o', linewidth=2, markersize=7, color='#c0392b')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=11)
        ax.set_ylabel('Mean Norm', fontsize=11)
        ax.set_xticks(layer_indices)
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Residual Evolution Summary', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    
    save_path = save_dir / 'residual_evolution_summary.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Residual evolution summary saved to {save_path}")


def plot_term_magnitudes(
    derivatives_results: Dict[str, Dict],
    save_dir: Path
) -> None:
    """
    Plot L2 norms of all terms across layers.
    
    Shows: ||h||, ||h_t||, ||h_xx||, ||nonlinear||, ||residual||
    
    Args:
        derivatives_results: Dict mapping layer_name -> results dict
        save_dir: Directory to save plot
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    layer_names = sorted(derivatives_results.keys())
    layer_indices = list(range(1, len(layer_names) + 1))
    
    # Extract norms for each term
    h_norms = [derivatives_results[ln]['norms']['h_norm'] for ln in layer_names]
    h_t_norms = [derivatives_results[ln]['norms']['h_t_norm'] for ln in layer_names]
    h_xx_norms = [derivatives_results[ln]['norms']['h_xx_norm'] for ln in layer_names]
    nonlinear_norms = [derivatives_results[ln]['norms']['nonlinear_norm'] for ln in layer_names]
    residual_norms = [derivatives_results[ln]['norms']['residual_norm'] for ln in layer_names]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(layer_indices, h_norms, marker='o', linewidth=2, label='||h||', color='blue')
    ax.plot(layer_indices, h_t_norms, marker='s', linewidth=2, label='||h_t||', color='green')
    ax.plot(layer_indices, h_xx_norms, marker='^', linewidth=2, label='||h_xx||', color='orange')
    ax.plot(layer_indices, nonlinear_norms, marker='d', linewidth=2, label='|||h|²h||', color='purple')
    ax.plot(layer_indices, residual_norms, marker='*', linewidth=2, markersize=10, 
            label='||residual||', color='crimson')
    
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean L2 Norm', fontsize=12, fontweight='bold')
    ax.set_title('Term Magnitudes Across Layers', fontsize=14, fontweight='bold')
    ax.set_xticks(layer_indices)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    save_path = save_dir / 'term_magnitudes.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Term magnitudes plot saved to {save_path}")


def plot_ic_profiles(
    ic_profile: Dict[str, np.ndarray],
    save_dir: Path
) -> None:
    """
    Plot h(x, 0) inferred at each layer alongside the analytic IC 2·sech(x).
    """
    if not ic_profile or not ic_profile.get('layers'):
        return
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    x = ic_profile['x']
    gt_real_func = ic_profile['gt_real']
    gt_imag_func = ic_profile['gt_imag']
    x_gt = np.linspace(x.min(), x.max(), 400)
    gt_real = gt_real_func(x_gt)
    gt_imag = gt_imag_func(x_gt)
    layers = sorted(ic_profile['layers'].keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers))) if layers else ['#2980b9']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    titles = ['Initial Condition (Real part)', 'Initial Condition (Imag part)']
    
    for ax_idx, (ax, comp_idx, title) in enumerate(zip(axes, [0, 1], titles)):
        for color, layer in zip(colors, layers):
            values = ic_profile['layers'][layer][:, comp_idx]
            label = layer if ax_idx == 0 else None
            ax.plot(x, values, color=color, alpha=0.85, label=label)
        if comp_idx == 0:
            ax.plot(x_gt, gt_real, color='black', linewidth=2.5, linestyle='--',
                    label='Ground Truth 2·sech(x)')
        else:
            ax.plot(x_gt, gt_imag, color='black', linewidth=2.0, linestyle='--',
                    label='Ground Truth 0')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('h(x, 0)', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    if layers:
        axes[0].legend(loc='upper right', fontsize=9, ncol=2)
        axes[1].legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    save_path = save_dir / 'ic_profiles.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Initial condition profiles saved to {save_path}")


def _plot_metrics_grid(
    layer_names,
    train_metrics: Dict[str, Dict[str, float]],
    eval_metrics: Dict[str, Dict[str, float]],
    title: str,
    save_path: Path
) -> None:
    layer_indices = list(range(1, len(layer_names) + 1))
    
    def extract(metrics_dict, key):
        return [metrics_dict.get(layer, {}).get(key, np.nan) for layer in layer_names]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panels = [
        (axes[0, 0], extract(train_metrics, 'l2'), 'Train L2'),
        (axes[0, 1], extract(eval_metrics, 'l2'), 'Eval L2'),
        (axes[1, 0], extract(train_metrics, 'linf'), 'Train L∞'),
        (axes[1, 1], extract(eval_metrics, 'linf'), 'Eval L∞'),
    ]
    
    for ax, values, subtitle in panels:
        ax.plot(layer_indices, values, marker='o', linewidth=2, color='#8e44ad')
        ax.set_title(subtitle, fontsize=13, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=11)
        ax.set_ylabel('Mean Norm', fontsize=11)
        ax.set_xticks(layer_indices)
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {title} saved to {save_path}")


def plot_derivative_heatmaps(
    derivatives_results: Dict[str, Dict],
    layer_name: str,
    x: np.ndarray,
    t: np.ndarray,
    ground_truth_derivatives: Dict[str, np.ndarray],
    save_dir: Path
) -> None:
    """
    Plot heatmaps of derivatives for a specific layer with ground truth comparison.
    
    Creates 6x3 grid:
    - Rows 1-2: Predicted (u, v components) of (h, h_t, h_xx)
    - Rows 3-4: Ground truth (u, v components)
    - Rows 5-6: Absolute error (u, v components)
    
    Args:
        derivatives_results: Dict mapping layer_name -> results dict
        layer_name: Which layer to visualize
        x: Spatial coordinates (N,) or (N, 1)
        t: Temporal coordinates (N,) or (N, 1)
        ground_truth_derivatives: Dict with 'h_gt', 'h_t_gt', 'h_xx_gt'
        save_dir: Directory to save plot
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if layer_name not in derivatives_results:
        print(f"  Warning: Layer {layer_name} not found, skipping heatmap")
        return
    
    results = derivatives_results[layer_name]
    
    # Flatten x and t if needed
    x_flat = x.flatten() if isinstance(x, np.ndarray) else x
    t_flat = t.flatten() if isinstance(t, np.ndarray) else t
    
    # Extract ground truth derivatives
    h_gt = ground_truth_derivatives['h_gt']
    h_t_gt = ground_truth_derivatives['h_t_gt']
    h_xx_gt = ground_truth_derivatives['h_xx_gt']
    
    # Create grid for interpolation
    x_grid = np.linspace(x_flat.min(), x_flat.max(), 200)
    t_grid = np.linspace(t_flat.min(), t_flat.max(), 200)
    X_grid, T_grid = np.meshgrid(x_grid, t_grid)
    
    # Terms to plot
    terms = ['h', 'h_t', 'h_xx']
    terms_gt = [h_gt, h_t_gt, h_xx_gt]
    term_labels = ['h', 'h_t', 'h_xx']
    
    fig, axes = plt.subplots(6, 3, figsize=(18, 28))
    fig.suptitle(f'Derivative Heatmaps with Ground Truth - {layer_name}', 
                 fontsize=16, fontweight='bold')
    
    for col_idx, (term, term_gt_data, label) in enumerate(zip(terms, terms_gt, term_labels)):
        term_pred = results[term]  # (N, 2)
        
        # Compute error
        term_error = np.abs(term_pred - term_gt_data)
        
        # Row 0-1: Predicted (u, v)
        for comp_idx, (comp_name, cmap, comp_label) in enumerate([
            ('u', 'viridis', 'u (real)'), ('v', 'plasma', 'v (imag)')
        ]):
            pred_data = term_pred[:, comp_idx]
            pred_grid = griddata((x_flat, t_flat), pred_data, 
                                (X_grid, T_grid), method='cubic')
            
            ax = axes[comp_idx, col_idx]
            im = ax.contourf(X_grid, T_grid, pred_grid, levels=50, cmap=cmap)
            
            # Add row label on leftmost column
            if col_idx == 0:
                ax.set_ylabel(f'{comp_label}\nPredicted\nt', 
                             fontsize=10, fontweight='bold')
            else:
                ax.set_ylabel('t', fontsize=10)
            
            # Column titles only on first row
            if comp_idx == 0:
                ax.set_title(f'{label}', fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax)
        
        # Row 2-3: Ground truth (u, v)
        for comp_idx, (comp_name, cmap, comp_label) in enumerate([
            ('u', 'viridis', 'u (real)'), ('v', 'plasma', 'v (imag)')
        ]):
            gt_data = term_gt_data[:, comp_idx]
            gt_grid = griddata((x_flat, t_flat), gt_data, 
                              (X_grid, T_grid), method='cubic')
            
            ax = axes[2 + comp_idx, col_idx]
            im = ax.contourf(X_grid, T_grid, gt_grid, levels=50, cmap=cmap)
            
            # Add row label on leftmost column
            if col_idx == 0:
                ax.set_ylabel(f'{comp_label}\nGround Truth\nt', 
                             fontsize=10, fontweight='bold')
            else:
                ax.set_ylabel('t', fontsize=10)
            plt.colorbar(im, ax=ax)
        
        # Row 4-5: Error (u, v)
        for comp_idx, (comp_name, cmap, comp_label) in enumerate([
            ('u', 'Reds', 'u (real)'), ('v', 'Reds', 'v (imag)')
        ]):
            error_data = term_error[:, comp_idx]
            error_grid = griddata((x_flat, t_flat), error_data, 
                                 (X_grid, T_grid), method='cubic')
            
            ax = axes[4 + comp_idx, col_idx]
            im = ax.contourf(X_grid, T_grid, error_grid, levels=50, cmap=cmap)
            
            # Add row label on leftmost column
            if col_idx == 0:
                ax.set_ylabel(f'{comp_label}\n|Error|\nt', 
                             fontsize=10, fontweight='bold')
            else:
                ax.set_ylabel('t', fontsize=10)
            ax.set_xlabel('x', fontsize=10)
            plt.colorbar(im, ax=ax)
    
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    
    save_path = save_dir / f'derivative_heatmaps_{layer_name}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Derivative heatmaps with GT for {layer_name} saved to {save_path}")


def plot_residual_heatmaps(
    derivatives_results: Dict[str, Dict],
    layer_name: str,
    x: np.ndarray,
    t: np.ndarray,
    save_dir: Path
) -> None:
    """
    Plot residual heatmaps for a specific layer.
    
    Creates 2x1 grid:
    - Top: f_u (real residual)
    - Bottom: f_v (imaginary residual)
    
    Args:
        derivatives_results: Dict mapping layer_name -> results dict
        layer_name: Which layer to visualize
        x: Spatial coordinates (N,) or (N, 1)
        t: Temporal coordinates (N,) or (N, 1)
        save_dir: Directory to save plot
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if layer_name not in derivatives_results:
        print(f"  Warning: Layer {layer_name} not found, skipping residual heatmap")
        return
    
    residual = derivatives_results[layer_name]['residual']  # (N, 2)
    
    # Flatten x and t if needed
    x_flat = x.flatten() if isinstance(x, np.ndarray) else x
    t_flat = t.flatten() if isinstance(t, np.ndarray) else t
    
    # Create grid for interpolation
    x_grid = np.linspace(x_flat.min(), x_flat.max(), 200)
    t_grid = np.linspace(t_flat.min(), t_flat.max(), 200)
    X_grid, T_grid = np.meshgrid(x_grid, t_grid)
    
    # Interpolate residual components
    f_u = residual[:, 0]
    f_v = residual[:, 1]
    
    f_u_grid = griddata((x_flat, t_flat), f_u, (X_grid, T_grid), method='cubic')
    f_v_grid = griddata((x_flat, t_flat), f_v, (X_grid, T_grid), method='cubic')
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'Residual Heatmaps - {layer_name}', fontsize=16, fontweight='bold')
    
    # Top: Real residual
    ax = axes[0]
    im = ax.contourf(X_grid, T_grid, f_u_grid, levels=50, cmap='RdBu_r')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('t', fontsize=12)
    ax.set_title('Residual f_u (real)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax)
    
    # Bottom: Imaginary residual
    ax = axes[1]
    im = ax.contourf(X_grid, T_grid, f_v_grid, levels=50, cmap='RdBu_r')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('t', fontsize=12)
    ax.set_title('Residual f_v (imag)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    save_path = save_dir / f'residual_heatmaps_{layer_name}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Residual heatmaps for {layer_name} saved to {save_path}")


def plot_residual_change_heatmaps(
    derivatives_results: Dict[str, Dict],
    x: np.ndarray,
    t: np.ndarray,
    save_dir: Path
) -> None:
    """
    Plot residual change heatmaps showing error evolution between layers.
    
    For each transition, computes: abs(residual_{i+1}) / abs(residual_i)
    - 0.0 (green): Error eliminated
    - 1.0 (white): No change
    - >1.0 (red): Error increased
    
    Args:
        derivatives_results: Dict mapping layer_name -> results dict
        x: Spatial coordinates
        t: Temporal coordinates
        save_dir: Directory to save plot
    """
    from matplotlib.colors import TwoSlopeNorm
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    layer_names = sorted(derivatives_results.keys(),
                        key=lambda x: int(x.split('_')[1]))
    n_layers = len(layer_names)
    
    if n_layers < 2:
        return  # Need at least 2 layers for changes
    
    # Store interpolated grids for all layers
    x_flat = x.flatten() if isinstance(x, np.ndarray) else x
    t_flat = t.flatten() if isinstance(t, np.ndarray) else t
    
    x_grid = np.linspace(x_flat.min(), x_flat.max(), 200)
    t_grid = np.linspace(t_flat.min(), t_flat.max(), 200)
    X_grid, T_grid = np.meshgrid(x_grid, t_grid)
    
    # Pre-compute all grids
    residual_grids = {}
    for layer_name in layer_names:
        residual = derivatives_results[layer_name]['residual']  # (N, 2)
        f_u = residual[:, 0]
        f_v = residual[:, 1]
        
        f_u_grid = griddata((x_flat, t_flat), f_u, (X_grid, T_grid),
                            method='cubic', fill_value=0)
        f_v_grid = griddata((x_flat, t_flat), f_v, (X_grid, T_grid),
                            method='cubic', fill_value=0)
        
        residual_grids[layer_name] = {
            'f_u': f_u_grid,
            'f_v': f_v_grid
        }
    
    # Create figure for changes (N-1 transitions, 2 subplots each)
    # Layout: imaginary below real, arrange transitions to make figure square
    n_changes = n_layers - 1
    
    # Calculate optimal grid layout for roughly square figure
    # Each transition needs 2 rows (u on top, v below)
    # n_cols_transitions = number of transitions per row
    import math
    n_cols_transitions = max(1, int(math.ceil(math.sqrt(n_changes))))
    n_rows_groups = int(math.ceil(n_changes / n_cols_transitions))
    n_rows_total = n_rows_groups * 2  # 2 rows per transition group
    
    fig, axes = plt.subplots(n_rows_total, n_cols_transitions,
                             figsize=(6 * n_cols_transitions, 5 * n_rows_groups))
    fig.suptitle('Residual Error Changes Between Layers\n'
                 '(Ratio: |layer_{i+1}| / |layer_i|, Green=Improved, Red=Worse)',
                 fontsize=14, fontweight='bold')
    
    # Ensure axes is 2D
    if n_rows_total == 1:
        axes = axes.reshape(1, -1)
    elif n_cols_transitions == 1:
        axes = axes.reshape(-1, 1)
    
    # Define colormap normalization (center at 1.0)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=3.0)
    
    for idx in range(n_changes):
        layer_prev = layer_names[idx]
        layer_curr = layer_names[idx + 1]
        
        prev_u = residual_grids[layer_prev]['f_u']
        prev_v = residual_grids[layer_prev]['f_v']
        curr_u = residual_grids[layer_curr]['f_u']
        curr_v = residual_grids[layer_curr]['f_v']
        
        # Compute ratio: abs(curr) / abs(prev)
        # Handle division by zero: set to large value (error got worse)
        eps = 1e-10
        ratio_u = np.abs(curr_u) / (np.abs(prev_u) + eps)
        ratio_v = np.abs(curr_v) / (np.abs(prev_v) + eps)
        
        # Cap extreme values for visualization
        ratio_u = np.clip(ratio_u, 0, 5)
        ratio_v = np.clip(ratio_v, 0, 5)
        
        # Calculate position in grid
        col_idx = idx % n_cols_transitions
        row_group = idx // n_cols_transitions
        row_u = row_group * 2  # Real part row
        row_v = row_group * 2 + 1  # Imaginary part row (below real)
        
        # Plot real part change (top)
        ax_u = axes[row_u, col_idx]
        im_u = ax_u.contourf(X_grid, T_grid, ratio_u, levels=50,
                             cmap='RdYlGn_r', norm=norm)
        ax_u.set_xlabel('x', fontsize=11)
        ax_u.set_ylabel('t', fontsize=11)
        ax_u.set_title(f'{layer_prev} → {layer_curr}\nReal part (u)',
                      fontsize=12, fontweight='bold')
        cbar_u = plt.colorbar(im_u, ax=ax_u)
        cbar_u.set_label('Error Ratio', fontsize=10)
        
        # Plot imaginary part change (below)
        ax_v = axes[row_v, col_idx]
        im_v = ax_v.contourf(X_grid, T_grid, ratio_v, levels=50,
                             cmap='RdYlGn_r', norm=norm)
        ax_v.set_xlabel('x', fontsize=11)
        ax_v.set_ylabel('t', fontsize=11)
        ax_v.set_title(f'{layer_prev} → {layer_curr}\nImaginary part (v)',
                      fontsize=12, fontweight='bold')
        cbar_v = plt.colorbar(im_v, ax=ax_v)
        cbar_v.set_label('Error Ratio', fontsize=10)
    
    # Hide unused subplots if any
    for row in range(n_rows_total):
        for col in range(n_cols_transitions):
            trans_idx = (row // 2) * n_cols_transitions + col
            if trans_idx >= n_changes:
                axes[row, col].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = save_dir / 'residual_change_heatmaps.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Residual change heatmaps saved to {save_path}")


def generate_all_derivative_plots(
    train_results: Dict[str, Dict],
    eval_results: Dict[str, Dict],
    ic_metrics: Dict[str, Dict[str, Dict[str, float]]],
    bc_value_metrics: Dict[str, Dict[str, Dict[str, float]]],
    bc_derivative_metrics: Dict[str, Dict[str, Dict[str, float]]],
    ic_profile: Dict[str, np.ndarray],
    x: np.ndarray,
    t: np.ndarray,
    ground_truth_derivatives: Dict[str, np.ndarray],
    save_dir: Path
) -> None:
    """
    Generate all derivative visualization plots.
    
    Args:
        derivatives_results: Dict mapping layer_name -> results dict
        x: Spatial coordinates
        t: Temporal coordinates
        ground_truth_derivatives: Dict with 'h_gt', 'h_t_gt', 'h_xx_gt'
        save_dir: Directory to save plots
    """
    print("\nGenerating derivative plots...")
    
    # Plot 1: Residual evolution summary (L2 / L_inf, train & eval)
    plot_residual_summary(train_results, eval_results, save_dir)
    
    # Plot 2: Term magnitudes (eval set for clarity)
    plot_term_magnitudes(eval_results, save_dir)
    
    layer_names = sorted(eval_results.keys())
    
    # Plot 3: IC / BC summaries
    _plot_metrics_grid(
        layer_names,
        ic_metrics['train'],
        ic_metrics['eval'],
        'Initial Condition Error Summary',
        save_dir / 'ic_summary.png'
    )
    _plot_metrics_grid(
        layer_names,
        bc_value_metrics['train'],
        bc_value_metrics['eval'],
        'Boundary Value Error Summary',
        save_dir / 'bc_value_summary.png'
    )
    _plot_metrics_grid(
        layer_names,
        bc_derivative_metrics['train'],
        bc_derivative_metrics['eval'],
        'Boundary Derivative Error Summary',
        save_dir / 'bc_derivative_summary.png'
    )
    
    # Plot IC profiles (eval set)
    plot_ic_profiles(ic_profile, save_dir)
    
    # Plot 4+: Heatmaps for every layer (stop-on-error would hide issues)
    total_layers = len(layer_names)
    for idx, layer_name in enumerate(layer_names, start=1):
        print(f"  Generating heatmaps for {layer_name} ({idx}/{total_layers})")
        try:
            plot_derivative_heatmaps(
                eval_results,
                layer_name,
                x,
                t,
                ground_truth_derivatives,
                save_dir
            )
        except Exception as exc:
            print(f"    WARNING: derivative heatmaps failed for {layer_name}: {exc}")
        try:
            plot_residual_heatmaps(
                eval_results,
                layer_name,
                x,
                t,
                save_dir
            )
        except Exception as exc:
            print(f"    WARNING: residual heatmaps failed for {layer_name}: {exc}")
    
    # Generate residual change heatmaps (N-1 transitions)
    try:
        plot_residual_change_heatmaps(eval_results, x, t, save_dir)
    except Exception as exc:
        print(f"    WARNING: residual change heatmaps failed: {exc}")
    
    print(f"All derivative plots generated in {save_dir}")


def plot_derivative_history_shaded(history: List[tuple], save_dir: Path) -> None:
    """
    Overlay derivative residual and IC/BC norms across epochs with shaded progression.
    history: list of (epoch, metrics_summary) returned by run_derivatives_tracker.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    history = sorted(history, key=lambda x: x[0])
    base_color = '#2ecc71'
    alphas = [min(0.45 + 0.15 * idx, 1.0) for idx in range(len(history))]

    def _plot_residual():
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        panels = [
            ('train_residual_l2', 'Train L2'),
            ('eval_residual_l2', 'Eval L2'),
            ('train_residual_linf', 'Train L∞'),
            ('eval_residual_linf', 'Eval L∞'),
        ]
        key_map = {
            'train_residual_l2': 'train_residual_l2',
            'eval_residual_l2': 'eval_residual_l2',
            'train_residual_linf': 'train_residual_linf',
            'eval_residual_linf': 'eval_residual_linf',
        }
        for ax, (key, title) in zip(axes.flat, panels):
            for (epoch, metrics), alpha in zip(history, alphas):
                layers = sorted(metrics.get('layer_norms', {}).keys(), key=lambda x: int(x.split('_')[-1]))
                values = [metrics['layer_norms'][ln].get(key_map[key], float('nan')) for ln in layers]
                ax.plot(range(1, len(values)+1), values, marker='o', color=base_color, alpha=alpha, label=f"Epoch {epoch}")
            ax.set_title(title)
            ax.set_xlabel('Layer')
            ax.set_ylabel('Mean Norm')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        axes[0,0].legend()
        plt.tight_layout()
        plt.savefig(save_dir / "residual_evolution_summary.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_ic_bc(key: str, title: str, filename: str):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        panels = [('l2', 'Train L2'), ('l2_eval', 'Eval L2'), ('linf', 'Train L∞'), ('linf_eval', 'Eval L∞')]
        for ax_idx, (metric_key, lbl) in enumerate(panels):
            for (epoch, metrics), alpha in zip(history, alphas):
                store = metrics.get(key, {})
                if not store:
                    continue
                # select train/eval
                split = 'train' if 'eval' not in metric_key else 'eval'
                base_key = 'l2' if 'l2' in metric_key else 'linf'
                per_layer = store.get(split, {})
                layers = sorted(per_layer.keys(), key=lambda x: int(x.split('_')[-1]))
                values = [per_layer[ln].get(base_key, float('nan')) for ln in layers]
                ax = axes.flatten()[ax_idx]
                ax.plot(range(1, len(values)+1), values, marker='o', color=base_color, alpha=alpha, label=f"Epoch {epoch}")
            ax = axes.flatten()[ax_idx]
            ax.set_title(lbl)
            ax.set_xlabel('Layer')
            ax.set_ylabel('Mean Norm')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        axes[0,0].legend()
        plt.tight_layout()
        plt.savefig(save_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

    _plot_residual()
    _plot_ic_bc('ic', 'Initial Condition Error Summary', 'ic_summary.png')
    _plot_ic_bc('bc_value', 'Boundary Value Error Summary', 'bc_value_summary.png')
    _plot_ic_bc('bc_derivative', 'Boundary Derivative Error Summary', 'bc_derivative_summary.png')

