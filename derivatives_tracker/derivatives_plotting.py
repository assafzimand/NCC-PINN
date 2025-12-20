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
    save_dir: Path,
    config: Dict = None
) -> None:
    """
    Plot L2 norms of terms relevant to the problem's residual formula.
    
    Only plots derivatives that are used in the specific problem's residual.
    
    Args:
        derivatives_results: Dict mapping layer_name -> results dict
        save_dir: Directory to save plot
        config: Configuration dict (used to determine relevant derivatives)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    layer_names = sorted(derivatives_results.keys())
    layer_indices = list(range(1, len(layer_names) + 1))
    
    # Get relevant derivatives and term metadata for this problem
    term_metadata = {}
    if config is not None:
        problem_name = config.get('problem', 'schrodinger')
        try:
            from derivatives_tracker.residuals import get_residual_module
            residual_module = get_residual_module(problem_name)
            relevant_derivatives = residual_module.get_relevant_derivatives()
            # Get problem-specific term metadata
            if hasattr(residual_module, 'get_term_metadata'):
                term_metadata = residual_module.get_term_metadata()
        except Exception as e:
            print(f"  Warning: Could not get relevant derivatives for {problem_name}: {e}")
            # Fallback: plot all available terms
            relevant_derivatives = ['h', 'h_t', 'h_tt', 'h_x', 'h_xx']
    else:
        # Fallback: plot all available terms
        relevant_derivatives = ['h', 'h_t', 'h_tt', 'h_x', 'h_xx']
    
    # Map derivative names to their norm keys, labels, and plotting styles
    derivative_mapping = {
        'h': ('h_norm', '||h||', 'o', 'blue'),
        'h_t': ('h_t_norm', '||h_t||', 's', 'green'),
        'h_tt': ('h_tt_norm', '||h_tt||', 'p', 'cyan'),
        'h_x': ('h_x_norm', '||h_x||', 'v', 'magenta'),
        'h_xx': ('h_xx_norm', '||h_xx||', '^', 'orange'),
    }
    
    # Add problem-specific terms from metadata
    for term_key, meta in term_metadata.items():
        derivative_mapping[term_key] = (
            f'{term_key}_norm',
            f"||{meta.get('label', term_key)}||",
            meta.get('marker', 'd'),
            meta.get('color', 'purple')
        )
        # Add to relevant_derivatives if not already there
        if term_key not in relevant_derivatives:
            relevant_derivatives.append(term_key)
    
    # Extract norms for residual (always present)
    residual_norms = [derivatives_results[ln]['norms']['residual_norm'] for ln in layer_names]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot only relevant terms
    first_layer = layer_names[0]
    norms_keys = derivatives_results[first_layer]['norms'].keys()
    
    for deriv_name in relevant_derivatives:
        if deriv_name in derivative_mapping:
            norm_key, label, marker, color = derivative_mapping[deriv_name]
            if norm_key in norms_keys:
                norms = [derivatives_results[ln]['norms'][norm_key] for ln in layer_names]
                ax.plot(layer_indices, norms, marker=marker, linewidth=2, label=label, color=color)
    
    # Always plot residual
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
    layers = sorted(ic_profile['layers'].keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers))) if layers else ['#2980b9']
    
    # Determine output_dim from first layer
    first_layer = layers[0] if layers else None
    output_dim = ic_profile['layers'][first_layer].shape[1] if first_layer else 1
    
    # Get ground truth functions
    gt_funcs = []
    gt_labels = []
    x_gt = np.linspace(x.min(), x.max(), 400)
    
    for comp_idx in range(output_dim):
        if comp_idx == 0:
            gt_funcs.append(ic_profile['gt_real'])
            gt_labels.append(ic_profile.get('gt_label_0', 'Ground Truth'))
        elif comp_idx == 1 and 'gt_imag' in ic_profile:
            gt_funcs.append(ic_profile['gt_imag'])
            gt_labels.append(ic_profile.get('gt_label_1', 'Ground Truth'))
    
    # Create subplots
    fig, axes = plt.subplots(1, output_dim, figsize=(7*output_dim, 6), sharex=True)
    if output_dim == 1:
        axes = [axes]
    
    titles = ['h(x, 0)'] if output_dim == 1 else ['Initial Condition (Real part)', 'Initial Condition (Imag part)']
    
    for comp_idx in range(output_dim):
        ax = axes[comp_idx]
        
        # Plot layer predictions
        for color, layer in zip(colors, layers):
            values = ic_profile['layers'][layer][:, comp_idx]
            label = layer if comp_idx == 0 else None
            ax.plot(x, values, color=color, alpha=0.85, label=label)
        
        # Plot ground truth
        if comp_idx < len(gt_funcs):
            gt_values = gt_funcs[comp_idx](x_gt)
            ax.plot(x_gt, gt_values, color='black', linewidth=2.5, linestyle='--',
                   label=gt_labels[comp_idx])
        
        ax.set_title(titles[comp_idx], fontsize=13, fontweight='bold')
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('h(x, 0)', fontsize=11)
        ax.grid(True, alpha=0.3)
        if layers or (comp_idx < len(gt_funcs)):
            ax.legend(loc='upper right', fontsize=9)
    
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
    save_dir: Path,
    config: Dict = None
) -> None:
    """
    Plot heatmaps of derivatives for a specific layer with ground truth comparison.
    Only plots derivatives that are relevant to the problem's residual formula.
    
    Creates grid with rows for each output component showing:
    - Predicted derivatives
    - Ground truth derivatives
    - Absolute error
    
    Args:
        derivatives_results: Dict mapping layer_name -> results dict
        layer_name: Which layer to visualize
        x: Spatial coordinates (N,) or (N, 1)
        t: Temporal coordinates (N,) or (N, 1)
        ground_truth_derivatives: Dict with 'h_gt' and optionally 'h_t_gt', 'h_tt_gt', 'h_xx_gt'
        save_dir: Directory to save plot
        config: Configuration dict (used to determine relevant derivatives)
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
    
    # Determine output_dim from h_gt
    h_gt = ground_truth_derivatives['h_gt']
    output_dim = h_gt.shape[1] if len(h_gt.shape) > 1 else 1
    
    # Get relevant derivatives for this problem
    if config is not None:
        problem_name = config.get('problem', 'schrodinger')
        try:
            from derivatives_tracker.residuals import get_residual_module
            residual_module = get_residual_module(problem_name)
            relevant_derivatives = residual_module.get_relevant_derivatives()
        except Exception as e:
            print(f"  Warning: Could not get relevant derivatives for {problem_name}: {e}")
            # Fallback: plot all available terms
            relevant_derivatives = ['h', 'h_t', 'h_tt', 'h_x', 'h_xx']
    else:
        # Fallback: plot all available terms
        relevant_derivatives = ['h', 'h_t', 'h_tt', 'h_x', 'h_xx']
    
    # Map derivative names to actual keys (only base derivatives, not computed terms)
    derivative_mapping = {
        'h': 'h',
        'h_t': 'h_t',
        'h_tt': 'h_tt',
        'h_x': 'h_x',
        'h_xx': 'h_xx'
    }
    
    # Collect terms that exist in both results and ground_truth, and are relevant
    terms_to_plot = []
    for deriv_name in relevant_derivatives:
        if deriv_name in derivative_mapping:
            term_key = derivative_mapping[deriv_name]
            if term_key in results and f'{term_key}_gt' in ground_truth_derivatives:
                terms_to_plot.append((term_key, ground_truth_derivatives[f'{term_key}_gt']))
    
    if not terms_to_plot:
        print(f"  Warning: No relevant terms found for {layer_name}, skipping heatmap")
        return
    
    n_terms = len(terms_to_plot)
    n_rows = output_dim * 3  # 3 types: predicted, GT, error
    
    # Create grid for interpolation
    x_grid = np.linspace(x_flat.min(), x_flat.max(), 200)
    t_grid = np.linspace(t_flat.min(), t_flat.max(), 200)
    X_grid, T_grid = np.meshgrid(x_grid, t_grid)
    
    fig, axes = plt.subplots(n_rows, n_terms, figsize=(6*n_terms, 4.5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_terms == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f'Derivative Heatmaps with Ground Truth - {layer_name}', 
                 fontsize=16, fontweight='bold')
    
    cmaps = ['viridis', 'plasma', 'inferno', 'magma']
    if output_dim == 1:
        comp_names = ['h(x,t)']
    elif output_dim == 2:
        comp_names = ['u (real)', 'v (imag)']
    else:
        comp_names = [f'h_{i}' for i in range(output_dim)]
    
    for col_idx, (term_key, term_gt_data) in enumerate(terms_to_plot):
        term_pred = results[term_key]  # (N, output_dim)
        term_error = np.abs(term_pred - term_gt_data)
        
        for comp_idx in range(output_dim):
            cmap = cmaps[comp_idx % len(cmaps)]
            
            # Predicted
            row = comp_idx
            pred_data = term_pred[:, comp_idx] if output_dim > 1 else term_pred.flatten()
            pred_grid = griddata((x_flat, t_flat), pred_data, (X_grid, T_grid),
                                 method='linear', fill_value=0.0)
            
            ax = axes[row, col_idx]
            if pred_grid.shape[0] < 2 or pred_grid.shape[1] < 2 or np.all(np.isnan(pred_grid)):
                ax.scatter(x_flat, t_flat, c=pred_data, cmap=cmap, s=5, alpha=0.5)
                ax.text(0.5, 0.95, '(Scatter)', ha='center', va='top', 
                       transform=ax.transAxes, fontsize=8, color='gray')
            else:
                im = ax.contourf(X_grid, T_grid, pred_grid, levels=50, cmap=cmap)
                plt.colorbar(im, ax=ax)
            
            if col_idx == 0:
                ax.set_ylabel(f'{comp_names[comp_idx]}\nPredicted\nt', fontsize=10, fontweight='bold')
            else:
                ax.set_ylabel('t', fontsize=10)
            if row == 0:
                ax.set_title(f'{term_key}', fontsize=12, fontweight='bold')
            
            # Ground truth
            row = output_dim + comp_idx
            gt_data = term_gt_data[:, comp_idx] if output_dim > 1 else term_gt_data.flatten()
            gt_grid = griddata((x_flat, t_flat), gt_data, (X_grid, T_grid),
                               method='linear', fill_value=0.0)
            
            ax = axes[row, col_idx]
            if gt_grid.shape[0] < 2 or gt_grid.shape[1] < 2 or np.all(np.isnan(gt_grid)):
                ax.scatter(x_flat, t_flat, c=gt_data, cmap=cmap, s=5, alpha=0.5)
                ax.text(0.5, 0.95, '(Scatter)', ha='center', va='top', 
                       transform=ax.transAxes, fontsize=8, color='gray')
            else:
                im = ax.contourf(X_grid, T_grid, gt_grid, levels=50, cmap=cmap)
                plt.colorbar(im, ax=ax)
            
            if col_idx == 0:
                ax.set_ylabel(f'{comp_names[comp_idx]}\nGround Truth\nt', fontsize=10, fontweight='bold')
            else:
                ax.set_ylabel('t', fontsize=10)
            
            # Error
            row = 2*output_dim + comp_idx
            error_data = term_error[:, comp_idx] if output_dim > 1 else term_error.flatten()
            error_grid = griddata((x_flat, t_flat), error_data, (X_grid, T_grid),
                                  method='linear', fill_value=0.0)
            
            ax = axes[row, col_idx]
            if error_grid.shape[0] < 2 or error_grid.shape[1] < 2 or np.all(np.isnan(error_grid)):
                ax.scatter(x_flat, t_flat, c=error_data, cmap='Reds', s=5, alpha=0.5)
                ax.text(0.5, 0.95, '(Scatter)', ha='center', va='top', 
                       transform=ax.transAxes, fontsize=8, color='gray')
            else:
                im = ax.contourf(X_grid, T_grid, error_grid, levels=50, cmap='Reds')
                plt.colorbar(im, ax=ax)
            
            if col_idx == 0:
                ax.set_ylabel(f'{comp_names[comp_idx]}\n|Error|\nt', fontsize=10, fontweight='bold')
            else:
                ax.set_ylabel('t', fontsize=10)
            ax.set_xlabel('x', fontsize=10)
    
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
    
    residual = derivatives_results[layer_name]['residual']  # (N, output_dim)
    
    # Determine output_dim
    output_dim = residual.shape[1] if len(residual.shape) > 1 else 1
    
    # Flatten x and t if needed
    x_flat = x.flatten() if isinstance(x, np.ndarray) else x
    t_flat = t.flatten() if isinstance(t, np.ndarray) else t
    
    # Create grid for interpolation
    x_grid = np.linspace(x_flat.min(), x_flat.max(), 200)
    t_grid = np.linspace(t_flat.min(), t_flat.max(), 200)
    X_grid, T_grid = np.meshgrid(x_grid, t_grid)
    
    fig, axes = plt.subplots(output_dim, 1, figsize=(12, 5*output_dim))
    if output_dim == 1:
        axes = [axes]
    
    fig.suptitle(f'Residual Heatmaps - {layer_name}', fontsize=16, fontweight='bold')
    
    if output_dim == 1:
        comp_names = ['Residual f']
    elif output_dim == 2:
        comp_names = ['f_u (real)', 'f_v (imag)']
    else:
        comp_names = [f'f_{i}' for i in range(output_dim)]
    
    for comp_idx in range(output_dim):
        f_comp = residual[:, comp_idx] if output_dim > 1 else residual.flatten()
        f_comp_grid = griddata((x_flat, t_flat), f_comp, (X_grid, T_grid),
                               method='linear', fill_value=0.0)
        
        ax = axes[comp_idx]
        if f_comp_grid.shape[0] < 2 or f_comp_grid.shape[1] < 2 or np.all(np.isnan(f_comp_grid)):
            ax.scatter(x_flat, t_flat, c=f_comp, cmap='RdBu_r', s=10, alpha=0.6)
            ax.text(0.5, 0.95, '(Scatter - insufficient data)', ha='center', va='top', 
                   transform=ax.transAxes, fontsize=9, color='gray')
        else:
            im = ax.contourf(X_grid, T_grid, f_comp_grid, levels=50, cmap='RdBu_r')
            plt.colorbar(im, ax=ax)
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('t', fontsize=12)
        ax.set_title(comp_names[comp_idx], fontsize=13, fontweight='bold')
    
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
    
    # Determine output_dim from first layer
    first_layer = layer_names[0]
    residual_sample = derivatives_results[first_layer]['residual']
    output_dim = residual_sample.shape[1] if len(residual_sample.shape) > 1 else 1
    
    # Store interpolated grids for all layers
    x_flat = x.flatten() if isinstance(x, np.ndarray) else x
    t_flat = t.flatten() if isinstance(t, np.ndarray) else t
    
    x_grid = np.linspace(x_flat.min(), x_flat.max(), 200)
    t_grid = np.linspace(t_flat.min(), t_flat.max(), 200)
    X_grid, T_grid = np.meshgrid(x_grid, t_grid)
    
    # Pre-compute all grids
    residual_grids = {}
    for layer_name in layer_names:
        residual = derivatives_results[layer_name]['residual']  # (N, output_dim)
        
        layer_grid = {}
        for comp_idx in range(output_dim):
            f_comp = residual[:, comp_idx] if output_dim > 1 else residual.flatten()
            f_comp_grid = griddata((x_flat, t_flat), f_comp, (X_grid, T_grid),
                                   method='linear', fill_value=0.0)
            layer_grid[f'f_{comp_idx}'] = f_comp_grid
        
        residual_grids[layer_name] = layer_grid
    
    # Create figure for changes (N-1 transitions, output_dim subplots each)
    # Layout: stack components vertically, arrange transitions to make figure square
    n_changes = n_layers - 1
    
    # Calculate optimal grid layout for roughly square figure
    # Each transition needs output_dim rows
    import math
    n_cols_transitions = max(1, int(math.ceil(math.sqrt(n_changes))))
    n_rows_groups = int(math.ceil(n_changes / n_cols_transitions))
    n_rows_total = n_rows_groups * output_dim  # output_dim rows per transition group
    
    fig, axes = plt.subplots(n_rows_total, n_cols_transitions,
                             figsize=(6 * n_cols_transitions, 5 * n_rows_groups))
    fig.suptitle('Residual Error Changes Between Layers\n'
                 '(Ratio: |layer_{i+1}| / |layer_i|, Green=Improved, Red=Worse)',
                 fontsize=14, fontweight='bold')
    
    # Ensure axes is 2D
    if n_rows_total == 1 and n_cols_transitions == 1:
        axes = np.array([[axes]])
    elif n_rows_total == 1:
        axes = axes.reshape(1, -1)
    elif n_cols_transitions == 1:
        axes = axes.reshape(-1, 1)
    
    # Define colormap normalization (center at 1.0)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=3.0)
    
    if output_dim == 1:
        comp_names = ['Residual f']
    elif output_dim == 2:
        comp_names = ['f_u (real)', 'f_v (imag)']
    else:
        comp_names = [f'f_{i}' for i in range(output_dim)]
    
    for idx in range(n_changes):
        layer_prev = layer_names[idx]
        layer_curr = layer_names[idx + 1]
        
        # Calculate position in grid
        col_idx = idx % n_cols_transitions
        row_group = idx // n_cols_transitions
        
        # Plot change for each output component
        for comp_idx in range(output_dim):
            prev_comp = residual_grids[layer_prev][f'f_{comp_idx}']
            curr_comp = residual_grids[layer_curr][f'f_{comp_idx}']
            
            # Compute ratio: abs(curr) / abs(prev)
            eps = 1e-10
            ratio = np.abs(curr_comp) / (np.abs(prev_comp) + eps)
            ratio = np.clip(ratio, 0, 5)  # Cap extreme values
            
            # Get subplot position
            row_idx = row_group * output_dim + comp_idx
            ax = axes[row_idx, col_idx]
            
            # Check if grid is valid for contourf
            if ratio.shape[0] < 2 or ratio.shape[1] < 2 or np.all(np.isnan(ratio)):
                # Skip this subplot if data is insufficient
                ax.text(0.5, 0.5, 'Insufficient data',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, color='gray')
                ax.axis('off')
                continue
            
            # Plot
            im = ax.contourf(X_grid, T_grid, ratio, levels=50,
                           cmap='RdYlGn_r', norm=norm)
            ax.set_xlabel('x', fontsize=11)
            ax.set_ylabel('t', fontsize=11)
            ax.set_title(f'{layer_prev} → {layer_curr}\n{comp_names[comp_idx]}',
                        fontsize=12, fontweight='bold')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Error Ratio', fontsize=10)
    
    # Hide unused subplots if any
    for row in range(n_rows_total):
        for col in range(n_cols_transitions):
            trans_idx = (row // output_dim) * n_cols_transitions + col
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
    save_dir: Path,
    config: Dict = None
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
    plot_term_magnitudes(eval_results, save_dir, config)
    
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
                save_dir,
                config
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

