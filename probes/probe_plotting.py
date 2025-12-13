"""Visualization for linear probe analysis."""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch


def plot_probe_metrics(probe_results: Dict, save_dir: Path):
    """
    Create line plots showing how probe metrics evolve across layers.
    
    Args:
        probe_results: Dictionary from probe_all_layers()
                      {layer_name: {'train_metrics', 'eval_metrics', ...}}
        save_dir: Directory to save plots
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract layer numbers and metrics
    layer_names = sorted(probe_results.keys())  # e.g., ['layer_1', 'layer_2', ...]
    layer_numbers = [int(name.split('_')[1]) for name in layer_names]
    
    train_rel_l2 = [probe_results[ln]['train_metrics']['rel_l2'] for ln in layer_names]
    train_inf_norm = [probe_results[ln]['train_metrics']['inf_norm'] for ln in layer_names]
    eval_rel_l2 = [probe_results[ln]['eval_metrics']['rel_l2'] for ln in layer_names]
    eval_inf_norm = [probe_results[ln]['eval_metrics']['inf_norm'] for ln in layer_names]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Relative L2 Error
    ax1.plot(layer_numbers, train_rel_l2, 'o-', linewidth=2, markersize=8, 
             label='Train', color='#3498db')
    ax1.plot(layer_numbers, eval_rel_l2, 's-', linewidth=2, markersize=8, 
             label='Eval', color='#e74c3c')
    ax1.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Relative L2 Error', fontsize=12, fontweight='bold')
    ax1.set_title('Linear Probe: Relative L2 Error vs Layer', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(layer_numbers)
    
    # Plot 2: Infinity Norm Error
    ax2.plot(layer_numbers, train_inf_norm, 'o-', linewidth=2, markersize=8, 
             label='Train', color='#3498db')
    ax2.plot(layer_numbers, eval_inf_norm, 's-', linewidth=2, markersize=8, 
             label='Eval', color='#e74c3c')
    ax2.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Infinity Norm Error', fontsize=12, fontweight='bold')
    ax2.set_title('Linear Probe: Infinity Norm Error vs Layer', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(layer_numbers)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = save_dir / "probe_metrics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Probe metrics plot saved to {plot_path}")


def plot_layer_dimensions(probe_results: Dict, save_dir: Path):
    """
    Plot the dimensionality of each layer alongside probe performance.
    
    Args:
        probe_results: Dictionary from probe_all_layers()
        save_dir: Directory to save plots
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    layer_names = sorted(probe_results.keys())
    layer_numbers = [int(name.split('_')[1]) for name in layer_names]
    dimensions = [probe_results[ln]['hidden_dim'] for ln in layer_names]
    eval_rel_l2 = [probe_results[ln]['eval_metrics']['rel_l2'] for ln in layer_names]
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot dimensions on left axis
    color = '#2ecc71'
    ax1.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Layer Dimension', fontsize=12, fontweight='bold', color=color)
    ax1.bar(layer_numbers, dimensions, alpha=0.6, color=color, label='Dimension')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(layer_numbers)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot performance on right axis
    ax2 = ax1.twinx()
    color = '#e74c3c'
    ax2.set_ylabel('Probe Eval Rel-L2 Error', fontsize=12, fontweight='bold', color=color)
    ax2.plot(layer_numbers, eval_rel_l2, 'o-', linewidth=2, markersize=8, 
             color=color, label='Eval Error')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Layer Dimension vs Probe Performance', fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    # Save plot
    plot_path = save_dir / "probe_dimensions.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Layer dimensions plot saved to {plot_path}")


def plot_probe_error_heatmaps(
    probe_results: Dict,
    eval_x: np.ndarray,
    eval_t: np.ndarray,
    eval_targets: np.ndarray,
    save_dir: Path
):
    """
    Generate error heatmaps for probe predictions (similar to residual heatmaps).
    
    Args:
        probe_results: Dictionary from probe_all_layers()
        eval_x: Spatial coordinates (N,)
        eval_t: Temporal coordinates (N,)
        eval_targets: Ground truth outputs (N, 2) - [u, v]
        save_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    
    save_dir.mkdir(parents=True, exist_ok=True)
    layer_names = sorted(probe_results.keys())
    
    for idx, layer_name in enumerate(layer_names, 1):
        print(f"  Generating error heatmaps for {layer_name} ({idx}/{len(layer_names)})")
        
        # Get predictions for this layer (ensure it's on CPU and numpy)
        eval_preds = probe_results[layer_name]['eval_predictions']  # (N, output_dim)
        if isinstance(eval_preds, torch.Tensor):
            eval_preds = eval_preds.cpu().numpy()
        output_dim = eval_preds.shape[1]
        
        # Compute errors
        error = eval_preds - eval_targets  # (N, output_dim)
        
        # Create grid for interpolation
        x_min, x_max = eval_x.min(), eval_x.max()
        t_min, t_max = eval_t.min(), eval_t.max()
        grid_x, grid_t = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(t_min, t_max, 200)
        )
        
        points = np.column_stack([eval_x, eval_t])
        
        # Create figure with output_dim subplots
        fig, axes = plt.subplots(output_dim, 1, figsize=(12, 5*output_dim))
        if output_dim == 1:
            axes = [axes]
        
        # Plot error for each output component
        if output_dim == 1:
            component_names = ['h(x,t)']
        elif output_dim == 2:
            component_names = ['u (real)', 'v (imag)']
        else:
            component_names = [f'h_{i}' for i in range(output_dim)]
        
        for i in range(output_dim):
            error_i = error[:, i]
            grid_error_i = griddata(points, error_i, (grid_x, grid_t),
                                    method='linear', fill_value=0.0)
            
            # Check if grid is valid for contourf
            if grid_error_i.shape[0] < 2 or grid_error_i.shape[1] < 2 or np.all(np.isnan(grid_error_i)):
                # Fall back to scatter plot
                axes[i].scatter(eval_x, eval_t, c=error_i, cmap='RdBu_r', s=10, alpha=0.6)
                axes[i].text(0.5, 0.95, '(Scatter - insufficient data for contour)',
                           ha='center', va='top', transform=axes[i].transAxes,
                           fontsize=9, color='gray')
            else:
                im = axes[i].contourf(grid_x, grid_t, grid_error_i, levels=50, cmap='RdBu_r')
                plt.colorbar(im, ax=axes[i])
            
            axes[i].set_xlabel('x', fontsize=11)
            axes[i].set_ylabel('t', fontsize=11)
            axes[i].set_title(f'Probe Error - {component_names[i]}', fontsize=12, fontweight='bold')
        
        # Overall title
        fig.suptitle(f'Probe Prediction Error Heatmaps - {layer_name}', 
                     fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        save_path = save_dir / f'probe_error_heatmaps_{layer_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    print(f"  Probe error heatmaps saved to {save_dir}")


def plot_probe_error_change_heatmaps(
    probe_results: Dict,
    eval_x: np.ndarray,
    eval_t: np.ndarray,
    eval_targets: np.ndarray,
    save_dir: Path
):
    """
    Plot probe error change heatmaps showing how errors evolve between layers.
    
    For each transition, computes: abs(error_{i+1}) / abs(error_i)
    - 0.0 (green): Error eliminated
    - 1.0 (white): No change
    - >1.0 (red): Error increased
    
    Args:
        probe_results: Dictionary from probe_all_layers()
        eval_x: Spatial coordinates (N,)
        eval_t: Temporal coordinates (N,)
        eval_targets: Ground truth outputs (N, 2) - [u, v]
        save_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from matplotlib.colors import TwoSlopeNorm
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    layer_names = sorted(probe_results.keys(),
                        key=lambda x: int(x.split('_')[1]))
    n_layers = len(layer_names)
    
    if n_layers < 2:
        return  # Need at least 2 layers for changes
    
    # Pre-compute error grids for all layers
    x_min, x_max = eval_x.min(), eval_x.max()
    t_min, t_max = eval_t.min(), eval_t.max()
    grid_x, grid_t = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(t_min, t_max, 200)
    )
    
    points = np.column_stack([eval_x, eval_t])
    error_grids = {}
    
    # Determine output dimension from first layer
    first_layer = layer_names[0]
    output_dim = probe_results[first_layer]['eval_predictions'].shape[1]
    
    for layer_name in layer_names:
        eval_preds = probe_results[layer_name]['eval_predictions']  # (N, output_dim)
        if isinstance(eval_preds, torch.Tensor):
            eval_preds = eval_preds.cpu().numpy()
        error = eval_preds - eval_targets  # (N, output_dim)
        
        layer_grid = {}
        for i in range(output_dim):
            error_i = error[:, i]
            grid_error_i = griddata(points, error_i, (grid_x, grid_t),
                                    method='linear', fill_value=0.0)
            layer_grid[f'error_{i}'] = grid_error_i
        
        error_grids[layer_name] = layer_grid
    
    # Create figure for changes (N-1 transitions, output_dim subplots each)
    # Layout: stack components vertically, arrange transitions to make figure square
    n_changes = n_layers - 1
    
    # Calculate optimal grid layout for roughly square figure
    # Each transition needs output_dim rows (one per component)
    import math
    n_cols_transitions = max(1, int(math.ceil(math.sqrt(n_changes))))
    n_rows_groups = int(math.ceil(n_changes / n_cols_transitions))
    n_rows_total = n_rows_groups * output_dim  # output_dim rows per transition group
    
    fig, axes = plt.subplots(n_rows_total, n_cols_transitions,
                             figsize=(6 * n_cols_transitions, 5 * n_rows_groups))
    fig.suptitle('Probe Error Changes Between Layers\n'
                 '(Ratio: |error_{i+1}| / |error_i|, Green=Improved, Red=Worse)',
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
    
    # Component names
    if output_dim == 1:
        component_names = ['h(x,t)']
    elif output_dim == 2:
        component_names = ['u (real)', 'v (imag)']
    else:
        component_names = [f'h_{i}' for i in range(output_dim)]
    
    for idx in range(n_changes):
        layer_prev = layer_names[idx]
        layer_curr = layer_names[idx + 1]
        
        # Calculate position in grid
        col_idx = idx % n_cols_transitions
        row_group = idx // n_cols_transitions
        
        # Plot change for each output component
        for comp_idx in range(output_dim):
            prev_error = error_grids[layer_prev][f'error_{comp_idx}']
            curr_error = error_grids[layer_curr][f'error_{comp_idx}']
            
            # Compute ratio: abs(curr) / abs(prev)
            eps = 1e-10
            ratio = np.abs(curr_error) / (np.abs(prev_error) + eps)
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
            im = ax.contourf(grid_x, grid_t, ratio, levels=50,
                           cmap='RdYlGn_r', norm=norm)
            ax.set_xlabel('x', fontsize=11)
            ax.set_ylabel('t', fontsize=11)
            ax.set_title(f'{layer_prev} → {layer_curr}\n{component_names[comp_idx]} error',
                        fontsize=12, fontweight='bold')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Error Ratio', fontsize=10)
    
    # Hide unused subplots if any
    for row in range(n_rows_total):
        for col in range(n_cols_transitions):
            trans_idx = (row // 2) * n_cols_transitions + col
            if trans_idx >= n_changes:
                axes[row, col].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = save_dir / 'probe_error_change_heatmaps.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Probe error change heatmaps saved to {save_path}")


def generate_all_probe_plots(
    probe_results: Dict, 
    save_dir: Path,
    eval_x: np.ndarray = None,
    eval_t: np.ndarray = None,
    eval_targets: np.ndarray = None
):
    """
    Generate all probe visualization plots.
    
    Args:
        probe_results: Dictionary from probe_all_layers()
        save_dir: Directory to save plots
        eval_x: Optional spatial coordinates for error heatmaps
        eval_t: Optional temporal coordinates for error heatmaps
        eval_targets: Optional ground truth for error heatmaps
    """
    print("\nGenerating probe plots...")
    plot_probe_metrics(probe_results, save_dir)
    plot_layer_dimensions(probe_results, save_dir)
    
    # Generate error heatmaps if data is provided
    if eval_x is not None and eval_t is not None and eval_targets is not None:
        plot_probe_error_heatmaps(probe_results, eval_x, eval_t, eval_targets, save_dir)
        # Also generate change heatmaps
        plot_probe_error_change_heatmaps(probe_results, eval_x, eval_t, eval_targets, save_dir)
    
    print("  All probe plots generated")


def plot_probe_history_shaded(history: List[tuple], save_dir: Path) -> None:
    """
    Overlay probe metrics across epochs with shaded progression.
    history: list of (epoch, metrics_summary) where metrics_summary matches run_probes return.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    history = sorted(history, key=lambda x: x[0])
    base_color = '#e74c3c'
    alphas = [min(0.45 + 0.15 * idx, 1.0) for idx in range(len(history))]

    # Rel-L2
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for (epoch, metrics), alpha in zip(history, alphas):
        layers = list(range(1, len(metrics['train']['rel_l2']) + 1))
        axes[0].plot(layers, metrics['train']['rel_l2'], marker='o', color=base_color, alpha=alpha, label=f"Epoch {epoch}")
        axes[1].plot(layers, metrics['eval']['rel_l2'], marker='s', color=base_color, alpha=alpha, label=f"Epoch {epoch}")
    axes[0].set_title('Train Rel-L2 (shaded)')
    axes[1].set_title('Eval Rel-L2 (shaded)')
    for ax in axes:
        ax.set_xlabel('Layer')
        ax.set_ylabel('Rel-L2')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(save_dir / "probe_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()

    # L_inf
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for (epoch, metrics), alpha in zip(history, alphas):
        layers = list(range(1, len(metrics['train']['inf_norm']) + 1))
        axes[0].plot(layers, metrics['train']['inf_norm'], marker='o', color=base_color, alpha=alpha, label=f"Epoch {epoch}")
        axes[1].plot(layers, metrics['eval']['inf_norm'], marker='s', color=base_color, alpha=alpha, label=f"Epoch {epoch}")
    axes[0].set_title('Train L∞ (shaded)')
    axes[1].set_title('Eval L∞ (shaded)')
    for ax in axes:
        ax.set_xlabel('Layer')
        ax.set_ylabel('L∞')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(save_dir / "probe_metrics_inf.png", dpi=150, bbox_inches='tight')
    plt.close()

