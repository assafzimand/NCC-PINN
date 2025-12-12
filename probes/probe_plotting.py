"""Visualization for linear probe analysis."""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import numpy as np


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
        
        # Get predictions for this layer
        eval_preds = probe_results[layer_name]['eval_predictions']  # (N, 2)
        
        # Compute errors
        error = eval_preds - eval_targets  # (N, 2)
        error_u = error[:, 0]  # real part error
        error_v = error[:, 1]  # imaginary part error
        
        # Create grid for interpolation
        x_min, x_max = eval_x.min(), eval_x.max()
        t_min, t_max = eval_t.min(), eval_t.max()
        grid_x, grid_t = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(t_min, t_max, 200)
        )
        
        # Interpolate errors onto grid
        points = np.column_stack([eval_x, eval_t])
        grid_error_u = griddata(points, error_u, (grid_x, grid_t), method='cubic', fill_value=0)
        grid_error_v = griddata(points, error_v, (grid_x, grid_t), method='cubic', fill_value=0)
        
        # Create figure with 2 subplots (u and v errors)
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot real part error
        im1 = axes[0].contourf(grid_x, grid_t, grid_error_u, levels=50, cmap='RdBu_r')
        axes[0].set_xlabel('x', fontsize=11)
        axes[0].set_ylabel('t', fontsize=11)
        axes[0].set_title(f'Probe Error - u (real)', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot imaginary part error
        im2 = axes[1].contourf(grid_x, grid_t, grid_error_v, levels=50, cmap='RdBu_r')
        axes[1].set_xlabel('x', fontsize=11)
        axes[1].set_ylabel('t', fontsize=11)
        axes[1].set_title(f'Probe Error - v (imag)', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=axes[1])
        
        # Overall title
        fig.suptitle(f'Probe Prediction Error Heatmaps - {layer_name}', 
                     fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        save_path = save_dir / f'probe_error_heatmaps_{layer_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    print(f"  Probe error heatmaps saved to {save_dir}")


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

