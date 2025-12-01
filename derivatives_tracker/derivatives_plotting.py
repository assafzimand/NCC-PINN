"""Plotting utilities for derivatives tracking."""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from pathlib import Path
from typing import Dict, List
from scipy.interpolate import griddata


def plot_residual_evolution(
    derivatives_results: Dict[str, Dict],
    save_dir: Path
) -> None:
    """
    Plot residual L2 norm evolution across layers.
    
    Expected to decrease as we go deeper (approaching physics solution).
    
    Args:
        derivatives_results: Dict mapping layer_name -> results dict
        save_dir: Directory to save plot
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract layer names and residual norms
    layer_names = sorted(derivatives_results.keys())
    residual_norms = [derivatives_results[ln]['norms']['residual_norm'] 
                      for ln in layer_names]
    
    # Create layer indices for x-axis
    layer_indices = list(range(1, len(layer_names) + 1))
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(layer_indices, residual_norms, marker='o', linewidth=2, 
            markersize=8, color='crimson', label='Residual L2 Norm')
    
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Residual L2 Norm', fontsize=12, fontweight='bold')
    ax.set_title('Residual Evolution Across Layers', fontsize=14, fontweight='bold')
    ax.set_xticks(layer_indices)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    save_path = save_dir / 'residual_evolution.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Residual evolution plot saved to {save_path}")


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


def plot_real_imag_combined(
    derivatives_results: Dict[str, Dict],
    save_dir: Path
) -> None:
    """
    Plot real and imaginary components of all terms in one plot.
    
    Single plot with 10 lines:
    - 5 colors (one per term: h, h_t, h_xx, nonlinear, residual)
    - Solid lines = real components
    - Dashed lines = imaginary components
    
    Args:
        derivatives_results: Dict mapping layer_name -> results dict
        save_dir: Directory to save plot
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    layer_names = sorted(derivatives_results.keys())
    layer_indices = list(range(1, len(layer_names) + 1))
    
    # Define colors for each term
    colors = {
        'h': 'blue',
        'h_t': 'green',
        'h_xx': 'orange',
        'nonlinear': 'purple',
        'residual': 'crimson'
    }
    
    # Extract mean components
    def get_mean_components(term_name):
        """Get mean real and imaginary components for a term."""
        real_means = []
        imag_means = []
        for ln in layer_names:
            term_data = derivatives_results[ln][term_name]  # (N, 2)
            real_means.append(np.mean(term_data[:, 0]))  # u component
            imag_means.append(np.mean(term_data[:, 1]))  # v component
        return real_means, imag_means
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for term_name, color in colors.items():
        real_means, imag_means = get_mean_components(term_name)
        
        # Solid line for real component
        ax.plot(layer_indices, real_means, color=color, linestyle='-', 
                linewidth=2, marker='o', label=f'{term_name} (real)')
        
        # Dashed line for imaginary component
        ax.plot(layer_indices, imag_means, color=color, linestyle='--', 
                linewidth=2, marker='s', label=f'{term_name} (imag)')
    
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Component Value', fontsize=12, fontweight='bold')
    ax.set_title('Real and Imaginary Component Evolution', fontsize=14, fontweight='bold')
    ax.set_xticks(layer_indices)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2, loc='best')
    
    plt.tight_layout()
    
    save_path = save_dir / 'real_imag_components.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Real/imaginary components plot saved to {save_path}")


def plot_derivative_heatmaps(
    derivatives_results: Dict[str, Dict],
    layer_name: str,
    x: np.ndarray,
    t: np.ndarray,
    save_dir: Path
) -> None:
    """
    Plot heatmaps of derivatives for a specific layer.
    
    Creates 2x3 grid:
    - Row 1: u-component of (h, h_t, h_xx)
    - Row 2: v-component of (h, h_t, h_xx)
    
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
        print(f"  Warning: Layer {layer_name} not found, skipping heatmap")
        return
    
    results = derivatives_results[layer_name]
    
    # Flatten x and t if needed
    x_flat = x.flatten() if isinstance(x, np.ndarray) else x
    t_flat = t.flatten() if isinstance(t, np.ndarray) else t
    
    # Create grid for interpolation
    x_grid = np.linspace(x_flat.min(), x_flat.max(), 200)
    t_grid = np.linspace(t_flat.min(), t_flat.max(), 200)
    X_grid, T_grid = np.meshgrid(x_grid, t_grid)
    
    # Terms to plot
    terms = ['h', 'h_t', 'h_xx']
    term_labels = ['h', 'h_t', 'h_xx']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Derivative Heatmaps - {layer_name}', fontsize=16, fontweight='bold')
    
    for col_idx, (term, label) in enumerate(zip(terms, term_labels)):
        term_data = results[term]  # (N, 2)
        
        # Row 0: u-component (real)
        u_data = term_data[:, 0]
        u_grid = griddata((x_flat, t_flat), u_data, (X_grid, T_grid), method='cubic')
        
        ax = axes[0, col_idx]
        im = ax.contourf(X_grid, T_grid, u_grid, levels=50, cmap='viridis')
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('t', fontsize=11)
        ax.set_title(f'{label}_u (real)', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax)
        
        # Row 1: v-component (imaginary)
        v_data = term_data[:, 1]
        v_grid = griddata((x_flat, t_flat), v_data, (X_grid, T_grid), method='cubic')
        
        ax = axes[1, col_idx]
        im = ax.contourf(X_grid, T_grid, v_grid, levels=50, cmap='plasma')
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('t', fontsize=11)
        ax.set_title(f'{label}_v (imag)', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    save_path = save_dir / f'derivative_heatmaps_{layer_name}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Derivative heatmaps for {layer_name} saved to {save_path}")


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


def plot_residual_balance(
    derivatives_results: Dict[str, Dict],
    save_dir: Path
) -> None:
    """
    Plot residual term contributions as stacked bars.
    
    Shows relative magnitudes of: ||i*h_t||, ||0.5*h_xx||, |||h|²h||
    
    Args:
        derivatives_results: Dict mapping layer_name -> results dict
        save_dir: Directory to save plot
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    layer_names = sorted(derivatives_results.keys())
    layer_indices = np.arange(len(layer_names))
    
    # Extract term magnitudes (these correspond to the residual components)
    h_t_norms = [derivatives_results[ln]['norms']['h_t_norm'] for ln in layer_names]
    h_xx_norms = [0.5 * derivatives_results[ln]['norms']['h_xx_norm'] for ln in layer_names]
    nonlinear_norms = [derivatives_results[ln]['norms']['nonlinear_norm'] for ln in layer_names]
    
    # Plot grouped bars
    fig, ax = plt.subplots(figsize=(12, 7))
    
    width = 0.25
    ax.bar(layer_indices - width, h_t_norms, width, label='||h_t||', color='green', alpha=0.8)
    ax.bar(layer_indices, h_xx_norms, width, label='||0.5*h_xx||', color='orange', alpha=0.8)
    ax.bar(layer_indices + width, nonlinear_norms, width, label='|||h|²h||', color='purple', alpha=0.8)
    
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Term Magnitude', fontsize=12, fontweight='bold')
    ax.set_title('Residual Term Contributions Per Layer', fontsize=14, fontweight='bold')
    ax.set_xticks(layer_indices)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    save_path = save_dir / 'residual_balance.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Residual balance plot saved to {save_path}")


def generate_all_derivative_plots(
    derivatives_results: Dict[str, Dict],
    x: np.ndarray,
    t: np.ndarray,
    save_dir: Path
) -> None:
    """
    Generate all derivative visualization plots.
    
    Args:
        derivatives_results: Dict mapping layer_name -> results dict
        x: Spatial coordinates
        t: Temporal coordinates
        save_dir: Directory to save plots
    """
    print("\nGenerating derivative plots...")
    
    # Plot 1: Residual evolution
    plot_residual_evolution(derivatives_results, save_dir)
    
    # Plot 2: Term magnitudes
    plot_term_magnitudes(derivatives_results, save_dir)
    
    # Plot 3: Real/imaginary combined
    plot_real_imag_combined(derivatives_results, save_dir)
    
    # Plot 4 & 5: Heatmaps for selected layers (first, middle, last)
    layer_names = sorted(derivatives_results.keys())
    if len(layer_names) > 0:
        # First layer
        plot_derivative_heatmaps(derivatives_results, layer_names[0], x, t, save_dir)
        plot_residual_heatmaps(derivatives_results, layer_names[0], x, t, save_dir)
        
        # Middle layer
        if len(layer_names) > 2:
            mid_idx = len(layer_names) // 2
            plot_derivative_heatmaps(derivatives_results, layer_names[mid_idx], x, t, save_dir)
            plot_residual_heatmaps(derivatives_results, layer_names[mid_idx], x, t, save_dir)
        
        # Last layer
        if len(layer_names) > 1:
            plot_derivative_heatmaps(derivatives_results, layer_names[-1], x, t, save_dir)
            plot_residual_heatmaps(derivatives_results, layer_names[-1], x, t, save_dir)
    
    # Plot 6: Residual balance
    plot_residual_balance(derivatives_results, save_dir)
    
    print(f"All derivative plots generated in {save_dir}")

