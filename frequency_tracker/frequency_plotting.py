"""Plotting utilities for frequency tracking."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from frequency_tracker.frequency_core import compute_binned_frequency_errors
from matplotlib.colors import LogNorm


def _safe_log_scale(ax, values_list):
    """
    Set log scale on y-axis only if all data has positive values.
    
    Returns:
        bool: True if log scale was applied, False if linear scale is used.
    """
    all_values = []
    for v in values_list:
        if isinstance(v, (list, np.ndarray)):
            all_values.extend(np.array(v).flatten())
        else:
            all_values.append(v)
    all_values = np.array(all_values)
    # Filter out NaN and zero values for the check
    valid_values = all_values[~np.isnan(all_values) & (all_values > 0)]
    if len(valid_values) > 0:
        ax.set_yscale('log')
        return True
    return False


def plot_learned_frequencies(
    freq_results: Dict[str, Dict],
    h_gt_spectrum: Dict,
    save_dir: Path,
    config: Dict
) -> None:
    """
    Generate the unified learned_frequencies visualization.
    
    Creates one figure with all layers overlaid:
    - Rows: Cumulative, Added, Leftover
    - Columns: k_x0, k_x1, k_t, |k| (or k_x, k_t, |k| for 2D)
    
    Args:
        freq_results: Dictionary mapping layer_name -> results dict
        h_gt_spectrum: Ground truth spectrum dict
        save_dir: Directory to save plot
        config: Configuration dictionary
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    problem = config['problem']
    problem_cfg = config[problem]
    spatial_dim = problem_cfg.get('spatial_dim', 1)
    
    layer_names = sorted(freq_results.keys())
    n_layers = len(layer_names)
    
    # Determine column names based on spatial_dim
    if spatial_dim == 2:
        dim_names = ['x0', 'x1', 't', 'radial']
        n_cols = 4
    else:
        dim_names = ['x', 't', 'radial']
        n_cols = 3
    
    n_rows = 3  # Cumulative, Added, Leftover
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Use colormap for layers
    cmap = plt.cm.viridis
    colors = [cmap(i / max(n_layers - 1, 1)) for i in range(n_layers)]
    
    # Compute binned errors for each layer and perspective
    for row_idx, perspective in enumerate(['cumulative', 'added', 'leftover']):
        for col_idx, dim_name in enumerate(dim_names):
            ax = axes[row_idx, col_idx]
            
            all_errors = []
            
            for layer_idx, layer_name in enumerate(layer_names):
                layer_data = freq_results[layer_name]
                
                if perspective == 'cumulative':
                    power_pred = layer_data['cumulative']['power']
                    power_gt = h_gt_spectrum['power']
                    freqs = layer_data['cumulative']['freqs']
                elif perspective == 'added':
                    power_pred = layer_data['added']['power']
                    # For "added", compare what layer added vs what remains to be learned
                    # Reference: what should have been added = remaining error from previous layer
                    if layer_idx == 0:
                        # First layer: compare against ground truth (everything needs to be learned)
                        power_gt = h_gt_spectrum['power']
                    else:
                        prev_layer = layer_names[layer_idx - 1]
                        prev_leftover = freq_results[prev_layer]['leftover']['power']
                        # What should be added = leftover from previous layer
                        power_gt = np.maximum(prev_leftover, 1e-10)
                    freqs = layer_data['added']['freqs']
                else:  # leftover
                    # Leftover: power of (h_gt - h_pred)
                    power_pred = layer_data['leftover']['power']
                    # Normalize by ground truth to get relative leftover
                    power_gt = h_gt_spectrum['power']
                    freqs = layer_data['leftover']['freqs']
                
                # Compute binned errors
                binned_errors = compute_binned_frequency_errors(
                    power_pred=power_pred,
                    power_gt=power_gt,
                    freqs=freqs,
                    spatial_dim=spatial_dim,
                    n_bins=20,
                    is_leftover=(perspective == 'leftover')
                )
                
                bin_centers, mean_error = binned_errors[dim_name]
                all_errors.append(mean_error)
                
                # Plot continuous line
                ax.plot(bin_centers, mean_error, color=colors[layer_idx], 
                       label=layer_name, linewidth=2, alpha=0.8)
            
            # Formatting
            if dim_name == 'radial':
                ax.set_xlabel('|k| (Radial Frequency)', fontsize=11)
            else:
                ax.set_xlabel(f'k_{dim_name}', fontsize=11)
            
            ax.set_ylabel('Mean Relative Error', fontsize=11)
            
            # Try log scale
            is_log = _safe_log_scale(ax, all_errors)
            scale_str = "[log]" if is_log else "[linear]"
            
            # Title
            title_map = {
                'cumulative': 'Approximation Error (up to layer i)\n(model output vs ground truth)',
                'added': 'Layer Contribution\n(what layer i added)',
                'leftover': 'Remaining Gap (after layer i)\n(what\'s left to learn)'
            }
            ax.set_title(f'{title_map[perspective]} - {dim_name} {scale_str}', 
                        fontsize=11, fontweight='bold')
            
            ax.grid(True, alpha=0.3)
            
            # Legend only on first subplot of each row
            if col_idx == 0:
                ax.legend(loc='upper right', fontsize=8, ncol=1)
    
    plt.suptitle('Frequency Learning Analysis - Relative Error by Frequency Band', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    save_path = save_dir / 'learned_frequencies.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Learned frequencies plot saved to {save_path}")


def plot_spectral_learning_efficiency(
    freq_results: Dict[str, Dict],
    h_gt_spectrum: Dict,
    save_dir: Path,
    config: Dict
) -> None:
    """
    Generate 2D heatmap showing spectral learning efficiency.
    
    X-axis: Layer index
    Y-axis: Radial frequency |k| (binned)
    Color: Leftover error at (layer, |k|) - lower = better learned
    
    Args:
        freq_results: Dictionary mapping layer_name -> results dict
        h_gt_spectrum: Ground truth spectrum dict
        save_dir: Directory to save plot
        config: Configuration dictionary
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    layer_names = sorted(freq_results.keys())
    n_layers = len(layer_names)
    
    # Get radial frequency bins and leftover errors for each layer
    n_bins = 20
    all_k_radial = []
    all_leftover_errors = []
    
    for layer_name in layer_names:
        layer_data = freq_results[layer_name]
        leftover = layer_data['leftover']
        
        # Compute binned errors for radial spectrum
        binned_errors = compute_binned_frequency_errors(
            power_pred=leftover['power'],
            power_gt=h_gt_spectrum['power'],
            freqs=leftover['freqs'],
            spatial_dim=config[config['problem']].get('spatial_dim', 1),
            n_bins=n_bins,
            is_leftover=True
        )
        
        k_radial, mean_error = binned_errors['radial']
        all_k_radial.append(k_radial)
        all_leftover_errors.append(mean_error)
    
    # Create 2D array: rows = frequency bins, cols = layers
    # Use the first layer's k_radial as reference (all should be same)
    k_radial_ref = all_k_radial[0]
    n_freq_bins = len(k_radial_ref)
    
    error_matrix = np.zeros((n_freq_bins, n_layers))
    for layer_idx, mean_error in enumerate(all_leftover_errors):
        error_matrix[:, layer_idx] = mean_error
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Layer indices for x-axis
    layer_indices = list(range(1, n_layers + 1))
    
    # Create heatmap
    im = ax.imshow(error_matrix, aspect='auto', cmap='viridis_r', 
                   interpolation='bilinear', origin='lower')
    
    # Set ticks
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    
    # Y-axis: frequency bins (show every 5th or so)
    y_ticks = np.linspace(0, n_freq_bins - 1, min(10, n_freq_bins))
    y_tick_labels = [f'{k_radial_ref[int(i)]:.2f}' for i in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    ax.set_ylabel('Radial Frequency |k|', fontsize=12, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Remaining Error (lower = better learned)', 
                   fontsize=11, rotation=270, labelpad=20)
    
    # Try log scale for colorbar
    if np.all(error_matrix > 0):
        im.set_norm(LogNorm(vmin=error_matrix[error_matrix > 0].min(), 
                           vmax=error_matrix.max()))
        cbar.set_label('Remaining Error (log scale) [log]', 
                       fontsize=11, rotation=270, labelpad=20)
    
    ax.set_title('Spectral Learning Efficiency\n(Layer vs Frequency: Remaining Error)', 
                 fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    save_path = save_dir / 'spectral_learning_efficiency.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Spectral learning efficiency plot saved to {save_path}")


def generate_all_frequency_plots(
    freq_results: Dict[str, Dict],
    h_gt_spectrum: Dict,
    save_dir: Path,
    config: Dict
) -> None:
    """
    Generate all frequency analysis plots.
    
    Args:
        freq_results: Dictionary mapping layer_name -> results dict
        h_gt_spectrum: Ground truth spectrum dict
        save_dir: Directory to save plots
        config: Configuration dictionary
    """
    print("  Generating frequency plots...")
    
    # Main learned frequencies plot
    plot_learned_frequencies(
        freq_results=freq_results,
        h_gt_spectrum=h_gt_spectrum,
        save_dir=save_dir,
        config=config
    )
    
    # Spectral learning efficiency heatmap
    plot_spectral_learning_efficiency(
        freq_results=freq_results,
        h_gt_spectrum=h_gt_spectrum,
        save_dir=save_dir,
        config=config
    )
    
    print(f"  All frequency plots saved to {save_dir}")
