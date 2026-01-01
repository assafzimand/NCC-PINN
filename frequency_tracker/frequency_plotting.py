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
    
    Shows absolute power spectrum (log scale) for each perspective.
    
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
    
    # Also add ground truth as reference
    gt_color = 'red'
    
    # Compute marginal spectra for each layer and perspective
    n_bins = 30  # Number of bins for marginal spectra
    
    for row_idx, perspective in enumerate(['cumulative', 'added', 'leftover']):
        for col_idx, dim_name in enumerate(dim_names):
            ax = axes[row_idx, col_idx]
            
            all_powers = []
            
            # Plot ground truth as reference (only for cumulative row)
            if perspective == 'cumulative' and col_idx == 0:
                # Will add to legend
                pass
            
            for layer_idx, layer_name in enumerate(layer_names):
                layer_data = freq_results[layer_name]
                
                if perspective == 'cumulative':
                    spectrum = layer_data['cumulative']
                elif perspective == 'added':
                    spectrum = layer_data['added']
                else:  # leftover
                    spectrum = layer_data['leftover']
                
                power = spectrum['power']
                freqs = spectrum['freqs']
                n_dims = len(freqs)
                
                # Handle multi-output: average over output dimension first
                if power.ndim > n_dims:
                    power_avg = power.mean(axis=-1)
                else:
                    power_avg = power
                
                # Compute marginal spectrum for the specified dimension
                if dim_name == 'radial':
                    # Radial spectrum
                    mesh_freqs = np.meshgrid(*freqs, indexing='ij')
                    k_magnitude = np.sqrt(sum(f**2 for f in mesh_freqs))
                    
                    k_max = k_magnitude.max()
                    if k_max == 0:
                        k_max = 1.0
                    k_bin_edges = np.linspace(0, k_max, n_bins + 1)
                    k_bin_centers = (k_bin_edges[:-1] + k_bin_edges[1:]) / 2
                    
                    power_binned = np.zeros(n_bins)
                    for bin_idx in range(n_bins):
                        mask = (k_magnitude >= k_bin_edges[bin_idx]) & (k_magnitude < k_bin_edges[bin_idx + 1])
                        if bin_idx == n_bins - 1:
                            mask = (k_magnitude >= k_bin_edges[bin_idx]) & (k_magnitude <= k_bin_edges[bin_idx + 1])
                        if mask.sum() > 0:
                            power_binned[bin_idx] = power_avg[mask].mean()
                    
                    bin_centers = k_bin_centers
                    power_1d = power_binned
                else:
                    # Marginal spectrum for specific dimension
                    if spatial_dim == 2:
                        dim_idx = {'x0': 0, 'x1': 1, 't': 2}.get(dim_name, 0)
                    else:
                        dim_idx = {'x': 0, 't': 1}.get(dim_name, 0)
                    
                    freq_1d = freqs[dim_idx]
                    axes_to_avg = tuple(j for j in range(n_dims) if j != dim_idx)
                    if axes_to_avg:
                        power_1d = power_avg.mean(axis=axes_to_avg)
                    else:
                        power_1d = power_avg
                    bin_centers = freq_1d
                
                all_powers.append(power_1d)
                
                # Plot continuous line
                ax.plot(bin_centers, power_1d, color=colors[layer_idx], 
                       label=layer_name, linewidth=2, alpha=0.8)
            
            # Plot ground truth reference for cumulative only
            if perspective == 'cumulative':
                gt_power = h_gt_spectrum['power']
                gt_freqs = h_gt_spectrum['freqs']
                n_dims_gt = len(gt_freqs)
                if gt_power.ndim > n_dims_gt:
                    gt_power_avg = gt_power.mean(axis=-1)
                else:
                    gt_power_avg = gt_power
                
                if dim_name == 'radial':
                    mesh_freqs = np.meshgrid(*gt_freqs, indexing='ij')
                    k_magnitude = np.sqrt(sum(f**2 for f in mesh_freqs))
                    k_max = k_magnitude.max()
                    if k_max == 0:
                        k_max = 1.0
                    k_bin_edges = np.linspace(0, k_max, n_bins + 1)
                    k_bin_centers = (k_bin_edges[:-1] + k_bin_edges[1:]) / 2
                    
                    gt_binned = np.zeros(n_bins)
                    for bin_idx in range(n_bins):
                        mask = (k_magnitude >= k_bin_edges[bin_idx]) & (k_magnitude < k_bin_edges[bin_idx + 1])
                        if bin_idx == n_bins - 1:
                            mask = (k_magnitude >= k_bin_edges[bin_idx]) & (k_magnitude <= k_bin_edges[bin_idx + 1])
                        if mask.sum() > 0:
                            gt_binned[bin_idx] = gt_power_avg[mask].mean()
                    
                    ax.plot(k_bin_centers, gt_binned, color=gt_color, linestyle='--', 
                           label='Ground Truth', linewidth=2, alpha=0.9)
                    all_powers.append(gt_binned)
                else:
                    if spatial_dim == 2:
                        dim_idx = {'x0': 0, 'x1': 1, 't': 2}.get(dim_name, 0)
                    else:
                        dim_idx = {'x': 0, 't': 1}.get(dim_name, 0)
                    
                    gt_freq_1d = gt_freqs[dim_idx]
                    axes_to_avg = tuple(j for j in range(n_dims_gt) if j != dim_idx)
                    if axes_to_avg:
                        gt_power_1d = gt_power_avg.mean(axis=axes_to_avg)
                    else:
                        gt_power_1d = gt_power_avg
                    
                    ax.plot(gt_freq_1d, gt_power_1d, color=gt_color, linestyle='--', 
                           label='Ground Truth', linewidth=2, alpha=0.9)
                    all_powers.append(gt_power_1d)
            
            # Formatting
            if dim_name == 'radial':
                ax.set_xlabel('|k| (Radial Frequency)', fontsize=11)
            else:
                ax.set_xlabel(f'k_{dim_name} (Frequency)', fontsize=11)
            
            ax.set_ylabel('Power |FFT|²', fontsize=11)
            
            # Try log scale
            is_log = _safe_log_scale(ax, all_powers)
            scale_str = "[log]" if is_log else "[linear]"
            
            # Math expression titles
            if perspective == 'cumulative':
                math_expr = r'$|\mathcal{F}[\hat{h}_i]|^2$'
                subtitle = f'Approximation Power (layer 1→i) {scale_str}'
            elif perspective == 'added':
                math_expr = r'$|\mathcal{F}[\hat{h}_i - \hat{h}_{i-1}]|^2$'
                subtitle = f'Added Power (layer i contribution) {scale_str}'
            else:  # leftover
                math_expr = r'$|\mathcal{F}[h_{gt} - \hat{h}_i]|^2$'
                subtitle = f'Leftover Power (error after layer i) {scale_str}'
            
            ax.set_title(f'{math_expr}\n{subtitle}', fontsize=11, fontweight='bold')
            
            ax.grid(True, alpha=0.3)
            
            # Legend only on first subplot of each row
            if col_idx == 0:
                ax.legend(loc='upper right', fontsize=8, ncol=1)
    
    plt.suptitle('Frequency Learning Analysis - Power Spectrum by Layer', 
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


def plot_spectral_learning_efficiency_comparison(
    frequency_data: Dict[str, Dict],
    output_dir: Path
) -> None:
    """
    Generate side-by-side comparison of spectral learning efficiency for all models.
    
    Creates a grid layout (2 columns, multiple rows) showing each model's
    spectral learning efficiency heatmap.
    
    Args:
        frequency_data: Dict mapping model_name -> frequency_metrics dict
        output_dir: Directory to save plot
    """
    model_names = list(frequency_data.keys())
    if not model_names:
        return
    
    # Check if all models have spectral efficiency data
    models_with_data = {}
    for model_name, metrics in frequency_data.items():
        if 'spectral_efficiency' in metrics:
            models_with_data[model_name] = metrics['spectral_efficiency']
    
    if not models_with_data:
        print("  No spectral efficiency data found in frequency metrics")
        return
    
    n_models = len(models_with_data)
    
    # Calculate grid layout: 2 columns, multiple rows
    n_cols = 2
    n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12 * n_cols, 8 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_cols > 1 else axes.reshape(1, 1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    for idx, (model_name, spectral_data) in enumerate(models_with_data.items()):
        ax = axes_flat[idx]
        
        # Extract error matrix and k_radial bins
        error_matrix = np.array(spectral_data['error_matrix']).T  # Transpose: rows = freq, cols = layers
        k_radial_ref = np.array(spectral_data['k_radial_bins'])
        
        n_freq_bins, n_layers = error_matrix.shape
        
        # Get layer names from frequency_data
        layers = frequency_data[model_name]['layers_analyzed']
        
        # Create heatmap
        im = ax.imshow(error_matrix, aspect='auto', cmap='viridis_r', 
                       interpolation='bilinear', origin='lower')
        
        # Set ticks
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
        ax.set_xlabel('Layer', fontsize=10, fontweight='bold')
        
        # Y-axis: frequency bins
        y_ticks = np.linspace(0, n_freq_bins - 1, min(10, n_freq_bins))
        y_tick_labels = [f'{k_radial_ref[int(i)]:.2f}' for i in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        ax.set_ylabel('Radial Frequency |k|', fontsize=10, fontweight='bold')
        
        # Colorbar for each subplot
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Remaining Error', fontsize=9, rotation=270, labelpad=15)
        
        # Try log scale for colorbar
        if np.all(error_matrix > 0):
            im.set_norm(LogNorm(vmin=error_matrix[error_matrix > 0].min(), 
                               vmax=error_matrix.max()))
            cbar.set_label('Remaining Error [log]', fontsize=9, rotation=270, labelpad=15)
        
        ax.set_title(f'{model_name}\nSpectral Learning Efficiency', 
                     fontsize=11, fontweight='bold', pad=10)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.suptitle('Spectral Learning Efficiency Comparison\n(Layer vs Frequency: Remaining Error)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    save_path = output_dir / 'spectral_learning_efficiency_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Spectral learning efficiency comparison saved to {save_path}")
