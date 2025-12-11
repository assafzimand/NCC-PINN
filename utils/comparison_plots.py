"""Shared comparison plotting functions for multi-experiment analysis."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib import colors as mcolors


def _build_color_map(model_names):
    """
    Deterministic color assignment per model, shared across all comparison plots.
    Sorting names ensures the same model always gets the same color in every figure.
    """
    sorted_names = sorted(model_names)
    cmap = plt.cm.get_cmap('tab20', len(sorted_names))
    color_map = {}
    for idx, name in enumerate(sorted_names):
        rgba = cmap(idx)
        color_map[name] = mcolors.to_hex(rgba)
    return color_map


def generate_ncc_classification_plot(save_dir, ncc_data):
    """Generate NCC classification accuracy comparison across layers and epochs."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    model_names = sorted(ncc_data.keys())
    color_map = _build_color_map(model_names)
    
    for model_idx, model_name in enumerate(model_names):
        epochs_data = ncc_data[model_name]
        
        # Sort epochs (numeric first, then 'final')
        sorted_epochs = sorted([e for e in epochs_data.keys() if isinstance(e, int)])
        if 'final' in epochs_data:
            sorted_epochs.append('final')
        
        for epoch_idx, epoch_key in enumerate(sorted_epochs):
            ncc_metrics = epochs_data[epoch_key]
            layers = ncc_metrics['layers_analyzed']
            accuracies = [ncc_metrics['layer_accuracies'][layer] for layer in layers]
            
            base_color = color_map[model_name]
            # Darker (higher alpha) for later epochs
            alpha = min(0.45 + 0.15 * epoch_idx, 1.0)
            
            epoch_label = f"Epoch {epoch_key}" if isinstance(epoch_key, int) else "Final"
            label = f"{model_name}" if epoch_idx == 0 else None
            
            ax.plot(
                layers,
                accuracies,
                marker='o',
                color=base_color,
                linewidth=2,
                markersize=6,
                alpha=alpha,
                label=label
            )
    
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('NCC Classification Accuracy Comparison\n(Darker shades = later epochs)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    save_path = Path(save_dir) / "ncc_classification_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  NCC classification comparison saved to {save_path}")


def generate_ncc_compactness_plot(save_dir, ncc_data):
    """Generate NCC compactness (margin) comparison across layers and epochs."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    model_names = sorted(ncc_data.keys())
    color_map = _build_color_map(model_names)
    
    for model_idx, model_name in enumerate(model_names):
        epochs_data = ncc_data[model_name]
        
        # Sort epochs (numeric first, then 'final')
        sorted_epochs = sorted([e for e in epochs_data.keys() if isinstance(e, int)])
        if 'final' in epochs_data:
            sorted_epochs.append('final')
        
        for epoch_idx, epoch_key in enumerate(sorted_epochs):
            ncc_metrics = epochs_data[epoch_key]
            layers = ncc_metrics['layers_analyzed']
            
            # Compute margin SNR for each layer
            margin_snrs = []
            for layer in layers:
                mean = ncc_metrics['layer_margins'][layer]['mean_margin']
                std = ncc_metrics['layer_margins'][layer]['std_margin']
                snr = mean / std if std > 0 else 0
                margin_snrs.append(snr)
            
            base_color = color_map[model_name]
            # Darker (higher alpha) for later epochs
            alpha = min(0.45 + 0.15 * epoch_idx, 1.0)
            
            epoch_label = f"Epoch {epoch_key}" if isinstance(epoch_key, int) else "Final"
            label = f"{model_name}" if epoch_idx == 0 else None
            
            ax.plot(
                layers,
                margin_snrs,
                marker='o',
                color=base_color,
                linewidth=2,
                markersize=6,
                alpha=alpha,
                label=label
            )
    
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Margin SNR (mean/std)', fontsize=12, fontweight='bold')
    ax.set_title('NCC Compactness (Margin SNR) Comparison\n(Darker shades = later epochs)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    save_path = Path(save_dir) / "ncc_compactness_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  NCC compactness comparison saved to {save_path}")


def generate_ncc_comparison_plots_only(save_dir, ncc_data):
    """
    Generate NCC comparison plots for multi-checkpoint evaluation.
    
    Args:
        save_dir: Directory to save plots
        ncc_data: Dict of {model_name: {epoch: ncc_metrics}}
    """
    print("\nGenerating NCC comparison plots...")
    generate_ncc_classification_plot(save_dir, ncc_data)
    generate_ncc_compactness_plot(save_dir, ncc_data)
    print(f"Comparison plots saved to {save_dir}")


def generate_probe_comparison_plots(save_dir, probe_data):
    """
    Generate linear probe comparison plots across experiments.
    
    Args:
        save_dir: Directory to save plots
        probe_data: Dict of {experiment_name: probe_metrics}
                   where probe_metrics has 'train'/'eval' with 'rel_l2' and 'inf_norm' lists
    """
    print("\nGenerating probe comparison plots...")
    
    exp_names = sorted(probe_data.keys())
    color_map = _build_color_map(exp_names)
    
    # Create figure with 2 subplots (one for train, one for eval)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for exp_name in exp_names:
        metrics = probe_data[exp_name]
        color = color_map[exp_name]
        
        # Number of layers
        num_layers = len(metrics['train']['rel_l2'])
        layer_numbers = list(range(1, num_layers + 1))
        
        # Plot training metrics
        ax1.plot(layer_numbers, metrics['train']['rel_l2'], 
                marker='o', color=color, linewidth=2, markersize=7,
                label=exp_name, alpha=0.8)
        
        # Plot evaluation metrics
        ax2.plot(layer_numbers, metrics['eval']['rel_l2'], 
                marker='s', color=color, linewidth=2, markersize=7,
                label=exp_name, alpha=0.8)
    
    # Configure training plot
    ax1.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probe Rel-L2 Error', fontsize=12, fontweight='bold')
    ax1.set_title('Linear Probe Performance - Training Data', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='best')
    
    # Configure evaluation plot
    ax2.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Probe Rel-L2 Error', fontsize=12, fontweight='bold')
    ax2.set_title('Linear Probe Performance - Evaluation Data', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    
    # Save plot
    save_path = Path(save_dir) / "probe_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Probe comparison plot saved to {save_path}")
    
    # Infinity norm comparison (train/eval)
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))
    for exp_name in exp_names:
        metrics = probe_data[exp_name]
        color = color_map[exp_name]
        num_layers = len(metrics['train']['inf_norm'])
        layer_numbers = list(range(1, num_layers + 1))
        
        ax3.plot(layer_numbers, metrics['train']['inf_norm'],
                 marker='o', color=color, linewidth=2, markersize=7,
                 label=exp_name, alpha=0.8)
        ax4.plot(layer_numbers, metrics['eval']['inf_norm'],
                 marker='s', color=color, linewidth=2, markersize=7,
                 label=exp_name, alpha=0.8)
    
    ax3.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Probe L∞ Error', fontsize=12, fontweight='bold')
    ax3.set_title('Linear Probe Infinity Norm - Training Data',
                  fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10, loc='best')
    
    ax4.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Probe L∞ Error', fontsize=12, fontweight='bold')
    ax4.set_title('Linear Probe Infinity Norm - Evaluation Data',
                  fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    
    inf_path = Path(save_dir) / "probe_comparison_inf.png"
    plt.savefig(inf_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Probe infinity-norm comparison saved to {inf_path}")


def _lookup_metric(exp_data, section_key, split, layer_name, metric_name):
    if section_key is None:
        layer_metrics = exp_data.get(split, {}).get(layer_name, {})
    else:
        layer_metrics = (
            exp_data.get(section_key, {})
            .get(split, {})
            .get(layer_name, {})
        )
    return layer_metrics.get(metric_name, np.nan)


def _plot_derivative_comparison_grid(save_dir, derivatives_data, section_key, key_map, title, filename):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    exp_names = sorted(derivatives_data.keys())
    color_map = _build_color_map(exp_names)
    max_layers = max(len(data['layers_analyzed']) for data in derivatives_data.values())
    
    panel_cfg = [
        (axes[0, 0], 'train', 'l2', 'Train L2'),
        (axes[0, 1], 'eval', 'l2', 'Eval L2'),
        (axes[1, 0], 'train', 'linf', 'Train L∞'),
        (axes[1, 1], 'eval', 'linf', 'Eval L∞'),
    ]
    
    for panel_idx, (ax, split, metric_id, panel_title) in enumerate(panel_cfg):
        for exp_name in exp_names:
            exp_data = derivatives_data[exp_name]
            layers = exp_data['layers_analyzed']
            layer_indices = list(range(1, len(layers) + 1))
            values = []
            metric_key = key_map[metric_id]
            for layer_name in layers:
                values.append(
                    _lookup_metric(exp_data, section_key, split, layer_name, metric_key)
                )
            ax.plot(
                layer_indices,
                values,
                marker='o',
                linewidth=2,
                markersize=6,
                label=exp_name,
                color=color_map[exp_name]
            )
        
        ax.set_title(panel_title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=11)
        ax.set_ylabel('Mean Norm', fontsize=11)
        ax.set_xticks(range(1, max_layers + 1))
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        if panel_idx == 0:
            ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    save_path = Path(save_dir) / filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {title} saved to {save_path}")


def generate_derivatives_comparison_plots(save_dir, derivatives_data):
    """
    Generate comparison plots for derivatives tracking across experiments.
    """
    print("\nGenerating derivatives comparison plots...")
    
    _plot_derivative_comparison_grid(
        save_dir,
        derivatives_data,
        section_key=None,
        key_map={'l2': 'residual_norm', 'linf': 'residual_inf_norm'},
        title='Derivatives residual comparison',
        filename="derivatives_residual_comparison.png"
    )
    _plot_derivative_comparison_grid(
        save_dir,
        derivatives_data,
        section_key='ic',
        key_map={'l2': 'l2', 'linf': 'linf'},
        title='Derivatives IC comparison',
        filename="derivatives_ic_comparison.png"
    )
    _plot_derivative_comparison_grid(
        save_dir,
        derivatives_data,
        section_key='bc_value',
        key_map={'l2': 'l2', 'linf': 'linf'},
        title='Derivatives BC value comparison',
        filename="derivatives_bc_value_comparison.png"
    )
    _plot_derivative_comparison_grid(
        save_dir,
        derivatives_data,
        section_key='bc_derivative',
        key_map={'l2': 'l2', 'linf': 'linf'},
        title='Derivatives BC derivative comparison',
        filename="derivatives_bc_derivative_comparison.png"
    )