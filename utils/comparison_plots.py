"""Shared comparison plotting functions for multi-experiment analysis."""

import matplotlib.pyplot as plt
from pathlib import Path


def generate_ncc_classification_plot(save_dir, ncc_data):
    """Generate NCC classification accuracy comparison across layers and epochs."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define color families for models
    color_families = [
        ['#ffcccc', '#ff9999', '#ff6666', '#ff3333', '#cc0000'],  # Reds
        ['#cce5ff', '#99ccff', '#66b3ff', '#3399ff', '#0066cc'],  # Blues
        ['#ccffcc', '#99ff99', '#66ff66', '#33cc33', '#009900'],  # Greens
        ['#ffe5cc', '#ffcc99', '#ffb366', '#ff9933', '#cc6600'],  # Oranges
        ['#e5ccff', '#cc99ff', '#b366ff', '#9933ff', '#6600cc'],  # Purples
    ]
    
    model_names = list(ncc_data.keys())
    
    for model_idx, (model_name, epochs_data) in enumerate(ncc_data.items()):
        color_family = color_families[model_idx % len(color_families)]
        
        # Sort epochs (numeric first, then 'final')
        sorted_epochs = sorted([e for e in epochs_data.keys() if isinstance(e, int)])
        if 'final' in epochs_data:
            sorted_epochs.append('final')
        
        for epoch_idx, epoch_key in enumerate(sorted_epochs):
            ncc_metrics = epochs_data[epoch_key]
            layers = ncc_metrics['layers_analyzed']
            accuracies = [ncc_metrics['layer_accuracies'][layer] for layer in layers]
            
            # Darker color for later epochs
            color_idx = min(epoch_idx, len(color_family) - 1)
            color = color_family[color_idx]
            
            epoch_label = f"Epoch {epoch_key}" if isinstance(epoch_key, int) else "Final"
            label = f"{model_name}" if epoch_idx == 0 else None
            
            ax.plot(layers, accuracies, marker='o', color=color, 
                   linewidth=2, markersize=6, alpha=0.8, label=label)
    
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
    
    # Define color families for models
    color_families = [
        ['#ffcccc', '#ff9999', '#ff6666', '#ff3333', '#cc0000'],  # Reds
        ['#cce5ff', '#99ccff', '#66b3ff', '#3399ff', '#0066cc'],  # Blues
        ['#ccffcc', '#99ff99', '#66ff66', '#33cc33', '#009900'],  # Greens
        ['#ffe5cc', '#ffcc99', '#ffb366', '#ff9933', '#cc6600'],  # Oranges
        ['#e5ccff', '#cc99ff', '#b366ff', '#9933ff', '#6600cc'],  # Purples
    ]
    
    model_names = list(ncc_data.keys())
    
    for model_idx, (model_name, epochs_data) in enumerate(ncc_data.items()):
        color_family = color_families[model_idx % len(color_families)]
        
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
            
            # Darker color for later epochs
            color_idx = min(epoch_idx, len(color_family) - 1)
            color = color_family[color_idx]
            
            epoch_label = f"Epoch {epoch_key}" if isinstance(epoch_key, int) else "Final"
            label = f"{model_name}" if epoch_idx == 0 else None
            
            ax.plot(layers, margin_snrs, marker='o', color=color, 
                   linewidth=2, markersize=6, alpha=0.8, label=label)
    
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
    
    # Define colors for different experiments
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    # Create figure with 2 subplots (one for train, one for eval)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for exp_idx, (exp_name, metrics) in enumerate(probe_data.items()):
        color = colors[exp_idx % len(colors)]
        
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


def generate_derivatives_comparison_plots(save_dir, derivatives_data):
    """
    Generate comparison plots for derivatives tracking across experiments.
    
    Args:
        save_dir: Directory to save plots
        derivatives_data: Dictionary mapping experiment name -> derivatives metrics
    """
    print("\nGenerating derivatives comparison plots...")
    
    # Plot 1: Residual evolution across layers for all experiments
    fig, ax = plt.subplots(figsize=(14, 8))
    
    exp_names = list(derivatives_data.keys())
    
    for exp_name in exp_names:
        deriv_metrics = derivatives_data[exp_name]
        layers = deriv_metrics['layers_analyzed']
        
        # Get residual norms for eval data
        residual_norms = []
        for layer_name in layers:
            if layer_name in deriv_metrics['eval']:
                residual_norms.append(deriv_metrics['eval'][layer_name]['residual_norm'])
            else:
                residual_norms.append(None)
        
        # Plot with layer indices
        layer_indices = list(range(1, len(layers) + 1))
        ax.plot(layer_indices, residual_norms, marker='o', linewidth=2, 
                markersize=6, label=exp_name)
    
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residual L2 Norm (Eval)', fontsize=12, fontweight='bold')
    ax.set_title('Residual Evolution Across Layers - All Experiments', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    save_path = Path(save_dir) / "derivatives_residual_evolution_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Derivatives residual evolution comparison saved to {save_path}")