"""Visualization for linear probe analysis."""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
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


def generate_all_probe_plots(probe_results: Dict, save_dir: Path):
    """
    Generate all probe visualization plots.
    
    Args:
        probe_results: Dictionary from probe_all_layers()
        save_dir: Directory to save plots
    """
    print("\nGenerating probe plots...")
    plot_probe_metrics(probe_results, save_dir)
    plot_layer_dimensions(probe_results, save_dir)
    print("  All probe plots generated")

