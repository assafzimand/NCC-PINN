"""Plotting utilities for training metrics."""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import numpy as np


def _safe_log_scale(ax, values_list):
    """Set log scale on y-axis only if all data has positive values.
    
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
    # Filter out NaN values for the check
    valid_values = all_values[~np.isnan(all_values)]
    if len(valid_values) > 0 and np.all(valid_values > 0):
        ax.set_yscale('log')
        return True
    return False


def plot_training_curves(
    metrics: Dict[str, List[float]], 
    save_dir: Path,
    optimizer_switch_epoch: int = None
) -> None:
    """
    Plot training and evaluation curves.

    Args:
        metrics: Dictionary with keys:
                - 'train_loss_epochs', 'train_loss' (all epochs)
                - 'epochs', 'eval_loss', 'train_rel_l2', 'eval_rel_l2' (eval epochs only)
                - Optional RNC keys: 'rnc_penalty', 'rnc_penalty_epochs', 'rnc_layer_terms'
        save_dir: Directory to save plots
        optimizer_switch_epoch: Epoch where optimizer switched (e.g., Adam to LBFGS).
                               If provided, a vertical line is drawn at this epoch.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loss_epochs = metrics['train_loss_epochs']
    eval_epochs = metrics['epochs']
    
    # Check if RNC metrics are present
    has_rnc = 'rnc_penalty' in metrics and len(metrics.get('rnc_penalty', [])) > 0
    
    # Create figure with 2x2 subplots if RNC enabled, otherwise 1x2
    if has_rnc:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss curves
    ax = axes[0]
    ax.plot(train_loss_epochs, metrics['train_loss'], 'b-', label='Train Loss',
            linewidth=2, alpha=0.8)
    ax.plot(eval_epochs, metrics['eval_loss'], 'r-', label='Eval Loss',
            linewidth=2, alpha=0.8)
    
    # Add optimizer switch marker
    if optimizer_switch_epoch is not None:
        ax.axvline(x=optimizer_switch_epoch, color='green', linestyle='--', 
                   linewidth=2, alpha=0.7, label='Optimizer Switch (Adam->LBFGS)')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    is_log_loss = _safe_log_scale(ax, [metrics['train_loss'], metrics['eval_loss']])
    scale_str_loss = "[log]" if is_log_loss else "[linear]"
    ax.set_title(f'Training and Evaluation Loss {scale_str_loss}', fontsize=14, fontweight='bold')

    # Plot 2: Relative L2 error
    ax = axes[1]
    ax.plot(eval_epochs, metrics['train_rel_l2'], 'b-', label='Train Rel. L2',
            linewidth=2, alpha=0.8)
    ax.plot(eval_epochs, metrics['eval_rel_l2'], 'r-', label='Eval Rel. L2',
            linewidth=2, alpha=0.8)
    
    # Add optimizer switch marker
    if optimizer_switch_epoch is not None:
        ax.axvline(x=optimizer_switch_epoch, color='green', linestyle='--', 
                   linewidth=2, alpha=0.7, label='Optimizer Switch (Adam->LBFGS)')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    is_log_l2 = _safe_log_scale(ax, [metrics['train_rel_l2'], metrics['eval_rel_l2']])
    scale_str_l2 = "[log]" if is_log_l2 else "[linear]"
    ax.set_title(f'Relative L2 Error {scale_str_l2}', fontsize=14, fontweight='bold')

    # Plot 3 & 4: RNC metrics (if enabled)
    if has_rnc:
        rnc_epochs = metrics['rnc_penalty_epochs']
        rnc_penalty = metrics['rnc_penalty']
        target_update_epochs = metrics.get('target_update_epochs', [])
        
        # Plot 3: Total RNC Penalty
        ax = axes[2]
        ax.plot(rnc_epochs, rnc_penalty, 'purple', label='RNC Penalty', linewidth=2, alpha=0.8)
        
        # Add target update markers
        for i, update_epoch in enumerate(target_update_epochs):
            label = 'Target Updates' if i == 0 else None
            ax.axvline(x=update_epoch, color='red', linestyle='--', 
                      linewidth=1, alpha=0.5, label=label)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('RNC Penalty', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        is_log_rnc = _safe_log_scale(ax, [rnc_penalty])
        scale_str_rnc = "[log]" if is_log_rnc else "[linear]"
        ax.set_title(f'RNC Penalty {scale_str_rnc}', fontsize=14, fontweight='bold')
        
        # Plot 4: Per-layer term norms
        ax = axes[3]
        rnc_layer_terms = metrics.get('rnc_layer_terms', {})
        
        # Use different colors for different layers/terms
        colors = plt.cm.tab10.colors
        for idx, (term_key, term_values) in enumerate(sorted(rnc_layer_terms.items())):
            # Clean up label (e.g., "layer_1_h_t_norm" -> "L1 h_t")
            parts = term_key.replace('_norm', '').split('_')
            if len(parts) >= 3:
                layer_num = parts[1]
                term_name = '_'.join(parts[2:])
                label = f'L{layer_num} {term_name}'
            else:
                label = term_key
            
            ax.plot(rnc_epochs, term_values, color=colors[idx % len(colors)], 
                   label=label, linewidth=1.5, alpha=0.8)
        
        # Add target update markers
        for i, update_epoch in enumerate(target_update_epochs):
            label = 'Target Updates' if i == 0 else None
            ax.axvline(x=update_epoch, color='red', linestyle='--', 
                      linewidth=1, alpha=0.3, label=label if i == 0 and not rnc_layer_terms else None)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Term Norm', fontsize=12)
        ax.legend(fontsize=9, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3)
        is_log_terms = _safe_log_scale(ax, list(rnc_layer_terms.values()))
        scale_str_terms = "[log]" if is_log_terms else "[linear]"
        ax.set_title(f'Per-Layer Term Norms {scale_str_terms}', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save figure
    save_path = save_dir / 'training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Training curves saved to {save_path}")


def plot_final_comparison(
    h_pred: np.ndarray,
    h_gt: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    save_dir: Path
) -> None:
    """
    Plot final predictions vs ground truth.

    Args:
        h_pred: Predicted values (N, output_dim)
        h_gt: Ground truth values (N, output_dim)
        x: Spatial coordinates (N, spatial_dim)
        t: Temporal coordinates (N, 1)
        save_dir: Directory to save plot
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # For 1D spatial + time, create scatter plots
    if x.shape[1] == 1:
        output_dim = h_pred.shape[1]
        cmaps = ['viridis', 'plasma', 'inferno', 'magma']
        
        fig, axes = plt.subplots(output_dim, 2, figsize=(14, 5*output_dim))
        if output_dim == 1:
            axes = axes.reshape(1, 2)

        for comp_idx in range(output_dim):
            # Prediction
            ax = axes[comp_idx, 0]
            scatter = ax.scatter(x[:, 0], t[:, 0], c=h_pred[:, comp_idx],
                               s=2, cmap=cmaps[comp_idx % len(cmaps)], alpha=0.6)
            ax.set_xlabel('x')
            ax.set_ylabel('t')
            ax.set_title(f'Prediction h_{comp_idx}(x,t)')
            plt.colorbar(scatter, ax=ax)

            # Ground Truth
            ax = axes[comp_idx, 1]
            scatter = ax.scatter(x[:, 0], t[:, 0], c=h_gt[:, comp_idx],
                               s=2, cmap=cmaps[comp_idx % len(cmaps)], alpha=0.6)
            ax.set_xlabel('x')
            ax.set_ylabel('t')
            ax.set_title(f'Ground Truth h_{comp_idx}(x,t)')
            plt.colorbar(scatter, ax=ax)

        plt.tight_layout()

        # Save
        save_path = save_dir / 'final_predictions.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Final predictions saved to {save_path}")

