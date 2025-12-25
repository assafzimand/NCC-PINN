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
        save_dir: Directory to save plots
        optimizer_switch_epoch: Epoch where optimizer switched (e.g., Adam to LBFGS).
                               If provided, a vertical line is drawn at this epoch.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loss_epochs = metrics['train_loss_epochs']
    eval_epochs = metrics['epochs']

    # Create figure with 2 subplots
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

