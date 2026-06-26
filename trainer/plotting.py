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
    optimizer_switch_epochs: List[int] = None,
    segment_start_epochs: List[int] = None
) -> None:
    """
    Plot training and evaluation curves.

    Args:
        metrics: Dictionary with keys:
                - 'train_loss_epochs', 'train_loss' (all epochs)
                - 'epochs', 'eval_loss', 'eval_rel_l2' (eval epochs only)
                - Optional: 'loss_components' dict with 'epochs', 'residual', 'ic', 'bc' lists
        save_dir: Directory to save plots
        optimizer_switch_epochs: List of epochs where optimizer switched.
                                Green dashed vertical lines drawn at each.
        segment_start_epochs: List of epochs where new training segments started.
                             Blue dotted vertical lines drawn at each.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loss_epochs = metrics['train_loss_epochs']
    eval_epochs = metrics['epochs']

    optimizer_switch_epochs = optimizer_switch_epochs or []
    segment_start_epochs = segment_start_epochs or []
    
    # Check if we have loss components for term-wise plot
    loss_comps = metrics.get('loss_components', {})
    has_components = (loss_comps.get('epochs') and 
                      len(loss_comps.get('epochs', [])) > 0 and
                      any(loss_comps.get(k) for k in ['residual', 'ic', 'bc']))

    # Create figure with 2 or 3 subplots depending on whether we have components
    n_plots = 3 if has_components else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))

    # Plot 1: Loss curves
    ax = axes[0]
    ax.plot(train_loss_epochs, metrics['train_loss'], 'b-', label='Train Loss',
            linewidth=2, alpha=0.8)
    ax.plot(eval_epochs, metrics['eval_loss'], 'r-', label='Eval Loss',
            linewidth=2, alpha=0.8)
    
    # Add optimizer switch markers (green dashed)
    for i, epoch in enumerate(optimizer_switch_epochs):
        label = 'Optimizer Switch' if i == 0 else None
        ax.axvline(x=epoch, color='green', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label=label)
    
    # Add segment boundary markers (blue dotted), skip epoch 1 (start of first segment)
    _seg_labeled = False
    for epoch in segment_start_epochs:
        if epoch <= 1:
            continue
        ax.axvline(x=epoch, color='blue', linestyle=':',
                   linewidth=1.5, alpha=0.6,
                   label='New Level Start' if not _seg_labeled else None)
        _seg_labeled = True
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    is_log_loss = _safe_log_scale(ax, [metrics['train_loss'], metrics['eval_loss']])
    scale_str_loss = "[log]" if is_log_loss else "[linear]"
    ax.set_title(f'Training and Evaluation Loss {scale_str_loss}', fontsize=14, fontweight='bold')

    # Plot 2: Relative L2 error
    ax = axes[1]
    ax.plot(eval_epochs, metrics['eval_rel_l2'], 'r-', label='Eval Rel. L2',
            linewidth=2, alpha=0.8)
    
    # Add optimizer switch markers (green dashed)
    for i, epoch in enumerate(optimizer_switch_epochs):
        label = 'Optimizer Switch' if i == 0 else None
        ax.axvline(x=epoch, color='green', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label=label)
    
    # Add segment boundary markers (blue dotted), skip epoch 1
    _seg_labeled = False
    for epoch in segment_start_epochs:
        if epoch <= 1:
            continue
        ax.axvline(x=epoch, color='blue', linestyle=':',
                   linewidth=1.5, alpha=0.6,
                   label='New Level Start' if not _seg_labeled else None)
        _seg_labeled = True
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    is_log_l2 = _safe_log_scale(ax, [metrics['eval_rel_l2']])
    scale_str_l2 = "[log]" if is_log_l2 else "[linear]"
    ax.set_title(f'Relative L2 Error {scale_str_l2}', fontsize=14, fontweight='bold')

    # Plot 3: Term-wise loss components (if available)
    if has_components:
        ax = axes[2]
        comp_epochs = loss_comps['epochs']
        
        # Color scheme for different loss terms
        term_colors = {
            'residual': '#e74c3c',  # red
            'ic': '#3498db',         # blue
            'bc': '#2ecc71',         # green
        }
        term_labels = {
            'residual': 'PDE Residual',
            'ic': 'Initial Condition',
            'bc': 'Boundary Condition',
        }
        
        values_for_log = []
        for term in ['residual', 'ic', 'bc']:
            if loss_comps.get(term) and len(loss_comps[term]) > 0:
                values = loss_comps[term]
                ax.plot(comp_epochs, values, '-', 
                       color=term_colors.get(term, 'gray'),
                       label=term_labels.get(term, term),
                       linewidth=1.5, alpha=0.8)
                values_for_log.append(values)
        
        # Add optimizer switch markers
        for i, epoch in enumerate(optimizer_switch_epochs):
            label = 'Optimizer Switch' if i == 0 else None
            ax.axvline(x=epoch, color='green', linestyle='--', 
                       linewidth=1.5, alpha=0.7, label=label)
        
        # Add segment boundary markers
        _seg_labeled = False
        for epoch in segment_start_epochs:
            if epoch <= 1:
                continue
            ax.axvline(x=epoch, color='blue', linestyle=':',
                       linewidth=1.5, alpha=0.6,
                       label='New Level Start' if not _seg_labeled else None)
            _seg_labeled = True
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss Component', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        is_log_comp = _safe_log_scale(ax, values_for_log) if values_for_log else False
        scale_str_comp = "[log]" if is_log_comp else "[linear]"
        ax.set_title(f'Loss Components {scale_str_comp}', fontsize=14, fontweight='bold')

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

