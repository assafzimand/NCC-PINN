"""
1D Wave equation specific visualizations.

Provides custom visualization for:
1. Dataset visualization: heatmap of h(x,t)
2. Evaluation visualization: comprehensive model performance analysis
3. NCC visualizations: classification heatmaps in output and input spaces
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
from matplotlib.colors import TwoSlopeNorm


def visualize_dataset(data_dict: Dict, save_dir: Path, config: Dict, split_name: str):
    """
    Visualize wave equation dataset with heatmap.
    
    Creates heatmap of h(x,t) - real-valued wave field.
    
    Args:
        data_dict: Dataset dictionary with 'x', 't', 'u_gt' tensors
        save_dir: Directory to save visualization
        config: Configuration dictionary
        split_name: Name of split ('training' or 'evaluation')
    """
    # Extract data
    x = data_dict['x'].cpu().numpy()  # (N, spatial_dim)
    t = data_dict['t'].cpu().numpy()  # (N, 1)
    u_gt = data_dict['u_gt'].cpu().numpy()  # (N, 1) - real-valued
    
    # Flatten coordinates
    x_flat = x[:, 0]
    t_flat = t[:, 0]
    h = u_gt[:, 0]  # Scalar wave field
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot: h(x,t)
    scatter = ax.scatter(x_flat, t_flat, c=h, cmap='RdBu_r', s=5, alpha=0.6,
                        vmin=-np.abs(h).max(), vmax=np.abs(h).max())
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('t', fontsize=12)
    ax.set_title(f'h(x,t) - {split_name.capitalize()}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('h', fontsize=11)
    
    plt.tight_layout()
    
    # Save
    save_path = save_dir / f"dataset_{split_name.lower()}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  {split_name.capitalize()} dataset visualization saved to {save_path}")


def visualize_evaluation(model: torch.nn.Module, eval_data_path: str, 
                        save_dir: Path, config: Dict):
    """
    Visualize wave equation model evaluation with smooth heatmap comparison plots.
    
    Creates a 3-column figure with continuous heatmaps:
    - Ground truth h(x,t)
    - Predicted h(x,t)
    - Error |h_pred - h_gt|
    
    Args:
        model: Trained neural network
        eval_data_path: Path to evaluation dataset
        save_dir: Directory to save visualization
        config: Configuration dictionary
    """
    # Get problem-specific config
    problem = config.get('problem', 'wave1d')
    problem_config = config[problem]
    spatial_domain = problem_config['spatial_domain'][0]  # [x_min, x_max]
    temporal_domain = problem_config['temporal_domain']  # [t_min, t_max]
    
    x_min, x_max = spatial_domain
    t_min, t_max = temporal_domain
    
    # Create dense evaluation grid for smooth heatmaps
    n_x = 256
    n_t = 200
    x_grid = np.linspace(x_min, x_max, n_x)
    t_grid = np.linspace(t_min, t_max, n_t)
    X, T = np.meshgrid(x_grid, t_grid)
    
    # Flatten for model input
    device = next(model.parameters()).device
    x_flat = torch.tensor(X.flatten(), dtype=torch.float32, device=device).view(-1, 1)
    t_flat = torch.tensor(T.flatten(), dtype=torch.float32, device=device).view(-1, 1)
    
    # Model predictions
    model.eval()
    with torch.no_grad():
        xt_input = torch.cat([x_flat, t_flat], dim=1)
        h_pred = model(xt_input)  # (N, 1)
        h_pred_np = h_pred[:, 0].cpu().numpy().reshape(n_t, n_x)
    
    # Get ground truth using the analytical solver
    from solvers.wave1d_solver import analytical_solution
    h_gt = analytical_solution(X, T)  # Already on grid, shape (n_t, n_x)
    
    # Compute error
    error = np.abs(h_pred_np - h_gt)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Determine common color scale for GT and Pred
    vmax = max(np.abs(h_gt).max(), np.abs(h_pred_np).max())
    vmin = -vmax
    
    # Plot 1: Ground Truth - continuous heatmap
    im1 = axes[0].contourf(X, T, h_gt, levels=50, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('t', fontsize=12)
    axes[0].set_title('Ground Truth h(x,t)', fontsize=14, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('h', fontsize=11)
    
    # Plot 2: Prediction - continuous heatmap
    im2 = axes[1].contourf(X, T, h_pred_np, levels=50, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('t', fontsize=12)
    axes[1].set_title('Prediction h_pred(x,t)', fontsize=14, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('h', fontsize=11)
    
    # Plot 3: Error - continuous heatmap
    im3 = axes[2].contourf(X, T, error, levels=50, cmap='Reds')
    axes[2].set_xlabel('x', fontsize=12)
    axes[2].set_ylabel('t', fontsize=12)
    axes[2].set_title('Error |h_pred - h_gt|', fontsize=14, fontweight='bold')
    cbar3 = plt.colorbar(im3, ax=axes[2])
    cbar3.set_label('|error|', fontsize=11)
    
    plt.tight_layout()
    
    # Save
    save_path = save_dir / "evaluation_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Evaluation comparison saved to {save_path}")


def visualize_ncc_dataset(ncc_data: Dict, dataset_dir: Path, config: Dict, prefix: str = ""):
    """
    Visualize NCC dataset distribution for wave equation (1D output).
    
    Args:
        ncc_data: NCC dataset dictionary
        dataset_dir: Directory to save visualization
        config: Configuration dictionary
        prefix: Prefix for filename
    """
    h_gt = ncc_data['u_gt'].cpu().numpy()  # (N, 1)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 1D histogram
    ax.hist(h_gt[:, 0], bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('h', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('NCC Dataset Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = dataset_dir / f"{prefix}ncc_dataset_distribution.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  NCC dataset distribution saved to {save_path}")


def visualize_ncc_classification(
    h_gt: np.ndarray,
    class_labels: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    bins: np.ndarray,
    save_path: Path
):
    """
    Visualize NCC classification in output space (1D for wave equation).
    
    For 1D output, creates scatter plot of h values colored by correctness.
    
    Args:
        h_gt: Ground truth values (N, 1)
        class_labels: True class labels (N,)
        predictions_dict: Dict mapping layer_name -> predictions (N,)
        bins: Bin edges for classification
        save_path: Path to save figure
    """
    n_layers = len(predictions_dict)
    fig, axes = plt.subplots(1, n_layers, figsize=(5*n_layers, 4))
    if n_layers == 1:
        axes = [axes]
    
    for ax, (layer_name, preds) in zip(axes, predictions_dict.items()):
        # Ensure numpy arrays
        preds_np = preds if isinstance(preds, np.ndarray) else preds.cpu().numpy()
        labels_np = class_labels if isinstance(class_labels, np.ndarray) else class_labels.cpu().numpy()
        correct = (preds_np == labels_np).astype(float)
        
        # Scatter plot: h values on x-axis, index on y-axis, color by correctness
        indices = np.arange(len(h_gt))
        colors = np.where(correct > 0.5, 'green', 'red')
        
        ax.scatter(h_gt[:, 0], indices, c=colors, s=2, alpha=0.5)
        ax.set_xlabel('h', fontsize=10)
        ax.set_ylabel('Sample index', fontsize=10)
        ax.set_title(f'{layer_name}\nAcc: {correct.mean():.3f}', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  NCC classification scatter saved to {save_path}")


def visualize_ncc_classification_input_space(
    x: np.ndarray,
    t: np.ndarray,
    class_labels: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    save_path: Path
):
    """
    Visualize NCC classification in input space (x, t).
    
    Args:
        x: Spatial coordinates (N, 1)
        t: Temporal coordinates (N, 1)
        class_labels: True class labels (N,)
        predictions_dict: Dict mapping layer_name -> predictions (N,)
        save_path: Path to save figure
    """
    n_layers = len(predictions_dict)
    fig, axes = plt.subplots(1, n_layers, figsize=(5*n_layers, 4))
    if n_layers == 1:
        axes = [axes]
    
    for ax, (layer_name, preds) in zip(axes, predictions_dict.items()):
        # Ensure numpy arrays
        preds_np = preds if isinstance(preds, np.ndarray) else preds.cpu().numpy()
        labels_np = class_labels if isinstance(class_labels, np.ndarray) else class_labels.cpu().numpy()
        correct = (preds_np == labels_np).astype(float)
        colors = np.where(correct > 0.5, 'green', 'red')
        
        ax.scatter(x[:, 0], t[:, 0], c=colors, s=2, alpha=0.5)
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('t', fontsize=10)
        ax.set_title(f'{layer_name}\nAcc: {correct.mean():.3f}', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  NCC classification (input space) scatter saved to {save_path}")


def visualize_ncc_classification_heatmap(
    h_gt: np.ndarray,
    class_labels: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    bins: np.ndarray,
    save_path: Path,
    config: Dict
):
    """
    Visualize NCC classification accuracy as 1D heatmap (line plot).
    
    For 1D output, bin the h values and show accuracy per bin.
    
    Args:
        h_gt: Ground truth values (N, 1)
        class_labels: True class labels (N,)
        predictions_dict: Dict mapping layer_name -> predictions (N,)
        bins: Bin edges for classification
        save_path: Path to save figure
        config: Configuration dictionary
    """
    n_bin_vis = config.get('n_bin_visualize_ncc', 100)
    min_samples = config.get('min_samples_threshold', 1)
    
    n_layers = len(predictions_dict)
    fig, axes = plt.subplots(1, n_layers, figsize=(6*n_layers, 4))
    if n_layers == 1:
        axes = [axes]
    
    for ax, (layer_name, preds) in zip(axes, predictions_dict.items()):
        # Create bins for visualization
        h_values = h_gt[:, 0]
        h_min, h_max = h_values.min(), h_values.max()
        bin_edges = np.linspace(h_min, h_max, n_bin_vis + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Ensure numpy arrays
        preds_np = preds if isinstance(preds, np.ndarray) else preds.cpu().numpy()
        labels_np = class_labels if isinstance(class_labels, np.ndarray) else class_labels.cpu().numpy()
        
        # Compute accuracy per bin
        accuracies = []
        for i in range(n_bin_vis):
            mask = (h_values >= bin_edges[i]) & (h_values < bin_edges[i+1])
            if mask.sum() >= min_samples:
                acc = (preds_np[mask] == labels_np[mask]).astype(float).mean()
                accuracies.append(acc)
            else:
                accuracies.append(np.nan)
        
        # Plot as line
        valid_mask = ~np.isnan(accuracies)
        ax.plot(bin_centers[valid_mask], np.array(accuracies)[valid_mask], 
               linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('h', fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_title(f'{layer_name}\nOverall Acc: {(preds_np == labels_np).astype(float).mean():.3f}', 
                    fontsize=11)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  NCC classification heatmap (1D) saved to {save_path}")


def visualize_ncc_classification_input_space_heatmap(
    x: np.ndarray,
    t: np.ndarray,
    class_labels: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    save_path: Path,
    config: Dict
):
    """
    Visualize NCC classification accuracy as smooth continuous heatmap in (x,t) space.
    
    Uses interpolation to create smooth heatmaps from scattered accuracy data.
    
    Args:
        x: Spatial coordinates (N, 1)
        t: Temporal coordinates (N, 1)
        class_labels: True class labels (N,)
        predictions_dict: Dict mapping layer_name -> predictions (N,)
        save_path: Path to save figure
        config: Configuration dictionary
    """
    from scipy.interpolate import griddata
    
    n_bin_vis = config.get('n_bin_visualize_ncc', 100)
    
    n_layers = len(predictions_dict)
    fig, axes = plt.subplots(1, n_layers, figsize=(6*n_layers, 5))
    if n_layers == 1:
        axes = [axes]
    
    # Create dense grid for smooth visualization
    x_vals = x[:, 0]
    t_vals = t[:, 0]
    x_grid = np.linspace(x_vals.min(), x_vals.max(), n_bin_vis)
    t_grid = np.linspace(t_vals.min(), t_vals.max(), n_bin_vis)
    X_grid, T_grid = np.meshgrid(x_grid, t_grid)
    
    for ax, (layer_name, preds) in zip(axes, predictions_dict.items()):
        # Ensure numpy arrays
        preds_np = preds if isinstance(preds, np.ndarray) else preds.cpu().numpy()
        labels_np = class_labels if isinstance(class_labels, np.ndarray) else class_labels.cpu().numpy()
        
        # Compute per-point correctness (1 or 0)
        correct = (preds_np == labels_np).astype(float)
        
        # Interpolate to dense grid for smooth heatmap
        points = np.column_stack([x_vals, t_vals])
        accuracy_grid = griddata(
            points, 
            correct, 
            (X_grid, T_grid), 
            method='cubic',
            fill_value=0.5  # Neutral value for extrapolated regions
        )
        
        # Clip to [0, 1] range (cubic interpolation can overshoot)
        accuracy_grid = np.clip(accuracy_grid, 0, 1)
        
        # Check if grid is valid for contourf (needs at least 2x2)
        if accuracy_grid.shape[0] < 2 or accuracy_grid.shape[1] < 2 or np.all(np.isnan(accuracy_grid)):
            # Fall back to scatter plot if interpolation failed
            ax.scatter(x_vals[correct.astype(bool)], t_vals[correct.astype(bool)], 
                      c='green', s=10, alpha=0.5, label='Correct')
            ax.scatter(x_vals[~correct.astype(bool)], t_vals[~correct.astype(bool)], 
                      c='red', s=10, alpha=0.5, label='Incorrect')
            ax.legend(fontsize=8)
        else:
            # Plot smooth heatmap using contourf
            contour = ax.contourf(X_grid, T_grid, accuracy_grid, levels=20, 
                                 cmap='RdYlGn', vmin=0, vmax=1)
            plt.colorbar(contour, ax=ax, label='Accuracy')
        
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('t', fontsize=10)
        ax.set_title(f'{layer_name}\nAcc: {correct.mean():.3f}', 
                    fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  NCC classification heatmap (input space) saved to {save_path}")


# Accuracy change heatmaps (similar structure to schrodinger)
def visualize_ncc_classification_accuracy_changes(
    h_gt: np.ndarray,
    class_labels: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    bins: np.ndarray,
    save_path: Path,
    config: Dict
):
    """Visualize NCC accuracy changes between layers in output space (1D)."""
    # For 1D output, this would show line plots of accuracy changes
    # Implementation similar to heatmap version but showing differences
    print(f"  Skipping accuracy changes (1D) for wave1d - similar to heatmap view")


def visualize_ncc_classification_input_space_accuracy_changes(
    x: np.ndarray,
    t: np.ndarray,
    class_labels: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    save_path: Path,
    config: Dict
):
    """
    Visualize NCC accuracy changes between layers as smooth continuous heatmaps in input space.
    
    Uses interpolation to create smooth transition heatmaps.
    """
    from scipy.interpolate import griddata
    from matplotlib.colors import TwoSlopeNorm
    
    n_bin_vis = config.get('n_bin_visualize_ncc', 100)
    
    layer_names = list(predictions_dict.keys())
    n_transitions = len(layer_names) - 1
    
    if n_transitions == 0:
        print("  Only one layer, skipping accuracy changes plot")
        return
    
    # Determine grid layout
    n_cols = min(3, n_transitions)
    n_rows = (n_transitions + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_transitions == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Create dense grid for smooth visualization
    x_vals = x[:, 0]
    t_vals = t[:, 0]
    x_grid = np.linspace(x_vals.min(), x_vals.max(), n_bin_vis)
    t_grid = np.linspace(t_vals.min(), t_vals.max(), n_bin_vis)
    X_grid, T_grid = np.meshgrid(x_grid, t_grid)
    
    # Compute interpolated accuracy grids for all layers
    accuracy_grids = {}
    labels_np = class_labels if isinstance(class_labels, np.ndarray) else class_labels.cpu().numpy()
    points = np.column_stack([x_vals, t_vals])
    
    for layer_name, preds in predictions_dict.items():
        preds_np = preds if isinstance(preds, np.ndarray) else preds.cpu().numpy()
        correct = (preds_np == labels_np).astype(float)
        
        # Interpolate to dense grid
        accuracy_grid = griddata(
            points,
            correct,
            (X_grid, T_grid),
            method='cubic',
            fill_value=0.5
        )
        accuracy_grid = np.clip(accuracy_grid, 0, 1)
        accuracy_grids[layer_name] = accuracy_grid
    
    # Plot changes with smooth heatmaps
    for idx in range(n_transitions):
        ax = axes[idx]
        layer_prev = layer_names[idx]
        layer_curr = layer_names[idx + 1]
        
        change = accuracy_grids[layer_curr] - accuracy_grids[layer_prev]
        
        # Check if grid is valid for contourf
        if change.shape[0] < 2 or change.shape[1] < 2 or np.all(np.isnan(change)):
            # Skip this subplot if data is insufficient
            ax.text(0.5, 0.5, 'Insufficient data\nfor change visualization',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='gray')
            ax.axis('off')
            continue
        
        # Use TwoSlopeNorm for diverging colormap centered at 0
        norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
        
        # Plot smooth change heatmap using contourf
        contour = ax.contourf(X_grid, T_grid, change, levels=20,
                             cmap='RdYlGn', norm=norm)
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('t', fontsize=10)
        ax.set_title(f'{layer_prev} → {layer_curr}', fontsize=11)
        plt.colorbar(contour, ax=ax, label='Accuracy Change')
    
    # Hide unused subplots
    for idx in range(n_transitions, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  NCC accuracy changes (input space) saved to {save_path}")

