"""Plotting utilities for dataset visualization."""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import numpy as np


def plot_dataset(data: Dict[str, torch.Tensor], save_path: str, title: str = "Dataset Visualization") -> None:
    """
    Visualize a PINN dataset showing point distributions and ground truth.
    
    Args:
        data: Dataset dictionary with x, t, u_gt, and mask keys
        save_path: Path to save the figure
        title: Title for the plot
    """
    # Move data to CPU for plotting
    x = data['x'].cpu().numpy()
    t = data['t'].cpu().numpy()
    u_gt = data['u_gt'].cpu().numpy()
    mask_residual = data['mask']['residual'].cpu().numpy()
    mask_ic = data['mask']['IC'].cpu().numpy()
    mask_bc = data['mask']['BC'].cpu().numpy()
    
    spatial_dim = x.shape[1]
    
    # Create figure based on spatial dimension
    if spatial_dim == 1:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot 1: Point distribution in (x, t) space
        ax = axes[0]
        ax.scatter(x[mask_residual, 0], t[mask_residual, 0], 
                  c='blue', s=1, alpha=0.5, label='Residual')
        ax.scatter(x[mask_ic, 0], t[mask_ic, 0], 
                  c='green', s=10, alpha=0.7, label='IC (t=0)')
        ax.scatter(x[mask_bc, 0], t[mask_bc, 0], 
                  c='red', s=10, alpha=0.7, label='BC (boundaries)')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_title('Point Distribution')
        ax.legend(markerscale=3)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Ground truth u_gt[:, 0] in (x, t) space
        ax = axes[1]
        scatter = ax.scatter(x[:, 0], t[:, 0], c=u_gt[:, 0], 
                           s=2, cmap='viridis', alpha=0.6)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_title('Ground Truth u₀(x,t)')
        plt.colorbar(scatter, ax=ax)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Ground truth u_gt[:, 1] in (x, t) space
        ax = axes[2]
        scatter = ax.scatter(x[:, 0], t[:, 0], c=u_gt[:, 1], 
                           s=2, cmap='plasma', alpha=0.6)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_title('Ground Truth u₁(x,t)')
        plt.colorbar(scatter, ax=ax)
        ax.grid(True, alpha=0.3)
        
    else:
        # For higher dimensions, show projections or simplified views
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Point distribution - first 2 spatial dims
        ax = axes[0, 0]
        ax.scatter(x[mask_residual, 0], x[mask_residual, min(1, spatial_dim-1)], 
                  c='blue', s=1, alpha=0.5, label='Residual')
        ax.scatter(x[mask_ic, 0], x[mask_ic, min(1, spatial_dim-1)], 
                  c='green', s=10, alpha=0.7, label='IC')
        ax.scatter(x[mask_bc, 0], x[mask_bc, min(1, spatial_dim-1)], 
                  c='red', s=10, alpha=0.7, label='BC')
        ax.set_xlabel('x₀')
        ax.set_ylabel(f'x₁' if spatial_dim > 1 else 'x₀')
        ax.set_title('Spatial Point Distribution')
        ax.legend(markerscale=3)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Time distribution
        ax = axes[0, 1]
        ax.hist(t[mask_residual, 0], bins=30, alpha=0.5, label='Residual', color='blue')
        ax.hist(t[mask_ic, 0], bins=30, alpha=0.7, label='IC', color='green')
        ax.hist(t[mask_bc, 0], bins=30, alpha=0.7, label='BC', color='red')
        ax.set_xlabel('t')
        ax.set_ylabel('Count')
        ax.set_title('Temporal Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Ground truth component 0
        ax = axes[1, 0]
        scatter = ax.scatter(x[:, 0], t[:, 0], c=u_gt[:, 0], 
                           s=2, cmap='viridis', alpha=0.6)
        ax.set_xlabel('x₀')
        ax.set_ylabel('t')
        ax.set_title('Ground Truth u₀')
        plt.colorbar(scatter, ax=ax)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Ground truth component 1
        ax = axes[1, 1]
        scatter = ax.scatter(x[:, 0], t[:, 0], c=u_gt[:, 1], 
                           s=2, cmap='plasma', alpha=0.6)
        ax.set_xlabel('x₀')
        ax.set_ylabel('t')
        ax.set_title('Ground Truth u₁')
        plt.colorbar(scatter, ax=ax)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save figure
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Dataset visualization saved to {save_path}")


def plot_dataset_statistics(data: Dict[str, torch.Tensor], save_path: str) -> None:
    """
    Plot statistical information about the dataset.
    
    Args:
        data: Dataset dictionary
        save_path: Path to save the figure
    """
    x = data['x'].cpu().numpy()
    t = data['t'].cpu().numpy()
    u_gt = data['u_gt'].cpu().numpy()
    
    mask_residual = data['mask']['residual'].cpu().numpy()
    mask_ic = data['mask']['IC'].cpu().numpy()
    mask_bc = data['mask']['BC'].cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Point type distribution
    ax = axes[0, 0]
    counts = [mask_residual.sum(), mask_ic.sum(), mask_bc.sum()]
    labels = ['Residual', 'IC', 'BC']
    colors = ['blue', 'green', 'red']
    ax.bar(labels, counts, color=colors, alpha=0.7)
    ax.set_ylabel('Number of Points')
    ax.set_title('Point Type Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    for i, (label, count) in enumerate(zip(labels, counts)):
        ax.text(i, count, f'{count}', ha='center', va='bottom')
    
    # Ground truth distribution
    ax = axes[0, 1]
    ax.hist(u_gt[:, 0], bins=50, alpha=0.6, label='u₀', color='purple')
    ax.hist(u_gt[:, 1], bins=50, alpha=0.6, label='u₁', color='orange')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Ground Truth Value Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Spatial coverage
    ax = axes[1, 0]
    for dim in range(x.shape[1]):
        ax.hist(x[:, dim], bins=50, alpha=0.5, label=f'x_{dim}')
    ax.set_xlabel('Spatial Coordinate')
    ax.set_ylabel('Frequency')
    ax.set_title('Spatial Coverage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Temporal coverage
    ax = axes[1, 1]
    ax.hist(t[:, 0], bins=50, alpha=0.7, color='teal')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Frequency')
    ax.set_title('Temporal Coverage')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Dataset statistics saved to {save_path}")

