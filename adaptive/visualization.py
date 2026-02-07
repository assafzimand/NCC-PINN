"""Visualization for adaptive expert PINN regions.

Provides plotting functions for:
1. Per-run expert region visualization
2. Cross-experiment comparison of subdomain partitioning
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.interpolate import griddata
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import torch

from adaptive.indicators import RegionDescriptor


def prepare_ground_truth_grid(
    eval_data: Dict[str, torch.Tensor],
    domain_bounds: Dict[str, List[float]],
    resolution: int = 100
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Prepare ground truth data on a regular grid for visualization.

    Args:
        eval_data: Dictionary with 'x', 't', 'h' tensors
        domain_bounds: {'lower': [x_min, t_min], 'upper': [x_max, t_max]}
        resolution: Grid resolution for each dimension

    Returns:
        (ground_truth, grid_x, grid_t) or (None, None, None) if preparation fails
    """
    try:
        # Extract data (ground truth key can be 'h', 'h_gt', or 'u')
        x = eval_data['x'].cpu().numpy()
        t = eval_data['t'].cpu().numpy()

        # Try different keys for ground truth (expanded search)
        h = None
        gt_key_used = None
        for key in ['h_gt', 'h', 'u', 'u_gt', 'y', 'y_gt']:
            if key in eval_data:
                h = eval_data[key].cpu().numpy()
                gt_key_used = key
                break

        if h is None:
            print(f"  [prepare_ground_truth_grid] Warning: No ground truth key found.")
            print(f"  [prepare_ground_truth_grid] Available keys: {list(eval_data.keys())}")
            print(f"  [prepare_ground_truth_grid] Tried: h_gt, h, u, u_gt, y, y_gt")
            return None, None, None

        print(f"  [prepare_ground_truth_grid] Using key '{gt_key_used}' for ground truth")
        
        # Only support 2D domains (x, t) for now
        if len(domain_bounds['lower']) != 2:
            return None, None, None
        
        # Flatten x if multi-dimensional spatial
        if x.ndim > 1:
            x = x[:, 0]  # Take first spatial dimension for 1D problems
        
        x = x.ravel()
        t = t.ravel()
        
        # Create regular grid
        x_min, t_min = domain_bounds['lower']
        x_max, t_max = domain_bounds['upper']
        
        grid_x = np.linspace(x_min, x_max, resolution)
        grid_t = np.linspace(t_min, t_max, resolution)
        X_grid, T_grid = np.meshgrid(grid_x, grid_t, indexing='ij')
        
        # Interpolate ground truth onto grid
        points = np.column_stack([x, t])
        
        # Handle multi-dimensional output
        if h.ndim > 1 and h.shape[1] > 1:
            # Multi-output: compute norm
            h_display = np.linalg.norm(h, axis=1)
        else:
            h_display = h.ravel()
        
        # Interpolate using linear method
        ground_truth = griddata(points, h_display, (X_grid, T_grid), method='linear')

        # Fill NaN values with nearest neighbor
        mask = np.isnan(ground_truth)
        if mask.any():
            ground_truth_nn = griddata(points, h_display, (X_grid, T_grid), method='nearest')
            ground_truth[mask] = ground_truth_nn[mask]

        # Verify output
        if ground_truth is None or grid_x is None or grid_t is None:
            print(f"  [prepare_ground_truth_grid] ERROR: Output is None (grid preparation failed)")
            return None, None, None

        print(f"  [prepare_ground_truth_grid] Success: grid shape {ground_truth.shape}, "
              f"x range [{grid_x.min():.2f}, {grid_x.max():.2f}], "
              f"t range [{grid_t.min():.2f}, {grid_t.max():.2f}]")

        return ground_truth, grid_x, grid_t

    except Exception as e:
        import traceback
        print(f"  [prepare_ground_truth_grid] ERROR: Could not prepare ground truth grid")
        print(f"  [prepare_ground_truth_grid] Exception: {e}")
        print(f"  [prepare_ground_truth_grid] Traceback:")
        traceback.print_exc()
        return None, None, None


# Color palette for expert regions by depth
# Depth 1: Warm colors (reds/oranges)
# Depth 2: Cool colors (blues/greens)
# Depth 3+: Mixed colors (purples/teals)
DEPTH_COLOR_PALETTES = {
    1: ['#e74c3c', '#c0392b', '#ff6b6b', '#ff8c8c', '#f39c12', '#e67e22'],  # Warm reds/oranges
    2: ['#3498db', '#2980b9', '#5dade2', '#85c1e9', '#2ecc71', '#27ae60'],  # Cool blues/greens
    3: ['#9b59b6', '#8e44ad', '#a569bd', '#bb8fce', '#1abc9c', '#16a085'],  # Purples/teals
    4: ['#e91e63', '#c2185b', '#f06292', '#f48fb1', '#00bcd4', '#0097a7'],  # Pinks/cyans
    5: ['#ff5722', '#e64a19', '#ff7043', '#ff8a65', '#607d8b', '#455a64'],  # Deep orange/grey
}

def _get_color_for_depth(depth: int, index_at_depth: int) -> str:
    """Get color for an expert based on its depth and index within that depth."""
    palette = DEPTH_COLOR_PALETTES.get(depth, DEPTH_COLOR_PALETTES[5])
    return palette[index_at_depth % len(palette)]

# Legacy color palette (for backwards compatibility)
EXPERT_COLORS = [
    '#e74c3c',  # Red
    '#3498db',  # Blue
    '#2ecc71',  # Green
    '#f39c12',  # Orange
    '#9b59b6',  # Purple
    '#1abc9c',  # Teal
    '#e91e63',  # Pink
    '#00bcd4',  # Cyan
    '#ff5722',  # Deep Orange
    '#607d8b',  # Blue Grey
]


def plot_expert_regions(
    regions: List[RegionDescriptor],
    domain_bounds: Dict[str, List[float]],
    output_path: Union[str, Path],
    problem_type: str = '2d',
    title: Optional[str] = None,
    ground_truth: Optional[np.ndarray] = None,
    grid_x: Optional[np.ndarray] = None,
    grid_t: Optional[np.ndarray] = None
) -> None:
    """
    Plot domain with expert region outlines.
    
    Args:
        regions: List of RegionDescriptor for each expert
        domain_bounds: {'lower': [x_min, t_min], 'upper': [x_max, t_max]}
        output_path: Path to save the plot
        problem_type: '2d' (x,t) or '3d' (x,y,t)
        title: Optional plot title
        ground_truth: Optional (Nx, Nt) or (Nx, Nt, output_dim) array of ground truth
        grid_x: Optional 1D array of x coordinates for ground truth
        grid_t: Optional 1D array of t coordinates for ground truth
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if problem_type == '2d':
        _plot_expert_regions_2d(regions, domain_bounds, output_path, title,
                                ground_truth, grid_x, grid_t)
    elif problem_type == '3d':
        _plot_expert_regions_3d(regions, domain_bounds, output_path, title)
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}. Use '2d' or '3d'.")


def _plot_expert_regions_2d(
    regions: List[RegionDescriptor],
    domain_bounds: Dict[str, List[float]],
    output_path: Path,
    title: Optional[str] = None,
    ground_truth: Optional[np.ndarray] = None,
    grid_x: Optional[np.ndarray] = None,
    grid_t: Optional[np.ndarray] = None
) -> None:
    """Plot 2D domain (x vs t) with expert regions colored by depth."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Domain bounds
    x_min, t_min = domain_bounds['lower']
    x_max, t_max = domain_bounds['upper']
    
    # Plot ground truth as background if provided
    if ground_truth is not None and grid_x is not None and grid_t is not None:
        # Handle multi-dimensional output (use norm)
        if ground_truth.ndim == 3:
            # (Nx, Nt, output_dim) -> compute norm
            gt_display = np.linalg.norm(ground_truth, axis=2)
        else:
            gt_display = ground_truth
        
        # Create meshgrid and plot
        T, X = np.meshgrid(grid_t, grid_x)
        im = ax.pcolormesh(X, T, gt_display, shading='auto', cmap='viridis', 
                          alpha=0.7, zorder=0)
        cbar = plt.colorbar(im, ax=ax, label='Ground Truth (amplitude)', shrink=0.8)
    else:
        # Draw domain rectangle as fallback
        domain_rect = patches.Rectangle(
            (x_min, t_min), x_max - x_min, t_max - t_min,
            linewidth=2, edgecolor='black', facecolor='#f0f0f0',
            label='Domain', zorder=1
        )
        ax.add_patch(domain_rect)
    
    # Draw expert regions as simple black outlines (no fill, no legend)
    for region in regions:
        rx_min, rt_min = region.bounds_lower
        rx_max, rt_max = region.bounds_upper
        
        # Simple black outline, no fill
        outline_rect = patches.Rectangle(
            (rx_min, rt_min), rx_max - rx_min, rt_max - rt_min,
            linewidth=1.5, edgecolor='black', facecolor='none',
            zorder=10
        )
        ax.add_patch(outline_rect)
    
    # Set axis limits with padding
    padding = 0.05
    x_range = x_max - x_min
    t_range = t_max - t_min
    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(t_min - padding * t_range, t_max + padding * t_range)
    
    # Add depth summary to title
    max_depth = max((getattr(r, 'depth', 1) for r in regions), default=0) if regions else 0
    depth_info = f", max depth={max_depth}" if max_depth > 0 else ""
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('t', fontsize=12)
    ax.set_title(title or f'Expert Regions ({len(regions)} experts{depth_info})', fontsize=14)
    ax.set_aspect('auto')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Expert regions plot saved to {output_path}")


def _plot_expert_regions_3d(
    regions: List[RegionDescriptor],
    domain_bounds: Dict[str, List[float]],
    output_path: Path,
    title: Optional[str] = None
) -> None:
    """Plot 3D domain (x, y, t) with expert regions colored by depth."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Domain bounds
    x_min, y_min, t_min = domain_bounds['lower']
    x_max, y_max, t_max = domain_bounds['upper']
    
    # Draw domain wireframe
    _draw_box_3d(ax, [x_min, y_min, t_min], [x_max, y_max, t_max],
                 color='black', alpha=0.3, linewidth=1, label='Domain')
    
    # Draw expert regions as simple black wireframes (no color, no legend)
    for region in regions:
        _draw_box_3d(ax, region.bounds_lower, region.bounds_upper,
                     color='black', alpha=1.0, linewidth=1.5, label=None)
    
    # Add depth summary to title
    max_depth = max((getattr(r, 'depth', 1) for r in regions), default=0) if regions else 0
    depth_info = f", max depth={max_depth}" if max_depth > 0 else ""
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('t', fontsize=12)
    ax.set_title(title or f'Expert Regions ({len(regions)} experts{depth_info})', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Expert regions 3D plot saved to {output_path}")


def _draw_box_3d(ax, lower, upper, color='blue', alpha=0.3, linewidth=1, label=None):
    """Draw a 3D wireframe box."""
    x_min, y_min, z_min = lower
    x_max, y_max, z_max = upper
    
    # Define the vertices of the box
    vertices = [
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max],
    ]
    
    # Define the 6 faces
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
    ]
    
    # Add faces with transparency
    face_collection = Poly3DCollection(faces, alpha=alpha, facecolor=color,
                                        edgecolor=color, linewidth=linewidth)
    ax.add_collection3d(face_collection)
    
    # Add a dummy line for the legend
    if label:
        ax.plot([], [], [], color=color, linewidth=2, label=label)


def plot_expert_regions_comparison(
    experiment_regions: Dict[str, List[RegionDescriptor]],
    domain_bounds: Dict[str, List[float]],
    output_path: Union[str, Path],
    problem_type: str = '2d',
    ground_truth: Optional[np.ndarray] = None,
    grid_x: Optional[np.ndarray] = None,
    grid_t: Optional[np.ndarray] = None
) -> None:
    """
    Side-by-side comparison of expert regions across experiments.
    
    Args:
        experiment_regions: Dict mapping experiment name to list of RegionDescriptors
        domain_bounds: {'lower': [x_min, t_min], 'upper': [x_max, t_max]}
        output_path: Path to save the comparison plot
        problem_type: '2d' or '3d'
        ground_truth: Optional (Nx, Nt) or (Nx, Nt, output_dim) array of ground truth
        grid_x: Optional 1D array of x coordinates for ground truth
        grid_t: Optional 1D array of t coordinates for ground truth
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    n_experiments = len(experiment_regions)
    if n_experiments == 0:
        print("  No experiments with expert regions to compare.")
        return
    
    # Calculate grid dimensions
    n_cols = min(3, n_experiments)
    n_rows = (n_experiments + n_cols - 1) // n_cols
    
    if problem_type == '2d':
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
        if n_experiments == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Domain bounds
        x_min, t_min = domain_bounds['lower']
        x_max, t_max = domain_bounds['upper']
        
        # Prepare ground truth display
        gt_display = None
        if ground_truth is not None and grid_x is not None and grid_t is not None:
            if ground_truth.ndim == 3:
                gt_display = np.linalg.norm(ground_truth, axis=2)
            else:
                gt_display = ground_truth
            T, X = np.meshgrid(grid_t, grid_x)
        
        for idx, (exp_name, regions) in enumerate(experiment_regions.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            # Draw ground truth background if available
            if gt_display is not None:
                im = ax.pcolormesh(X, T, gt_display, shading='auto', cmap='viridis',
                                  alpha=0.7, zorder=0)
            else:
                # Draw domain rectangle as fallback
                domain_rect = patches.Rectangle(
                    (x_min, t_min), x_max - x_min, t_max - t_min,
                    linewidth=2, edgecolor='black', facecolor='#f0f0f0', zorder=1
                )
                ax.add_patch(domain_rect)
            
            # Draw expert regions colored by depth, with recency-based transparency
            depth_counts = {}
            
            # Compute recency for this experiment's regions
            if regions:
                spawn_epochs = [r.spawn_epoch for r in regions]
                min_epoch = min(spawn_epochs)
                max_epoch = max(spawn_epochs)
                epoch_range = max_epoch - min_epoch if max_epoch > min_epoch else 1
            
            for i, region in enumerate(regions):
                depth = getattr(region, 'depth', 1)
                
                # Get index within this depth
                if depth not in depth_counts:
                    depth_counts[depth] = 0
                index_at_depth = depth_counts[depth]
                depth_counts[depth] += 1
                
                color = _get_color_for_depth(depth, index_at_depth)
                
                rx_min, rt_min = region.bounds_lower
                rx_max, rt_max = region.bounds_upper
                
                # Recency factor: older = more transparent
                recency = (region.spawn_epoch - min_epoch) / epoch_range if epoch_range > 0 else 1.0
                fill_alpha = 0.1 + 0.2 * recency
                outline_alpha = 0.4 + 0.6 * recency
                outline_linewidth = 1.0 + 2.0 * recency
                
                expert_rect = patches.Rectangle(
                    (rx_min, rt_min), rx_max - rx_min, rt_max - rt_min,
                    linewidth=2, edgecolor=color, facecolor=color,
                    alpha=fill_alpha, zorder=2 + depth
                )
                ax.add_patch(expert_rect)
                
                outline_rect = patches.Rectangle(
                    (rx_min, rt_min), rx_max - rx_min, rt_max - rt_min,
                    linewidth=outline_linewidth, edgecolor=color, facecolor='none',
                    alpha=outline_alpha,
                    label=f'E{i+1} (d={depth})', zorder=10 + depth
                )
                ax.add_patch(outline_rect)
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(t_min, t_max)
            ax.set_xlabel('x', fontsize=10)
            ax.set_ylabel('t', fontsize=10)
            ax.set_title(f'{exp_name}\n({len(regions)} experts)', fontsize=11)
            ax.set_aspect('auto')
            ax.grid(True, alpha=0.3)
            
            # Add legend to each subplot
            if len(regions) > 0:
                ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        
        # Hide unused subplots
        for idx in range(n_experiments, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        fig.suptitle('Expert Regions Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    elif problem_type == '3d':
        # For 3D, create separate subplots
        fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
        
        x_min, y_min, t_min = domain_bounds['lower']
        x_max, y_max, t_max = domain_bounds['upper']
        
        for idx, (exp_name, regions) in enumerate(experiment_regions.items()):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
            
            # Draw domain wireframe
            _draw_box_3d(ax, domain_bounds['lower'], domain_bounds['upper'],
                         color='black', alpha=0.1, linewidth=1)
            
            # Draw expert regions colored by depth, with recency-based transparency
            depth_counts = {}
            
            # Compute recency for this experiment's regions
            if regions:
                spawn_epochs = [r.spawn_epoch for r in regions]
                min_epoch = min(spawn_epochs)
                max_epoch = max(spawn_epochs)
                epoch_range = max_epoch - min_epoch if max_epoch > min_epoch else 1
            
            for i, region in enumerate(regions):
                depth = getattr(region, 'depth', 1)
                
                if depth not in depth_counts:
                    depth_counts[depth] = 0
                index_at_depth = depth_counts[depth]
                depth_counts[depth] += 1
                
                color = _get_color_for_depth(depth, index_at_depth)
                
                # Recency factor: older = more transparent
                recency = (region.spawn_epoch - min_epoch) / epoch_range if epoch_range > 0 else 1.0
                alpha = 0.1 + 0.2 * recency
                linewidth = 1.0 + 2.0 * recency
                
                _draw_box_3d(ax, region.bounds_lower, region.bounds_upper,
                             color=color, alpha=alpha, linewidth=linewidth,
                             label=f'E{i+1} (d={depth})')
            
            ax.set_xlabel('x', fontsize=9)
            ax.set_ylabel('y', fontsize=9)
            ax.set_zlabel('t', fontsize=9)
            ax.set_title(f'{exp_name}\n({len(regions)} experts)', fontsize=10)
            
            # Add legend to each 3D subplot
            if len(regions) > 0:
                ax.legend(loc='upper left', fontsize=8)
        
        fig.suptitle('Expert Regions Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  Expert regions comparison saved to {output_path}")


def save_regions_metadata(
    regions: List[RegionDescriptor],
    output_path: Union[str, Path]
) -> None:
    """
    Save expert regions metadata to JSON file.
    
    Args:
        regions: List of RegionDescriptor
        output_path: Path to save JSON file
    """
    import json
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'n_experts': len(regions),
        'regions': [r.to_dict() for r in regions]
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  Expert regions metadata saved to {output_path}")


def load_regions_metadata(input_path: Union[str, Path]) -> List[RegionDescriptor]:
    """
    Load expert regions metadata from JSON file.
    
    Args:
        input_path: Path to JSON file
        
    Returns:
        List of RegionDescriptor
    """
    import json
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    return [RegionDescriptor.from_dict(r) for r in data['regions']]


def plot_expert_soft_weights(
    model,  # AdaptiveExpertPINN with soft blending
    domain_bounds: Dict[str, List[float]],
    output_path: Union[str, Path],
    resolution: int = 100,
    ground_truth: Optional[np.ndarray] = None,
    grid_x: Optional[np.ndarray] = None,
    grid_t: Optional[np.ndarray] = None,
    title_prefix: str = ""
) -> None:
    """
    Plot heatmaps of soft blending weights (partition of unity) for each expert.
    
    Uses a red colormap where:
    - Darker red = higher weight (more influence)
    - Lighter/white = lower weight (less influence)
    
    Args:
        model: AdaptiveExpertPINN instance with soft blending enabled
        domain_bounds: {'lower': [x_min, t_min], 'upper': [x_max, t_max]}
        output_path: Path to save the plot
        resolution: Grid resolution for each dimension
        ground_truth: Optional ground truth array for reference
        grid_x: Optional x grid for ground truth
        grid_t: Optional t grid for ground truth
        title_prefix: Optional prefix for plot title
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Only support 2D domains (x, t) for now
    if len(domain_bounds['lower']) != 2:
        print(f"  Warning: Soft weight visualization only supports 2D domains")
        return
    
    # Check if model has soft blending
    if not hasattr(model, 'base_indicator') or model.base_indicator is None:
        print(f"  Warning: Model does not have soft blending enabled")
        return
    
    # Create evaluation grid
    x_min, t_min = domain_bounds['lower']
    x_max, t_max = domain_bounds['upper']
    
    eval_grid_x = np.linspace(x_min, x_max, resolution)
    eval_grid_t = np.linspace(t_min, t_max, resolution)
    X, T = np.meshgrid(eval_grid_x, eval_grid_t, indexing='ij')
    
    # Flatten for model input
    device = next(model.parameters()).device
    inputs = torch.tensor(
        np.column_stack([X.ravel(), T.ravel()]),
        dtype=torch.float32, 
        device=device
    )
    
    # Compute normalized weights
    with torch.no_grad():
        # Get decomposed output with weights
        decomposed = model.forward_decomposed(inputs)
        weights_norm = decomposed.get('weights_normalized', {})
        
        if not weights_norm:
            print(f"  Warning: No normalized weights available (model may use hard blending)")
            return
        
        # Extract weights and reshape to grid
        psi_base_norm = weights_norm['base'].cpu().numpy().reshape(X.shape)
        psi_experts_norm = []
        for i in range(model.num_experts):
            key = f'expert_{i}'
            if key in weights_norm:
                psi_k_norm = weights_norm[key].cpu().numpy().reshape(X.shape)
                psi_experts_norm.append(psi_k_norm)
    
    # Create subplots: base + each expert
    n_plots = 1 + len(psi_experts_norm)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    axes = axes.flatten()
    
    # Red colormap (darker = higher weight)
    cmap = plt.cm.Reds
    
    # Plot base model weight
    im = axes[0].pcolormesh(eval_grid_x, eval_grid_t, psi_base_norm.T,
                            cmap=cmap, vmin=0, vmax=1, shading='auto')
    axes[0].set_title('Base Model Weight (ψ̃₀)', fontsize=11)
    axes[0].set_xlabel('x', fontsize=10)
    axes[0].set_ylabel('t', fontsize=10)
    plt.colorbar(im, ax=axes[0], label='Normalized Weight')
    axes[0].grid(True, alpha=0.3)
    
    # Plot expert weights
    for i, psi_norm in enumerate(psi_experts_norm):
        ax = axes[i + 1]
        im = ax.pcolormesh(eval_grid_x, eval_grid_t, psi_norm.T,
                          cmap=cmap, vmin=0, vmax=1, shading='auto')
        
        # Get region info for title
        if i < len(model.regions):
            region = model.regions[i]
            depth = region.depth
            ax.set_title(f'Expert {i+1} Weight (ψ̃_{i+1}, depth={depth})', fontsize=11)
            
            # Draw region boundary as dashed black line
            lo, hi = region.bounds_lower, region.bounds_upper
            rect = patches.Rectangle(
                (lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1],
                linewidth=2, edgecolor='black', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)
        else:
            ax.set_title(f'Expert {i+1} Weight (ψ̃_{i+1})', fontsize=11)
        
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('t', fontsize=10)
        plt.colorbar(im, ax=ax, label='Normalized Weight')
        ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)
    
    # Overall title
    title = f'{title_prefix}Soft Blending Weights (Partition of Unity)' if title_prefix else 'Soft Blending Weights (Partition of Unity)'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Soft blending weights plot saved to {output_path}")