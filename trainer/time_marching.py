"""
Time Marching Module for PINN Training.

Implements sequential training over temporal windows, essential for chaotic PDEs
like Kuramoto-Sivashinsky where standard PINNs fail due to error accumulation.

Key idea: Split temporal domain into windows, train AToE on each window sequentially,
using the previous window's terminal prediction as the next window's initial condition.
"""

import copy
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import importlib


@dataclass
class TimeWindow:
    """Represents a single time window for time marching."""
    idx: int          # 0, 1, 2, ...
    t_start: float    # start of window
    t_end: float      # end of window
    is_first: bool    # True for window 0
    M: int            # experts allocated to this window


def compute_m_per_window(global_M: int, num_windows: int, distribution: str) -> List[int]:
    """
    Distribute global_M experts across windows based on distribution strategy.
    
    Args:
        global_M: Total number of experts to distribute
        num_windows: Number of time windows
        distribution: 'equal' | 'linear' | 'quadratic'
    
    Returns:
        List of M values for each window, summing to global_M
    
    Examples (global_M=40, num_windows=5):
        - equal: [8, 8, 8, 8, 8]
        - linear: [3, 5, 8, 11, 13]
        - quadratic: [1, 4, 7, 13, 15]
    """
    if distribution == 'equal':
        base = global_M // num_windows
        remainder = global_M % num_windows
        result = [base] * num_windows
        result[-1] += remainder  # add remainder to last window
        return result
    
    elif distribution == 'linear':
        # M_i proportional to (i+1)
        # Sum of 1+2+...+n = n*(n+1)/2
        weights = [(i + 1) for i in range(num_windows)]
        total_weight = sum(weights)
        raw = [global_M * w / total_weight for w in weights]
        result = [max(1, round(r)) for r in raw]
        # Adjust to ensure sum equals global_M
        diff = global_M - sum(result)
        result[-1] += diff
        return result
    
    elif distribution == 'quadratic':
        # M_i proportional to (i+1)^2
        # Sum of 1^2+2^2+...+n^2 = n*(n+1)*(2n+1)/6
        weights = [(i + 1) ** 2 for i in range(num_windows)]
        total_weight = sum(weights)
        raw = [global_M * w / total_weight for w in weights]
        result = [max(1, round(r)) for r in raw]
        # Adjust to ensure sum equals global_M
        diff = global_M - sum(result)
        result[-1] += diff
        return result
    
    else:
        raise ValueError(f"Unknown m_distribution: {distribution}. Use 'equal', 'linear', or 'quadratic'.")


def compute_time_windows(
    temporal_domain: List[float], 
    num_windows: int,
    global_M: int,
    m_distribution: str
) -> List[TimeWindow]:
    """
    Split [t_min, t_max] into num_windows equal, non-overlapping windows with M allocation.
    
    Args:
        temporal_domain: [t_min, t_max] from config
        num_windows: Number of windows to create
        global_M: Total experts to distribute
        m_distribution: Distribution strategy
    
    Returns:
        List of TimeWindow objects
    """
    t_min, t_max = temporal_domain
    dt = (t_max - t_min) / num_windows
    m_values = compute_m_per_window(global_M, num_windows, m_distribution)
    
    windows = []
    for i in range(num_windows):
        windows.append(TimeWindow(
            idx=i,
            t_start=t_min + i * dt,
            t_end=t_min + (i + 1) * dt,
            is_first=(i == 0),
            M=m_values[i]
        ))
    return windows


def narrow_config_for_window(cfg: Dict, window: TimeWindow, prev_model: nn.Module = None) -> Dict:
    """
    Create a copy of cfg with temporal_domain and M narrowed for this window.
    
    This is the key trick that makes everything work:
    - Dataset generation uses temporal_domain → generates points in [t_start, t_end]
    - Tree spawning uses domain_bounds from data → automatically matches window
    - Resampling uses temporal_domain → stays within window
    - M_experts_num is set per-window for variable expert allocation
    
    Args:
        cfg: Full configuration dictionary
        window: TimeWindow to narrow to
        prev_model: Model from previous window (for IC override during resampling)
    
    Returns:
        Deep copy of cfg with temporal_domain and M_experts_num updated
    """
    window_cfg = copy.deepcopy(cfg)
    problem = window_cfg['problem']
    
    # Narrow temporal domain
    window_cfg[problem]['temporal_domain'] = [window.t_start, window.t_end]
    
    # Set window-specific M
    window_cfg['adaptive_pinn']['M_experts_num'] = window.M
    
    # Add flag to indicate time marching is active (for eval filtering and IC override)
    # prev_model is stored as reference for IC override after resampling
    window_cfg['_time_marching_window'] = {
        'enabled': True,
        't_start': window.t_start,
        't_end': window.t_end,
        'idx': window.idx,
        'prev_model': prev_model  # None for window 0, model for windows 1+
    }
    
    return window_cfg


def override_ic_with_model(
    dataset: Dict[str, torch.Tensor],
    prev_model: nn.Module,
    window: TimeWindow,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Replace h_gt for IC points with predictions from prev_model.
    
    This is the key trick: IC loss in *_loss.py uses h_gt from batch.
    By replacing h_gt for IC points, we get predicted IC loss for free.
    
    The existing loss_weights.ic from the problem config is used automatically
    since the loss computation mechanism stays unchanged.
    
    Args:
        dataset: Training or eval dataset dict with 'x', 't', 'h_gt', 'mask'
        prev_model: Model from previous window to query
        window: Current window (to check if first)
        device: Device to run inference on
    
    Returns:
        Modified dataset with h_gt overridden for IC points
    """
    if window.is_first:
        return dataset  # Window 0 uses analytical IC
    
    # Get IC mask
    ic_mask = dataset['mask']['IC']
    if ic_mask.sum() == 0:
        print(f"    [IC Override] Window {window.idx}: No IC points found in dataset, skipping")
        return dataset
    
    x_ic = dataset['x'][ic_mask]  # (n_ic, spatial_dim)
    t_ic = dataset['t'][ic_mask]  # (n_ic, 1) - all at window.t_start
    h_gt_original = dataset['h_gt'][ic_mask].clone()
    
    # Diagnostic: print input stats
    print(f"    [IC Override] Window {window.idx}: Overriding {ic_mask.sum().item()} IC points")
    print(f"      x_ic: shape={x_ic.shape}, min={x_ic.min().item():.4f}, max={x_ic.max().item():.4f}, mean={x_ic.mean().item():.4f}")
    print(f"      t_ic: min={t_ic.min().item():.4f}, max={t_ic.max().item():.4f}")
    print(f"      h_gt (original): min={h_gt_original.min().item():.4f}, max={h_gt_original.max().item():.4f}, mean={h_gt_original.mean().item():.4f}")
    
    # Query previous model (no gradients)
    prev_model.eval()
    with torch.no_grad():
        inputs = torch.cat([x_ic, t_ic], dim=1).to(device)
        h_pred = prev_model(inputs)
    
    # Diagnostic: print prediction stats
    has_nan = torch.isnan(h_pred).any().item()
    has_inf = torch.isinf(h_pred).any().item()
    print(f"      h_pred: min={h_pred.min().item():.4f}, max={h_pred.max().item():.4f}, mean={h_pred.mean().item():.4f}")
    print(f"      h_pred contains NaN: {has_nan}, Inf: {has_inf}")
    
    if has_nan or has_inf:
        print(f"      [WARNING] Previous model produced invalid values! This will cause NaN divergence.")
        num_nan = torch.isnan(h_pred).sum().item()
        num_inf = torch.isinf(h_pred).sum().item()
        print(f"      Number of NaN: {num_nan}, Number of Inf: {num_inf}")
    
    # Override h_gt for IC points
    dataset['h_gt'][ic_mask] = h_pred.to(dataset['h_gt'].device)
    
    print(f"    [IC Override] Completed: overrode {ic_mask.sum().item()} IC points")
    
    return dataset


def _plot_combined_loss_curves(windows: List[TimeWindow], run_dir: Path) -> None:
    """
    Create a combined loss curve plot showing all windows concatenated.
    
    Reads metrics.json from each window and creates a single plot with:
    - Train loss and eval loss curves concatenated across windows
    - Vertical lines showing window boundaries
    
    Args:
        windows: List of TimeWindow objects
        run_dir: Root run directory containing window subdirectories
    """
    print(f"\n  Creating combined loss curve plot...")
    
    all_train_epochs = []
    all_train_loss = []
    all_eval_epochs = []
    all_eval_loss = []
    all_eval_rel_l2 = []
    
    epoch_offset = 0
    window_boundaries = [0]  # Epoch boundaries between windows
    
    for window in windows:
        window_metrics_path = run_dir / f"window_{window.idx}" / "metrics.json"
        if not window_metrics_path.exists():
            print(f"    Warning: metrics.json not found for window {window.idx}")
            continue
        
        with open(window_metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Offset epochs to create continuous timeline
        train_epochs = np.array(metrics['train_loss_epochs']) + epoch_offset
        eval_epochs = np.array(metrics['epochs']) + epoch_offset
        
        all_train_epochs.extend(train_epochs)
        all_train_loss.extend(metrics['train_loss'])
        all_eval_epochs.extend(eval_epochs)
        all_eval_loss.extend(metrics['eval_loss'])
        all_eval_rel_l2.extend(metrics['eval_rel_l2'])
        
        # Update offset for next window
        if len(train_epochs) > 0:
            epoch_offset = train_epochs[-1]
            window_boundaries.append(epoch_offset)
    
    if len(all_train_epochs) == 0:
        print(f"    Warning: No metrics found for any window")
        return
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Plot 1: Loss curves
    ax = axes[0]
    ax.plot(all_train_epochs, all_train_loss, 'b-', label='Train Loss',
            linewidth=2, alpha=0.8)
    ax.plot(all_eval_epochs, all_eval_loss, 'r-', label='Eval Loss',
            linewidth=2, alpha=0.8)
    
    # Add window boundary markers
    for i, boundary in enumerate(window_boundaries[1:-1], start=1):
        ax.axvline(x=boundary, color='gray', linestyle='--', 
                   linewidth=1.5, alpha=0.5)
        ax.text(boundary, ax.get_ylim()[1]*0.95, f'W{i}', 
                ha='center', va='top', fontsize=9, alpha=0.7)
    
    ax.set_xlabel('Epoch (Cumulative)', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_title(f'Time Marching: Combined Loss Curves [{len(windows)} windows]', 
                 fontsize=14, fontweight='bold')
    
    # Plot 2: Relative L2 error
    ax = axes[1]
    ax.plot(all_eval_epochs, all_eval_rel_l2, 'r-', label='Eval Rel. L2',
            linewidth=2, alpha=0.8)
    
    # Add window boundary markers
    for i, boundary in enumerate(window_boundaries[1:-1], start=1):
        ax.axvline(x=boundary, color='gray', linestyle='--', 
                   linewidth=1.5, alpha=0.5)
        ax.text(boundary, ax.get_ylim()[1]*0.95, f'W{i}', 
                ha='center', va='top', fontsize=9, alpha=0.7)
    
    ax.set_xlabel('Epoch (Cumulative)', fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_title(f'Time Marching: Combined Relative L2 Error', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    save_path = run_dir / 'time_marching_combined_loss_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Combined loss curves saved to {save_path}")


def _plot_combined_heatmap(
    combined_model: nn.Module,
    config: Dict,
    run_dir: Path,
    device: torch.device
) -> None:
    """
    Create a global heatmap showing prediction and error vs ground truth.
    
    Uses the combined TimeMarchingModel to generate predictions across the
    entire temporal domain and compares with ground truth.
    
    Args:
        combined_model: TimeMarchingModel wrapping all windows
        config: Full configuration dictionary
        run_dir: Directory to save the plot
        device: Device for inference
    """
    from utils.problem_specific.generic_viz import plot_predictions_and_error_maps
    
    print(f"\n  Creating combined prediction heatmap...")
    
    try:
        plot_predictions_and_error_maps(
            model=combined_model,
            save_dir=run_dir,
            config=config,
            filename="time_marching_combined_heatmap.png",
            n_x=256,
            n_t=200
        )
        print(f"    Combined heatmap saved")
    except Exception as e:
        print(f"    Warning: Could not create combined heatmap: {e}")


def train_with_time_marching(
    model_class,
    architecture: List[int],
    activation: str,
    config: Dict,
    adaptive_cfg: Dict,
    run_dir: Path,
    device: torch.device,
) -> Tuple[nn.Module, Path]:
    """
    Train separate AToE models for each time window, then combine.
    
    This orchestrator:
    1. Computes time windows with M allocation
    2. For each window:
       - Narrows config (temporal_domain, M_experts_num)
       - Generates datasets for narrowed domain
       - If not first window: overrides IC h_gt with prev_model predictions
       - Creates fresh model
       - Calls existing train() as black box
       - Optionally freezes model
    3. Wraps all window models in TimeMarchingModel
    
    Args:
        model_class: Class to instantiate (AToE, ANT, or AToELeaves)
        architecture: Base architecture
        activation: Activation function name
        config: Full configuration dictionary
        adaptive_cfg: Adaptive PINN config section
        run_dir: Output directory for this run
        device: CUDA device
    
    Returns:
        Tuple of (combined_model, best_checkpoint_path)
    """
    from trainer.trainer import train
    from utils.dataset_gen import generate_and_save_datasets
    from models.time_marching_model import TimeMarchingModel
    
    problem = config['problem']
    tm_cfg = config[problem]['time_marching']
    global_M = config['adaptive_pinn']['M_experts_num']
    
    # Compute time windows with M allocation
    windows = compute_time_windows(
        config[problem]['temporal_domain'],
        tm_cfg['num_windows'],
        global_M,
        tm_cfg['m_distribution']
    )
    
    # Log M distribution
    print(f"\n{'='*60}")
    print(f"  TIME MARCHING: {len(windows)} windows, global_M={global_M}")
    print(f"  Distribution ({tm_cfg['m_distribution']}): {[w.M for w in windows]}")
    print(f"  Temporal ranges:")
    for w in windows:
        print(f"    Window {w.idx}: t in [{w.t_start:.4f}, {w.t_end:.4f}], M={w.M}")
    print(f"{'='*60}")
    
    window_models: List[Tuple[TimeWindow, nn.Module]] = []
    prev_model = None
    last_checkpoint_path = None
    
    for window in windows:
        print(f"\n{'='*60}")
        print(f"  WINDOW {window.idx + 1}/{len(windows)}: t in [{window.t_start:.4f}, {window.t_end:.4f}]")
        print(f"  M_experts_num = {window.M}")
        print(f"{'='*60}")
        
        # 1. Narrow config for this window (pass prev_model for IC override during resampling)
        window_cfg = narrow_config_for_window(config, window, prev_model=prev_model)
        window_run_dir = run_dir / f"window_{window.idx}"
        window_run_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Generate datasets with narrowed domain
        print(f"\n  Generating datasets for window {window.idx}...")
        generate_and_save_datasets(window_cfg)
        
        # NOTE: IC override for windows 1+ is now handled in-memory by trainer.py
        # (_override_ic_for_time_marching) which runs before filtering and after each resample.
        # This avoids corrupting the disk dataset if a previous window diverged with NaN.
        
        # 3. Create fresh model for this window
        print(f"\n  Creating model for window {window.idx}...")
        window_model = model_class(architecture, activation, window_cfg, window_cfg['adaptive_pinn'])
        window_model = window_model.to(device)
        
        # Convert to double precision if configured
        precision = window_cfg.get('precision', 'float32')
        if precision == 'float64':
            window_model = window_model.double()
        
        print(f"  {type(window_model).__name__} created")
        
        # 4. Build loss function for this window
        loss_module = importlib.import_module(f"losses.{problem}_loss")
        loss_fn = loss_module.build_loss(**window_cfg)
        
        # 5. Call existing train() as black box
        print(f"\n  Training window {window.idx}...")
        train_data_path = f"datasets/{problem}/training_data.pt"
        eval_data_path = f"datasets/{problem}/eval_data.pt"
        
        checkpoint_path = train(
            model=window_model,
            loss_fn=loss_fn,
            train_data_path=train_data_path,
            eval_data_path=eval_data_path,
            cfg=window_cfg,
            run_dir=window_run_dir,
        )
        
        # 6. Save window-specific checkpoint
        window_checkpoint = {
            'window_idx': window.idx,
            'window': {
                't_start': window.t_start,
                't_end': window.t_end,
                'M': window.M,
            },
            'model_state_dict': window_model.state_dict(),
            'is_adaptive': True,
            'adaptive_state': window_model.get_state_dict_extended() if hasattr(window_model, 'get_state_dict_extended') else None,
        }
        window_checkpoint_path = window_run_dir / f"window_{window.idx}_final.pt"
        torch.save(window_checkpoint, window_checkpoint_path)
        print(f"  Window checkpoint saved: {window_checkpoint_path}")
        
        # 7. Optionally freeze for memory savings
        if tm_cfg['freeze_previous_windows']:
            print(f"  Freezing window {window.idx} model parameters")
            for p in window_model.parameters():
                p.requires_grad = False
            window_model.eval()
        
        window_models.append((window, window_model))
        prev_model = window_model
        last_checkpoint_path = checkpoint_path
    
    # 9. Combine into TimeMarchingModel
    print(f"\n{'='*60}")
    print(f"  Creating combined TimeMarchingModel")
    print(f"{'='*60}")
    combined_model = TimeMarchingModel(window_models)
    
    # 10. Save combined model checkpoint
    combined_checkpoint = {
        'is_time_marching': True,
        'num_windows': len(windows),
        'windows': [
            {'idx': w.idx, 't_start': w.t_start, 't_end': w.t_end, 'M': w.M}
            for w, _ in window_models
        ],
        'window_checkpoints': [
            str(run_dir / f"window_{w.idx}" / f"window_{w.idx}_final.pt")
            for w, _ in window_models
        ],
    }
    combined_checkpoint_path = run_dir / "time_marching_combined.pt"
    torch.save(combined_checkpoint, combined_checkpoint_path)
    print(f"  Combined checkpoint saved: {combined_checkpoint_path}")
    
    # 11. Create combined visualizations
    print(f"\n{'='*60}")
    print(f"  Generating time marching visualizations")
    print(f"{'='*60}")
    
    # Plot combined loss curves from all windows
    _plot_combined_loss_curves(windows, run_dir)
    
    # Plot combined prediction heatmap vs ground truth
    _plot_combined_heatmap(combined_model, config, run_dir, device)
    
    print(f"\n{'='*60}")
    print(f"  Time marching training complete!")
    print(f"{'='*60}")
    
    return combined_model, last_checkpoint_path
