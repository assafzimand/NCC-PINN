"""Training loop for PINN models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple
import json
import time
import numpy as np

from trainer.plotting import plot_training_curves, plot_final_comparison
from trainer.utils import compute_relative_l2_error, compute_infinity_norm_error


def _build_expert_tree_from_pretrained(
    model: nn.Module,
    eval_data: Dict,
    cfg: Dict,
    run_dir: Path
) -> int:
    """
    Build the expert tree using ONLY the pretrained base model's predictions.
    
    This function spawns all experts based on the frozen base model, without
    training any expert during the tree-building phase. Tree building stops
    when a spawn step produces 0 new experts.
    
    Args:
        model: AdaptiveExpertPINN with pretrained, frozen base model
        eval_data: Evaluation data dictionary
        cfg: Configuration dictionary
        run_dir: Output directory for plots
        
    Returns:
        Number of experts spawned
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Building Expert Tree (Pretrained Base Model)")
    print("=" * 60)
    print("  Base model is frozen - spawning decisions based ONLY on base predictions")
    print("  No expert training during this phase")
    print("=" * 60)
    
    adaptive_cfg = cfg.get('adaptive_pinn', {})
    max_experts = adaptive_cfg.get('max_experts', 5)
    max_tree_depth = adaptive_cfg.get('max_depth', 5)
    overlap_threshold = adaptive_cfg.get('overlap_threshold', 0.5)
    min_samples_per_region = adaptive_cfg.get('min_samples_per_region', 50)
    max_children_coverage = adaptive_cfg.get('max_children_coverage', 0.95)
    spawn_by_depth = adaptive_cfg.get('spawn_by_depth', False)
    wavelet_threshold = adaptive_cfg.get('wavelet_threshold', None)
    
    # Import adaptive modules
    from adaptive.region_detector import RegionDetector
    from adaptive.visualization import (
        plot_expert_regions, save_regions_metadata, prepare_ground_truth_grid,
        plot_expert_soft_weights
    )
    from adaptive.residual_utils import compute_pde_residuals
    
    device = next(model.parameters()).device
    domain_bounds = model.get_domain_bounds()
    
    # Prepare ground truth grid for visualization
    gt_grid, gt_x, gt_t = prepare_ground_truth_grid(eval_data, domain_bounds)
    
    region_detector = RegionDetector(
        n_estimators=adaptive_cfg.get('rf_n_estimators', 100),
        max_depth=adaptive_cfg.get('rf_max_depth', 10),
        min_samples_leaf=adaptive_cfg.get('rf_min_samples_leaf', 50),
        domain_bounds=domain_bounds
    )
    
    # Create directory for adaptive outputs
    adaptive_plots_dir = run_dir / "adaptive_plots"
    adaptive_plots_dir.mkdir(exist_ok=True)
    
    # Depth-locked spawning tracking
    current_spawn_depth = 1
    
    # Get base model predictions ONCE (frozen, won't change)
    model.eval()
    eval_inputs = torch.cat([eval_data['x'], eval_data['t']], dim=1)
    
    # CRITICAL: Use base model ONLY for predictions (not the full composed model)
    # This ensures spawning is based only on base model, not early-spawned experts
    with torch.no_grad():
        u_pred_base = model.base_model(eval_inputs)
    
    # Compute PDE residuals using base model only
    # We need to temporarily make the composed forward use only base
    problem = cfg.get('problem', 'schrodinger')
    pde_residuals = compute_pde_residuals(
        model=model.base_model,  # Use base model directly for residuals
        x=eval_data['x'],
        t=eval_data['t'],
        problem=problem,
        config=cfg
    )
    
    # Convert to numpy for RF
    X_eval_full = eval_inputs.cpu().numpy()
    y_eval_full = u_pred_base.cpu().numpy()
    residuals_full = pde_residuals.detach().cpu().numpy()
    
    spawn_step = 0
    total_experts_spawned = 0
    
    print(f"\nStarting tree building loop (max_experts={max_experts}, max_depth={max_tree_depth})")
    
    while model.num_experts < max_experts:
        spawn_step += 1
        print(f"\n{'='*60}")
        print(f"Tree Build Step {spawn_step}")
        print(f"  Current experts: {model.num_experts}/{max_experts}")
        print(f"  Highest depth populated: {model.get_highest_depth()}")
        if spawn_by_depth:
            print(f"  Depth-locked mode: searching depth {current_spawn_depth}")
        print(f"{'='*60}")
        
        experts_spawned_this_step = 0
        
        # Determine which depths to search
        if spawn_by_depth:
            # Depth-locked mode: check if we should advance depth
            if current_spawn_depth > 1:
                parent_regions = model.get_regions_at_depth(current_spawn_depth - 1)
                all_parents_saturated = True
                for parent_region in parent_regions:
                    parent_idx = model.regions.index(parent_region)
                    coverage = model.compute_children_coverage(eval_inputs, parent_idx)
                    if coverage <= max_children_coverage:
                        all_parents_saturated = False
                        break
                
                if all_parents_saturated and len(parent_regions) > 0 and current_spawn_depth < max_tree_depth:
                    current_spawn_depth += 1
                    print(f"\n  All parents at depth {current_spawn_depth - 1} saturated, advancing to depth {current_spawn_depth}")
            else:
                coverage = model.compute_children_coverage(eval_inputs, parent_idx=-1)
                if coverage > max_children_coverage and current_spawn_depth < max_tree_depth:
                    current_spawn_depth += 1
                    print(f"\n  Base model saturated ({coverage*100:.1f}% covered), advancing to depth {current_spawn_depth}")
            
            depths_to_search = [current_spawn_depth]
        else:
            highest_depth = model.get_highest_depth()
            current_max_search_depth = min(max_tree_depth, highest_depth + 1)
            if highest_depth == 0:
                current_max_search_depth = 1
            depths_to_search = list(range(1, current_max_search_depth + 1))
        
        # Search selected depths
        for depth in depths_to_search:
            if model.num_experts >= max_experts:
                print(f"  Max experts reached, stopping search")
                break
            
            print(f"\n  Searching depth {depth}...")
            
            if depth == 1:
                # Depth 1: Search entire domain (children of base model)
                parent_idx = -1
                
                coverage = model.compute_children_coverage(eval_inputs, parent_idx=-1)
                if coverage > max_children_coverage:
                    print(f"    Base model already {coverage*100:.1f}% covered by depth-1 children, skipping")
                    continue
                
                X_search = X_eval_full
                y_search = y_eval_full
                res_search = residuals_full
                
                print(f"    Search domain: entire domain ({len(X_search)} points, {coverage*100:.1f}% covered)")
                
                if len(X_search) < min_samples_per_region:
                    print(f"    Too few points ({len(X_search)} < {min_samples_per_region}), skipping")
                    continue
                
                sibling_regions = model.get_children_of_parent(parent_idx=-1)
                print(f"    Siblings (children of base): {len(sibling_regions)}")
                
                region = region_detector.detect(
                    X=X_search,
                    y=y_search,
                    residuals=res_search,
                    sibling_regions=sibling_regions,
                    overlap_threshold=overlap_threshold,
                    wavelet_threshold=wavelet_threshold,
                    spawn_epoch=0,  # Tree building phase
                    depth=depth,
                    parent_idx=parent_idx
                )
                
                if region is not None:
                    expert_idx = model.spawn_expert(region)
                    if expert_idx >= 0:
                        experts_spawned_this_step += 1
                else:
                    print(f"    No suitable region found at depth {depth}")
            
            else:
                # Depth > 1: Search inside each parent region
                parent_regions = model.get_regions_at_depth(depth - 1)
                
                if not parent_regions:
                    print(f"    No depth-{depth-1} regions yet, skipping")
                    continue
                
                print(f"    Searching inside {len(parent_regions)} parent region(s) from depth {depth-1}...")
                
                for parent_region in parent_regions:
                    if model.num_experts >= max_experts:
                        break
                    
                    parent_idx = model.regions.index(parent_region)
                    
                    coverage = model.compute_children_coverage(eval_inputs, parent_idx)
                    if coverage > max_children_coverage:
                        print(f"      Parent E{parent_idx+1} already {coverage*100:.1f}% covered by children, skipping")
                        continue
                    
                    parent_mask = model.get_mask_for_expert(eval_inputs, parent_idx)
                    parent_mask_np = parent_mask.cpu().numpy()
                    
                    if not parent_mask_np.any():
                        continue
                    
                    X_search = X_eval_full[parent_mask_np]
                    y_search = y_eval_full[parent_mask_np]
                    res_search = residuals_full[parent_mask_np]
                    
                    print(f"      Parent E{parent_idx+1} (depth {depth-1}): {len(X_search)} points, {coverage*100:.1f}% covered")
                    
                    if len(X_search) < min_samples_per_region:
                        print(f"        Too few points, skipping")
                        continue
                    
                    sibling_regions = model.get_children_of_parent(parent_idx=parent_idx)
                    print(f"        Siblings (children of E{parent_idx+1}): {len(sibling_regions)}")
                    
                    region = region_detector.detect(
                        X=X_search,
                        y=y_search,
                        residuals=res_search,
                        sibling_regions=sibling_regions,
                        overlap_threshold=overlap_threshold,
                        wavelet_threshold=wavelet_threshold,
                        spawn_epoch=0,  # Tree building phase
                        depth=depth,
                        parent_idx=parent_idx
                    )
                    
                    if region is not None:
                        expert_idx = model.spawn_expert(region)
                        if expert_idx >= 0:
                            experts_spawned_this_step += 1
                    else:
                        print(f"        No suitable region found in parent E{parent_idx+1}")
        
        total_experts_spawned += experts_spawned_this_step
        
        if experts_spawned_this_step > 0:
            print(f"\n  Spawned {experts_spawned_this_step} expert(s) this step")
            
            # Plot expert regions
            problem_type = '2d' if len(domain_bounds['lower']) == 2 else '3d'
            plot_expert_regions(
                regions=model.regions,
                domain_bounds=domain_bounds,
                output_path=adaptive_plots_dir / f"expert_regions_tree_step_{spawn_step}.png",
                problem_type=problem_type,
                title=f"Expert Tree Build Step {spawn_step} ({model.num_experts} experts)",
                ground_truth=gt_grid,
                grid_x=gt_x,
                grid_t=gt_t
            )
            
            # Plot soft weights if using soft blending
            if adaptive_cfg.get('blending_mode', 'hard') == 'soft' and problem_type == '2d':
                plot_expert_soft_weights(
                    model=model,
                    domain_bounds=domain_bounds,
                    output_path=adaptive_plots_dir / f"soft_weights_tree_step_{spawn_step}.png",
                    title_prefix=f"Tree Step {spawn_step}: "
                )
        else:
            print(f"\n  No experts spawned this step - tree building complete!")
            break  # Stop immediately when no new experts (no cooldown needed)
    
    # Save tree building summary
    print(f"\n{'='*60}")
    print(f"PHASE 1 COMPLETE: Expert Tree Built")
    print(f"  Total experts spawned: {model.num_experts}")
    print(f"  Tree building steps: {spawn_step}")
    print(f"  Highest depth: {model.get_highest_depth()}")
    print(f"{'='*60}")
    
    # Save final tree structure
    save_regions_metadata(
        regions=model.regions,
        output_path=adaptive_plots_dir / "expert_tree_structure.json"
    )
    
    return model.num_experts


def _create_adam_optimizer(model: nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    """Create Adam optimizer with config parameters.
    
    Only includes trainable parameters (requires_grad=True) to avoid
    wasting memory/compute on frozen parameters (e.g., pretrained base model).
    """
    betas = tuple(cfg.get('adam_betas', [0.9, 0.999]))
    eps = cfg.get('adam_eps', 1e-8)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(
        trainable_params,
        lr=cfg['lr'],
        betas=betas,
        eps=eps
    )


def _create_lbfgs_optimizer(model: nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    """Create LBFGS optimizer. Should be used with full-batch training.
    
    Only includes trainable parameters (requires_grad=True) to avoid
    wasting memory/compute on frozen parameters (e.g., pretrained base model).
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.LBFGS(
        trainable_params,
        lr=cfg.get('lbfgs_lr', 1.0),
        max_iter=cfg.get('lbfgs_max_iter', 20),
        max_eval=None,  # Default: max_iter * 1.25
        history_size=cfg.get('lbfgs_history_size', 100),
        line_search_fn=cfg.get('lbfgs_line_search', 'strong_wolfe'),
        tolerance_grad=cfg.get('lbfgs_tolerance_grad', 1e-7),
        tolerance_change=cfg.get('lbfgs_tolerance_change', 1e-9)
    )


def train(
    model: nn.Module,
    loss_fn: Callable,
    train_data_path: str,
    eval_data_path: str,
    cfg: Dict,
    run_dir: Path
) -> Path:
    """
    Train a PINN model with CUDA acceleration and vectorized operations.

    Args:
        model: Neural network model
        loss_fn: Loss function (model, batch) -> scalar
        train_data_path: Path to training_data.pt
        eval_data_path: Path to eval_data.pt
        cfg: Configuration dictionary
        run_dir: Output directory for this run

    Returns:
        Path to best checkpoint
    """
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    # Setup device
    device = torch.device('cuda' if cfg['cuda'] and
                          torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # GPU optimization and monitoring
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Initial GPU Memory Allocated: {torch.cuda.memory_allocated()/1e9:.3f} GB")
        torch.backends.cudnn.benchmark = True
        print("CUDNN benchmark enabled for GPU optimization")

    # Set seed for reproducibility
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg['seed'])

    # Move model to device
    model = model.to(device)

    # DIAGNOSTIC: Verify model is on correct device
    print(f"\n{'='*40} GPU DIAGNOSTIC {'='*40}")
    print(f"Target device: {device}")
    if hasattr(model, 'base_model'):
        print(f"Base model device: {next(model.base_model.parameters()).device}")
    else:
        print(f"Model device: {next(model.parameters()).device}")
    print(f"{'='*80}\n")

    # Load datasets
    print(f"\nLoading datasets...")
    train_data = torch.load(train_data_path)
    eval_data = torch.load(eval_data_path)

    # Move data to device
    train_data = _move_batch_to_device(train_data, device)
    eval_data = _move_batch_to_device(eval_data, device)

    print(f"  Train size: {train_data['x'].shape[0]}")
    print(f"  Eval size: {eval_data['x'].shape[0]}")
    print(f"  Train data device: {train_data['x'].device}")
    print(f"  Eval data device: {eval_data['x'].device}")

    # Create DataLoaders
    train_loader = _create_dataloader(train_data, cfg['batch_size'],
                                      shuffle=True)
    eval_loader = _create_dataloader(eval_data, cfg['batch_size'],
                                     shuffle=False)

    # Determine switch epoch and optimizer strategy
    switch_at_fraction = cfg.get('optimizer_switch_at', 1.0)
    epochs = cfg['epochs']
    switch_epoch = int(epochs * switch_at_fraction) + 1

    # Setup initial optimizer
    if switch_at_fraction == 0.0:
        optimizer = _create_lbfgs_optimizer(model, cfg)
        current_optimizer_name = 'LBFGS'
        print(f"Using LBFGS optimizer (full-batch) for all epochs")
    else:
        optimizer = _create_adam_optimizer(model, cfg)
        current_optimizer_name = 'Adam'
        if switch_at_fraction < 1.0:
            print(f"Using Adam (mini-batch) until epoch {switch_epoch}, then LBFGS (full-batch)")
        else:
            print(f"Using Adam optimizer (mini-batch) for all epochs")

    # Training setup
    print_every = cfg['print_every']
    eval_every = cfg.get('eval_every', print_every)
    inner_metrics_every = cfg.get('inner_metrics_eval_every', 0)
    save_every = cfg['save_every']

    # Metrics storage
    # Note: train_loss is stored every epoch, eval metrics only every print_every
    metrics = {
        'train_loss_epochs': [],  # All epochs
        'train_loss': [],          # All epochs
        'epochs': [],              # Evaluation epochs only
        'eval_loss': [],
        'train_rel_l2': [],
        'eval_rel_l2': [],
        'train_inf_norm': [],
        'eval_inf_norm': []
    }

    best_eval_loss = float('inf')
    best_checkpoint_path = None

    # Create checkpoint directory (aligned with outputs naming: <problem>-<layers>-<act>)
    architecture_str = "-".join(map(str, cfg['architecture']))
    checkpoint_dir = Path("checkpoints") / cfg['problem'] / f"{cfg['problem']}-{architecture_str}-{cfg['activation']}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create ncc_plots directory for periodic NCC analysis
    ncc_plots_parent = run_dir / "ncc_plots"
    ncc_plots_parent.mkdir(exist_ok=True)

    # Adaptive PINN setup
    adaptive_cfg = cfg.get('adaptive_pinn', {})
    is_adaptive = adaptive_cfg.get('enabled', False)
    region_detector = None
    spawn_every = adaptive_cfg.get('spawn_every_epochs', 2000)
    max_experts = adaptive_cfg.get('max_experts', 5)
    wavelet_threshold = adaptive_cfg.get('wavelet_threshold', None)
    adaptive_inner_metrics = adaptive_cfg.get('inner_metrics_calculation', False)
    
    # Pretrained base model mode
    pretrained_base_model = adaptive_cfg.get('pretrained_base_model', False)
    pretrained_base_path = adaptive_cfg.get('pretrained_base_path', None)
    disable_spawning_during_training = False  # Will be set to True after tree building
    
    if is_adaptive:
        # Extract hierarchical tree parameters
        max_tree_depth = adaptive_cfg.get('max_depth', 5)
        overlap_threshold = adaptive_cfg.get('overlap_threshold', 0.5)
        min_samples_per_region = adaptive_cfg.get('min_samples_per_region', 50)
        max_children_coverage = adaptive_cfg.get('max_children_coverage', 0.9)
        spawn_by_depth = adaptive_cfg.get('spawn_by_depth', False)  # Depth-locked spawning mode
        
        print(f"\nAdaptive PINN enabled (Hierarchical Expert Tree):")
        print(f"  Max experts: {max_experts}")
        print(f"  Max depth: {max_tree_depth}")
        print(f"  Spawn every: {spawn_every} epochs")
        print(f"  Overlap threshold: {overlap_threshold}")
        print(f"  Min samples per region: {min_samples_per_region}")
        print(f"  Max children coverage: {max_children_coverage}")
        print(f"  Spawn by depth: {spawn_by_depth}")
        print(f"  Blending mode: {adaptive_cfg.get('blending_mode', 'hard')}")
        print(f"  Freeze mode: {adaptive_cfg.get('freeze_mode', 'none')}")
        print(f"  Pretrained base model: {pretrained_base_model}")
        
        # Handle pretrained base model mode
        if pretrained_base_model:
            if pretrained_base_path is None:
                raise ValueError("pretrained_base_model is True but pretrained_base_path is not set")
            
            # Load pretrained base and freeze it
            model.load_pretrained_base(pretrained_base_path)
            
            # Precompute base outputs for all data points ONCE
            # This avoids repeated forward passes through frozen base during training
            print("\n  Precomputing base model outputs for all data...")
            with torch.no_grad():
                train_inputs = torch.cat([train_data['x'], train_data['t']], dim=1)
                eval_inputs = torch.cat([eval_data['x'], eval_data['t']], dim=1)
                train_data['u_base'] = model.base_model(train_inputs)
                eval_data['u_base'] = model.base_model(eval_inputs)
            print(f"    Train u_base shape: {train_data['u_base'].shape}")
            print(f"    Eval u_base shape: {eval_data['u_base'].shape}")
            
            # Build expert tree based ONLY on pretrained base
            num_experts_built = _build_expert_tree_from_pretrained(
                model=model,
                eval_data=eval_data,
                cfg=cfg,
                run_dir=run_dir
            )
            
            print(f"\n{'='*60}")
            print(f"PHASE 2: Training Experts (Base Frozen)")
            print(f"  Experts to train: {num_experts_built}")
            print(f"  Base model: FROZEN (pretrained)")
            print(f"  Spawning: DISABLED (tree already built)")
            print(f"{'='*60}\n")
            
            # Disable spawning during training - tree is already built
            disable_spawning_during_training = True
            
            # Ensure base stays frozen and experts are trainable
            model.freeze_base_model()
            model.unfreeze_experts()
            
            # DIAGNOSTIC: Verify all models on correct device after tree building
            print(f"\n{'='*40} POST-TREE GPU CHECK {'='*40}")
            print(f"Base model device: {next(model.base_model.parameters()).device}")
            for i, expert in enumerate(model.experts):
                print(f"Expert {i} device: {next(expert.parameters()).device}")
            print(f"{'='*80}\n")
            
            # Recreate optimizer to include all expert parameters
            if switch_at_fraction == 0.0:
                optimizer = _create_lbfgs_optimizer(model, cfg)
                current_optimizer_name = 'LBFGS'
            else:
                optimizer = _create_adam_optimizer(model, cfg)
                current_optimizer_name = 'Adam'
        
        # Import and create region detector (only needed if not pretrained mode)
        if not pretrained_base_model:
            from adaptive.region_detector import RegionDetector
            from adaptive.visualization import (
                plot_expert_regions, save_regions_metadata, prepare_ground_truth_grid,
                plot_expert_soft_weights
            )
            from adaptive.residual_utils import compute_pde_residuals
            
            # Get domain bounds
            domain_bounds = model.get_domain_bounds()
            
            # Prepare ground truth grid for visualization
            gt_grid, gt_x, gt_t = prepare_ground_truth_grid(eval_data, domain_bounds)
            
            region_detector = RegionDetector(
                n_estimators=adaptive_cfg.get('rf_n_estimators', 100),
                max_depth=adaptive_cfg.get('rf_max_depth', 10),
                min_samples_leaf=adaptive_cfg.get('rf_min_samples_leaf', 50),
                domain_bounds=domain_bounds
            )
        else:
            # For pretrained mode, still need these imports for final plots
            from adaptive.visualization import (
                plot_expert_regions, save_regions_metadata, prepare_ground_truth_grid,
                plot_expert_soft_weights
            )
            domain_bounds = model.get_domain_bounds()
            gt_grid, gt_x, gt_t = prepare_ground_truth_grid(eval_data, domain_bounds)
        
        # Create directory for adaptive outputs
        adaptive_plots_dir = run_dir / "adaptive_plots"
        adaptive_plots_dir.mkdir(exist_ok=True)
    
    # Adaptive spawn cooldown: skip N spawn attempts after finding 0 experts
    spawn_skip_counter = 0  # When > 0, skip spawn attempts and decrement
    spawn_cooldown_steps = 3  # How many steps to skip after finding 0 experts
    
    # Depth-locked spawning: track current depth when spawn_by_depth is enabled
    current_spawn_depth = 1  # Start at depth 1 (children of base model)

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Timing debug: accumulators for pretrained base case
        if disable_spawning_during_training:
            t_epoch_start = time.perf_counter()
            t_train_loss_fn, t_train_backward, t_train_step = 0.0, 0.0, 0.0
            t_eval_train_metrics, t_eval_loop_loss_fn, t_eval_loop_h_pred = 0.0, 0.0, 0.0

        # Train phase
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        if current_optimizer_name == 'Adam':
            # Adam: Mini-batch training (GPU parallelized)
            for batch in train_loader:
                optimizer.zero_grad()
                if disable_spawning_during_training:
                    _t0 = time.perf_counter()
                loss = loss_fn(model, batch)
                if disable_spawning_during_training:
                    _t1 = time.perf_counter()
                    t_train_loss_fn += _t1 - _t0
                loss.backward()
                if disable_spawning_during_training:
                    _t2 = time.perf_counter()
                    t_train_backward += _t2 - _t1
                optimizer.step()
                if disable_spawning_during_training:
                    t_train_step += time.perf_counter() - _t2
                
                train_loss += loss.item()
                n_train_batches += 1

        else:
            # LBFGS: Full-batch training with memory error handling
            # Process entire dataset in single forward pass (no batching)
            def closure():
                optimizer.zero_grad()
                # Single forward pass with ALL training data at once
                loss = loss_fn(model, train_data)
                loss.backward()
                return loss
            
            try:
                if disable_spawning_during_training:
                    _t_lbfgs_closure = time.perf_counter()
                # LBFGS step processes entire dataset via closure
                loss = optimizer.step(closure)
                if disable_spawning_during_training:
                    t_train_loss_fn = time.perf_counter() - _t_lbfgs_closure  # closure dominates
                    t_train_backward = 0.0
                    t_train_step = 0.0
                train_loss = loss.item()
                n_train_batches = 1
            
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # GPU OOM - fallback to Adam with persistent warning
                    error_msg = (
                        f"\n{'='*60}\n"
                        f"MEMORY ERROR at epoch {epoch}\n"
                        f"LBFGS ran out of GPU memory. Falling back to Adam.\n"
                        f"Consider: reducing batch_size, dataset size, or\n"
                        f"setting optimizer_switch_at=1.0 to disable LBFGS.\n"
                        f"{'='*60}\n"
                    )
                    print(error_msg)
                    
                    # Save warning to persistent file
                    warning_log = run_dir / "optimizer_fallback_warning.txt"
                    with open(warning_log, 'a') as f:
                        from datetime import datetime
                        f.write(f"[{datetime.now()}] Epoch {epoch}:\n")
                        f.write(error_msg)
                        f.write(f"Error details: {str(e)}\n\n")
                    
                    # Clear GPU cache and fallback
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    optimizer = _create_adam_optimizer(model, cfg)
                    current_optimizer_name = 'Adam'
                    
                    # Continue with Adam on first batch
                    optimizer.zero_grad()
                    batch = next(iter(train_loader))
                    loss = loss_fn(model, batch)
                    loss.backward()
                    optimizer.step()
                    train_loss = loss.item()
                    n_train_batches = 1
                else:
                    raise  # Re-raise other errors

        train_loss /= n_train_batches
        
        # Check for optimizer switch
        if epoch == switch_epoch and switch_at_fraction < 1.0 and current_optimizer_name == 'Adam':
            print(f"\n{'='*60}")
            print(f"OPTIMIZER SWITCH: Adam -> LBFGS at epoch {epoch}")
            print(f"Switching to full-batch LBFGS for fine-tuning")
            print(f"{'='*60}\n")
            
            optimizer = _create_lbfgs_optimizer(model, cfg)
            current_optimizer_name = 'LBFGS'
        
        # Store train loss every epoch
        metrics['train_loss_epochs'].append(epoch)
        metrics['train_loss'].append(train_loss)

        # Compute evaluation metrics only every print_every epochs or last epoch
        # This speeds up training significantly for physics-informed losses
        should_evaluate = (epoch % eval_every == 0 or epoch == 1 or epoch == epochs)
        
        # Initialize metrics for this epoch (will be updated if we evaluate)
        eval_loss = None
        eval_rel_l2 = None
        train_rel_l2 = None
        train_inf_norm = None
        eval_inf_norm = None
        
        if should_evaluate:
            # Compute train rel-L2 and infinity norm errors
            model.train()
            train_rel_l2 = 0.0
            train_inf_norm = 0.0
            n_train_batches_l2 = 0

            if disable_spawning_during_training:
                _t_train_metrics_start = time.perf_counter()
            for batch in train_loader:
                with torch.no_grad():
                    inputs = torch.cat([batch['x'], batch['t']], dim=1)
                    u_base = batch.get('u_base')
                    h_pred = model(inputs, u_base_precomputed=u_base)
                    rel_l2 = compute_relative_l2_error(h_pred, batch['h_gt'])
                    inf_norm = compute_infinity_norm_error(h_pred, batch['h_gt'])
                    train_rel_l2 += rel_l2.item()
                    train_inf_norm += inf_norm.item()
                    n_train_batches_l2 += 1
            if disable_spawning_during_training:
                t_eval_train_metrics = time.perf_counter() - _t_train_metrics_start
            
            train_rel_l2 /= n_train_batches_l2
            train_inf_norm /= n_train_batches_l2
            
            # Eval phase
            model.eval()
            eval_loss = 0.0
            eval_rel_l2 = 0.0
            eval_inf_norm = 0.0
            n_eval_batches = 0

            for batch in eval_loader:
                # Note: For physics-informed losses, we need gradients w.r.t. inputs
                # even during evaluation (for computing derivatives in PDE residuals).
                # We still use model.eval() to disable dropout/batchnorm training behavior.
                if disable_spawning_during_training:
                    _t0 = time.perf_counter()
                loss = loss_fn(model, batch)
                if disable_spawning_during_training:
                    _t1 = time.perf_counter()
                    t_eval_loop_loss_fn += _t1 - _t0

                with torch.no_grad():
                    inputs = torch.cat([batch['x'], batch['t']], dim=1)
                    u_base = batch.get('u_base')
                    _t2 = time.perf_counter() if disable_spawning_during_training else None
                    h_pred = model(inputs, u_base_precomputed=u_base)
                    if disable_spawning_during_training:
                        t_eval_loop_h_pred += time.perf_counter() - _t2
                    rel_l2 = compute_relative_l2_error(h_pred, batch['h_gt'])
                    inf_norm = compute_infinity_norm_error(h_pred, batch['h_gt'])

                eval_loss += loss.item()
                eval_rel_l2 += rel_l2.item()
                eval_inf_norm += inf_norm.item()
                n_eval_batches += 1

            eval_loss /= n_eval_batches
            eval_rel_l2 /= n_eval_batches
            eval_inf_norm /= n_eval_batches

            # Store evaluation metrics (train_loss already stored above for all epochs)
            metrics['epochs'].append(epoch)
            metrics['eval_loss'].append(eval_loss)
            metrics['train_rel_l2'].append(train_rel_l2)
            metrics['eval_rel_l2'].append(eval_rel_l2)
            metrics['train_inf_norm'].append(train_inf_norm)
            metrics['eval_inf_norm'].append(eval_inf_norm)

        # Timing debug: print breakdown for pretrained base case
        if disable_spawning_during_training:
            t_epoch_total = time.perf_counter() - t_epoch_start
            if should_evaluate:
                print(f"  [TIMING] Epoch {epoch} total: {t_epoch_total:.3f}s")
                print(f"    Train: loss_fn={t_train_loss_fn:.3f}s backward={t_train_backward:.3f}s step={t_train_step:.3f}s "
                      f"(batches={n_train_batches})")
                print(f"    Eval:  train_metrics={t_eval_train_metrics:.3f}s | "
                      f"eval_loop: loss_fn={t_eval_loop_loss_fn:.3f}s model_h_pred={t_eval_loop_h_pred:.3f}s "
                      f"(batches={n_eval_batches})")
            else:
                print(f"  [TIMING] Epoch {epoch} total: {t_epoch_total:.3f}s | "
                      f"loss_fn={t_train_loss_fn:.3f}s backward={t_train_backward:.3f}s step={t_train_step:.3f}s")

        # Print progress
        if should_evaluate:
            elapsed = time.time() - start_time
            batch_mode = "mini" if current_optimizer_name == 'Adam' else "full"
            print(f"Epoch [{epoch}/{epochs}] ({elapsed:.1f}s) [{current_optimizer_name}/{batch_mode}] | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Eval Loss: {eval_loss:.6f} | "
                  f"Train Rel-L2: {train_rel_l2:.6f} | "
                  f"Eval Rel-L2: {eval_rel_l2:.6f} | "
                  f"Train Inf: {train_inf_norm:.6f} | "
                  f"Eval Inf: {eval_inf_norm:.6f}")

        # Save checkpoint periodically (only when we have eval metrics)
        if epoch % save_every == 0 and eval_loss is not None:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            _save_checkpoint(checkpoint_path, model, optimizer, current_optimizer_name, epoch,
                           train_loss, eval_loss, cfg, metrics)
            print(f"  Checkpoint saved: {checkpoint_path}")

        # Save best model (only when we have eval metrics)
        if eval_loss is not None and eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_checkpoint_path = checkpoint_dir / "best_model.pt"
            _save_checkpoint(best_checkpoint_path, model, optimizer, current_optimizer_name, epoch,
                           train_loss, eval_loss, cfg, metrics)

        # Periodic inner metrics (NCC + probes + derivatives + frequency)
        # Skip if adaptive PINN and inner_metrics_calculation is disabled
        should_run_inner_metrics = inner_metrics_every > 0 and epoch % inner_metrics_every == 0
        if is_adaptive and not adaptive_inner_metrics:
            should_run_inner_metrics = False
            
        if should_run_inner_metrics:
            print(f"\n  Running inner metrics at epoch {epoch} (NCC/Probes/Derivatives/Frequency)...")
            ncc_metrics = _run_intermediate_ncc(model, cfg, run_dir, epoch)
            probe_metrics = _run_intermediate_probes(model, cfg, run_dir, epoch)
            deriv_metrics = _run_intermediate_derivatives(model, cfg, run_dir, epoch)
            freq_metrics = _run_intermediate_frequency(model, cfg, run_dir, epoch)
            # Cache for post-run shading overlays
            if 'ncc_history' not in metrics:
                metrics['ncc_history'] = []
            metrics['ncc_history'].append((epoch, ncc_metrics))
            if 'probe_history' not in metrics:
                metrics['probe_history'] = []
            if probe_metrics is not None:
                metrics['probe_history'].append((epoch, probe_metrics))
            if 'deriv_history' not in metrics:
                metrics['deriv_history'] = []
            if deriv_metrics is not None:
                metrics['deriv_history'].append((epoch, deriv_metrics))
            if 'freq_history' not in metrics:
                metrics['freq_history'] = []
            if freq_metrics is not None:
                metrics['freq_history'].append((epoch, freq_metrics))

        # Adaptive PINN: Hierarchical expert spawning (with cooldown after 0-spawn steps)
        # Disabled if tree was pre-built from pretrained base model
        spawn_check_triggered = (is_adaptive and 
                                 epoch % spawn_every == 0 and 
                                 model.num_experts < max_experts and
                                 not disable_spawning_during_training)
        
        if spawn_check_triggered and spawn_skip_counter > 0:
            # In cooldown period - skip this spawn attempt
            spawn_skip_counter -= 1
            print(f"\n  [Adaptive] Skipping spawn attempt (cooldown: {spawn_skip_counter} steps remaining)")
            spawn_check_triggered = False  # Don't proceed with spawn
        
        if spawn_check_triggered:
            print(f"\n{'='*60}")
            print(f"Adaptive PINN: Hierarchical search at epoch {epoch}")
            print(f"  Current experts: {model.num_experts}/{max_experts}")
            print(f"  Highest depth populated: {model.get_highest_depth()}")
            if spawn_by_depth:
                print(f"  Depth-locked mode: searching depth {current_spawn_depth}")
            print(f"{'='*60}")
            
            # Get predictions on eval data for region detection
            model.eval()
            with torch.no_grad():
                eval_inputs = torch.cat([eval_data['x'], eval_data['t']], dim=1)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                u_pred = model(eval_inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            # Compute PDE residuals for residual-weighted wavelet norms
            problem = cfg.get('problem', 'schrodinger')
            pde_residuals = compute_pde_residuals(
                model=model,
                x=eval_data['x'],
                t=eval_data['t'],
                problem=problem,
                config=cfg
            )
            
            # Convert to numpy for RF
            X_eval_full = eval_inputs.cpu().numpy()
            y_eval_full = u_pred.cpu().numpy()
            residuals_full = pde_residuals.detach().cpu().numpy()
            
            experts_spawned_this_step = 0
            
            # Determine which depths to search
            if spawn_by_depth:
                # Depth-locked mode: only search at current_spawn_depth
                # But first, check if we should advance to next depth
                if current_spawn_depth > 1:
                    # Check if ALL parents at (current_spawn_depth - 1) are saturated
                    parent_regions = model.get_regions_at_depth(current_spawn_depth - 1, before_epoch=epoch)
                    all_parents_saturated = True
                    for parent_region in parent_regions:
                        parent_idx = model.regions.index(parent_region)
                        coverage = model.compute_children_coverage(eval_inputs, parent_idx, before_epoch=epoch)
                        if coverage <= max_children_coverage:
                            all_parents_saturated = False
                            break
                    
                    if all_parents_saturated and len(parent_regions) > 0 and current_spawn_depth < max_tree_depth:
                        current_spawn_depth += 1
                        print(f"\n  All parents at depth {current_spawn_depth - 1} saturated, advancing to depth {current_spawn_depth}")
                else:
                    # At depth 1: check if base model is saturated
                    coverage = model.compute_children_coverage(eval_inputs, parent_idx=-1, before_epoch=epoch)
                    if coverage > max_children_coverage and current_spawn_depth < max_tree_depth:
                        current_spawn_depth += 1
                        print(f"\n  Base model saturated ({coverage*100:.1f}% covered), advancing to depth {current_spawn_depth}")
                
                depths_to_search = [current_spawn_depth]
            else:
                # Original mode: search all depths up to highest + 1
                highest_depth = model.get_highest_depth()
                current_max_search_depth = min(max_tree_depth, highest_depth + 1)
                # If no experts yet, search depth 1
                if highest_depth == 0:
                    current_max_search_depth = 1
                depths_to_search = list(range(1, current_max_search_depth + 1))
            
            # Search selected depths
            for depth in depths_to_search:
                if model.num_experts >= max_experts:
                    print(f"  Max experts reached, stopping search")
                    break
                
                print(f"\n  Searching depth {depth}...")
                
                if depth == 1:
                    # ====== DEPTH 1: Search entire domain ======
                    # Children of base model (parent_idx = -1)
                    parent_idx = -1
                    
                    # Check if base model is already mostly covered by depth-1 children
                    coverage = model.compute_children_coverage(eval_inputs, parent_idx=-1, before_epoch=epoch)
                    if coverage > max_children_coverage:
                        print(f"    Base model already {coverage*100:.1f}% covered by depth-1 children, skipping")
                        continue
                    
                    X_search = X_eval_full
                    y_search = y_eval_full
                    res_search = residuals_full
                    
                    print(f"    Search domain: entire domain ({len(X_search)} points, {coverage*100:.1f}% covered)")
                    
                    # Check if enough samples
                    if len(X_search) < min_samples_per_region:
                        print(f"    Too few points ({len(X_search)} < {min_samples_per_region}), skipping")
                        continue
                    
                    # Sibling check: other children of base model (same parent_idx=-1)
                    sibling_regions = model.get_children_of_parent(parent_idx=-1, before_epoch=epoch)
                    print(f"    Siblings (children of base): {len(sibling_regions)}")
                    
                    # Detect refinement region
                    region = region_detector.detect(
                        X=X_search,
                        y=y_search,
                        residuals=res_search,
                        sibling_regions=sibling_regions,
                        overlap_threshold=overlap_threshold,
                        wavelet_threshold=wavelet_threshold,
                        spawn_epoch=epoch,
                        depth=depth,
                        parent_idx=parent_idx
                    )
                    
                    if region is not None:
                        # Spawn new expert
                        expert_idx = model.spawn_expert(region)
                        
                        if expert_idx >= 0:
                            experts_spawned_this_step += 1
                            
                            # Store expert spawn history
                            if 'expert_spawns' not in metrics:
                                metrics['expert_spawns'] = []
                            metrics['expert_spawns'].append({
                                'epoch': epoch,
                                'expert_idx': expert_idx,
                                'region': region.to_dict(),
                                'num_experts': model.num_experts,
                                'depth': depth,
                                'parent_idx': parent_idx
                            })
                    else:
                        print(f"    No suitable region found at depth {depth}")
                
                else:
                    # ====== DEPTH > 1: Search inside each parent region individually ======
                    # Get all trained parent regions (depth - 1) that existed before this epoch
                    parent_regions = model.get_regions_at_depth(depth - 1, before_epoch=epoch)
                    
                    if not parent_regions:
                        print(f"    No trained depth-{depth-1} regions yet (spawned before epoch {epoch}), skipping")
                        continue
                    
                    print(f"    Searching inside {len(parent_regions)} parent region(s) from depth {depth-1}...")
                    
                    # Try each parent region as a separate search domain
                    # Can spawn 1 expert per parent (not limited to 1 per depth)
                    for parent_region in parent_regions:
                        if model.num_experts >= max_experts:
                            break
                        
                        # Find the index of this parent region
                        parent_idx = model.regions.index(parent_region)
                        
                        # Check if parent is already mostly covered by children
                        # If so, skip - no room for new children
                        coverage = model.compute_children_coverage(eval_inputs, parent_idx, before_epoch=epoch)
                        if coverage > max_children_coverage:
                            print(f"      Parent E{parent_idx+1} already {coverage*100:.1f}% covered by children, skipping")
                            continue
                        
                        # Filter data to points inside this parent region
                        parent_mask = model.get_mask_for_expert(eval_inputs, parent_idx)
                        parent_mask_np = parent_mask.cpu().numpy()
                        
                        if not parent_mask_np.any():
                            continue
                        
                        X_search = X_eval_full[parent_mask_np]
                        y_search = y_eval_full[parent_mask_np]
                        res_search = residuals_full[parent_mask_np]
                        
                        print(f"      Parent E{parent_idx+1} (depth {depth-1}): {len(X_search)} points, {coverage*100:.1f}% covered")
                        
                        # Check if enough samples
                        if len(X_search) < min_samples_per_region:
                            print(f"        Too few points, skipping")
                            continue
                        
                        # Sibling check: only children of THIS SAME PARENT
                        sibling_regions = model.get_children_of_parent(parent_idx=parent_idx, before_epoch=epoch)
                        print(f"        Siblings (children of E{parent_idx+1}): {len(sibling_regions)}")
                        
                        # Detect refinement region within this parent
                        region = region_detector.detect(
                            X=X_search,
                            y=y_search,
                            residuals=res_search,
                            sibling_regions=sibling_regions,
                            overlap_threshold=overlap_threshold,
                            wavelet_threshold=wavelet_threshold,
                            spawn_epoch=epoch,
                            depth=depth,
                            parent_idx=parent_idx
                        )
                        
                        if region is not None:
                            # Spawn new expert
                            expert_idx = model.spawn_expert(region)
                            
                            if expert_idx >= 0:
                                experts_spawned_this_step += 1
                                
                                # Store expert spawn history
                                if 'expert_spawns' not in metrics:
                                    metrics['expert_spawns'] = []
                                metrics['expert_spawns'].append({
                                    'epoch': epoch,
                                    'expert_idx': expert_idx,
                                    'region': region.to_dict(),
                                    'num_experts': model.num_experts,
                                    'depth': depth,
                                    'parent_idx': parent_idx
                                })
                        else:
                            print(f"        No suitable region found in parent E{parent_idx+1}")
            
            # If any experts were spawned, update optimizer and plot
            if experts_spawned_this_step > 0:
                print(f"\n  Spawned {experts_spawned_this_step} expert(s) this step")
                
                # Recreate optimizer to include new expert parameters
                if current_optimizer_name == 'Adam':
                    optimizer = _create_adam_optimizer(model, cfg)
                else:
                    optimizer = _create_lbfgs_optimizer(model, cfg)
                
                # Apply freezing strategy
                model.freeze_models()
                
                # Plot expert regions with depth info
                problem_type = '2d' if len(domain_bounds['lower']) == 2 else '3d'
                plot_expert_regions(
                    regions=model.regions,
                    domain_bounds=domain_bounds,
                    output_path=adaptive_plots_dir / f"expert_regions_epoch_{epoch}.png",
                    problem_type=problem_type,
                    title=f"Expert Regions at Epoch {epoch} ({model.num_experts} experts)",
                    ground_truth=gt_grid,
                    grid_x=gt_x,
                    grid_t=gt_t
                )
                
                # Plot soft blending weights if using soft blending mode
                if adaptive_cfg.get('blending_mode', 'hard') == 'soft' and problem_type == '2d':
                    plot_expert_soft_weights(
                        model=model,
                        domain_bounds=domain_bounds,
                        output_path=adaptive_plots_dir / f"soft_weights_epoch_{epoch}.png",
                        title_prefix=f"Epoch {epoch}: "
                    )
            else:
                print(f"\n  No experts spawned this step")
                # Enter cooldown: skip next N spawn attempts
                spawn_skip_counter = spawn_cooldown_steps
                print(f"  Entering spawn cooldown for {spawn_cooldown_steps} steps")
            
            model.train()

    # Save final model
    final_checkpoint_path = checkpoint_dir / "final_model.pt"
    _save_checkpoint(final_checkpoint_path, model, optimizer, current_optimizer_name, epochs,
                    train_loss, eval_loss, cfg, metrics)

    print(f"\nTraining completed in {time.time() - start_time:.1f}s")
    print(f"  Best eval loss: {best_eval_loss:.6f}")
    print(f"  Best checkpoint: {best_checkpoint_path}")
    print(f"  Final checkpoint: {final_checkpoint_path}")

    # Plot training curves
    print(f"\nGenerating training plots...")
    training_plots_dir = run_dir / "training_plots"
    # Pass optimizer switch epoch if there was a switch
    switch_epoch_to_plot = switch_epoch if switch_at_fraction < 1.0 else None
    plot_training_curves(metrics, training_plots_dir, optimizer_switch_epoch=switch_epoch_to_plot)

    # Plot final predictions
    model.eval()
    with torch.no_grad():
        inputs_eval = torch.cat([eval_data['x'], eval_data['t']], dim=1)
        u_base_eval = eval_data.get('u_base')
        h_pred_eval = model(inputs_eval, u_base_precomputed=u_base_eval)

    plot_final_comparison(
        h_pred_eval.cpu().numpy(),
        eval_data['h_gt'].cpu().numpy(),
        eval_data['x'].cpu().numpy(),
        eval_data['t'].cpu().numpy(),
        training_plots_dir
    )

    # Run final probes, derivatives, and frequency analysis (without epoch_suffix for main directory)
    print("\n" + "=" * 60)
    print("Running Final Probe, Derivative, and Frequency Analysis")
    print("=" * 60)
    
    from probes.probe_runner import run_probes
    from derivatives_tracker.derivatives_runner import run_derivatives_tracker
    from frequency_tracker.frequency_runner import run_frequency_tracker
    
    train_data_path = Path("datasets") / cfg['problem'] / "training_data.pt"
    eval_data_path = Path("datasets") / cfg['problem'] / "eval_data.pt"
    
    # Final probes (saves to main probe_plots/ directory)
    print("\nRunning final probe analysis...")
    final_probe_metrics = run_probes(
        model=model,
        train_data_path=str(train_data_path),
        eval_data_path=str(eval_data_path),
        cfg=cfg,
        run_dir=run_dir
    )
    
    # Final derivatives (saves to main derivatives_plots/ directory)
    print("\nRunning final derivatives analysis...")
    final_deriv_metrics = run_derivatives_tracker(
        model=model,
        train_data_path=str(train_data_path),
        eval_data_path=str(eval_data_path),
        cfg=cfg,
        run_dir=run_dir
    )
    
    # Final frequency (saves to main frequency_plots/ directory)
    print("\nRunning final frequency analysis...")
    final_freq_metrics = run_frequency_tracker(
        model=model,
        train_data_path=str(train_data_path),
        eval_data_path=str(eval_data_path),
        cfg=cfg,
        run_dir=run_dir
    )
    
    # Add final results to history for shaded plotting
    if 'probe_history' not in metrics:
        metrics['probe_history'] = []
    if final_probe_metrics is not None:
        metrics['probe_history'].append((epochs, final_probe_metrics))
    
    if 'deriv_history' not in metrics:
        metrics['deriv_history'] = []
    if final_deriv_metrics is not None:
        metrics['deriv_history'].append((epochs, final_deriv_metrics))
    
    if 'freq_history' not in metrics:
        metrics['freq_history'] = []
    if final_freq_metrics is not None:
        metrics['freq_history'].append((epochs, final_freq_metrics))
    
    # Post-run shaded overlays for mid-training metrics (if collected)
    _maybe_plot_ncc_history(metrics, run_dir)
    _maybe_plot_probe_history(metrics, run_dir)
    _maybe_plot_deriv_history(metrics, run_dir)

    # Final adaptive PINN outputs
    if is_adaptive and model.num_experts > 0:
        print("\n" + "=" * 60)
        print("Adaptive PINN Final Summary")
        print("=" * 60)
        print(f"  Total experts spawned: {model.num_experts}")
        
        # Final expert regions plot
        problem_type = '2d' if len(domain_bounds['lower']) == 2 else '3d'
        plot_expert_regions(
            regions=model.regions,
            domain_bounds=domain_bounds,
            output_path=adaptive_plots_dir / "expert_regions_final.png",
            problem_type=problem_type,
            title=f"Final Expert Regions ({model.num_experts} experts)",
            ground_truth=gt_grid,
            grid_x=gt_x,
            grid_t=gt_t
        )
        
        # Final soft blending weights plot if using soft blending mode
        if adaptive_cfg.get('blending_mode', 'hard') == 'soft' and problem_type == '2d':
            plot_expert_soft_weights(
                model=model,
                domain_bounds=domain_bounds,
                output_path=adaptive_plots_dir / "soft_weights_final.png",
                title_prefix="Final: "
            )
        
        # Save regions metadata
        save_regions_metadata(
            regions=model.regions,
            output_path=adaptive_plots_dir / "expert_regions.json"
        )
        
        # Store final regions in metrics
        metrics['adaptive_pinn'] = {
            'num_experts': model.num_experts,
            'max_experts': max_experts,
            'regions': [r.to_dict() for r in model.regions]
        }

    # Save metrics to JSON
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    # Save summary
    summary_path = run_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Training Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Problem: {cfg['problem']}\n")
        f.write(f"Architecture: {cfg['architecture']}\n")
        f.write(f"Activation: {cfg['activation']}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch size: {cfg['batch_size']}\n")
        f.write(f"Learning rate: {cfg['lr']}\n")
        f.write(f"Device: {device}\n\n")
        f.write(f"Final train loss: {train_loss:.6f}\n")
        f.write(f"Final eval loss: {eval_loss:.6f}\n")
        f.write(f"Final train rel-L2: {train_rel_l2:.6f}\n")
        f.write(f"Final eval rel-L2: {eval_rel_l2:.6f}\n")
        f.write(f"Final train inf-norm: {train_inf_norm:.6f}\n")
        f.write(f"Final eval inf-norm: {eval_inf_norm:.6f}\n")
        f.write(f"Best eval loss: {best_eval_loss:.6f}\n\n")
        f.write(f"Best checkpoint: {best_checkpoint_path}\n")
        f.write(f"Final checkpoint: {final_checkpoint_path}\n")
    print(f"  Summary saved to {summary_path}")

    # Save config used
    config_path = run_dir / "config_used.yaml"
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"  Config saved to {config_path}")
    
    # Problem-specific final evaluation visualization
    print("\nGenerating problem-specific evaluation visualizations...")
    try:
        from utils.problem_specific import get_visualization_module
        viz_module = get_visualization_module(cfg['problem'])
        visualize_evaluation = viz_module[1]  # Second element is visualize_evaluation
        visualize_evaluation(model, eval_data_path, run_dir, cfg)
    except ValueError as e:
        print(f"  (No custom evaluation visualization for {cfg['problem']})")
        print(f"  ValueError details: {e}")
    except Exception as e:
        print(f"  Warning: Could not generate evaluation visualization: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    return best_checkpoint_path


def _move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    """Move a batch dictionary to specified device."""
    result = {
        'x': batch['x'].to(device),
        't': batch['t'].to(device),
        'h_gt': batch['h_gt'].to(device),
        'mask': {
            'residual': batch['mask']['residual'].to(device),
            'IC': batch['mask']['IC'].to(device),
            'BC': batch['mask']['BC'].to(device)
        }
    }
    # Include precomputed base output if available (for pretrained base mode)
    if 'u_base' in batch:
        result['u_base'] = batch['u_base'].to(device)
    return result


def _create_dataloader(
    data: Dict,
    batch_size: int,
    shuffle: bool
) -> DataLoader:
    """
    Create DataLoader from data dictionary.

    Args:
        data: Dictionary with 'x', 't', 'h_gt', 'mask', and optionally 'u_base'
        batch_size: Batch size
        shuffle: Whether to shuffle

    Returns:
        DataLoader
    """
    # Check if precomputed base outputs are available
    has_u_base = 'u_base' in data
    
    # Create TensorDataset - include u_base if available
    if has_u_base:
        dataset = TensorDataset(
            data['x'],
            data['t'],
            data['h_gt'],
            data['mask']['residual'],
            data['mask']['IC'],
            data['mask']['BC'],
            data['u_base']
        )
    else:
        dataset = TensorDataset(
            data['x'],
            data['t'],
            data['h_gt'],
            data['mask']['residual'],
            data['mask']['IC'],
            data['mask']['BC']
        )

    # Custom collate function to reconstruct dict format
    def collate_fn(batch_list):
        x_batch = torch.stack(tuple(item[0] for item in batch_list))
        t_batch = torch.stack(tuple(item[1] for item in batch_list))
        h_gt_batch = torch.stack(tuple(item[2] for item in batch_list))
        mask_res_batch = torch.stack(tuple(item[3] for item in batch_list))
        mask_ic_batch = torch.stack(tuple(item[4] for item in batch_list))
        mask_bc_batch = torch.stack(tuple(item[5] for item in batch_list))

        result = {
            'x': x_batch,
            't': t_batch,
            'h_gt': h_gt_batch,
            'mask': {
                'residual': mask_res_batch,
                'IC': mask_ic_batch,
                'BC': mask_bc_batch
            }
        }
        
        # Include u_base if available (for pretrained base mode)
        if has_u_base:
            u_base_batch = torch.stack(tuple(item[6] for item in batch_list))
            result['u_base'] = u_base_batch
        
        return result

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        pin_memory=False,  # Data already on device
        num_workers=0  # Keep data on GPU
    )


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_name: str,
    epoch: int,
    train_loss: float,
    eval_loss: float,
    cfg: Dict,
    metrics: Dict
) -> None:
    """Save model checkpoint with full information."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer_name,
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'eval_loss': eval_loss,
        'config': cfg,
        'metrics': metrics
    }
    
    # For AdaptiveExpertPINN, also save extended state
    if hasattr(model, 'state_dict_extended'):
        checkpoint['adaptive_state'] = model.state_dict_extended()
        checkpoint['is_adaptive'] = True
    
    torch.save(checkpoint, path)


def _run_intermediate_ncc(model, cfg, run_dir, epoch):
    """Run NCC analysis at intermediate epoch."""
    from ncc.ncc_runner import run_ncc
    
    # Get NCC data path
    ncc_data_path = Path("datasets") / cfg['problem'] / "ncc_data.pt"
    
    # Run NCC with epoch-specific output dir nested inside ncc_plots
    return run_ncc(
        model=model,
        eval_data_path=str(ncc_data_path),
        cfg=cfg,
        run_dir=run_dir,
        epoch_suffix=f"_epoch_{epoch}"
    )


def _run_intermediate_probes(model, cfg, run_dir, epoch):
    """Run probe analysis at intermediate epoch."""
    from probes.probe_runner import run_probes
    train_data_path = Path("datasets") / cfg['problem'] / "training_data.pt"
    eval_data_path = Path("datasets") / cfg['problem'] / "eval_data.pt"
    return run_probes(
        model=model,
        train_data_path=str(train_data_path),
        eval_data_path=str(eval_data_path),
        cfg=cfg,
        run_dir=run_dir,
        epoch_suffix=f"_epoch_{epoch}"
    )


def _run_intermediate_derivatives(model, cfg, run_dir, epoch):
    """Run derivatives tracker at intermediate epoch."""
    from derivatives_tracker.derivatives_runner import run_derivatives_tracker
    train_data_path = Path("datasets") / cfg['problem'] / "training_data.pt"
    eval_data_path = Path("datasets") / cfg['problem'] / "eval_data.pt"
    return run_derivatives_tracker(
        model=model,
        train_data_path=str(train_data_path),
        eval_data_path=str(eval_data_path),
        cfg=cfg,
        run_dir=run_dir,
        epoch_suffix=f"_epoch_{epoch}"
    )


def _run_intermediate_frequency(model, cfg, run_dir, epoch):
    """Run frequency tracker at intermediate epoch."""
    from frequency_tracker.frequency_runner import run_frequency_tracker
    train_data_path = Path("datasets") / cfg['problem'] / "training_data.pt"
    eval_data_path = Path("datasets") / cfg['problem'] / "eval_data.pt"
    return run_frequency_tracker(
        model=model,
        train_data_path=str(train_data_path),
        eval_data_path=str(eval_data_path),
        cfg=cfg,
        run_dir=run_dir,
        epoch_suffix=f"_epoch_{epoch}"
    )


def _maybe_plot_ncc_history(metrics: Dict, run_dir: Path):
    history = metrics.get('ncc_history')
    if not history:
        return
    from ncc.ncc_plotting import plot_ncc_history_shaded
    plot_ncc_history_shaded(history, run_dir / "ncc_plots")


def _maybe_plot_probe_history(metrics: Dict, run_dir: Path):
    history = metrics.get('probe_history')
    if not history:
        return
    from probes.probe_plotting import plot_probe_history_shaded
    plot_probe_history_shaded(history, run_dir / "probe_plots")


def _maybe_plot_deriv_history(metrics: Dict, run_dir: Path):
    history = metrics.get('deriv_history')
    if not history:
        return
    from derivatives_tracker.derivatives_plotting import plot_derivative_history_shaded
    plot_derivative_history_shaded(history, run_dir / "derivatives_plots")

