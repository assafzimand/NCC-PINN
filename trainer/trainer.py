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
from trainer.timing import EpochTimer


def _build_expert_tree_from_pretrained(
    model: nn.Module,
    eval_data: Dict,
    cfg: Dict,
    run_dir: Path,
    loss_fn: Callable
) -> int:
    """
    Build the expert tree using single decision tree traversal.
    (Pretrained case: build entire tree upfront)

    Args:
        model: AdaptiveExpertPINN with pretrained, frozen base model
        eval_data: Evaluation data dictionary
        cfg: Configuration dictionary
        run_dir: Output directory for plots
        loss_fn: Loss function (model, batch) -> scalar

    Returns:
        Number of experts spawned
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Building Expert Tree (Tree-Based Spawning)")
    print("=" * 60)
    print("  Base model is frozen - spawning decisions based ONLY on base predictions")
    print("  Using single decision tree traversal (BFS)")
    print("=" * 60)

    adaptive_cfg = cfg.get('adaptive_pinn', {})
    max_experts = adaptive_cfg.get('max_experts', 5)
    wavelet_threshold = adaptive_cfg.get('wavelet_threshold', None)
    tree_max_depth = adaptive_cfg.get('tree_max_depth', 15)
    tree_min_samples_leaf = adaptive_cfg.get('tree_min_samples_leaf', 10)

    # Import adaptive modules
    from adaptive.region_detector import RegionDetector
    from adaptive.visualization import (
        plot_expert_regions, save_regions_metadata, prepare_ground_truth_grid,
        plot_expert_soft_weights
    )
    from adaptive.residual_utils import compute_loss_components
    from adaptive.indicators import RegionDescriptor

    device = next(model.parameters()).device
    domain_bounds = model.get_domain_bounds()

    # Prepare ground truth grid for visualization
    gt_grid, gt_x, gt_t = prepare_ground_truth_grid(eval_data, domain_bounds)

    # Create RegionDetector with n_estimators=1 (single tree)
    region_detector = RegionDetector(
        n_estimators=1,  # CRITICAL: single tree only
        max_depth=tree_max_depth,
        min_samples_leaf=tree_min_samples_leaf,
        domain_bounds=domain_bounds
    )

    # Create directory for adaptive outputs
    adaptive_plots_dir = run_dir / "adaptive_plots"
    adaptive_plots_dir.mkdir(exist_ok=True)

    # Get base model predictions on eval_data (frozen, won't change)
    # CRITICAL: Always use eval_data, not train data
    model.eval()
    eval_inputs = torch.cat([eval_data['x'], eval_data['t']], dim=1)

    with torch.no_grad():
        u_pred_base = model.base_model(eval_inputs)  # Base only (pretrained, frozen)

    # Compute per-sample loss components for total loss weighting
    problem = cfg.get('problem', 'schrodinger')
    loss_components = compute_loss_components(
        model=model.base_model,  # Use base model directly for loss components
        x=eval_data['x'],
        t=eval_data['t'],
        target=eval_data.get('h_gt', eval_data.get('u_gt')),
        masks=eval_data['mask'],
        loss_fn=loss_fn,
        weights={
            'residual': cfg[problem]['loss_weights']['residual'],
            'ic': cfg[problem]['loss_weights']['ic'],
            'bc': cfg[problem]['loss_weights']['bc']
        }
    )

    # Convert to numpy for RF
    X_eval = eval_inputs.cpu().numpy()
    y_eval = u_pred_base.cpu().numpy()

    # Fit tree once on entire domain (using eval_data)
    print(f"\nFitting single decision tree (max_depth={tree_max_depth}, min_samples_leaf={tree_min_samples_leaf})...")
    region_detector.fit(X=X_eval, y=y_eval, loss_components=loss_components)

    # Traverse tree (BFS) to get candidate regions
    print(f"\nTraversing tree (BFS) to extract regions...")
    traversal_result = region_detector.extract_regions_from_tree(
        wavelet_threshold=wavelet_threshold,
        verbose=True
    )

    if not traversal_result:
        print(f"\nNo regions to spawn (all below threshold or tree is empty)")
        return 0

    # Build mapping from tree node IDs to expert indices
    tree_node_to_expert = {-1: -1}  # Root/base maps to expert -1
    experts_spawned = 0

    print(f"\n{'='*60}")
    print(f"Spawning experts (max_experts={max_experts})...")
    print(f"{'='*60}")

    # Spawn experts in BFS order
    for node_info, parent_tree_node_id in traversal_result:
        if experts_spawned >= max_experts:
            print(f"\nMax experts reached ({max_experts}), stopping spawn process")
            break

        # Map parent tree node to expert index
        if parent_tree_node_id == -1:
            # Root node (base model)
            parent_idx = -1
            depth = 1
        elif parent_tree_node_id in tree_node_to_expert:
            parent_idx = tree_node_to_expert[parent_tree_node_id]
            # Compute depth based on parent
            if parent_idx == -1:
                depth = 1
            else:
                parent_region = model.regions[parent_idx]
                depth = parent_region.depth + 1
        else:
            # Parent wasn't spawned (shouldn't happen with correct traversal)
            print(f"  WARNING: Parent tree node {parent_tree_node_id} not found, using base as parent")
            parent_idx = -1
            depth = 1

        # Create RegionDescriptor
        region = RegionDescriptor(
            bounds_lower=node_info.bounds_lower,
            bounds_upper=node_info.bounds_upper,
            wavelet_norm=node_info.wavelet_norm,
            spawn_epoch=0,  # Tree building phase
            depth=depth,
            parent_idx=parent_idx
        )

        # Spawn expert
        expert_idx = model.spawn_expert(region)
        if expert_idx >= 0:
            # Track mapping for future children
            tree_node_to_expert[node_info.node_id] = expert_idx
            experts_spawned += 1
        else:
            print(f"  Failed to spawn expert for node {node_info.node_id}")
            break

    # Save tree building summary
    print(f"\n{'='*60}")
    print(f"PHASE 1 COMPLETE: Expert Tree Built (Tree-Based Spawning)")
    print(f"  Total experts spawned: {model.num_experts}")
    print(f"  Highest depth: {model.get_highest_depth()}")
    print(f"{'='*60}")

    # Visualize final tree structure
    problem_type = '2d' if len(domain_bounds['lower']) == 2 else '3d'
    plot_expert_regions(
        regions=model.regions,
        domain_bounds=domain_bounds,
        output_path=adaptive_plots_dir / f"expert_regions_final.png",
        problem_type=problem_type,
        title=f"Expert Tree ({model.num_experts} experts)",
        ground_truth=gt_grid,
        grid_x=gt_x,
        grid_t=gt_t
    )

    # Plot soft weights if using soft blending
    if adaptive_cfg.get('blending_mode', 'hard') == 'soft' and problem_type == '2d':
        plot_expert_soft_weights(
            model=model,
            domain_bounds=domain_bounds,
            output_path=adaptive_plots_dir / f"soft_weights_final.png",
            title_prefix="Final Tree: "
        )

    # Save final tree structure
    save_regions_metadata(
        regions=model.regions,
        output_path=adaptive_plots_dir / "expert_tree_structure.json"
    )

    # DIAGNOSTIC: Verify zero-initialization
    print(f"\n{'='*60}")
    print("DIAGNOSTIC: Verifying Expert Initialization")
    print(f"{'='*60}")
    with torch.no_grad():
        sample_inputs = eval_inputs[:100]  # Use eval data sample
        for i, expert in enumerate(model.experts):
            out = expert(sample_inputs)
            out_norm = out.norm().item()
            out_mean = out.abs().mean().item()
            out_max = out.abs().max().item()
            print(f"Expert {i}: norm={out_norm:.8f}, mean={out_mean:.8f}, max={out_max:.8f}")
    print(f"{'='*60}\n")

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
        # Extract tree-based spawning parameters
        tree_max_depth = adaptive_cfg.get('tree_max_depth', 15)
        tree_min_samples_leaf = adaptive_cfg.get('tree_min_samples_leaf', 10)

        print(f"\nAdaptive PINN enabled (Tree-Based Spawning):")
        print(f"  Max experts: {max_experts}")
        print(f"  Spawn every: {spawn_every} epochs")
        print(f"  Wavelet threshold: {wavelet_threshold}")
        print(f"  Tree max depth: {tree_max_depth}")
        print(f"  Tree min samples leaf: {tree_min_samples_leaf}")
        print(f"  Blending mode: {adaptive_cfg.get('blending_mode', 'hard')}")
        print(f"  Freeze mode: {adaptive_cfg.get('freeze_mode', 'none')}")
        print(f"  Pretrained base model: {pretrained_base_model}")
        enable_timing_cfg = adaptive_cfg.get('enable_timing', False)
        print(f"  Timing profiling: {'enabled' if enable_timing_cfg else 'disabled'}")
        
        # Handle pretrained base model mode
        if pretrained_base_model:
            if pretrained_base_path is None:
                raise ValueError("pretrained_base_model is True but pretrained_base_path is not set")
            
            # Load pretrained base and freeze it
            model.load_pretrained_base(pretrained_base_path)
            
            # Build expert tree based ONLY on pretrained base
            num_experts_built = _build_expert_tree_from_pretrained(
                model=model,
                eval_data=eval_data,
                cfg=cfg,
                run_dir=run_dir,
                loss_fn=loss_fn
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
            from adaptive.residual_utils import compute_loss_components
            from adaptive.indicators import RegionDescriptor

            # Get domain bounds
            domain_bounds = model.get_domain_bounds()

            # Prepare ground truth grid for visualization
            gt_grid, gt_x, gt_t = prepare_ground_truth_grid(eval_data, domain_bounds)

            # Create RegionDetector with n_estimators=1 (single tree for each spawn)
            tree_min_samples_leaf = adaptive_cfg.get('tree_min_samples_leaf', 10)
            region_detector = RegionDetector(
                n_estimators=1,  # Single tree for each parent split
                max_depth=1,     # Single binary split per spawn
                min_samples_leaf=tree_min_samples_leaf,
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

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    start_time = time.time()
    
    # Epoch timer for fine-grained performance profiling
    enable_timing = adaptive_cfg.get('enable_timing', False) if is_adaptive else False
    timer = EpochTimer(enabled=enable_timing, print_every=eval_every)
    if enable_timing:
        model._timer = timer

    for epoch in range(1, epochs + 1):
        timer.start_epoch(epoch, num_experts=model.num_experts if is_adaptive else 0)

        # Train phase
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        if current_optimizer_name == 'Adam':
            # Adam: Mini-batch training (GPU parallelized)
            for batch in train_loader:
                optimizer.zero_grad()
                timer.start('train.loss_fn')
                loss = loss_fn(model, batch)
                timer.stop('train.loss_fn')
                timer.start('train.backward')
                loss.backward()
                timer.stop('train.backward')
                timer.start('train.optim_step')
                optimizer.step()
                timer.stop('train.optim_step')

                train_loss += loss.item()
                n_train_batches += 1

                # DIAGNOSTIC: Track expert gradients and outputs (first batch only per epoch)
                if n_train_batches == 1 and model.num_experts > 0:
                    with torch.no_grad():
                        # Check expert gradients
                        expert_grad_norms = []
                        for i, expert in enumerate(model.experts):
                            layer_names = expert.get_layer_names()
                            if layer_names and hasattr(expert.network[layer_names[0]], 'weight'):
                                first_layer = expert.network[layer_names[0]]
                                if first_layer.weight.grad is not None:
                                    grad_norm = first_layer.weight.grad.norm().item()
                                    expert_grad_norms.append(grad_norm)

                        # Check expert outputs vs base
                        inputs = torch.cat([batch['x'], batch['t']], dim=1)
                        decomp = model.forward_decomposed(inputs)
                        base_norm = decomp['base'].norm().item()
                        expert_norms = [decomp[f'expert_{i}'].norm().item() for i in range(model.num_experts)]
                        total_expert_contrib = sum(expert_norms)

                        # Store for this epoch
                        if not hasattr(model, '_diag_data'):
                            model._diag_data = []
                        model._diag_data.append({
                            'epoch': epoch,
                            'base_norm': base_norm,
                            'expert_norms': expert_norms,
                            'expert_grad_norms': expert_grad_norms,
                            'total_expert_contrib': total_expert_contrib
                        })

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
                timer.start('train.lbfgs_step')
                # LBFGS step processes entire dataset via closure
                loss = optimizer.step(closure)
                timer.stop('train.lbfgs_step')
                train_loss = loss.item()
                n_train_batches = 1

                # DIAGNOSTIC: Track expert gradients and outputs (LBFGS)
                if model.num_experts > 0:
                    with torch.no_grad():
                        # Check expert gradients
                        expert_grad_norms = []
                        for i, expert in enumerate(model.experts):
                            layer_names = expert.get_layer_names()
                            if layer_names and hasattr(expert.network[layer_names[0]], 'weight'):
                                first_layer = expert.network[layer_names[0]]
                                if first_layer.weight.grad is not None:
                                    grad_norm = first_layer.weight.grad.norm().item()
                                    expert_grad_norms.append(grad_norm)

                        # Check expert outputs vs base
                        inputs = torch.cat([train_data['x'][:512], train_data['t'][:512]], dim=1)  # Sample for speed
                        decomp = model.forward_decomposed(inputs)
                        base_norm = decomp['base'].norm().item()
                        expert_norms = [decomp[f'expert_{i}'].norm().item() for i in range(model.num_experts)]
                        total_expert_contrib = sum(expert_norms)

                        # Store for this epoch
                        if not hasattr(model, '_diag_data'):
                            model._diag_data = []
                        model._diag_data.append({
                            'epoch': epoch,
                            'base_norm': base_norm,
                            'expert_norms': expert_norms,
                            'expert_grad_norms': expert_grad_norms,
                            'total_expert_contrib': total_expert_contrib
                        })
            
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

            timer.start('eval.train_metrics')
            for batch in train_loader:
                with torch.no_grad():
                    inputs = torch.cat([batch['x'], batch['t']], dim=1)
                    h_pred = model(inputs)
                    rel_l2 = compute_relative_l2_error(h_pred, batch['h_gt'])
                    inf_norm = compute_infinity_norm_error(h_pred, batch['h_gt'])
                    train_rel_l2 += rel_l2.item()
                    train_inf_norm += inf_norm.item()
                    n_train_batches_l2 += 1
            timer.stop('eval.train_metrics')
            
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
                timer.start('eval.loss_fn')
                loss = loss_fn(model, batch)
                timer.stop('eval.loss_fn')

                with torch.no_grad():
                    inputs = torch.cat([batch['x'], batch['t']], dim=1)
                    timer.start('eval.h_pred')
                    h_pred = model(inputs)
                    timer.stop('eval.h_pred')
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

        # End epoch timing (handles printing based on print_every)
        timer.end_epoch()

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

            # DIAGNOSTIC: Print expert contributions
            if model.num_experts > 0 and hasattr(model, '_diag_data') and model._diag_data:
                latest_diag = model._diag_data[-1]
                base_norm = latest_diag['base_norm']
                total_expert = latest_diag['total_expert_contrib']
                expert_norms = latest_diag['expert_norms']
                expert_grads = latest_diag['expert_grad_norms']

                print(f"  [DIAG] Base norm: {base_norm:.6f} | Expert contrib: {total_expert:.6f} | Ratio: {total_expert/base_norm if base_norm > 0 else 0:.4f}")
                print(f"  [DIAG] Expert norms: {[f'{x:.4f}' for x in expert_norms[:5]]}" + ("..." if len(expert_norms) > 5 else ""))
                if expert_grads:
                    print(f"  [DIAG] Expert grad norms: {[f'{x:.6f}' for x in expert_grads[:5]]}" + ("..." if len(expert_grads) > 5 else ""))

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
            print(f"Adaptive PINN: Adding depth level at epoch {epoch}")
            print(f"  Current experts: {model.num_experts}/{max_experts}")
            print(f"  Highest depth populated: {model.get_highest_depth()}")
            print(f"{'='*60}")

            # Get GLOBAL model predictions on eval_data ONCE (not per parent)
            # CRITICAL: eval_data is the fixed evaluation dataset, not training batch
            model.eval()
            with torch.no_grad():
                eval_inputs = torch.cat([eval_data['x'], eval_data['t']], dim=1)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                u_pred = model(eval_inputs)  # Global blended solution (all experts)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            # Compute per-sample loss components on global solution
            problem = cfg.get('problem', 'schrodinger')
            loss_components = compute_loss_components(
                model=model,
                x=eval_data['x'],
                t=eval_data['t'],
                target=eval_data.get('h_gt', eval_data.get('u_gt')),
                masks=eval_data['mask'],
                loss_fn=loss_fn,
                weights={
                    'residual': cfg[problem]['loss_weights']['residual'],
                    'ic': cfg[problem]['loss_weights']['ic'],
                    'bc': cfg[problem]['loss_weights']['bc']
                }
            )

            # Convert to numpy ONCE
            X_eval = eval_inputs.cpu().numpy()
            y_eval = u_pred.cpu().numpy()  # Global predictions

            experts_spawned_this_step = 0

            # Get all nodes at current deepest depth
            deepest_depth = model.get_highest_depth()

            # Build list of parents to process: both spawned experts AND skipped regions
            parent_regions_to_process = []  # List of (region_or_none, parent_idx)

            if deepest_depth == 0:
                # No experts yet - spawn children of base
                parent_regions_to_process.append((None, -1))  # (region, parent_idx)
            else:
                # Add spawned experts at deepest depth
                spawned_at_depth = model.get_regions_at_depth(deepest_depth)
                for region in spawned_at_depth:
                    parent_idx = model.regions.index(region)
                    parent_regions_to_process.append((region, parent_idx))

                # Add skipped regions at deepest depth (tracked from previous iteration)
                # These are regions that failed wavelet threshold but should still have children checked
                if 'skipped_regions_at_depth' in locals() and deepest_depth in skipped_regions_at_depth:
                    for skipped_region, skipped_parent_idx in skipped_regions_at_depth[deepest_depth]:
                        parent_regions_to_process.append((skipped_region, skipped_parent_idx))

            print(f"\n  [Spawning] Adding depth level {deepest_depth + 1}")
            print(f"  [Spawning] Processing {len(parent_regions_to_process)} parents at depth {deepest_depth}")

            # Track skipped regions at next depth for future iterations
            if 'skipped_regions_at_depth' not in locals():
                skipped_regions_at_depth = {}
            skipped_regions_at_depth[deepest_depth + 1] = []

            # For each parent at deepest depth (both spawned and skipped)
            for parent_region, parent_idx in parent_regions_to_process:
                if model.num_experts >= max_experts:
                    print(f"\n  Max experts reached ({max_experts}), stopping spawn process")
                    break

                # Determine parent_str for logging
                if parent_region is None:
                    parent_str = "Base Model"
                elif parent_idx >= 0 and parent_idx < len(model.regions):
                    parent_str = f"Expert {parent_idx+1} (spawned)"
                else:
                    parent_str = f"Skipped region (parent={parent_idx})"

                print(f"\n    [Spawning] Parent {parent_str} (depth {deepest_depth})")

                # Spawn children for this parent
                # IMPORTANT: X_eval is filtered to subdomain, but y_eval is GLOBAL solution
                children = region_detector.spawn_children_for_node(
                    parent_region=parent_region if parent_region is not None else
                                 RegionDescriptor(
                                     bounds_lower=list(domain_bounds['lower']),
                                     bounds_upper=list(domain_bounds['upper']),
                                     wavelet_norm=0.0,
                                     spawn_epoch=0,
                                     depth=0,
                                     parent_idx=-1
                                 ),
                    X=X_eval,  # Full eval_data coordinates
                    y=y_eval,  # Global solution predictions
                    loss_components=loss_components,
                    verbose=True
                )

                # For each child, check wavelet threshold and spawn or track as skipped
                for child_node, _ in children:
                    if model.num_experts >= max_experts:
                        break

                    # Create region descriptor for this child
                    child_region = RegionDescriptor(
                        bounds_lower=child_node.bounds_lower,
                        bounds_upper=child_node.bounds_upper,
                        wavelet_norm=child_node.wavelet_norm,
                        spawn_epoch=epoch,
                        depth=deepest_depth + 1,
                        parent_idx=parent_idx
                    )

                    # Check wavelet threshold
                    if wavelet_threshold is not None and child_node.wavelet_norm < wavelet_threshold:
                        print(f"      [Spawning] Skip child (wavelet={child_node.wavelet_norm:.6f} < threshold={wavelet_threshold})")
                        print(f"                 → Will check its children at depth {deepest_depth + 2}")
                        # Track this skipped region so we can check its children in next depth
                        skipped_regions_at_depth[deepest_depth + 1].append((child_region, parent_idx))
                        continue

                    # Spawn child
                    expert_idx = model.spawn_expert(child_region)
                    if expert_idx >= 0:
                        experts_spawned_this_step += 1

                        # Store expert spawn history
                        if 'expert_spawns' not in metrics:
                            metrics['expert_spawns'] = []
                        metrics['expert_spawns'].append({
                            'epoch': epoch,
                            'expert_idx': expert_idx,
                            'region': child_region.to_dict(),
                            'num_experts': model.num_experts,
                            'depth': deepest_depth + 1,
                            'parent_idx': parent_idx
                        })

            # If any experts were spawned, update optimizer and plot
            if experts_spawned_this_step > 0:
                print(f"\n  [Spawning] Spawned {experts_spawned_this_step} experts in this step")

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
                print(f"\n  [Spawning] No experts spawned this step")
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
    
    # Save timing data and print summary
    timer.save(run_dir / "timing.json")
    timer.print_summary()

    # Save expert diagnostics to CSV
    if model.num_experts > 0 and hasattr(model, '_diag_data') and model._diag_data:
        import pandas as pd
        diag_csv_path = run_dir / "expert_diagnostics.csv"

        # Flatten diagnostic data for CSV
        diag_rows = []
        for diag in model._diag_data:
            row = {
                'epoch': diag['epoch'],
                'base_norm': diag['base_norm'],
                'total_expert_contrib': diag['total_expert_contrib'],
                'ratio_expert_to_base': diag['total_expert_contrib'] / diag['base_norm'] if diag['base_norm'] > 0 else 0
            }
            # Add individual expert norms
            for i, norm in enumerate(diag['expert_norms']):
                row[f'expert_{i}_norm'] = norm
            # Add individual expert gradient norms
            for i, grad_norm in enumerate(diag['expert_grad_norms']):
                row[f'expert_{i}_grad_norm'] = grad_norm
            diag_rows.append(row)

        df = pd.DataFrame(diag_rows)
        df.to_csv(diag_csv_path, index=False)
        print(f"  Expert diagnostics saved: {diag_csv_path}")

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
        h_pred_eval = model(inputs_eval)

    plot_final_comparison(
        h_pred_eval.cpu().numpy(),
        eval_data['h_gt'].cpu().numpy(),
        eval_data['x'].detach().cpu().numpy(),
        eval_data['t'].detach().cpu().numpy(),
        training_plots_dir
    )

    # Run final probes, derivatives, and frequency analysis (without epoch_suffix for main directory)
    # Skip if adaptive PINN with inner_metrics_calculation disabled
    skip_final_inner_metrics = is_adaptive and not adaptive_inner_metrics
    
    if skip_final_inner_metrics:
        print("\n  [Skipping final inner metrics (probes/derivatives/frequency) — inner_metrics_calculation=false]")
    else:
        print("\n" + "=" * 60)
        print("Running Final Probe, Derivative, and Frequency Analysis")
        print("=" * 60)
    
    from probes.probe_runner import run_probes
    from derivatives_tracker.derivatives_runner import run_derivatives_tracker
    from frequency_tracker.frequency_runner import run_frequency_tracker
    
    train_data_path = Path("datasets") / cfg['problem'] / "training_data.pt"
    eval_data_path = Path("datasets") / cfg['problem'] / "eval_data.pt"
    
    if not skip_final_inner_metrics:
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
    if not skip_final_inner_metrics:
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
    return result


def _create_dataloader(
    data: Dict,
    batch_size: int,
    shuffle: bool
) -> DataLoader:
    """
    Create DataLoader from data dictionary.

    Args:
        data: Dictionary with 'x', 't', 'h_gt', 'mask'
        batch_size: Batch size
        shuffle: Whether to shuffle

    Returns:
        DataLoader
    """
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

