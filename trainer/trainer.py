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
from models.atoe import AToE
from models.atoe_leaves import AToELeaves
from models.ant import ANT
from utils.dataset_gen import regenerate_training_data
from losses.causal_weighting import advance_causal_schedule
from losses.lra import LRAWeights


class _NumpySafeEncoder(json.JSONEncoder):
    """Handles numpy scalars that stdlib json cannot serialize."""
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


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
        history_size=cfg.get('lbfgs_history_size', 20),
        line_search_fn=cfg.get('lbfgs_line_search', 'strong_wolfe'),
        tolerance_grad=cfg.get('lbfgs_tolerance_grad', 0.0),
        tolerance_change=cfg.get('lbfgs_tolerance_change', 0.0)
    )


def _create_soap_optimizer(model: nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    """Create SOAP optimizer (quasi-second-order, Shampoo-preconditioned Adam).

    Only includes trainable parameters (requires_grad=True).
    """
    from optimizers.soap import SOAP
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return SOAP(
        trainable_params,
        lr=cfg['lr'],
        betas=tuple(cfg.get('soap_betas', [0.95, 0.95])),
        eps=cfg.get('adam_eps', 1e-8),
        precondition_frequency=cfg.get('soap_precondition_frequency', 10),
        weight_decay=cfg.get('soap_weight_decay', 0.0),
    )


def _create_ssbroyden_optimizer(model: nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    """Create SSBroyden (Self-Scaled Broyden) quasi-Newton optimizer via torchmin.

    Falls back to LBFGS with a warning if torchmin is not installed.
    Only includes trainable parameters.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    try:
        from torchmin import Broyden
        return Broyden(
            trainable_params,
            lr=cfg.get('lr', 1e-3),
            history_size=cfg.get('ssbroyden_history_size', 10),
            max_iter=cfg.get('ssbroyden_max_iter', 1),
        )
    except ImportError:
        print("  [Warning] torchmin not installed — SSBroyden unavailable, falling back to LBFGS.")
        print("            Install with: pip install torchmin")
        return _create_lbfgs_optimizer(model, cfg)


def _create_optimizer_by_name(name: str, model: nn.Module, cfg: Dict) -> Tuple[torch.optim.Optimizer, str]:
    """Create an optimizer by name string. Returns (optimizer, display_name)."""
    name = name.lower()
    if name == 'soap':
        return _create_soap_optimizer(model, cfg), 'SOAP'
    elif name == 'lbfgs':
        return _create_lbfgs_optimizer(model, cfg), 'LBFGS'
    elif name == 'ssbroyden':
        opt = _create_ssbroyden_optimizer(model, cfg)
        return opt, 'SSBroyden' if opt.__class__.__name__ != 'LBFGS' else 'LBFGS'
    else:
        return _create_adam_optimizer(model, cfg), 'Adam'


def _create_primary_optimizer(model: nn.Module, cfg: Dict) -> Tuple[torch.optim.Optimizer, str]:
    """Create the primary (first-order) optimizer based on config.

    Supports new optimizer_1 key and legacy optimizer key.
    Returns (optimizer, name_string).
    """
    opt_name = cfg.get('optimizer_1', cfg.get('optimizer', 'adam')).lower()
    return _create_optimizer_by_name(opt_name, model, cfg)


def _create_lr_scheduler(optimizer, cfg, total_steps):
    """Create an LR scheduler composed of optional warmup + decay.

    Uses standard PyTorch schedulers:
    - LinearLR for warmup (ramps from ~0 to base lr)
    - StepLR for exponential decay (multiplies lr by decay_rate every decay_steps)
    - CosineAnnealingLR for cosine schedule

    Returns None if no scheduling is configured.
    """
    from torch.optim.lr_scheduler import LinearLR, StepLR, CosineAnnealingLR, SequentialLR

    schedule = cfg.get('lr_schedule', 'exponential')
    warmup_steps = cfg.get('lr_warmup_steps', 0)

    if schedule == 'none' and warmup_steps <= 0:
        return None

    schedulers = []
    milestones = []

    if warmup_steps > 0:
        schedulers.append(LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps))
        milestones.append(warmup_steps)

    if schedule == 'exponential':
        decay_rate = cfg.get('lr_decay_rate', 0.9)
        decay_steps = cfg.get('lr_decay_steps', 2000)
        schedulers.append(StepLR(optimizer, step_size=decay_steps, gamma=decay_rate))
    elif schedule == 'cosine':
        remaining = max(total_steps - warmup_steps, 1)
        schedulers.append(CosineAnnealingLR(optimizer, T_max=remaining))

    if len(schedulers) == 0:
        return None
    elif len(schedulers) == 1:
        return schedulers[0]
    else:
        return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)


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

    # DIAGNOSTIC: Verify model is on correct device (configurable)
    if cfg.get('adaptive_pinn', {}).get('enable_gradient_diagnostics', False):
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

    # ── 3-phase logic for full_tree_by_norm / use_perfect_trees ──
    adaptive_cfg_init = cfg.get('adaptive_pinn', {})
    spawning_method_init = adaptive_cfg_init.get('spawning_method', 'by_mean_loss')
    initial_train_cfg = adaptive_cfg_init.get('initial_train', None)
    use_three_phase = (spawning_method_init == 'full_tree_by_norm' and initial_train_cfg is not None)
    use_perfect_trees = (spawning_method_init == 'use_perfect_trees')
    reinit_base_after_spawn = adaptive_cfg_init.get('reinitialize_base_after_spawn', False)

    if use_perfect_trees:
        # Skip Phase 1: spawn from pre-computed tree, then Phase 3
        phase3_epochs = cfg['epochs']
        active_cfg = cfg
        epochs = phase3_epochs
        current_phase = 3
        print(f"\n  [PerfectTree] Skipping Phase 1: loading pre-computed tree")
        print(f"  [PerfectTree] Phase 3 will run for {phase3_epochs} epochs")
    elif use_three_phase:
        phase1_cfg = dict(cfg)
        for k, v in initial_train_cfg.items():
            phase1_cfg[k] = v
        phase1_epochs = initial_train_cfg.get('epochs', cfg['epochs'])
        phase3_epochs = cfg['epochs']
        active_cfg = phase1_cfg
        epochs = phase1_epochs
        current_phase = 1
        print(f"\n  [3-Phase] Phase 1: initial training for {phase1_epochs} epochs")
        print(f"  [3-Phase] Phase 3 will run for {phase3_epochs} epochs after spawning")
        if reinit_base_after_spawn:
            print(f"  [3-Phase] Base model will be reinitialized after spawning")
    else:
        active_cfg = cfg
        epochs = cfg['epochs']
        current_phase = 0  # single-phase (legacy)

    # Determine optimizer strategy (new config: optimizer_1/optimizer_2/optimizer_switch_epoch)
    optimizer_1_name = active_cfg.get('optimizer_1', active_cfg.get('optimizer', 'adam')).lower()
    optimizer_2_name_cfg = active_cfg.get('optimizer_2', None)
    optimizer_2_name = optimizer_2_name_cfg.lower() if optimizer_2_name_cfg else None

    # optimizer_switch_epoch: when to switch. None / ignored when optimizer_2 is null.
    if optimizer_2_name is not None:
        switch_epoch = active_cfg.get('optimizer_switch_epoch', None)
        if switch_epoch is None:
            switch_epoch = epochs + 1  # no switch if not specified
    else:
        switch_epoch = epochs + 1  # never switch

    # Estimate total optimizer steps for LR scheduler
    n_train_samples = train_data['x'].shape[0]
    batches_per_epoch = max(1, (n_train_samples + cfg['batch_size'] - 1) // cfg['batch_size'])
    total_steps_estimate = epochs * batches_per_epoch

    # Patience tracking: only active after switch (or from epoch 1 if no switch)
    patience_start_epoch = switch_epoch if optimizer_2_name is not None else 1

    # Setup initial optimizer + scheduler
    full_batch_opt1 = optimizer_1_name in ('lbfgs', 'ssbroyden')
    if full_batch_opt1:
        optimizer, current_optimizer_name = _create_optimizer_by_name(optimizer_1_name, model, active_cfg)
        lr_scheduler = None
        print(f"Using {current_optimizer_name} optimizer (full-batch) for all epochs")
    else:
        optimizer, current_optimizer_name = _create_primary_optimizer(model, active_cfg)
        lr_scheduler = _create_lr_scheduler(optimizer, active_cfg, total_steps_estimate)
        if optimizer_2_name is not None:
            print(f"Using {current_optimizer_name} until epoch {switch_epoch}, "
                  f"then {optimizer_2_name.upper()} (full-batch)")
            print(f"  Patience early-stopping active from epoch {patience_start_epoch}")
        else:
            print(f"Using {current_optimizer_name} optimizer (mini-batch) for all epochs")
        if lr_scheduler is not None:
            sched_name = active_cfg.get('lr_schedule', 'exponential')
            warmup = active_cfg.get('lr_warmup_steps', 0)
            print(f"  LR schedule: {sched_name} (warmup={warmup} steps, ~{total_steps_estimate} total steps)")

    step_count = 0  # global optimizer step counter for LR scheduler

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
        'eval_rel_l2': [],
        'eval_inf_norm': [],
        'causal_history': [],      # Causal training state (tol, min_weight, stage) at eval epochs
        'lra_history': [],         # LRA weights and grad norms at eval epochs
        'resample_events': []      # Track resampling/skipping events
    }

    best_eval_loss = float('inf')
    best_train_loss = float('inf')
    best_checkpoint_path = None
    patience_epochs = cfg.get('patience_epochs', 0)
    min_epochs = cfg.get('min_epochs', 0)
    epochs_without_improvement = 0

    # LRA: adaptive loss component weighting
    lra_cfg = cfg.get('lra', {})
    lra_enabled = lra_cfg.get('enabled', False)
    if lra_enabled:
        # Initialize LRA weights from problem's loss_weights
        problem = cfg.get('problem', '')
        problem_cfg = cfg.get(problem, {})
        initial_loss_weights = problem_cfg.get('loss_weights', {})
        lra_weights = LRAWeights(
            alpha=lra_cfg.get('alpha', 0.1),
            update_every=lra_cfg.get('update_every', 100),
            initial_weights=initial_loss_weights,
        )
    else:
        lra_weights = None

    # Wrap loss_fn to apply LRA weights when enabled
    if lra_weights is not None:
        _orig_loss_fn = loss_fn

        def _lra_loss_fn(model, batch, for_tree_spawning=False, return_components=False):
            if for_tree_spawning or return_components:
                return _orig_loss_fn(model, batch,
                                     for_tree_spawning=for_tree_spawning,
                                     return_components=return_components)
            comps = _orig_loss_fn(model, batch, return_components=True)
            w = lra_weights.weights
            return (w.get('residual', 1.0) * comps['residual']
                    + w.get('ic', 1.0) * comps['ic']
                    + w.get('bc', 1.0) * comps['bc'])

        _lra_loss_fn.causal_state = getattr(loss_fn, 'causal_state', None)
        loss_fn = _lra_loss_fn

    checkpoint_dir = run_dir / "checkpoints"
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
    spawning_method = adaptive_cfg.get('spawning_method', 'by_mean_loss')
    problem_cfg = cfg.get(cfg['problem'], {})
    wavelet_threshold = problem_cfg.get('wavelet_threshold', 0.0)
    adaptive_inner_metrics = adaptive_cfg.get('inner_metrics_calculation', False)
    spawning_complete = False
    
    # Read configurable norm variables
    variable_for_node_accept = adaptive_cfg.get('variable_for_node_accept', 'norm')
    variable_for_expert_size = adaptive_cfg.get('variable_for_expert_size', 'norm')
    
    # Build thresholds dict
    thresholds = {
        'norm': problem_cfg.get('wavelet_threshold', 0.0),
        'new_norm': problem_cfg.get('new_norm_threshold', 0.0),
        'smoothness': problem_cfg.get('tree_smoothness_threshold', 0.7),
    }

    if is_adaptive:
        tree_max_depth = adaptive_cfg.get('tree_max_depth', 15)
        tree_min_samples_leaf = adaptive_cfg.get('tree_min_samples_leaf', 10)

        print(f"\nAdaptive PINN enabled (spawning_method={spawning_method}):")
        print(f"  Max experts: {max_experts}")
        print(f"  Spawn every: {spawn_every} epochs")
        print(f"  Tree max depth: {tree_max_depth}")
        print(f"  Tree min samples leaf: {tree_min_samples_leaf}")
        if spawning_method in ('accept_split_by_norm', 'full_tree_by_norm', 'use_perfect_trees'):
            print(f"  Wavelet threshold: {wavelet_threshold}")
        print(f"  Blending mode: {adaptive_cfg.get('blending_mode', 'hard')}")
        print(f"  Freeze mode: {adaptive_cfg.get('freeze_mode', 'none')}")
        print(f"  Model type: {type(model).__name__}")
        enable_timing_cfg = adaptive_cfg.get('enable_timing', False)
        print(f"  Timing profiling: {'enabled' if enable_timing_cfg else 'disabled'}")

        from adaptive.region_detector import RegionDetector
        from adaptive.visualization import (
            plot_expert_regions, save_regions_metadata, prepare_ground_truth_grid,
            plot_expert_soft_weights
        )
        from adaptive.residual_utils import compute_loss_components
        from adaptive.indicators import RegionDescriptor

        domain_bounds = model.get_domain_bounds()
        gt_grid, gt_x, gt_t = prepare_ground_truth_grid(eval_data, domain_bounds)

        tree_min_samples_leaf = adaptive_cfg.get('tree_min_samples_leaf', 10)
        region_detector = RegionDetector(
            n_estimators=1,
            max_depth=tree_max_depth if spawning_method in ('full_tree_by_norm', 'use_perfect_trees') else 1,
            min_samples_leaf=tree_min_samples_leaf,
            domain_bounds=domain_bounds
        )
        
        # Create directory for adaptive outputs
        adaptive_plots_dir = run_dir / "adaptive_plots"
        adaptive_plots_dir.mkdir(exist_ok=True)
    
    rejected_regions = []
    leaf_loss_history = []

    # ── use_perfect_trees: spawn experts from JSON before training ──
    if use_perfect_trees and is_adaptive:
        import json as _json
        perfect_trees_path = adaptive_cfg.get(
            'perfect_trees_path',
            'perfect_tree_examples/perfect_trees.json')
        problem_name = cfg.get('problem', '')
        print(f"\n  [PerfectTree] Loading tree from: {perfect_trees_path}")

        with open(perfect_trees_path, 'r') as _f:
            all_perfect_trees = _json.load(_f)

        if problem_name not in all_perfect_trees:
            raise ValueError(
                f"Problem '{problem_name}' not found in "
                f"{perfect_trees_path}. "
                f"Available: {list(all_perfect_trees.keys())}")

        pt_data = all_perfect_trees[problem_name]
        pt_nodes = pt_data['accepted_nodes_bfs']
        pt_summary = pt_data['summary']
        print(f"  [PerfectTree] Tree has "
              f"{pt_summary['accepted_nodes']} accepted nodes "
              f"({pt_summary['pruned_tree_leaves']} leaves)")

        is_copy_spawn = isinstance(model, AToELeaves)
        is_atoe_plain = isinstance(model, AToE) and not isinstance(model, AToELeaves)
        atoe_zero_init = not reinit_base_after_spawn
        if isinstance(model, AToELeaves):
            nodes_to_spawn = [
                n for n in pt_nodes
                if n['is_leaf_in_pruned_tree']]
        else:
            nodes_to_spawn = pt_nodes

        node_to_expert = {}
        experts_spawned_pt = 0
        for nd in nodes_to_spawn:
            parent_tree_nid = nd['parent_tree_node_id']
            parent_expert_idx = node_to_expert.get(
                parent_tree_nid, -1)
            depth = nd['tree_depth']

            child_region = RegionDescriptor(
                bounds_lower=nd['bounds_lower'],
                bounds_upper=nd['bounds_upper'],
                wavelet_norm_squared=nd['wavelet_norm_squared'],
                new_wavelet_norm_squared=nd.get('new_wavelet_norm_squared', 0.0),
                spawn_epoch=0,
                depth=depth,
                parent_idx=parent_expert_idx,
            )

            if is_copy_spawn:
                expert_idx = model.spawn_expert(
                    child_region,
                    copy_from_idx=parent_expert_idx)
            elif is_atoe_plain:
                expert_idx = model.spawn_expert(
                    child_region, zero_init=False)
            else:
                expert_idx = model.spawn_expert(child_region)
            if expert_idx >= 0:
                node_to_expert[nd['node_id']] = expert_idx
                experts_spawned_pt += 1

        print(f"  [PerfectTree] Spawned {experts_spawned_pt} "
              f"experts from perfect tree")

        if reinit_base_after_spawn:
            model.reinitialize_base()

        spawning_complete = True
        model.freeze_models()

        # Save metrics
        if 'spawning_diagnostics' not in metrics:
            metrics['spawning_diagnostics'] = []
        metrics['spawning_diagnostics'].append({
            'epoch': 0,
            'method': 'use_perfect_trees',
            'source_file': perfect_trees_path,
            'accepted_count': len(nodes_to_spawn),
            'spawned_count': experts_spawned_pt,
            'nodes': pt_data.get('all_nodes', []),
            'wavelet_threshold': pt_data['tree_params'].get(
                'wavelet_threshold', 0),
        })

        # Plot initial expert regions
        problem_type = (
            '2d' if len(domain_bounds['lower']) == 2
            else '3d')
        leaf_info = model.get_leaf_info()
        leaf_expert_indices = [
            idx for _, idx in leaf_info if idx >= 0]
        regions_to_plot = (
            [model.regions[i] for i in leaf_expert_indices]
            if isinstance(model, (AToELeaves, ANT))
            else model.regions)
        plot_expert_regions(
            regions=regions_to_plot,
            domain_bounds=domain_bounds,
            output_path=(
                adaptive_plots_dir
                / "expert_regions_perfect_tree.png"),
            problem_type=problem_type,
            title=(
                f"Perfect Tree Regions "
                f"({len(regions_to_plot)} experts)"),
            ground_truth=gt_grid,
            grid_x=gt_x,
            grid_t=gt_t,
        )

        # Recreate optimizer for Phase 3 (fresh state)
        optimizer, current_optimizer_name = (
            _create_primary_optimizer(model, active_cfg))
        total_steps_p3 = phase3_epochs * batches_per_epoch
        lr_scheduler = _create_lr_scheduler(
            optimizer, active_cfg, total_steps_p3)
        # Recalculate switch epoch for Phase 3 with new config keys
        _pt_opt2_cfg = active_cfg.get('optimizer_2', None)
        optimizer_2_name = _pt_opt2_cfg.lower() if _pt_opt2_cfg else None
        if optimizer_2_name is not None:
            _pt_switch = active_cfg.get('optimizer_switch_epoch', None)
            switch_epoch = _pt_switch if _pt_switch else epochs + 1
            patience_start_epoch = switch_epoch
        else:
            switch_epoch = epochs + 1
            patience_start_epoch = 1
        step_count = 0
        print(f"  [PerfectTree] Phase 3 optimizer: "
              f"{current_optimizer_name}, "
              f"lr={active_cfg.get('lr')}, "
              f"schedule={active_cfg.get('lr_schedule', 'exponential')}")

    # Training loop
    total_epochs = epochs  # may extend when transitioning to Phase 3
    print(f"\nTraining for {total_epochs} epochs...")
    start_time = time.time()
    
    # Epoch timer for fine-grained performance profiling
    enable_timing = adaptive_cfg.get('enable_timing', False) if is_adaptive else False
    timer = EpochTimer(enabled=enable_timing, print_every=eval_every)
    if enable_timing:
        model._timer = timer

    train_loss = 0.0
    eval_loss = 0.0
    eval_rel_l2 = 0.0
    eval_inf_norm = 0.0

    resample_every = cfg.get('sampling', {}).get('resample_every_epochs', 0)
    base_seed = cfg.get('seed', 42)
    
    # Consolidated feature summary
    print("\n" + "=" * 60)
    print("FEATURE SUMMARY")
    print("=" * 60)
    
    # Fourier Features
    ff_cfg = cfg.get('fourier_features', {})
    ff_enabled = ff_cfg.get('enabled', False)
    if ff_enabled:
        ff_dim = ff_cfg.get('dim', 64)
        ff_scale = ff_cfg.get('scale', 1.0)
        print(f"  Fourier Features: enabled (dim={ff_dim}, scale={ff_scale}, output_dim={2*ff_dim})")
    else:
        print(f"  Fourier Features: disabled")
    
    # RWF
    rwf_enabled = cfg.get('rwf', False)
    if rwf_enabled:
        print(f"  RWF: enabled")
    else:
        print(f"  RWF: disabled")
    
    # Causal Training
    _cs_init = getattr(loss_fn, 'causal_state', None)
    if _cs_init is not None:
        print(f"  Causal Training: enabled (schedule={_cs_init['schedule']}, chunks={_cs_init['num_chunks']}, threshold={_cs_init['threshold']})")
    else:
        print(f"  Causal Training: disabled")
    
    # LRA
    if lra_enabled:
        init_w = lra_weights.weights
        print(f"  LRA: enabled (alpha={lra_weights.alpha}, update_every={lra_weights.update_every}, "
              f"init_weights={{res={init_w['residual']:.1f}, ic={init_w['ic']:.1f}, bc={init_w['bc']:.1f}}})")
    else:
        print(f"  LRA: disabled")
    
    # Resampling & Adaptive Sampling
    adaptive_sampling_enabled = cfg.get('sampling', {}).get('adaptive_sampling', {}).get('enabled', False)
    if resample_every > 0:
        if adaptive_sampling_enabled:
            as_ratio = cfg.get('sampling', {}).get('adaptive_sampling', {}).get('adaptive_ratio', 0.5)
            print(f"  Resampling: every {resample_every} epochs (adaptive: enabled, ratio={as_ratio})")
        else:
            print(f"  Resampling: every {resample_every} epochs (adaptive: disabled)")
    else:
        print(f"  Resampling: disabled")
    
    # Optimizer schedule
    opt1_name = cfg.get('optimizer_1', cfg.get('optimizer', 'adam'))
    opt2_name = cfg.get('optimizer_2', 'null')
    if opt2_name and opt2_name != 'null':
        switch_epoch = cfg.get('optimizer_switch_epoch', total_epochs + 1)
        print(f"  Optimizer: {opt1_name} → {opt2_name} at epoch {switch_epoch}")
    else:
        print(f"  Optimizer: {opt1_name}")
    
    # Early stopping
    if patience_epochs > 0:
        print(f"  Early stopping: enabled (patience={patience_epochs}, min_epochs={min_epochs})")
    else:
        print(f"  Early stopping: disabled")
    
    print("=" * 60 + "\n")

    epoch = 0
    while epoch < total_epochs:
        epoch += 1
        timer.start_epoch(epoch, num_experts=model.num_experts if (is_adaptive and hasattr(model, 'num_experts')) else 0)

        # Enable residual caching for adaptive sampling if needed
        # Cache THIS epoch's residuals for NEXT epoch's resampling
        # Only activate adaptive sampling when causal training reaches final stage (Option B)
        adaptive_sampling_enabled = cfg.get('sampling', {}).get('adaptive_sampling', {}).get('enabled', False)
        causal_state = getattr(loss_fn, 'causal_state', None)
        causal_at_final_stage = True  # default if causal disabled
        if causal_state is not None:
            causal_at_final_stage = (causal_state['schedule_idx'] >= len(causal_state['schedule']) - 1)
        
        will_cache_for_resample = (
            adaptive_sampling_enabled and causal_at_final_stage
            and resample_every > 0
            and epoch > 0 and epoch % resample_every == 0
        )
        if will_cache_for_resample:
            model._residual_cache = []
            model._residual_cache_enabled = True
            # Log when adaptive sampling first activates
            if not hasattr(model, '_adaptive_sampling_activated'):
                model._adaptive_sampling_activated = True
                print(f"  [Adaptive Sampling] Activated at epoch {epoch} (causal training reached final stage)")

        # Resample training data periodically (in-memory, no disk I/O)
        # Skip resampling during L-BFGS/SSBroyden (they need stable loss landscape)
        allow_resample_optimizer = current_optimizer_name not in ('LBFGS', 'SSBroyden')
        if resample_every > 0 and epoch > 1 and (epoch - 1) % resample_every == 0 and allow_resample_optimizer:
            resample_seed = base_seed + epoch
            print(f"  [Resample] Regenerating training data (epoch {epoch}, seed {resample_seed})...")
            # Get cached residuals from previous epoch (if available)
            cached_residuals = getattr(model, '_residual_cache', [])
            model._residual_cache_enabled = False
            train_data = regenerate_training_data(
                cfg, device, resample_seed=resample_seed,
                cached_residuals=cached_residuals,
                run_dir=run_dir,
                epoch=epoch
            )
            train_loader = _create_dataloader(train_data, cfg['batch_size'], shuffle=True)
            # Save resample event to metrics
            metrics['resample_events'].append({
                'epoch': epoch,
                'action': 'resampled',
                'optimizer': current_optimizer_name
            })
        elif resample_every > 0 and epoch > 1 and (epoch - 1) % resample_every == 0 and not allow_resample_optimizer:
            # Log when resampling is skipped due to optimizer
            if not hasattr(model, '_resample_skip_logged'):
                model._resample_skip_logged = True
                print(f"  [Resample] Skipping resampling during {current_optimizer_name} (loss landscape stability required)")
            # Save skip event to metrics
            metrics['resample_events'].append({
                'epoch': epoch,
                'action': 'skipped',
                'optimizer': current_optimizer_name,
                'reason': 'optimizer_stability'
            })

        # Train phase
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        if current_optimizer_name in ('Adam', 'SOAP'):
            # Adam/SOAP: Mini-batch training (GPU parallelized)
            for batch in train_loader:
                optimizer.zero_grad()
                timer.start('train.loss_fn')
                loss = loss_fn(model, batch)
                timer.stop('train.loss_fn')
                timer.start('train.backward')
                loss.backward()
                timer.stop('train.backward')

                # DIAGNOSTIC: Check gradients immediately after backward (early epochs only, configurable)
                enable_grad_diag = adaptive_cfg.get('enable_gradient_diagnostics', False) if is_adaptive else False
                if enable_grad_diag and n_train_batches == 0 and hasattr(model, 'num_experts') and model.num_experts > 0 and epoch <= 10:
                    print(f"\n[DIAG Epoch {epoch}] Checking gradients after backward pass:")
                    for i, expert in enumerate(model.experts):
                        layer_names = expert.get_layer_names()
                        if layer_names:
                            first_layer = expert.network[layer_names[0]]
                            final_layer = expert.network[layer_names[-1]]
                            first_grad = first_layer.weight.grad
                            final_grad = final_layer.weight.grad
                            print(f"  Expert {i}: first_layer.grad={'None' if first_grad is None else f'norm={first_grad.norm().item():.6f}'}, "
                                  f"final_layer.grad={'None' if final_grad is None else f'norm={final_grad.norm().item():.6f}'}")

                timer.start('train.optim_step')
                optimizer.step()
                timer.stop('train.optim_step')

                step_count += 1
                if lr_scheduler is not None and current_optimizer_name != 'LBFGS':
                    lr_scheduler.step()

                train_loss += loss.item()
                n_train_batches += 1

                # DIAGNOSTIC: Track expert gradients and outputs (first batch only per epoch, configurable)
                enable_grad_diag = adaptive_cfg.get('enable_gradient_diagnostics', False) if is_adaptive else False
                if enable_grad_diag and n_train_batches == 1 and is_adaptive and hasattr(model, 'num_experts') and model.num_experts > 0:
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

                # DIAGNOSTIC: Track expert gradients and outputs (LBFGS, configurable)
                enable_grad_diag = adaptive_cfg.get('enable_gradient_diagnostics', False) if is_adaptive else False
                if enable_grad_diag and is_adaptive and hasattr(model, 'num_experts') and model.num_experts > 0:
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
                    
                    optimizer, current_optimizer_name = _create_primary_optimizer(model, active_cfg)
                    lr_scheduler = _create_lr_scheduler(optimizer, active_cfg, total_steps_estimate)
                    
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
        
        # Check for optimizer switch (optimizer_1 → optimizer_2)
        if epoch == switch_epoch and optimizer_2_name is not None:
            print(f"\n{'='*60}")
            print(f"OPTIMIZER SWITCH: {current_optimizer_name} -> {optimizer_2_name.upper()} at epoch {epoch}")
            print(f"{'='*60}\n")
            optimizer, current_optimizer_name = _create_optimizer_by_name(
                optimizer_2_name, model, active_cfg)
            lr_scheduler = None  # optimizer_2 uses its own LR / line search
            # Reset patience counter at switch point
            epochs_without_improvement = 0
            best_train_loss = float('inf')
        
        # Store train loss every epoch
        metrics['train_loss_epochs'].append(epoch)
        metrics['train_loss'].append(train_loss)

        # LRA: update adaptive loss weights periodically
        if lra_weights is not None and epoch % lra_weights.update_every == 0:
            try:
                batch_for_lra = next(iter(train_loader))
                lra_weights.update(model, loss_fn, batch_for_lra)
                if epoch % print_every == 0:
                    print(f"  [LRA] weights: residual={lra_weights.weights['residual']:.4f}, "
                          f"ic={lra_weights.weights['ic']:.4f}, "
                          f"bc={lra_weights.weights['bc']:.4f}")
            except Exception as e:
                print(f"  [LRA] Weight update failed at epoch {epoch}: {e}")

        # Causal weighting: check if epsilon should advance
        causal_state = getattr(loss_fn, 'causal_state', None)
        if advance_causal_schedule(causal_state):
            cs = loss_fn.causal_state
            print(f"  [Causal] epsilon advanced to "
                  f"{cs['tol']:.2f} "
                  f"(stage {cs['schedule_idx']+1}/"
                  f"{len(cs['schedule'])}, "
                  f"min_w={cs['min_weight']:.4f})")

        # Compute evaluation metrics only every print_every epochs or last epoch
        # This speeds up training significantly for physics-informed losses
        should_evaluate = (epoch % eval_every == 0 or epoch == 1 or epoch == total_epochs)
        
        # Initialize metrics for this epoch (will be updated if we evaluate)
        eval_loss = None
        eval_rel_l2 = None
        eval_inf_norm = None

        if should_evaluate:
            # Compute train rel-L2 and infinity norm errors

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
            metrics['eval_rel_l2'].append(eval_rel_l2)
            metrics['eval_inf_norm'].append(eval_inf_norm)

        # End epoch timing (handles printing based on print_every)
        timer.end_epoch()

        # Print progress
        if should_evaluate:
            elapsed = time.time() - start_time
            batch_mode = "mini" if current_optimizer_name in ('Adam', 'SOAP') else "full"
            print(f"Epoch [{epoch}/{total_epochs}] ({elapsed:.1f}s) [{current_optimizer_name}/{batch_mode}] | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Eval Loss: {eval_loss:.6f} | "
                  f"Eval Rel-L2: {eval_rel_l2:.6f} | "
                  f"Eval Inf: {eval_inf_norm:.6f}")

            # DIAGNOSTIC: Causal weight progression
            causal_state = getattr(loss_fn, 'causal_state', None)
            if causal_state is not None:
                cs = causal_state
                stage_str = f"{cs['schedule_idx']+1}/{len(cs['schedule'])}"
                print(f"  [Causal] tol={cs['tol']:.2f}, stage={stage_str}, min_weight={cs['min_weight']:.6f}")
                # Save to metrics
                metrics['causal_history'].append({
                    'epoch': epoch,
                    'tol': float(cs['tol']),
                    'stage': int(cs['schedule_idx']),
                    'stage_total': len(cs['schedule']),
                    'min_weight': float(cs['min_weight']),
                    'threshold': float(cs['threshold'])
                })

            # DIAGNOSTIC: LRA weights and gradient norms
            if lra_weights is not None:
                w = lra_weights.weights
                g = lra_weights.last_grad_norms
                print(f"  [LRA] weights: res={w['residual']:.4f}, ic={w['ic']:.4f}, bc={w['bc']:.4f} | "
                      f"grad_norms: res={g.get('residual', 0):.6f}, ic={g.get('ic', 0):.6f}, bc={g.get('bc', 0):.6f}")
                # Save to metrics
                metrics['lra_history'].append({
                    'epoch': epoch,
                    'weights': {
                        'residual': float(w['residual']),
                        'ic': float(w['ic']),
                        'bc': float(w['bc'])
                    },
                    'grad_norms': {
                        'residual': float(g.get('residual', 0)),
                        'ic': float(g.get('ic', 0)),
                        'bc': float(g.get('bc', 0))
                    }
                })

            # DIAGNOSTIC: Unweighted loss component breakdown
            # Compute on a sample eval batch
            try:
                sample_batch = next(iter(eval_loader))
                # Get the original loss function (unwrap LRA if present)
                orig_loss_fn = getattr(loss_fn, '__wrapped__', loss_fn)
                if hasattr(orig_loss_fn, '__self__'):  # It's a method/closure
                    # For the LRA wrapper, we need to access _orig_loss_fn from the closure
                    if '_orig_loss_fn' in dir(loss_fn):
                        orig_loss_fn = loss_fn.__code__.co_consts  # This won't work, need different approach
                # Actually, just call with return_components=True which the wrapper forwards
                with torch.no_grad():
                    components = loss_fn(model, sample_batch, return_components=True)
                    print(f"  [Loss] components: residual={components['residual']:.6f}, "
                          f"ic={components['ic']:.6f}, bc={components['bc']:.6f} (unweighted)")
            except Exception as e:
                # Don't crash if component breakdown fails
                pass

            # DIAGNOSTIC: Print expert contributions (configurable)
            enable_grad_diag = adaptive_cfg.get('enable_gradient_diagnostics', False) if is_adaptive else False
            if enable_grad_diag and is_adaptive and hasattr(model, 'num_experts') and model.num_experts > 0 and hasattr(model, '_diag_data') and model._diag_data:
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

        # Patience-based early stopping on train loss (only active from patience_start_epoch)
        if train_loss is not None and patience_epochs > 0 and epoch >= patience_start_epoch:
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if (epoch >= min_epochs
                    and epochs_without_improvement >= patience_epochs):
                print(f"\n  [EarlyStop] No train loss improvement "
                      f"for {epochs_without_improvement} epochs "
                      f"(best={best_train_loss:.6f}). "
                      f"Stopping at epoch {epoch}.")
                break

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

        # Adaptive PINN: Hierarchical expert spawning from leaf nodes
        if spawning_method in ('full_tree_by_norm', 'use_perfect_trees'):
            spawn_check_triggered = (is_adaptive and
                                     epoch % spawn_every == 0 and
                                     not spawning_complete)
        else:
            spawn_check_triggered = (is_adaptive and
                                     epoch % spawn_every == 0 and
                                     hasattr(model, 'num_experts') and
                                     model.num_experts < max_experts)
        
        if spawn_check_triggered:
            print(f"\n{'='*60}")
            print(f"Adaptive PINN: Spawning check at epoch {epoch}")
            if hasattr(model, 'num_experts'):
                print(f"  Current experts: {model.num_experts}/{max_experts}")
            leaf_nodes = model.get_leaf_info()
            print(f"  Current leaf nodes: {len(leaf_nodes)}")
            print(f"{'='*60}")

            model.eval()
            with torch.no_grad():
                eval_inputs = torch.cat([eval_data['x'], eval_data['t']], dim=1)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                u_pred = model(eval_inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            X_eval = eval_inputs.cpu().numpy()
            y_eval = u_pred.cpu().numpy()

            # Only compute per-sample losses for by_mean_loss (expensive)
            loss_components = None
            if spawning_method == 'by_mean_loss':
                problem = cfg.get('problem', 'schrodinger')
                loss_weights = cfg[problem].get('loss_weights', {})
                loss_components = compute_loss_components(
                    model=model,
                    x=eval_data['x'],
                    t=eval_data['t'],
                    target=eval_data.get('h_gt', eval_data.get('u_gt')),
                    masks=eval_data['mask'],
                    loss_fn=loss_fn,
                    weights={
                        'residual': loss_weights.get('residual', 1.0),
                        'ic': loss_weights.get('ic', 1.0),
                        'bc': loss_weights.get('bc', 1.0),
                    }
                )

            import numpy as np
            is_copy_spawn = isinstance(model, AToELeaves)
            experts_spawned_this_step = 0

            if 'spawning_diagnostics' not in metrics:
                metrics['spawning_diagnostics'] = []

            # =============================================================
            # Dispatch on spawning_method
            # =============================================================

            if spawning_method == 'by_mean_loss':
                # Pick the single leaf with highest mean loss, split it
                _spawned_children_diag = []
                w_res = loss_components['weights'].get('residual', 1.0)
                w_ic  = loss_components['weights'].get('ic', 1.0)
                w_bc  = loss_components['weights'].get('bc', 1.0)
                per_sample_total = (w_res * loss_components['residual']
                                    + w_ic * loss_components['ic']
                                    + w_bc * loss_components['bc'])

                leaf_mean_losses = []
                for leaf_region, leaf_idx in leaf_nodes:
                    if leaf_region is None:
                        mask = np.ones(len(X_eval), dtype=bool)
                    else:
                        mask = np.ones(len(X_eval), dtype=bool)
                        for dim in range(len(leaf_region.bounds_lower)):
                            mask &= (X_eval[:, dim] >= leaf_region.bounds_lower[dim])
                            mask &= (X_eval[:, dim] <= leaf_region.bounds_upper[dim])
                    n_in_region = mask.sum()
                    if n_in_region > 0:
                        mean_loss = float(per_sample_total[mask].mean())
                    else:
                        mean_loss = 0.0
                    leaf_mean_losses.append((mean_loss, leaf_region, leaf_idx, n_in_region))
                    leaf_str = f"Expert {leaf_idx+1}" if leaf_idx >= 0 else "Base Model"
                    print(f"    Leaf {leaf_str}: mean_loss={mean_loss:.6f} ({n_in_region} samples)")

                leaf_loss_history.append({
                    'epoch': epoch,
                    'leaves': [
                        {
                            'leaf_idx': idx,
                            'mean_loss': ml,
                            'n_samples': int(n),
                            'bounds_lower': list(reg.bounds_lower) if reg is not None else list(domain_bounds['lower']),
                            'bounds_upper': list(reg.bounds_upper) if reg is not None else list(domain_bounds['upper']),
                        }
                        for ml, reg, idx, n in leaf_mean_losses
                    ]
                })

                leaf_mean_losses.sort(key=lambda x: x[0], reverse=True)

                for candidate_loss, candidate_region, candidate_idx, candidate_n in leaf_mean_losses:
                    candidate_str = f"Expert {candidate_idx+1}" if candidate_idx >= 0 else "Base Model"
                    print(f"\n  [Spawning] Trying leaf: {candidate_str} "
                          f"(mean_loss={candidate_loss:.6f}, {candidate_n} samples)")

                    parent_region = candidate_region
                    parent_idx = candidate_idx
                    parent_depth = 0 if parent_region is None else parent_region.depth

                    children = region_detector.spawn_children_for_node(
                        parent_region=parent_region if parent_region is not None else
                                     RegionDescriptor(
                                         bounds_lower=list(domain_bounds['lower']),
                                         bounds_upper=list(domain_bounds['upper']),
                                         wavelet_norm_squared=0.0,
                                         spawn_epoch=0,
                                         depth=0,
                                         parent_idx=-1
                                     ),
                        X=X_eval,
                        y=y_eval,
                        loss_components=loss_components,
                        verbose=True
                    )

                    if not children:
                        print(f"      [Spawning] Could not split {candidate_str} "
                              f"(too few samples?), trying next leaf...")
                        continue

                    child_depth = parent_depth + 1
                    for child_node, _ in children:
                        child_region = RegionDescriptor(
                            bounds_lower=child_node.bounds_lower,
                            bounds_upper=child_node.bounds_upper,
                            wavelet_norm_squared=child_node.wavelet_norm_squared,
                            new_wavelet_norm_squared=child_node.new_wavelet_norm_squared,
                            spawn_epoch=epoch,
                            depth=child_depth,
                            parent_idx=parent_idx,
                            smoothness_alpha=child_node.smoothness_alpha,
                        )

                        if is_copy_spawn:
                            expert_idx = model.spawn_expert(child_region, copy_from_idx=parent_idx)
                        else:
                            expert_idx = model.spawn_expert(child_region)
                        if expert_idx >= 0:
                            experts_spawned_this_step += 1
                            if 'expert_spawns' not in metrics:
                                metrics['expert_spawns'] = []
                            metrics['expert_spawns'].append({
                                'epoch': epoch,
                                'expert_idx': expert_idx,
                                'region': child_region.to_dict(),
                                'depth': child_depth,
                                'parent_idx': parent_idx,
                                **(({'num_experts': model.num_experts} if hasattr(model, 'num_experts') else {}))
                            })

                    _spawned_children_diag = [
                        {
                            'node_id': c.node_id,
                            'wavelet_norm_squared': c.wavelet_norm_squared,
                            'n_samples': c.n_samples,
                            'bounds_lower': c.bounds_lower,
                            'bounds_upper': c.bounds_upper,
                        }
                        for c, _ in children
                    ]
                    if experts_spawned_this_step > 0:
                        print(f"      [Spawning] Spawned {experts_spawned_this_step} children from {candidate_str}")
                    break

                # Save diagnostics for by_mean_loss
                diag = {
                    'epoch': epoch,
                    'method': 'by_mean_loss',
                    'evaluated_leaves': [
                        {
                            'leaf_idx': idx,
                            'mean_loss': ml,
                            'n_samples': int(n),
                            'selected': (idx == leaf_mean_losses[0][2]),
                            'bounds_lower': list(
                                reg.bounds_lower) if reg is not None else list(domain_bounds['lower']),
                            'bounds_upper': list(
                                reg.bounds_upper) if reg is not None else list(domain_bounds['upper']),
                        }
                        for ml, reg, idx, n in leaf_mean_losses
                    ],
                    'spawned_children': _spawned_children_diag,
                }
                metrics['spawning_diagnostics'].append(diag)

            elif spawning_method == 'accept_split_by_norm':
                # Iterate all leaves, split each, accept if wavelet norm above threshold
                norm_diag_leaves = []
                for leaf_region, leaf_idx in leaf_nodes:
                    if hasattr(model, 'num_experts') and model.num_experts >= max_experts:
                        break

                    parent_region = leaf_region
                    parent_idx = leaf_idx
                    parent_depth = 0 if parent_region is None else parent_region.depth
                    leaf_str = f"Expert {leaf_idx+1}" if leaf_idx >= 0 else "Base Model"

                    children = region_detector.spawn_children_for_node(
                        parent_region=parent_region if parent_region is not None else
                                     RegionDescriptor(
                                         bounds_lower=list(domain_bounds['lower']),
                                         bounds_upper=list(domain_bounds['upper']),
                                         wavelet_norm_squared=0.0,
                                         spawn_epoch=0,
                                         depth=0,
                                         parent_idx=-1
                                     ),
                        X=X_eval,
                        y=y_eval,
                        loss_components=loss_components,
                        verbose=True
                    )

                    parent_bounds_lower = list(parent_region.bounds_lower) if parent_region is not None else list(domain_bounds['lower'])
                    parent_bounds_upper = list(parent_region.bounds_upper) if parent_region is not None else list(domain_bounds['upper'])

                    if not children:
                        norm_diag_leaves.append({
                            'leaf_idx': leaf_idx,
                            'parent_bounds_lower': parent_bounds_lower,
                            'parent_bounds_upper': parent_bounds_upper,
                            'children': [],
                            'accepted': False,
                            'reason': 'no_split',
                        })
                        continue

                    # Check acceptance based on configured variable
                    def _get_child_metric_value(c):
                        if variable_for_node_accept == 'norm':
                            return c.wavelet_norm_squared
                        elif variable_for_node_accept == 'new_norm':
                            return c.new_wavelet_norm_squared
                        elif variable_for_node_accept == 'smoothness':
                            return c.smoothness_alpha if c.smoothness_alpha is not None else 0.0
                        return c.wavelet_norm_squared
                    
                    threshold = thresholds.get(variable_for_node_accept, 0.0)
                    
                    if variable_for_node_accept == 'smoothness':
                        # For smoothness: lower alpha = rougher = keep (alpha < threshold)
                        above = any(_get_child_metric_value(c) < threshold and c.smoothness_r2 is not None and c.smoothness_r2 >= 0.5 for c, _ in children)
                    else:
                        # For norm/new_norm: higher = more variation = keep (value >= threshold)
                        above = any(_get_child_metric_value(c) >= threshold for c, _ in children)
                    
                    child_diags = [
                        {
                            'node_id': c.node_id,
                            'wavelet_norm_squared': c.wavelet_norm_squared,
                            'new_wavelet_norm_squared': c.new_wavelet_norm_squared,
                            'smoothness_alpha': c.smoothness_alpha,
                            'n_samples': c.n_samples,
                            'bounds_lower': c.bounds_lower,
                            'bounds_upper': c.bounds_upper,
                            'is_leaf': bool(c.is_leaf),
                        }
                        for c, _ in children
                    ]
                    norm_diag_leaves.append({
                        'leaf_idx': leaf_idx,
                        'parent_bounds_lower': parent_bounds_lower,
                        'parent_bounds_upper': parent_bounds_upper,
                        'children': child_diags,
                        'accepted': above,
                        'reason': 'above_threshold' if above else 'below_threshold',
                    })

                    if not above:
                        print(f"    [Spawning] {leaf_str}: children below {variable_for_node_accept} threshold "
                              f"({threshold}), skipping")
                        continue

                    child_depth = parent_depth + 1
                    for child_node, _ in children:
                        if hasattr(model, 'num_experts') and model.num_experts >= max_experts:
                            break
                        child_region = RegionDescriptor(
                            bounds_lower=child_node.bounds_lower,
                            bounds_upper=child_node.bounds_upper,
                            wavelet_norm_squared=child_node.wavelet_norm_squared,
                            new_wavelet_norm_squared=child_node.new_wavelet_norm_squared,
                            spawn_epoch=epoch,
                            depth=child_depth,
                            parent_idx=parent_idx,
                            smoothness_alpha=child_node.smoothness_alpha,
                        )
                        if is_copy_spawn:
                            expert_idx = model.spawn_expert(child_region, copy_from_idx=parent_idx)
                        else:
                            expert_idx = model.spawn_expert(child_region)
                        if expert_idx >= 0:
                            experts_spawned_this_step += 1
                            if 'expert_spawns' not in metrics:
                                metrics['expert_spawns'] = []
                            metrics['expert_spawns'].append({
                                'epoch': epoch,
                                'expert_idx': expert_idx,
                                'region': child_region.to_dict(),
                                'depth': child_depth,
                                'parent_idx': parent_idx,
                                **(({'num_experts': model.num_experts} if hasattr(model, 'num_experts') else {}))
                            })

                metrics['spawning_diagnostics'].append({
                    'epoch': epoch,
                    'method': 'accept_split_by_norm',
                    'wavelet_threshold': wavelet_threshold,
                    'evaluated_leaves': norm_diag_leaves,
                })

            elif spawning_method == 'full_tree_by_norm':
                # One-shot: fit full tree, prune bottom-up, spawn all accepted
                print(f"  [FullTree] Fitting full tree (max_depth={region_detector.max_depth}, "
                      f"min_samples_leaf={region_detector.min_samples_leaf})...")
                accepted_nodes, prune_depth_stats = \
                    region_detector.fit_full_tree_and_prune(
                        X=X_eval,
                        y=y_eval,
                        variable_for_node_accept=variable_for_node_accept,
                        thresholds=thresholds,
                        verbose=True,
                    )

                # Determine which nodes to spawn based on model type
                tree = region_detector.rf.estimators_[0].tree_
                children_left = tree.children_left

                if isinstance(model, AToELeaves):
                    # Only leaf nodes of the pruned tree
                    accepted_ids = {n.node_id for n, _ in accepted_nodes}
                    nodes_to_spawn = [
                        (node, parent_id) for node, parent_id in accepted_nodes
                        if children_left[node.node_id] == -1
                        or children_left[node.node_id] not in accepted_ids
                    ]
                else:
                    # AToE and ANT: all accepted nodes
                    nodes_to_spawn = accepted_nodes

                # Pre-compute tree depth for each node and parent map
                from collections import deque as _deque
                _parent_map = {}
                _node_tree_depth = {0: 0}
                _bfs = _deque([0])
                while _bfs:
                    nid = _bfs.popleft()
                    l, r = children_left[nid], tree.children_right[nid]
                    for child in (l, r):
                        if child != -1:
                            _parent_map[child] = nid
                            _node_tree_depth[child] = _node_tree_depth[nid] + 1
                            _bfs.append(child)

                node_to_expert = {}
                is_atoe_plain = isinstance(model, AToE) and not isinstance(model, AToELeaves)
                atoe_zero_init = not reinit_base_after_spawn

                for node, parent_tree_id in nodes_to_spawn:
                    parent_expert_idx = node_to_expert.get(parent_tree_id, -1)
                    depth = _node_tree_depth.get(node.node_id, 1)

                    child_region = RegionDescriptor(
                        bounds_lower=node.bounds_lower,
                        bounds_upper=node.bounds_upper,
                        wavelet_norm_squared=node.wavelet_norm_squared,
                        new_wavelet_norm_squared=node.new_wavelet_norm_squared,
                        spawn_epoch=epoch,
                        depth=depth,
                        parent_idx=parent_expert_idx,
                        smoothness_alpha=node.smoothness_alpha,
                    )

                    if is_copy_spawn:
                        expert_idx = model.spawn_expert(child_region, copy_from_idx=parent_expert_idx)
                    elif is_atoe_plain:
                        expert_idx = model.spawn_expert(
                            child_region, zero_init=atoe_zero_init)
                    else:
                        expert_idx = model.spawn_expert(child_region)
                    if expert_idx >= 0:
                        node_to_expert[node.node_id] = expert_idx
                        experts_spawned_this_step += 1
                        if 'expert_spawns' not in metrics:
                            metrics['expert_spawns'] = []
                        metrics['expert_spawns'].append({
                            'epoch': epoch,
                            'expert_idx': expert_idx,
                            'region': child_region.to_dict(),
                            'depth': depth,
                            'parent_idx': parent_expert_idx,
                            **(({'num_experts': model.num_experts} if hasattr(model, 'num_experts') else {}))
                        })

                # Save diagnostics for all tree nodes
                accepted_ids_diag = {n.node_id for n, _ in accepted_nodes}
                spawned_ids_diag = {n.node_id for n, _ in nodes_to_spawn}
                all_tree_nodes = region_detector.compute_wavelet_norms()
                tree_diag_nodes = []
                for nd in all_tree_nodes:
                    if nd.node_id == 0:
                        continue
                    tree_diag_nodes.append({
                        'node_id': nd.node_id,
                        'parent_node_id': _parent_map.get(nd.node_id, -1),
                        'wavelet_norm_squared': nd.wavelet_norm_squared,
                        'n_samples': nd.n_samples,
                        'is_leaf': bool(nd.is_leaf),
                        'bounds_lower': nd.bounds_lower,
                        'bounds_upper': nd.bounds_upper,
                        'accepted': bool(
                            nd.node_id in accepted_ids_diag),
                        'spawned_as_expert': bool(
                            nd.node_id in spawned_ids_diag),
                        'tree_depth': _node_tree_depth.get(
                            nd.node_id, -1),
                    })
                # Convert depth_stats keys to strings for JSON
                depth_stats_json = {
                    str(k): v for k, v in
                    prune_depth_stats.items()
                }
                metrics['spawning_diagnostics'].append({
                    'epoch': epoch,
                    'method': 'full_tree_by_norm',
                    'wavelet_threshold': wavelet_threshold,
                    'total_tree_nodes': int(tree.node_count),
                    'accepted_count': len(accepted_ids_diag),
                    'spawned_count': len(spawned_ids_diag),
                    'depth_stats': depth_stats_json,
                    'nodes': tree_diag_nodes,
                })

                spawning_complete = True
                print(f"  [FullTree] Spawning complete. No further spawning steps.")

            if experts_spawned_this_step > 0:
                print(f"\n  [Spawning] Spawned {experts_spawned_this_step} experts in this step")

                # ── 3-phase: reinitialize base + transition to Phase 3 ──
                if use_three_phase and spawning_complete and current_phase == 1:
                    if reinit_base_after_spawn:
                        model.reinitialize_base()
                    current_phase = 3
                    active_cfg = cfg
                    total_epochs = epoch + phase3_epochs  # extend loop
                    # Recalculate optimizer strategy from top-level config
                    _p3_opt1 = active_cfg.get('optimizer_1', active_cfg.get('optimizer', 'adam')).lower()
                    _p3_opt2_cfg = active_cfg.get('optimizer_2', None)
                    optimizer_2_name = _p3_opt2_cfg.lower() if _p3_opt2_cfg else None
                    total_steps_p3 = phase3_epochs * batches_per_epoch
                    if optimizer_2_name is not None:
                        _p3_switch = active_cfg.get('optimizer_switch_epoch', None)
                        switch_epoch = (epoch + _p3_switch) if _p3_switch else (epoch + phase3_epochs + 1)
                        patience_start_epoch = switch_epoch
                    else:
                        switch_epoch = epoch + phase3_epochs + 1
                        patience_start_epoch = epoch + 1
                    print(f"\n  [3-Phase] Transitioning to Phase 3: {phase3_epochs} epochs of full model training")
                    print(f"  [3-Phase] Total epochs now: {total_epochs} (Phase 1: {epoch}, Phase 3: {phase3_epochs})")
                    print(f"  [3-Phase] Optimizer: {_p3_opt1}, lr: {active_cfg.get('lr')}, schedule: {active_cfg.get('lr_schedule', 'exponential')}")

                optimizer, current_optimizer_name = _create_primary_optimizer(model, active_cfg)
                if current_phase == 3:
                    lr_scheduler = _create_lr_scheduler(optimizer, active_cfg, total_steps_p3)
                else:
                    lr_scheduler = _create_lr_scheduler(optimizer, active_cfg, total_steps_estimate)
                step_count = 0

                model.freeze_models()

                problem_type = '2d' if len(domain_bounds['lower']) == 2 else '3d'
                num_experts_str = f" ({model.num_experts} experts)" if hasattr(model, 'num_experts') else ""
                leaf_info = model.get_leaf_info()
                leaf_expert_indices = [idx for _, idx in leaf_info if idx >= 0]
                regions_to_plot = (
                    [model.regions[i] for i in leaf_expert_indices]
                    if isinstance(model, (AToELeaves, ANT)) else model.regions
                )
                plot_expert_regions(
                    regions=regions_to_plot,
                    domain_bounds=domain_bounds,
                    output_path=adaptive_plots_dir / f"expert_regions_epoch_{epoch}.png",
                    problem_type=problem_type,
                    title=f"Expert Regions at Epoch {epoch}{num_experts_str}",
                    ground_truth=gt_grid,
                    grid_x=gt_x,
                    grid_t=gt_t
                )

                if adaptive_cfg.get('blending_mode', 'hard') == 'soft' and problem_type == '2d':
                    leaf_indices_set = (
                        set(leaf_expert_indices) if isinstance(model, (AToELeaves, ANT)) else None
                    )
                    plot_expert_soft_weights(
                        model=model,
                        domain_bounds=domain_bounds,
                        output_path=adaptive_plots_dir / f"soft_weights_epoch_{epoch}.png",
                        title_prefix=f"Epoch {epoch}: ",
                        leaf_indices=leaf_indices_set
                    )
            else:
                print(f"\n  [Spawning] No experts spawned this step")

            model.train()

    # Save final model
    final_checkpoint_path = checkpoint_dir / "final_model.pt"
    _save_checkpoint(final_checkpoint_path, model, optimizer, current_optimizer_name, total_epochs,
                    train_loss, eval_loss, cfg, metrics)

    print(f"\nTraining completed in {time.time() - start_time:.1f}s")
    print(f"  Best eval loss: {best_eval_loss:.6f}")
    print(f"  Best checkpoint: {best_checkpoint_path}")
    print(f"  Final checkpoint: {final_checkpoint_path}")
    
    # Save timing data and print summary
    timer.save(run_dir / "timing.json")
    timer.print_summary()

    # Save expert diagnostics to CSV
    if is_adaptive and hasattr(model, 'num_experts') and model.num_experts > 0 and hasattr(model, '_diag_data') and model._diag_data:
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
    switch_epoch_to_plot = switch_epoch if (optimizer_2_name is not None and switch_epoch <= epochs) else None
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
    if is_adaptive and hasattr(model, 'num_experts') and model.num_experts > 0:
        print("\n" + "=" * 60)
        print("Adaptive PINN Final Summary")
        print("=" * 60)
        print(f"  Total experts spawned: {model.num_experts}")
        
        problem_type = '2d' if len(domain_bounds['lower']) == 2 else '3d'
        is_leaves_model = isinstance(model, (AToELeaves, ANT))
        leaf_info = model.get_leaf_info()
        leaf_expert_indices = [idx for _, idx in leaf_info if idx >= 0]
        regions_to_plot = (
            [model.regions[i] for i in leaf_expert_indices]
            if is_leaves_model else model.regions
        )
        label = 'leaves' if is_leaves_model else 'experts'
        plot_expert_regions(
            regions=regions_to_plot,
            domain_bounds=domain_bounds,
            output_path=adaptive_plots_dir / "expert_regions_final.png",
            problem_type=problem_type,
            title=f"Final Expert Regions ({len(regions_to_plot)} {label})",
            ground_truth=gt_grid,
            grid_x=gt_x,
            grid_t=gt_t
        )

        if adaptive_cfg.get('blending_mode', 'hard') == 'soft' and problem_type == '2d':
            leaf_indices_set = set(leaf_expert_indices) if is_leaves_model else None
            plot_expert_soft_weights(
                model=model,
                domain_bounds=domain_bounds,
                output_path=adaptive_plots_dir / "soft_weights_final.png",
                title_prefix="Final: ",
                leaf_indices=leaf_indices_set
            )
        
        save_regions_metadata(
            regions=model.regions,
            output_path=adaptive_plots_dir / "expert_regions.json",
            rejected_regions=rejected_regions,
            leaf_loss_history=leaf_loss_history,
            spawning_method=spawning_method,
            spawning_diagnostics=metrics.get('spawning_diagnostics', []),
        )
        
        base_params = sum(p.numel() for p in model.base_model.parameters())
        expert_full_params = []
        for i, expert in enumerate(model.experts):
            expert_full_params.append(
                sum(p.numel() for p in expert.parameters()))
        expert_archs = [
            e.layers if hasattr(e, 'layers') else []
            for e in model.experts]

        leaf_info = model.get_leaf_info()
        leaf_expert_indices = set(
            idx for _, idx in leaf_info if idx >= 0)

        is_ant = isinstance(model, ANT)
        is_leaves_only = isinstance(model, AToELeaves) and not is_ant

        # For ANT: non-leaf experts' output layers (last_hidden → output_dim)
        # are unused in inference (only activations propagate to children).
        # Count full params for leaves, subtract output layer for non-leaves.
        if is_ant:
            expert_params = []
            for i, full_p in enumerate(expert_full_params):
                arch = expert_archs[i]
                if i not in leaf_expert_indices and len(arch) >= 2:
                    out_layer = arch[-2] * arch[-1] + arch[-1]
                    expert_params.append(full_p - out_layer)
                else:
                    expert_params.append(full_p)
        else:
            expert_params = expert_full_params

        leaf_params = sum(
            expert_params[i] for i in leaf_expert_indices
            if i < len(expert_params))

        metrics['adaptive_pinn'] = {
            'num_experts': model.num_experts,
            'max_experts': max_experts,
            'spawning_method': spawning_method,
            'wavelet_threshold': wavelet_threshold,
            'regions': [r.to_dict() for r in model.regions],
            'base_params': base_params,
            'expert_params': expert_params,
            'expert_architectures': expert_archs,
            'total_params': base_params + sum(expert_params),
            'leaf_expert_indices': sorted(leaf_expert_indices),
            'leaf_params': leaf_params,
            'forward_params': base_params + (
                leaf_params if is_leaves_only
                else sum(expert_params)),
        }

    total_model_params = sum(p.numel() for p in model.parameters())
    metrics['total_params'] = total_model_params

    # Save metrics to JSON
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, cls=_NumpySafeEncoder)
    print(f"  Metrics saved to {metrics_path}")

    # Save summary
    summary_path = run_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Training Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Problem: {cfg['problem']}\n")
        f.write(f"Architecture: {cfg['base_architecture']}\n")
        f.write(f"Activation: {cfg['activation']}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch size: {cfg['batch_size']}\n")
        f.write(f"Learning rate: {cfg['lr']}\n")
        f.write(f"Device: {device}\n\n")
        f.write(f"Final train loss: {train_loss:.6f}\n")
        f.write(f"Final eval loss: {eval_loss:.6f}\n")
        f.write(f"Final eval rel-L2: {eval_rel_l2:.6f}\n")
        f.write(f"Final eval inf-norm: {eval_inf_norm:.6f}\n")
        f.write(f"Best eval loss: {best_eval_loss:.6f}\n\n")
        f.write(f"Best checkpoint: {best_checkpoint_path}\n")
        f.write(f"Final checkpoint: {final_checkpoint_path}\n")
    print(f"  Summary saved to {summary_path}")

    # Save config used
    from utils.io import get_git_info
    cfg['git'] = get_git_info()
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

