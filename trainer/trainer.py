"""Training loop for PINN models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple
import json
import math
import time
import numpy as np

from trainer.plotting import plot_training_curves, plot_final_comparison
from trainer.utils import compute_infinity_norm_error
from trainer.timing import EpochTimer
from models.atoe import AToE
from models.atoe_leaves import AToELeaves
from models.ant import ANT
from utils.dataset_gen import regenerate_training_data, _save_adaptive_sampling_heatmap
from utils.dataset_plotting import save_spawn_prediction_plot
from utils.config_validation import validate_problem_config
from losses.causal_weighting import advance_causal_schedule, create_causal_state
from losses.lra import LRAWeights
import losses.ks_loss as _ks_loss_module


def _override_ic_for_time_marching(
    train_data: Dict[str, torch.Tensor],
    cfg: Dict,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Override IC h_gt values with previous model predictions for time marching.
    Also updates IC t values to window.t_start so they aren't filtered out.
    
    Called after each regenerate_training_data to ensure IC values come from
    the previous window's model, not the analytical IC.
    
    Args:
        train_data: Freshly resampled training data
        cfg: Config with _time_marching_window info
        device: Device for inference
    
    Returns:
        train_data with IC h_gt overridden (if time marching window > 0)
    """
    tm_window = cfg.get('_time_marching_window', {})
    if not tm_window.get('enabled', False):
        return train_data
    
    window_idx = tm_window.get('idx', 0)
    prev_model = tm_window.get('prev_model', None)
    t_start = tm_window.get('t_start', 0)
    
    # Window 0 uses analytical IC, no override needed
    if window_idx == 0 or prev_model is None:
        return train_data
    
    # Get IC mask and points
    ic_mask = train_data['mask']['IC']
    if ic_mask.sum() == 0:
        print(f"  [IC Override] Window {window_idx}: No IC points found in dataset, skipping")
        return train_data
    
    x_ic = train_data['x'][ic_mask]
    h_gt_original = train_data['h_gt'][ic_mask].clone()
    
    # Create t values at window.t_start for querying previous model
    t_query = torch.full_like(train_data['t'][ic_mask], t_start)
    
    # Diagnostic: print input stats
    print(f"  [IC Override] Window {window_idx}: Overriding {ic_mask.sum().item()} IC points at t={t_start:.4f}")
    print(f"    x_ic: shape={x_ic.shape}, min={x_ic.min().item():.4f}, max={x_ic.max().item():.4f}, mean={x_ic.mean().item():.4f}")
    print(f"    h_gt (original): min={h_gt_original.min().item():.4f}, max={h_gt_original.max().item():.4f}, mean={h_gt_original.mean().item():.4f}")
    
    # Query previous model for IC values
    prev_model.eval()
    with torch.no_grad():
        inputs = torch.cat([x_ic, t_query], dim=1).to(device)
        h_pred = prev_model(inputs)
    
    # Diagnostic: print prediction stats
    has_nan = torch.isnan(h_pred).any().item()
    has_inf = torch.isinf(h_pred).any().item()
    print(f"    h_pred: min={h_pred.min().item():.4f}, max={h_pred.max().item():.4f}, mean={h_pred.mean().item():.4f}")
    print(f"    h_pred contains NaN: {has_nan}, Inf: {has_inf}")
    
    if has_nan or has_inf:
        print(f"    [WARNING] Previous model produced invalid values! This will cause NaN divergence.")
        num_nan = torch.isnan(h_pred).sum().item()
        num_inf = torch.isinf(h_pred).sum().item()
        print(f"    Number of NaN: {num_nan}, Number of Inf: {num_inf}")
    
    # Override h_gt AND t for IC points
    train_data['h_gt'][ic_mask] = h_pred.to(train_data['h_gt'].device)
    train_data['t'][ic_mask] = t_start  # Set IC t to window start
    
    return train_data


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
    betas = tuple(cfg['adam_betas'])
    eps = cfg['adam_eps']
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
        lr=cfg['lbfgs_lr'],
        max_iter=cfg['lbfgs_max_iter'],
        max_eval=None,  # Default: max_iter * 1.25
        history_size=cfg['lbfgs_history_size'],
        line_search_fn=cfg['lbfgs_line_search'],
        tolerance_grad=cfg['lbfgs_tolerance_grad'],
        tolerance_change=cfg['lbfgs_tolerance_change']
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
        betas=tuple(cfg['soap_betas']),
        eps=cfg['adam_eps'],
        precondition_frequency=cfg['soap_precondition_frequency'],
        weight_decay=cfg['soap_weight_decay'],
    )


def _create_ssbroyden_optimizer(model: nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    """Create SSBroyden (Self-Scaled Broyden) quasi-Newton optimizer via scimba.

    Falls back to LBFGS with a warning if scimba is not installed.
    Only includes trainable parameters.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    try:
        from scimba_torch.optimizers.ssbroyden import SSBroyden
        return SSBroyden(
            trainable_params,
            lr=cfg.get('ssbroyden_lr', 1.0),
            tolerance_grad=cfg.get('ssbroyden_tolerance_grad', 1e-10),
            method='ssbroyden',
        )
    except ImportError:
        print("  [Warning] scimba not installed — SSBroyden unavailable, falling back to LBFGS.")
        print("            Install with: pip install scimba")
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
    opt_name = cfg['optimizer_1'].lower()
    return _create_optimizer_by_name(opt_name, model, cfg)


def _create_grouped_optimizer(param_groups: list, cfg: Dict) -> torch.optim.Optimizer:
    """Create an optimizer with explicit param groups, each with its own LR.

    Used at spawn/unfreeze to give ancestors, untouched leaves, and new experts
    separate LRs and fresh/preserved state independently.
    Supports Adam and SOAP; raises ValueError for others.
    """
    opt_name = cfg['optimizer_1'].lower()
    if opt_name == 'adam':
        betas = tuple(cfg['adam_betas'])
        eps = cfg['adam_eps']
        return torch.optim.Adam(param_groups, betas=betas, eps=eps)
    elif opt_name == 'soap':
        from optimizers.soap import SOAP
        return SOAP(
            param_groups,
            betas=tuple(cfg['soap_betas']),
            eps=cfg['adam_eps'],
            precondition_frequency=cfg['soap_precondition_frequency'],
            weight_decay=cfg['soap_weight_decay'],
        )
    else:
        raise ValueError(f"_create_grouped_optimizer not supported for optimizer: {opt_name}")


def _create_lr_scheduler(optimizer, cfg, total_steps):
    """Create an LR scheduler composed of optional warmup + decay.

    Uses standard PyTorch schedulers:
    - LinearLR for warmup (ramps from ~0 to base lr)
    - StepLR for exponential decay (multiplies lr by decay_rate every decay_steps)
    - CosineAnnealingLR for cosine schedule

    Returns None if no scheduling is configured.
    """
    from torch.optim.lr_scheduler import LinearLR, StepLR, CosineAnnealingLR, SequentialLR

    schedule = cfg['lr_schedule']
    warmup_steps = cfg['lr_warmup_steps']

    if schedule == 'none' and warmup_steps <= 0:
        return None

    schedulers = []
    milestones = []

    if warmup_steps > 0:
        start_factor = cfg['lr_warmup_start_factor']
        schedulers.append(LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_steps))
        milestones.append(warmup_steps)

    if schedule == 'exponential':
        decay_rate = cfg['lr_decay_rate']
        decay_steps = cfg['lr_decay_steps']
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


def _get_optimizer_snapshot(optimizer, lr_scheduler, step_count):
    """Return a compact dict of optimizer/scheduler state for metrics logging at key events."""
    lrs = [pg['lr'] for pg in optimizer.param_groups]
    sched_type = type(lr_scheduler).__name__ if lr_scheduler is not None else None
    sched_last_epoch = getattr(lr_scheduler, 'last_epoch', None) if lr_scheduler is not None else None
    sched_base_lrs = None
    if lr_scheduler is not None:
        sched_base_lrs = getattr(lr_scheduler, 'base_lrs', None)
        if sched_base_lrs is None:
            first_sub = (getattr(lr_scheduler, '_schedulers', None) or [None])[0]
            sched_base_lrs = getattr(first_sub, 'base_lrs', None) if first_sub else None
    return {
        'step_count': step_count,
        'num_param_groups': len(optimizer.param_groups),
        'lr_per_group': lrs,
        'scheduler_type': sched_type,
        'scheduler_last_epoch': sched_last_epoch,
        'scheduler_base_lrs': sched_base_lrs,
    }


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

    # Validate per-problem config (all features must be explicitly specified)
    validate_problem_config(cfg)
    problem = cfg['problem']
    problem_cfg = cfg[problem]
    
    # Copy per-problem features to top-level for backward compatibility with
    # functions that read cfg['init'], cfg['fourier_features'], etc.
    for key in ['rwf', 'fourier_features', 'init', 'lra', 'adaptive_sampling',
                'grad_clip_norm', 'expert_grad_clip_norm']:
        if key in problem_cfg:
            cfg[key] = problem_cfg[key]

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
    if cfg['adaptive_pinn']['enable_gradient_diagnostics']:
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

    # Cast data to configured precision (float32 or float64)
    precision = cfg.get('precision', 'float32')
    target_dtype = torch.float64 if precision == 'float64' else torch.float32
    train_data = _cast_data_to_dtype(train_data, target_dtype)
    eval_data = _cast_data_to_dtype(eval_data, target_dtype)

    # Filter train and eval data by window temporal bounds if time marching is enabled
    time_marching_window = cfg.get('_time_marching_window', {})
    if time_marching_window.get('enabled', False):
        t_start = time_marching_window['t_start']
        t_end = time_marching_window['t_end']
        window_idx = time_marching_window['idx']
        
        # IMPORTANT: For windows 1+, override IC BEFORE filtering
        # This updates IC t values from t=0 to t=window.t_start, so they survive filtering
        train_data = _override_ic_for_time_marching(train_data, cfg, device)
        eval_data = _override_ic_for_time_marching(eval_data, cfg, device)
        
        # --- Filter TRAINING data ---
        t_train = train_data['t'].squeeze()
        train_mask = (t_train >= t_start) & (t_train < t_end)
        # Handle edge case: last window should include t_end
        if t_train.max() <= t_end:
            train_mask = train_mask | (t_train == t_end)
        
        n_train_original = train_data['x'].shape[0]
        n_train_filtered = train_mask.sum().item()
        
        print(f"  [Time Marching] Filtering train data for window {window_idx}: "
              f"t in [{t_start:.4f}, {t_end:.4f}]")
        print(f"  [Time Marching] Train data: {n_train_original} → {n_train_filtered} points")
        
        # Apply mask to train_data
        filtered_train_data = {}
        for key, value in train_data.items():
            if torch.is_tensor(value):
                filtered_train_data[key] = value[train_mask]
            elif key == 'mask':
                filtered_train_data[key] = {
                    k: v[train_mask] if torch.is_tensor(v) else v
                    for k, v in value.items()
                }
            else:
                filtered_train_data[key] = value
        train_data = filtered_train_data
        
        # --- Filter EVAL data ---
        t_eval = eval_data['t'].squeeze()
        eval_mask = (t_eval >= t_start) & (t_eval < t_end)
        # Handle edge case: last window should include t_end
        if t_eval.max() <= t_end:
            eval_mask = eval_mask | (t_eval == t_end)
        
        n_eval_original = eval_data['x'].shape[0]
        n_eval_filtered = eval_mask.sum().item()
        
        print(f"  [Time Marching] Filtering eval data for window {window_idx}: "
              f"t in [{t_start:.4f}, {t_end:.4f}]")
        print(f"  [Time Marching] Eval data: {n_eval_original} → {n_eval_filtered} points")
        
        # Apply mask to eval_data
        filtered_eval_data = {}
        for key, value in eval_data.items():
            if torch.is_tensor(value):
                filtered_eval_data[key] = value[eval_mask]
            elif key == 'mask':
                filtered_eval_data[key] = {
                    k: v[eval_mask] if torch.is_tensor(v) else v
                    for k, v in value.items()
                }
            else:
                filtered_eval_data[key] = value
        eval_data = filtered_eval_data

    print(f"  Train size: {train_data['x'].shape[0]}")
    print(f"  Eval size: {eval_data['x'].shape[0]}")
    print(f"  Train data device: {train_data['x'].device}")
    print(f"  Eval data device: {eval_data['x'].device}")

    # Create DataLoaders
    train_loader = _create_dataloader(train_data, cfg['batch_size'],
                                      shuffle=True)
    eval_loader = _create_dataloader(eval_data, cfg['batch_size'],
                                     shuffle=False)

    # ── 3-phase logic for M_term_tree_by_norm / use_perfect_trees ──
    adaptive_cfg_init = cfg['adaptive_pinn']
    spawning_method_init = adaptive_cfg_init['spawning_method']
    initial_train_cfg = adaptive_cfg_init.get('initial_train', None)
    use_three_phase = (spawning_method_init == 'M_term_tree_by_norm' and initial_train_cfg is not None)
    use_perfect_trees = (spawning_method_init == 'use_perfect_trees')
    reinit_base_after_spawn = adaptive_cfg_init['reinitialize_base_after_spawn']

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
        phase1_epochs = initial_train_cfg['epochs']
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
    optimizer_1_name = active_cfg['optimizer_1'].lower()
    optimizer_2_name_cfg = active_cfg.get('optimizer_2', None)
    optimizer_2_name = optimizer_2_name_cfg.lower() if optimizer_2_name_cfg else None

    # optimizer_switch_epoch: when to switch. None / ignored when optimizer_2 is null.
    if optimizer_2_name is not None:
        switch_epoch = active_cfg['optimizer_switch_epoch']
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
            sched_name = active_cfg['lr_schedule']
            warmup = active_cfg['lr_warmup_steps']
            print(f"  LR schedule: {sched_name} (warmup={warmup} steps, ~{total_steps_estimate} total steps)")

    step_count = 0  # global optimizer step counter for LR scheduler

    # Training setup
    print_every = cfg['print_every']
    eval_every = cfg['eval_every']
    inner_metrics_every = cfg['inner_metrics_eval_every']
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
        'resample_events': [],     # Track resampling/skipping events
        'freeze_events': [],       # Freeze/unfreeze events with epoch and reason
        'plateau_events': [],      # Plateau check outcomes (deferred / triggered)
        'optimizer_events': [],    # Optimizer switch events
        'optimizer_snapshots': [],  # Optimizer/scheduler state at spawn, freeze, unfreeze, resample
        'loss_components_history': [],  # Per-component losses at eval epochs
        'exception_events': [],    # Caught Python exceptions with traceback
    }

    best_eval_loss = float('inf')
    best_train_loss = float('inf')
    best_checkpoint_path = None
    patience_epochs = cfg['patience_epochs']
    min_epochs = cfg['min_epochs']
    epochs_without_improvement = 0

    # LRA: adaptive loss component weighting (read from per-problem config)
    lra_cfg = problem_cfg['lra']
    lra_enabled = lra_cfg['enabled']
    if lra_enabled:
        initial_loss_weights = problem_cfg['loss_weights']
        lra_weights = LRAWeights(
            alpha=lra_cfg['alpha'],
            update_every=lra_cfg['update_every'],
            initial_weights=initial_loss_weights,
            scheme=lra_cfg['scheme'],
            scheme_cfg=lra_cfg,
        )
    else:
        lra_weights = None

    # Wrap loss_fn to apply LRA weights when enabled
    if lra_weights is not None:
        _orig_loss_fn = loss_fn

        def _lra_loss_fn(model, batch, for_tree_spawning=False, return_components=False, update_causal_state=True):
            if for_tree_spawning or return_components:
                return _orig_loss_fn(model, batch,
                                     for_tree_spawning=for_tree_spawning,
                                     return_components=return_components,
                                     update_causal_state=False)
            comps = _orig_loss_fn(model, batch, return_components=True, update_causal_state=update_causal_state)
            w = lra_weights.weights
            return sum(w.get(k, 1.0) * v for k, v in comps.items())

        _lra_loss_fn.causal_state = getattr(loss_fn, 'causal_state', None)
        _lra_loss_fn._leaf_state = getattr(loss_fn, '_leaf_state', None)
        loss_fn = _lra_loss_fn

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create ncc_plots directory for periodic NCC analysis
    ncc_plots_parent = run_dir / "ncc_plots"
    ncc_plots_parent.mkdir(exist_ok=True)

    # Adaptive PINN setup
    adaptive_cfg = cfg['adaptive_pinn']
    is_adaptive = adaptive_cfg['enabled']
    region_detector = None
    spawn_every = adaptive_cfg['spawn_every_epochs']
    max_experts = adaptive_cfg['max_experts']
    spawning_method = adaptive_cfg['spawning_method']
    wavelet_threshold = problem_cfg['wavelet_threshold']
    adaptive_inner_metrics = adaptive_cfg['inner_metrics_calculation']
    spawning_complete = False
    _retries_before_stop = adaptive_cfg['spawn_retries_before_stop']
    _stop_on_no_spawn = _retries_before_stop is not False  # False = feature disabled
    _no_spawn_retries_max = int(_retries_before_stop) if _stop_on_no_spawn else 0
    _no_spawn_retries_remaining = _no_spawn_retries_max
    _spawn_retry_after = adaptive_cfg.get('spawn_retry_after', None)
    _spawn_last_fail_epoch = -1  # epoch of last failed spawn; -1 = no pending retry
    _per_leaf_causal = problem_cfg.get('causal_training', {}).get('per_leaf_causal', False)  # Optional feature
    _per_leaf_sampling = problem_cfg['adaptive_sampling'].get('per_leaf_sampling', False)  # Optional feature
    
    # Read configurable norm variables
    variable_for_node_accept = adaptive_cfg['variable_for_node_accept']
    variable_for_expert_size = adaptive_cfg['variable_for_expert_size']
    
    # Build thresholds dict
    thresholds = {
        'norm': problem_cfg['wavelet_threshold'],
        'new_norm': problem_cfg['new_norm_threshold'],
        'smoothness': problem_cfg['tree_smoothness_threshold'],
    }

    if is_adaptive:
        tree_max_depth = adaptive_cfg['tree_max_depth']
        tree_min_samples_leaf = adaptive_cfg['tree_min_samples_leaf']

        print(f"\nAdaptive PINN enabled (spawning_method={spawning_method}):")
        print(f"  Max experts: {max_experts}")
        print(f"  Spawn every: {spawn_every} epochs")
        print(f"  Tree max depth: {tree_max_depth}")
        print(f"  Tree min samples leaf: {tree_min_samples_leaf}")
        if spawning_method in ('accept_split_by_norm', 'M_term_tree_by_norm', 'use_perfect_trees'):
            if spawning_method == 'M_term_tree_by_norm':
                print(f"  M experts num: {adaptive_cfg['M_experts_num']}")
            print(f"  Wavelet threshold: {wavelet_threshold}")
        print(f"  Blending mode: {adaptive_cfg['blending_mode']}")
        print(f"  Freeze mode: {adaptive_cfg['freeze_mode']}")
        print(f"  Model type: {type(model).__name__}")
        enable_timing_cfg = adaptive_cfg['enable_timing']
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

        region_detector = RegionDetector(
            n_estimators=1,
            max_depth=tree_max_depth if spawning_method in ('M_term_tree_by_norm', 'use_perfect_trees') else 1,
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
              f"lr={active_cfg['lr']}, "
              f"schedule={active_cfg['lr_schedule']}")

    # Training loop
    total_epochs = epochs  # may extend when transitioning to Phase 3
    print(f"\nTraining for {total_epochs} epochs...")
    start_time = time.time()
    
    # Epoch timer for fine-grained performance profiling
    enable_timing = adaptive_cfg['enable_timing'] if is_adaptive else False
    timer = EpochTimer(enabled=enable_timing, print_every=eval_every)
    if enable_timing:
        model._timer = timer

    train_loss = 0.0
    eval_loss = 0.0
    eval_rel_l2 = 0.0
    eval_inf_norm = 0.0

    resample_every = cfg['sampling']['resample_every_epochs']
    base_seed = cfg['seed']
    # Gradient clipping (read from per-problem config)
    grad_clip_norm = problem_cfg['grad_clip_norm']
    # Tighter clip for all expert params (separate from base); only active when experts exist.
    # When no experts exist (base-only phase), grad_clip_norm applies to all params as usual.
    expert_grad_clip_norm = problem_cfg['expert_grad_clip_norm']

    # Freeze-after-spawn state (Fix 2)
    # freeze_mode: none takes priority — if explicitly set to none, disable post-spawn freeze entirely.
    freeze_epochs_after_spawn = adaptive_cfg['freeze_epochs_after_spawn'] if is_adaptive else 0
    if adaptive_cfg['freeze_mode'] == 'none':
        freeze_epochs_after_spawn = 0
    _unfreeze_at_epoch = None           # set after each spawn
    _pre_freeze_lr = None               # LR of all groups before freeze (restored to ancestors at unfreeze)
    _ancestor_indices: set = set()      # expert indices frozen as ancestors (includes -1 for base)
    _ancestor_params: list = []         # ancestor parameter tensors (saved at freeze, used at unfreeze)
    _ancestor_state_snapshot: dict = {} # {id(p): state} for ancestor params
    _untouched_leaf_params: list = []   # leaf params kept in the freeze-period optimizer

    # Plateau-gated spawning state (Fix 3)
    # spawn_every acts as minimum interval; once elapsed, plateau is checked every epoch until met.
    _spawn_require_plateau = adaptive_cfg['spawn_require_plateau'] if is_adaptive else False
    _spawn_plateau_epochs = adaptive_cfg['spawn_plateau_epochs']
    _spawn_plateau_delta = adaptive_cfg['spawn_plateau_delta']
    _last_spawn_epoch = 0           # epoch of last successful spawn
    
    # Consolidated feature summary
    print("\n" + "=" * 60)
    print("FEATURE SUMMARY")
    print("=" * 60)
    
    # Fourier Features (read from per-problem config)
    ff_cfg = problem_cfg['fourier_features']
    ff_enabled = ff_cfg['enabled']
    if ff_enabled:
        ff_dim = ff_cfg['dim']
        ff_scale = ff_cfg['scale']
        _base_for_ff = model.base_model if hasattr(model, 'base_model') else model
        _ff_out = _base_for_ff.ff_emb.output_dim if (hasattr(_base_for_ff, 'ff_emb') and _base_for_ff.ff_emb is not None) else 2 * ff_dim
        _periodic = ff_cfg['periodic']
        print(f"  Fourier Features: enabled (dim={ff_dim}, scale={ff_scale}, output_dim={_ff_out}, periodic={_periodic})")
    else:
        print(f"  Fourier Features: disabled")
    
    # RWF (read from per-problem config)
    rwf_enabled = problem_cfg['rwf']
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
        init_w_str = ', '.join(f'{k}={v:.1f}' for k, v in init_w.items())
        print(f"  LRA: enabled (scheme={lra_weights.scheme}, alpha={lra_weights.alpha}, update_every={lra_weights.update_every}, "
              f"init_weights={{{init_w_str}}})")
    else:
        print(f"  LRA: disabled")
    
    # Resampling & Adaptive Sampling (read from per-problem config)
    adaptive_sampling_cfg = problem_cfg['adaptive_sampling']
    adaptive_sampling_enabled = adaptive_sampling_cfg['enabled']
    if resample_every > 0:
        if adaptive_sampling_enabled:
            as_ratio = adaptive_sampling_cfg['adaptive_ratio']
            print(f"  Resampling: every {resample_every} epochs (adaptive: enabled, ratio={as_ratio})")
        else:
            print(f"  Resampling: every {resample_every} epochs (adaptive: disabled)")
    else:
        print(f"  Resampling: disabled")
    
    # Optimizer schedule
    opt1_name = cfg['optimizer_1']
    opt2_name = cfg.get('optimizer_2', 'null')
    if opt2_name and opt2_name != 'null':
        switch_epoch = cfg['optimizer_switch_epoch']
        print(f"  Optimizer: {opt1_name} → {opt2_name} at epoch {switch_epoch}")
    else:
        print(f"  Optimizer: {opt1_name}")
    
    # Early stopping
    if patience_epochs > 0:
        print(f"  Early stopping: enabled (patience={patience_epochs}, min_epochs={min_epochs})")
    else:
        print(f"  Early stopping: disabled")
    
    print("=" * 60 + "\n")

    # Smart initialization (Glorot hidden + zero/LS output) — base model only
    from trainer.init import apply_hidden_init, apply_output_init, apply_expert_init, apply_parent_copy_init, apply_spectral_norm
    _init_target = model.base_model if is_adaptive else model
    _init_cfg = cfg.get('init', {})
    if _init_cfg.get('hidden', 'default') != 'default' or _init_cfg.get('output', 'default') != 'default' or _init_cfg.get('spectral_norm', False):
        print("[Init] Applying smart initialization to base model...")
        # parent_weights is expert-only; use glorot for base model unless architecture is
        # resnet (glorot zeros fc2 in ResBlocks → spectral_norm wraps it → sigma=0 → NaN)
        _base_init_cfg = cfg
        _expert_type = cfg['adaptive_pinn']['expert_type']
        if cfg['init']['hidden'] == 'parent_weights' and _expert_type != 'resnet':
            _base_init_cfg = {**cfg, 'init': {**cfg['init'], 'hidden': 'glorot'}}
        apply_hidden_init(_init_target, _base_init_cfg)
        apply_output_init(_init_target, train_data, cfg, device)
        apply_spectral_norm(_init_target, cfg)
        print()

    epoch = 0

    # Emergency save: fires on any unhandled exception (or process exit) so metrics.json
    # is never lost even if the training loop crashes with a Python exception.
    import atexit as _atexit

    def _emergency_metrics_save():
        if _emergency_metrics_save.done:
            return
        import traceback as _tb_mod
        exc = _tb_mod.format_exc()
        metrics['exception_events'].append({
            'epoch': epoch,
            'note': 'process_exit_or_exception',
            'traceback': exc if exc.strip() != 'NoneType: None' else None,
        })
        metrics['training_time_seconds'] = time.time() - start_time
        _p = run_dir / "metrics.json"
        try:
            with open(_p, 'w') as _f:
                json.dump(metrics, _f, indent=2, cls=_NumpySafeEncoder)
            print(f"\n[Emergency] Metrics saved to {_p}")
        except Exception as _se:
            print(f"\n[Emergency] Could not save metrics: {_se}")

    _emergency_metrics_save.done = False
    _atexit.register(_emergency_metrics_save)

    _nan_detected = False
    while epoch < total_epochs:
        epoch += 1
        timer.start_epoch(epoch, num_experts=model.num_experts if (is_adaptive and hasattr(model, 'num_experts')) else 0)

        # Unfreeze ancestors after freeze_epochs_after_spawn has elapsed
        if _unfreeze_at_epoch is not None and epoch == _unfreeze_at_epoch:
            print(f"\n  [Freeze] Unfreezing ancestors at epoch {epoch} — restoring full training")
            metrics['freeze_events'].append({'epoch': epoch, 'action': 'unfreeze', 'reason': 'freeze_epochs_elapsed'})
            _unfreeze_at_epoch = None
            model.freeze_models(mode='none')
            # Add ancestor params back as a NEW param group at their pre-freeze LR.
            # Existing groups (untouched leaves, new experts) keep their current LRs — no rebuild,
            # no scheduler restart, no warmup re-applied to anything.
            if _ancestor_params and _pre_freeze_lr is not None:
                optimizer.add_param_group({
                    'params': _ancestor_params,
                    'lr': _pre_freeze_lr,
                })
                if _ancestor_state_snapshot:
                    for p in _ancestor_params:
                        saved = _ancestor_state_snapshot.get(id(p))
                        if saved:
                            optimizer.state[p] = copy.deepcopy(saved)
                n_anc = len([i for i in _ancestor_indices if i >= 0])
                base_note = '+base' if -1 in _ancestor_indices else ''
                print(f"  [Freeze] Ancestors ({n_anc} expert(s){base_note}) restored "
                      f"at lr={_pre_freeze_lr:.2e} with preserved moments")
            _ancestor_params = []
            _ancestor_state_snapshot = {}
            _ancestor_indices = set()
            _pre_freeze_lr = None
            metrics['optimizer_snapshots'].append({
                'epoch': epoch,
                'event': 'unfreeze',
                **_get_optimizer_snapshot(optimizer, lr_scheduler, step_count),
            })

        # Enable residual caching for adaptive sampling if needed
        # Cache THIS epoch's residuals for NEXT epoch's resampling
        # (adaptive_sampling_enabled already set from problem_cfg above)
        causal_state = getattr(loss_fn, 'causal_state', None)
        
        will_cache_for_resample = (
            adaptive_sampling_enabled
            and resample_every > 0
            and epoch > 0 and epoch % resample_every == 0
        )
        # Cache residuals for the diagnostic heatmap even when adaptive sampling is off
        _problem_spatial_dim = problem_cfg['spatial_dim']
        will_cache_for_plot = (
            not adaptive_sampling_enabled
            and resample_every > 0
            and epoch > 0 and epoch % resample_every == 0
            and _problem_spatial_dim == 1
        )
        if will_cache_for_resample or will_cache_for_plot:
            model._residual_cache = []
            model._residual_cache_enabled = True
            # Log when adaptive sampling first activates
            if will_cache_for_resample and not hasattr(model, '_adaptive_sampling_activated'):
                model._adaptive_sampling_activated = True
                print(f"  [Adaptive Sampling] Activated at epoch {epoch} (causal training reached final stage)")

        # Resample training data periodically (in-memory, no disk I/O)
        # Skip resampling during L-BFGS/SSBroyden (they need stable loss landscape)
        allow_resample_optimizer = current_optimizer_name not in ('LBFGS', 'SSBroyden')
        if resample_every > 0 and epoch > 1 and (epoch - 1) % resample_every == 0 and allow_resample_optimizer:
            resample_seed = base_seed + epoch
            print(f"  [Resample] Regenerating training data (epoch {epoch}, seed {resample_seed})...")
            cached_residuals = getattr(model, '_residual_cache', [])
            model._residual_cache_enabled = False
            _leaf_info_for_sampling = None
            if _per_leaf_sampling and is_adaptive and hasattr(model, 'get_leaf_info'):
                # Filter out base model entry (region=None); only pass real expert leaf regions.
                # Before first spawn, get_leaf_info() returns [(None, -1)] — passing that to
                # regenerate_training_data would crash when accessing region.bounds_lower.
                _raw_leaf_info = model.get_leaf_info()
                _leaf_info_for_sampling = [(r, idx) for r, idx in _raw_leaf_info if r is not None] or None
            _leaf_causal_states_for_plot = (
                loss_fn._leaf_state.get('causal_states', {})
                if _per_leaf_causal and hasattr(loss_fn, '_leaf_state') else None
            )
            # Save residual heatmap when adaptive sampling is off (adaptive path saves it internally)
            if not adaptive_sampling_enabled and cached_residuals and _problem_spatial_dim == 1:
                all_x = torch.cat([r[0] for r in cached_residuals], dim=0)
                all_t = torch.cat([r[1] for r in cached_residuals], dim=0)
                all_r2 = torch.cat([r[2] for r in cached_residuals], dim=0)
                _save_adaptive_sampling_heatmap(
                    all_x, all_t, all_r2,
                    None, None,
                    run_dir, epoch, cfg,
                    causal_state=causal_state,
                    leaf_info=_leaf_info_for_sampling,
                    leaf_causal_states=_leaf_causal_states_for_plot,
                )
            train_data = regenerate_training_data(
                cfg, device, resample_seed=resample_seed,
                cached_residuals=cached_residuals,
                run_dir=run_dir,
                epoch=epoch,
                causal_state=causal_state,
                leaf_info=_leaf_info_for_sampling,
                leaf_causal_states=_leaf_causal_states_for_plot,
            )
            # Override IC h_gt for time marching (windows 1+)
            train_data = _override_ic_for_time_marching(train_data, cfg, device)
            train_loader = _create_dataloader(train_data, cfg['batch_size'], shuffle=True)
            metrics['resample_events'].append({
                'epoch': epoch,
                'action': 'resampled',
                'optimizer': current_optimizer_name
            })
            metrics['optimizer_snapshots'].append({
                'epoch': epoch,
                'event': 'resample',
                **_get_optimizer_snapshot(optimizer, lr_scheduler, step_count),
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

        _ks_loss_module._nan_ctx[0] = f"epoch {epoch}"

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

                # Split clip: experts at expert_grad_clip_norm (tighter), base at grad_clip_norm.
                # When no experts exist (base-only phase), falls back to grad_clip_norm for all.
                _exp_clip_ps = ([p for exp in model.experts for p in exp.parameters()
                                  if p.requires_grad]
                                 if hasattr(model, 'experts') and model.experts else [])
                _base_clip_ps = ([p for p in model.base_model.parameters() if p.requires_grad]
                                  if hasattr(model, 'base_model') else [])
                if _exp_clip_ps and expert_grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(_exp_clip_ps, expert_grad_clip_norm)
                    if _base_clip_ps and grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(_base_clip_ps, grad_clip_norm)
                elif grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], grad_clip_norm)
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
                # Split clip: experts at expert_grad_clip_norm (tighter), base at grad_clip_norm.
                # When no experts exist (base-only phase), falls back to grad_clip_norm for all.
                _exp_clip_ps = ([p for exp in model.experts for p in exp.parameters()
                                  if p.requires_grad]
                                 if hasattr(model, 'experts') and model.experts else [])
                _base_clip_ps = ([p for p in model.base_model.parameters() if p.requires_grad]
                                  if hasattr(model, 'base_model') else [])
                if _exp_clip_ps and expert_grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(_exp_clip_ps, expert_grad_clip_norm)
                    if _base_clip_ps and grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(_base_clip_ps, grad_clip_norm)
                elif grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], grad_clip_norm)
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
                    if grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad], grad_clip_norm)
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
            _prev_opt = current_optimizer_name
            optimizer, current_optimizer_name = _create_optimizer_by_name(
                optimizer_2_name, model, active_cfg)
            lr_scheduler = None  # optimizer_2 uses its own LR / line search
            # Reset patience counter at switch point
            epochs_without_improvement = 0
            best_train_loss = float('inf')
            metrics['optimizer_events'].append({
                'epoch': epoch,
                'from': _prev_opt,
                'to': current_optimizer_name,
            })
        
        # Store train loss every epoch
        metrics['train_loss_epochs'].append(epoch)
        metrics['train_loss'].append(train_loss)

        # NaN early-stop: save everything and break so the next experiment can run
        if math.isnan(train_loss) or math.isinf(train_loss):
            print(f"\n{'!'*60}")
            print(f"  [NaN] Training diverged at epoch {epoch} — saving diagnostics and stopping.")

            # Diagnose which loss component went NaN
            try:
                with torch.no_grad():
                    _diag_batch = next(iter(train_loader))
                    _comps = loss_fn(model, _diag_batch, return_components=True)
                    print(f"  [NaN] Loss components: " +
                          ", ".join(f"{k}={v.item():.6g}" for k, v in _comps.items()))
                    metrics['nan_components'] = {k: float(v.item()) for k, v in _comps.items()}
            except Exception as _e:
                print(f"  [NaN] Could not compute loss components: {_e}")

            metrics['nan_divergence'] = {'epoch': epoch, 'train_loss': train_loss}
            metrics['training_time_seconds'] = time.time() - start_time

            # Save metrics JSON so the run is inspectable
            _nan_metrics_path = run_dir / "metrics.json"
            with open(_nan_metrics_path, 'w') as _f:
                json.dump(metrics, _f, indent=2, cls=_NumpySafeEncoder)
            print(f"  [NaN] Metrics saved to {_nan_metrics_path}")

            # Save a NaN-state checkpoint for post-mortem inspection
            _nan_ckpt_path = checkpoint_dir / f"nan_checkpoint_epoch_{epoch}.pt"
            _save_checkpoint(_nan_ckpt_path, model, optimizer, current_optimizer_name,
                             epoch, train_loss, eval_loss, cfg, metrics)
            print(f"  [NaN] Checkpoint saved to {_nan_ckpt_path}")
            print(f"{'!'*60}\n")
            _nan_detected = True
            break

        # LRA: update adaptive loss weights periodically
        if lra_weights is not None and epoch > 0 and epoch % lra_weights.update_every == 0:
            try:
                batch_for_lra = next(iter(train_loader))
                lra_weights.update(model, loss_fn, batch_for_lra)
                if epoch % print_every == 0:
                    w_str = ', '.join(f'{k}={v:.4f}' for k, v in lra_weights.weights.items())
                    print(f"  [LRA] weights: {w_str}")
            except Exception as e:
                print(f"  [LRA] Weight update failed at epoch {epoch}: {e}")

        # Causal weighting: check if epsilon should advance
        causal_state = getattr(loss_fn, 'causal_state', None)
        causal_epoch_min_weight = None
        if _per_leaf_causal and hasattr(loss_fn, '_leaf_state'):
            leaf_states = loss_fn._leaf_state.get('causal_states', {})
            if leaf_states:
                for _expert_idx, _leaf_cs in leaf_states.items():
                    causal_epoch_min_weight = min(
                        causal_epoch_min_weight if causal_epoch_min_weight is not None else 1.0,
                        _leaf_cs.get('min_weight', 1.0))
                    if advance_causal_schedule(_leaf_cs):
                        print(f"  [PerLeafCausal] Expert {_expert_idx}: epsilon advanced to "
                              f"{_leaf_cs['tol']:.2f} "
                              f"(stage {_leaf_cs['schedule_idx']+1}/{len(_leaf_cs['schedule'])})")
                    _leaf_cs['min_weight'] = 1.0
            else:
                # No leaves yet (pre-first-spawn); fall back to global causal
                if causal_state is not None:
                    causal_epoch_min_weight = causal_state['min_weight']
                if advance_causal_schedule(causal_state):
                    cs = loss_fn.causal_state
                    print(f"  [Causal] epsilon advanced to "
                          f"{cs['tol']:.2f} "
                          f"(stage {cs['schedule_idx']+1}/{len(cs['schedule'])}, "
                          f"prev_min_w={causal_epoch_min_weight:.6f})")
                if causal_state is not None:
                    causal_state['min_weight'] = 1.0
        else:
            if causal_state is not None:
                causal_epoch_min_weight = causal_state['min_weight']
            if advance_causal_schedule(causal_state):
                cs = loss_fn.causal_state
                print(f"  [Causal] epsilon advanced to "
                      f"{cs['tol']:.2f} "
                      f"(stage {cs['schedule_idx']+1}/"
                      f"{len(cs['schedule'])}, "
                      f"prev_min_w={causal_epoch_min_weight:.6f})")
            # Reset min_weight AFTER advance check so it sees the true minimum.
            if causal_state is not None:
                causal_state['min_weight'] = 1.0

        # Compute evaluation metrics only every print_every epochs or last epoch
        # This speeds up training significantly for physics-informed losses
        should_evaluate = (epoch % eval_every == 0 or epoch == 1 or epoch == total_epochs)
        
        if should_evaluate:
            # Compute train rel-L2 and infinity norm errors

            # Eval phase
            model.eval()
            eval_loss = 0.0
            # Accumulate squared sums for correct global rel-L2 computation
            # (averaging per-batch rel-L2 is mathematically incorrect)
            total_diff_sq = 0.0
            total_gt_sq = 0.0
            eval_inf_norm = 0.0  # Track max across all batches
            n_eval_batches = 0

            for batch in eval_loader:
                # Note: For physics-informed losses, we need gradients w.r.t. inputs
                # even during evaluation (for computing derivatives in PDE residuals).
                # We still use model.eval() to disable dropout/batchnorm training behavior.
                timer.start('eval.loss_fn')
                loss = loss_fn(model, batch, update_causal_state=False)
                timer.stop('eval.loss_fn')

                with torch.no_grad():
                    inputs = torch.cat([batch['x'], batch['t']], dim=1)
                    timer.start('eval.h_pred')
                    h_pred = model(inputs)
                    timer.stop('eval.h_pred')
                    # Accumulate squared differences and GT norms for global rel-L2
                    diff = h_pred - batch['h_gt']
                    total_diff_sq += (diff ** 2).sum().item()
                    total_gt_sq += (batch['h_gt'] ** 2).sum().item()
                    # Track max inf_norm across all batches
                    inf_norm = compute_infinity_norm_error(h_pred, batch['h_gt'])
                    eval_inf_norm = max(eval_inf_norm, inf_norm.item())

                eval_loss += loss.item()
                n_eval_batches += 1

            eval_loss /= n_eval_batches
            # Compute global rel-L2: ||pred - gt||_2 / ||gt||_2
            eval_rel_l2 = math.sqrt(total_diff_sq) / (math.sqrt(total_gt_sq) + 1e-10)

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
            if causal_state is not None and causal_epoch_min_weight is not None:
                cs = causal_state
                stage_str = f"{cs['schedule_idx']+1}/{len(cs['schedule'])}"
                print(f"  [Causal] tol={cs['tol']:.2f}, stage={stage_str}, min_weight={causal_epoch_min_weight:.6f}")
                metrics['causal_history'].append({
                    'epoch': epoch,
                    'tol': float(cs['tol']),
                    'stage': int(cs['schedule_idx']),
                    'stage_total': len(cs['schedule']),
                    'min_weight': float(causal_epoch_min_weight),
                    'threshold': float(cs['threshold'])
                })

            # DIAGNOSTIC: LRA weights and gradient norms
            if lra_weights is not None:
                w = lra_weights.weights
                g = lra_weights.last_grad_norms
                w_str = ', '.join(f'{k}={v:.4f}' for k, v in w.items())
                g_str = ', '.join(f'{k}={g.get(k, 0):.6f}' for k in w)
                print(f"  [LRA] weights: {w_str} | grads: {g_str}")
                # Save to metrics
                metrics['lra_history'].append({
                    'epoch': epoch,
                    'weights': {k: float(v) for k, v in w.items()},
                    'grad_norms': {k: float(g.get(k, 0)) for k in w},
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
                    comps_str = ', '.join(f'{k}={v:.6f}' for k, v in components.items())
                    print(f"  [Loss] components: {comps_str} (unweighted)")
                    metrics['loss_components_history'].append({
                        'epoch': epoch,
                        'residual': float(components['residual'].item()),
                        'ic': float(components['ic'].item()),
                        'bc': float(components['bc'].item()),
                    })
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
        # Normal trigger: epoch % spawn_every == 0.
        # After a failed spawn: also trigger spawn_retry_after epochs after the failure
        # (tracked from _spawn_last_fail_epoch). Plateau gating still applies at retries.
        if spawning_method in ('M_term_tree_by_norm', 'use_perfect_trees'):
            _base_spawn_eligible = is_adaptive and not spawning_complete
        else:
            _base_spawn_eligible = (is_adaptive and
                                    hasattr(model, 'num_experts') and
                                    model.num_experts < max_experts)

        _at_spawn_interval = epoch % spawn_every == 0
        _at_retry = (_spawn_retry_after is not None
                     and _spawn_last_fail_epoch >= 0
                     and epoch == _spawn_last_fail_epoch + _spawn_retry_after)
        _at_interval = _at_spawn_interval or _at_retry

        if _base_spawn_eligible and _spawn_require_plateau:
            if _at_interval:
                # Check training-loss plateau over the look-back window
                _lookback = max(1, _spawn_plateau_epochs)
                _recent = metrics['train_loss'][-_lookback:]
                _recent_valid = [r for r in _recent if not math.isnan(r) and not math.isinf(r)]
                if len(_recent_valid) >= 2:
                    _rel_drop = (_recent_valid[0] - _recent_valid[-1]) / (abs(_recent_valid[0]) + 1e-8)
                    _plateau_met = _rel_drop <= _spawn_plateau_delta
                else:
                    _plateau_met = False

                _drop_str = f"{_rel_drop*100:.2f}%" if len(_recent_valid) >= 2 else "n/a"
                if _plateau_met:
                    spawn_check_triggered = True
                    metrics['plateau_events'].append({
                        'epoch': epoch,
                        'action': 'triggered',
                        'rel_drop_pct': float(_rel_drop * 100) if len(_recent_valid) >= 2 else None,
                        'threshold_pct': float(_spawn_plateau_delta * 100),
                    })
                else:
                    spawn_check_triggered = False
                    _spawn_last_fail_epoch = epoch
                    print(f"  [Plateau] Spawn deferred — loss still dropping "
                          f"({_drop_str} over last {_spawn_plateau_epochs} epochs, "
                          f"threshold={_spawn_plateau_delta*100:.2f}%)")
                    metrics['plateau_events'].append({
                        'epoch': epoch,
                        'action': 'deferred',
                        'rel_drop_pct': float(_rel_drop * 100) if len(_recent_valid) >= 2 else None,
                        'threshold_pct': float(_spawn_plateau_delta * 100),
                    })
            else:
                spawn_check_triggered = False
        else:
            # No plateau gating: fire at spawn_every, and at retry interval after a failure
            spawn_check_triggered = _base_spawn_eligible and _at_interval
        
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

            # Save prediction heatmap BEFORE spawning so it reflects the state that
            # drove the spawn decision. Deleted below if spawn yields 0 experts.
            _pending_spawn_plot = None
            if _problem_spatial_dim == 1 and gt_grid is not None:
                _pending_spawn_plot = adaptive_plots_dir / f"spawn_pred_epoch_{epoch}.png"
                save_spawn_prediction_plot(
                    model=model,
                    domain_bounds=domain_bounds,
                    gt_grid=gt_grid,
                    grid_x=gt_x,
                    grid_t=gt_t,
                    output_path=_pending_spawn_plot,
                    epoch=epoch,
                    cfg=cfg,
                )

            # Only compute per-sample losses for by_mean_residual (expensive)
            loss_components = None
            if spawning_method == 'by_mean_residual':
                problem = cfg['problem']
                loss_weights = cfg[problem]['loss_weights']
                loss_components = compute_loss_components(
                    model=model,
                    x=eval_data['x'],
                    t=eval_data['t'],
                    target=eval_data.get('h_gt', eval_data.get('u_gt')),
                    masks=eval_data['mask'],
                    loss_fn=loss_fn,
                    weights={
                        'residual': loss_weights['residual'],
                        'ic': loss_weights['ic'],
                        'bc': loss_weights['bc'],
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

            if spawning_method == 'by_mean_residual':
                # Pick the single leaf with highest mean residual, split it
                _spawned_children_diag = []
                per_sample_total = loss_components['residual']

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

                # Save diagnostics for by_mean_residual
                diag = {
                    'epoch': epoch,
                    'method': 'by_mean_residual',
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

            elif spawning_method == 'M_term_tree_by_norm':
                # One-shot: fit full tree, select top M by norm, spawn all accepted
                M = adaptive_cfg['M_experts_num']
                print(f"  [M-term Tree] Fitting full tree (max_depth={region_detector.max_depth}, "
                      f"min_samples_leaf={region_detector.min_samples_leaf}), selecting top M={M}...")
                accepted_nodes, prune_depth_stats = \
                    region_detector.fit_full_tree_and_prune(
                        X=X_eval,
                        y=y_eval,
                        M=M,
                        variable_for_node_accept=variable_for_node_accept,
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
                    'method': 'M_term_tree_by_norm',
                    'M_experts_num': M,
                    'variable_for_node_accept': variable_for_node_accept,
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
                _last_spawn_epoch = epoch
                _spawn_last_fail_epoch = -1
                _no_spawn_retries_remaining = _no_spawn_retries_max  # reset on success

                # Apply smart init to newly spawned experts (Glorot hidden + zero output,
                # or parent_weights: copy hidden layers from parent expert for stability).
                _init_mode = problem_cfg['init']['hidden']
                _new_exp_start_idx = len(model.experts) - experts_spawned_this_step
                for _ei, _new_exp in enumerate(model.experts[-experts_spawned_this_step:]):
                    _new_exp_idx = _new_exp_start_idx + _ei
                    if _init_mode == 'parent_weights':
                        # Resolve parent model: AToE uses model.regions, ANT uses model.parent_indices
                        if hasattr(model, 'regions') and _new_exp_idx < len(model.regions):
                            _par_idx = model.regions[_new_exp_idx].parent_idx
                        elif hasattr(model, 'parent_indices') and _new_exp_idx < len(model.parent_indices):
                            _par_idx = model.parent_indices[_new_exp_idx]
                        else:
                            _par_idx = -1
                        _parent_model = (model.base_model if _par_idx == -1
                                         else model.experts[_par_idx])
                        apply_parent_copy_init(
                            _new_exp, _parent_model, cfg,
                            copy_output=isinstance(model, (AToELeaves, ANT)),
                        )
                    else:
                        apply_expert_init(_new_exp, cfg)
                    apply_spectral_norm(_new_exp, cfg)

                # ── 3-phase: reinitialize base + transition to Phase 3 ──
                if use_three_phase and spawning_complete and current_phase == 1:
                    if reinit_base_after_spawn:
                        model.reinitialize_base()
                    current_phase = 3
                    active_cfg = cfg
                    total_epochs = epoch + phase3_epochs  # extend loop
                    # Recalculate optimizer strategy from top-level config
                    _p3_opt1 = active_cfg['optimizer_1'].lower()
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
                    print(f"  [3-Phase] Optimizer: {_p3_opt1}, lr: {active_cfg['lr']}, schedule: {active_cfg['lr_schedule']}")

                # Collect new expert parameters before any freeze/optimizer logic.
                import copy
                _new_expert_params = [
                    p for _exp in model.experts[-experts_spawned_this_step:]
                    for p in _exp.parameters() if p.requires_grad
                ]

                if freeze_epochs_after_spawn > 0:
                    # freeze>0: freeze only the ANCESTORS of newly spawned experts.
                    # Ancestors are the only experts with overlapping regions.
                    # Sibling leaves on other branches keep training uninterrupted.

                    _new_expert_indices = list(range(
                        len(model.experts) - experts_spawned_this_step,
                        len(model.experts)
                    ))
                    if hasattr(model, 'get_ancestor_indices'):
                        _ancestor_indices = model.get_ancestor_indices(_new_expert_indices)
                    else:
                        # Fallback for model types without ancestor tracking
                        _ancestor_indices = set(range(len(model.experts) - experts_spawned_this_step)) | {-1}

                    # Collect ancestor params (all currently trainable, before any freezing)
                    _ancestor_params = []
                    if -1 in _ancestor_indices:
                        _ancestor_params.extend(model.base_model.parameters())
                    for _ai in sorted(i for i in _ancestor_indices if i >= 0):
                        _ancestor_params.extend(model.experts[_ai].parameters())
                    _ancestor_param_ids = {id(p) for p in _ancestor_params}
                    _new_expert_param_ids = {id(p) for p in _new_expert_params}

                    # Untouched leaf params: in current optimizer but not ancestor or new expert
                    _untouched_leaf_params = [
                        p for pg in optimizer.param_groups for p in pg['params']
                        if id(p) not in _ancestor_param_ids and id(p) not in _new_expert_param_ids
                    ]
                    _untouched_state = {
                        id(p): copy.deepcopy(optimizer.state.get(p, {}))
                        for p in _untouched_leaf_params
                    }

                    _pre_freeze_lr = optimizer.param_groups[0]['lr']
                    _ancestor_state_snapshot = {
                        id(p): copy.deepcopy(optimizer.state.get(p, {}))
                        for p in _ancestor_params
                    }

                    _unfreeze_at_epoch = epoch + freeze_epochs_after_spawn
                    n_anc = len([i for i in _ancestor_indices if i >= 0])
                    base_note = '+base' if -1 in _ancestor_indices else ''
                    print(f"  [Freeze] Freezing {n_anc} ancestor(s){base_note} for "
                          f"{freeze_epochs_after_spawn} epochs (unfreeze at epoch {_unfreeze_at_epoch})")
                    print(f"  [Freeze] {len(_untouched_leaf_params)} untouched leaf params continue training")
                    metrics['freeze_events'].append({
                        'epoch': epoch,
                        'action': 'freeze',
                        'reason': 'post_spawn',
                        'freeze_mode_applied': 'ancestors',
                        'ancestor_indices': sorted(_ancestor_indices),
                        'unfreeze_at': _unfreeze_at_epoch,
                        'experts_spawned': experts_spawned_this_step,
                    })
                    metrics['optimizer_snapshots'].append({
                        'epoch': epoch,
                        'event': 'freeze',
                        'experts_spawned': experts_spawned_this_step,
                        **_get_optimizer_snapshot(optimizer, lr_scheduler, step_count),
                    })

                    if hasattr(model, 'freeze_ancestors'):
                        model.freeze_ancestors(_ancestor_indices)
                    else:
                        model.freeze_models(mode='previous')

                    # Build freeze-period optimizer with two groups:
                    # Group 0: untouched leaves (restored state, same LR — keep training)
                    # Group 1: new expert params (fresh state, initial LR)
                    # No scheduler: LRs stay fixed during freeze period (no warmup restart).
                    _freeze_groups = []
                    if _untouched_leaf_params:
                        _freeze_groups.append({'params': _untouched_leaf_params, 'lr': _pre_freeze_lr})
                    if _new_expert_params:
                        _new_expert_lr = _pre_freeze_lr * active_cfg['new_expert_lr_decay']
                        _freeze_groups.append({'params': _new_expert_params, 'lr': _new_expert_lr})
                        if _new_expert_lr != _pre_freeze_lr:
                            print(f"  [SpawnGroup] New expert LR: {_new_expert_lr:.2e} "
                                  f"({active_cfg['new_expert_lr_decay']:.2f}× current {_pre_freeze_lr:.2e})")
                    if _freeze_groups:
                        try:
                            optimizer = _create_grouped_optimizer(_freeze_groups, active_cfg)
                            for p in _untouched_leaf_params:
                                saved = _untouched_state.get(id(p))
                                if saved:
                                    optimizer.state[p] = saved
                        except ValueError:
                            optimizer, current_optimizer_name = _create_primary_optimizer(model, active_cfg)
                    else:
                        optimizer, current_optimizer_name = _create_primary_optimizer(model, active_cfg)
                    lr_scheduler = None  # no scheduler during freeze; LRs are fixed per group
                else:
                    # freeze==0: keep existing optimizer intact so old params preserve their
                    # current state and LR without any restart. Add new expert params as a
                    # separate fresh param group at the initial LR (AB-PINNs pattern: new
                    # subdomains declared under their own optimizer entry with independent LRs).
                    model.freeze_models()  # applies configured freeze_mode (typically 'none')
                    if _new_expert_params:
                        _current_lr = optimizer.param_groups[0]['lr']
                        _new_expert_lr = _current_lr * active_cfg['new_expert_lr_decay']
                        optimizer.add_param_group({
                            'params': _new_expert_params,
                            'lr': _new_expert_lr,
                        })
                        print(f"  [SpawnGroup] New expert params added as fresh param group "
                              f"{len(optimizer.param_groups) - 1} "
                              f"at lr={_new_expert_lr:.2e} "
                              f"({active_cfg['new_expert_lr_decay']:.2f}× current {_current_lr:.2e}); "
                              f"old params unchanged")

                metrics['optimizer_snapshots'].append({
                    'epoch': epoch,
                    'event': 'spawn',
                    'experts_spawned': experts_spawned_this_step,
                    'freeze_epochs_after_spawn': freeze_epochs_after_spawn,
                    **_get_optimizer_snapshot(optimizer, lr_scheduler, step_count),
                })

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

                # spawn_pred plot was already saved before spawn (pre-spawn state)

                if adaptive_cfg['blending_mode'] == 'soft' and problem_type == '2d':
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
                _spawn_last_fail_epoch = epoch
                if _pending_spawn_plot is not None and _pending_spawn_plot.exists():
                    _pending_spawn_plot.unlink()
                    _pending_spawn_plot = None
                if _stop_on_no_spawn:
                    if _no_spawn_retries_remaining > 0:
                        _no_spawn_retries_remaining -= 1
                        print(f"  [SpawnRetry] Zero experts spawned — "
                              f"{_no_spawn_retries_remaining} retries remaining before stop.")
                    else:
                        print(f"  [SpawnRetry] Zero experts spawned and no retries remaining — "
                              f"saving final checkpoint and stopping.")
                        final_checkpoint_path = checkpoint_dir / "final_model.pt"
                        _save_checkpoint(final_checkpoint_path, model, optimizer,
                                         current_optimizer_name, epoch,
                                         train_loss, eval_loss, cfg, metrics)
                        model.train()
                        break

            # Per-leaf causal: update leaf causal states after successful spawn
            if experts_spawned_this_step > 0 and _per_leaf_causal and hasattr(loss_fn, '_leaf_state'):
                if hasattr(model, 'get_leaf_info'):
                    _new_leaf_info = model.get_leaf_info()
                    _existing_leaf_states = loss_fn._leaf_state.get('causal_states', {})
                    _new_states = {}
                    for _region, _expert_idx in _new_leaf_info:
                        if _expert_idx in _existing_leaf_states:
                            _new_states[_expert_idx] = _existing_leaf_states[_expert_idx]
                        else:
                            _new_states[_expert_idx] = create_causal_state(problem_cfg)
                    loss_fn._leaf_state['causal_states'] = _new_states
                    loss_fn._leaf_state['leaf_info'] = _new_leaf_info
                    print(f"  [PerLeafCausal] Updated leaf states: "
                          f"{list(_new_states.keys())} ({len(_new_states)} leaves)")

            # Post-spawn resample: immediately rebuild dataset with new leaf structure so new leaves
            # get their fair share of adaptive points instead of waiting for the next scheduled resample.
            # Without this, newly spawned experts in tiny regions are starved of training points
            # for up to resample_every_epochs batches, causing empty causal chunks and NaN.
            if experts_spawned_this_step > 0 and _per_leaf_sampling and hasattr(model, 'get_leaf_info'):
                _spawn_raw_leaf_info = model.get_leaf_info()
                _spawn_leaf_info = [(r, idx) for r, idx in _spawn_raw_leaf_info if r is not None] or None
                _spawn_cached = getattr(model, '_residual_cache', [])
                _spawn_causal_states = (
                    loss_fn._leaf_state.get('causal_states', {})
                    if _per_leaf_causal and hasattr(loss_fn, '_leaf_state') else None
                )
                _spawn_train_data = regenerate_training_data(
                    cfg, device, resample_seed=epoch,
                    cached_residuals=_spawn_cached,
                    run_dir=run_dir,
                    epoch=epoch,
                    causal_state=causal_state,
                    leaf_info=_spawn_leaf_info,
                    leaf_causal_states=_spawn_causal_states,
                )
                # Override IC h_gt for time marching (windows 1+)
                _spawn_train_data = _override_ic_for_time_marching(_spawn_train_data, cfg, device)
                train_loader = _create_dataloader(_spawn_train_data, cfg['batch_size'], shuffle=True)
                n_new_leaves = len(_spawn_leaf_info) if _spawn_leaf_info else 0
                print(f"  [PostSpawnResample] Rebuilt dataset for {n_new_leaves} leaves")

            model.train()

    # Disable emergency save (loop done or NaN exit)
    _emergency_metrics_save.done = True
    _atexit.unregister(_emergency_metrics_save)

    if _nan_detected:
        print("[NaN] Generating partial training curves before exit...")
        try:
            training_plots_dir = run_dir / "training_plots"
            switch_epoch_to_plot = switch_epoch if (optimizer_2_name is not None and switch_epoch <= epochs) else None
            plot_training_curves(metrics, training_plots_dir, optimizer_switch_epoch=switch_epoch_to_plot)
        except Exception as _plot_err:
            print(f"  [NaN] Could not generate training curves: {_plot_err}")
        print("[NaN] Skipping remaining post-training cleanup — moving to next experiment.")
        return

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

        if adaptive_cfg['blending_mode'] == 'soft' and problem_type == '2d':
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
    metrics['training_time_seconds'] = time.time() - start_time

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
        f.write(f"Final eval loss: {eval_loss:.6f}\n" if eval_loss is not None else "Final eval loss: N/A\n")
        f.write(f"Final eval rel-L2: {eval_rel_l2:.6f}\n" if eval_rel_l2 is not None else "Final eval rel-L2: N/A\n")
        f.write(f"Final eval inf-norm: {eval_inf_norm:.6f}\n" if eval_inf_norm is not None else "Final eval inf-norm: N/A\n")
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


def _cast_data_to_dtype(batch: Dict, dtype: torch.dtype) -> Dict:
    """Cast floating-point tensors in a batch dictionary to specified dtype."""
    result = {
        'x': batch['x'].to(dtype) if batch['x'].is_floating_point() else batch['x'],
        't': batch['t'].to(dtype) if batch['t'].is_floating_point() else batch['t'],
        'h_gt': batch['h_gt'].to(dtype) if batch['h_gt'].is_floating_point() else batch['h_gt'],
        'mask': batch['mask']  # masks are boolean, don't cast
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

