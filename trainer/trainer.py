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
from utils.dataset_gen import (
    regenerate_training_data,
    resample_residual_inplace,
    _save_adaptive_sampling_heatmap,
)
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

    # Reset default device context to CPU before creating DataLoaders.
    # This fixes a PyTorch issue where CUDA inference (e.g., prev_model forward pass
    # in time marching) can corrupt the global device context, causing RandomSampler
    # to fail with "Expected 'cuda' device type for generator but found 'cpu'".
    # This does NOT affect training device - model and data are already on CUDA
    # via explicit .to(device) calls; this only affects internal generator creation.
    torch.set_default_device(None)

    # Create DataLoaders
    train_loader = _create_dataloader(train_data, cfg['batch_size'],
                                      shuffle=True)
    eval_loader = _create_dataloader(eval_data, cfg['batch_size'],
                                     shuffle=False)

    # ── 3-phase logic for M_term_tree_by_norm (the only spawning method) ──
    adaptive_cfg_init = cfg['adaptive_pinn']
    is_adaptive_init = adaptive_cfg_init['enabled']
    spawning_method_init = adaptive_cfg_init['spawning_method']
    if is_adaptive_init and spawning_method_init != 'M_term_tree_by_norm':
        raise ValueError(
            f"spawning_method must be 'M_term_tree_by_norm' "
            f"(got '{spawning_method_init}'). The other spawning methods were "
            f"removed in the M_term cleanup.")
    initial_train_cfg = adaptive_cfg_init.get('initial_train', None)
    reinit_base_after_spawn = adaptive_cfg_init['reinitialize_base_after_spawn']
    pretrained_base_checkpoint = problem_cfg.get('pretrained_base_checkpoint', None)
    _pretrained_force_spawn = False  # set True to force first-epoch spawn (checkpoint flow)

    if not is_adaptive_init:
        # Non-adaptive base-only training: single phase, no spawning.
        active_cfg = cfg
        epochs = cfg['epochs']
        phase3_epochs = 0
        current_phase = 0
        use_three_phase = False
    elif pretrained_base_checkpoint is not None:
        # Phase 1 supplied as a checkpoint: load the base, skip Phase-1 training,
        # force the spawn on the first loop epoch, then transition to Phase 3.
        if reinit_base_after_spawn:
            raise ValueError(
                "reinitialize_base_after_spawn must be false when "
                "pretrained_base_checkpoint is set (loading then reinitializing "
                "would discard the checkpoint).")
        _load_pretrained_base(model, pretrained_base_checkpoint, cfg)
        phase3_epochs = cfg['epochs']
        active_cfg = cfg
        epochs = 1            # one loop epoch to trigger the forced spawn
        current_phase = 1
        use_three_phase = True
        _pretrained_force_spawn = True
        print(f"\n  [3-Phase] Phase 1 skipped: base loaded from "
              f"{pretrained_base_checkpoint}")
        print(f"  [3-Phase] Phase 3 will run for {phase3_epochs} epochs after spawning")
    else:
        # Phase 1 trains the base for initial_train.epochs.
        if initial_train_cfg is None:
            raise ValueError(
                "adaptive_pinn.initial_train is required for M_term_tree_by_norm "
                "when pretrained_base_checkpoint is null.")
        phase1_cfg = dict(cfg)
        for k, v in initial_train_cfg.items():
            phase1_cfg[k] = v
        phase1_epochs = initial_train_cfg['epochs']
        phase3_epochs = cfg['epochs']
        active_cfg = phase1_cfg
        epochs = phase1_epochs
        current_phase = 1
        use_three_phase = True
        print(f"\n  [3-Phase] Phase 1: initial training for {phase1_epochs} epochs")
        print(f"  [3-Phase] Phase 3 will run for {phase3_epochs} epochs after spawning")
        if reinit_base_after_spawn:
            print(f"  [3-Phase] Base model will be reinitialized after spawning")

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
        # 'freeze_events' removed - staged freezing now uses requires_grad=False per level
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
    # Relative-improvement threshold for the plateau test: an epoch only counts as
    # an improvement if it beats the anchored best by at least this fraction.
    patience_rel_delta = cfg.get('patience_rel_delta', 0.0)
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

    if is_adaptive:
        tree_max_depth = adaptive_cfg['tree_max_depth']
        tree_min_samples_leaf = adaptive_cfg['tree_min_samples_leaf']

        print(f"\nAdaptive PINN enabled (spawning_method={spawning_method}):")
        print(f"  Max experts: {max_experts}")
        print(f"  Spawn every: {spawn_every} epochs")
        print(f"  Tree max depth: {tree_max_depth}")
        print(f"  Tree min samples leaf: {tree_min_samples_leaf}")
        print(f"  M experts num: {adaptive_cfg['M_experts_num']}")
        print(f"  Wavelet threshold: {wavelet_threshold}")
        print(f"  Blending mode: {adaptive_cfg['blending_mode']}")
        print(f"  Model type: {type(model).__name__}")
        enable_timing_cfg = adaptive_cfg['enable_timing']
        print(f"  Timing profiling: {'enabled' if enable_timing_cfg else 'disabled'}")

        from adaptive.region_detector import RegionDetector
        from adaptive.visualization import (
            plot_expert_regions, save_regions_metadata, prepare_ground_truth_grid,
            plot_expert_soft_weights
        )
        from adaptive.indicators import RegionDescriptor

        domain_bounds = model.get_domain_bounds()
        gt_grid, gt_x, gt_t = prepare_ground_truth_grid(eval_data, domain_bounds)

        region_detector = RegionDetector(
            n_estimators=1,
            max_depth=tree_max_depth,
            min_samples_leaf=tree_min_samples_leaf,
            domain_bounds=domain_bounds
        )
        
        # Create directory for adaptive outputs
        adaptive_plots_dir = run_dir / "adaptive_plots"
        adaptive_plots_dir.mkdir(exist_ok=True)
    
    rejected_regions = []
    leaf_loss_history = []

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
            train_data = resample_residual_inplace(
                train_data, cfg, device,
                resample_seed=resample_seed,
                cached_residuals=cached_residuals,
                run_dir=run_dir,
                epoch=epoch,
                causal_state=causal_state,
                leaf_info=_leaf_info_for_sampling,
                leaf_causal_states=_leaf_causal_states_for_plot,
            )
            torch.set_default_device(None)  # Reset device context after CUDA inference
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

                # DIAGNOSTIC: Gradient flow analysis (gated by debug_prints, every 100 epochs)
                if cfg.get('debug_prints', False) and n_train_batches == 0 and epoch % 100 == 0:
                    _net = getattr(model, 'base_model', model)
                    
                    # Alpha gradients (PirateNet specific)
                    _alpha_grads = []
                    _alpha_vals = []
                    for name, param in _net.named_parameters():
                        if 'alpha' in name and param.grad is not None:
                            _alpha_grads.append((name, param.grad.norm().item(), param.item()))
                            _alpha_vals.append(param.item())
                    if _alpha_grads:
                        _ag_str = ', '.join(f'{g:.2e}' for _, g, _ in _alpha_grads)
                        print(f"  [GradDiag] alpha grads: [{_ag_str}]")
                    
                    # Per-layer gradient norms (top 5 smallest non-zero)
                    _layer_grads = []
                    for name, param in _net.named_parameters():
                        if param.grad is not None:
                            _gn = param.grad.norm().item()
                            if _gn > 0:
                                _layer_grads.append((name, _gn, param.data.norm().item()))
                    if _layer_grads:
                        _layer_grads.sort(key=lambda x: x[1])  # sort by grad norm
                        _smallest = _layer_grads[:3]
                        _largest = _layer_grads[-3:]
                        _sm_str = ', '.join(f'{n.split(".")[-1]}={g:.2e}' for n, g, _ in _smallest)
                        _lg_str = ', '.join(f'{n.split(".")[-1]}={g:.2e}' for n, g, _ in _largest)
                        print(f"  [GradDiag] smallest grads: [{_sm_str}]")
                        print(f"  [GradDiag] largest grads: [{_lg_str}]")
                        
                        # Gradient/weight ratio (indicates update magnitude)
                        _ratios = [(n, g/w if w > 0 else 0) for n, g, w in _layer_grads]
                        _ratios.sort(key=lambda x: x[1])
                        _ratio_str = ', '.join(f'{n.split(".")[-1]}={r:.2e}' for n, r in _ratios[:3])
                        print(f"  [GradDiag] grad/weight ratios (smallest): [{_ratio_str}]")

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
                
                # DIAGNOSTIC: Track parameter values before step for update magnitude calculation
                _param_before = None
                if cfg.get('debug_prints', False) and n_train_batches == 0 and epoch % 100 == 0:
                    _net = getattr(model, 'base_model', model)
                    _param_before = {name: param.data.clone() for name, param in _net.named_parameters() if param.requires_grad}
                
                optimizer.step()
                timer.stop('train.optim_step')
                
                # DIAGNOSTIC: Compute actual parameter update magnitudes
                if _param_before is not None:
                    _net = getattr(model, 'base_model', model)
                    _update_norms = []
                    _alpha_updates = []
                    for name, param in _net.named_parameters():
                        if name in _param_before:
                            _delta = (param.data - _param_before[name]).norm().item()
                            _update_norms.append((name, _delta, param.data.norm().item()))
                            if 'alpha' in name:
                                _alpha_updates.append((name, _delta, param.item()))
                    
                    # Report alpha updates specifically
                    if _alpha_updates:
                        _au_str = ', '.join(f'{d:.2e}' for _, d, _ in _alpha_updates)
                        print(f"  [UpdateDiag] alpha update magnitudes: [{_au_str}]")
                    
                    # Overall update stats
                    if _update_norms:
                        _total_update = sum(d for _, d, _ in _update_norms)
                        _total_weight = sum(w for _, _, w in _update_norms)
                        print(f"  [UpdateDiag] total update norm: {_total_update:.4e}, "
                              f"total weight norm: {_total_weight:.2f}, "
                              f"ratio: {_total_update/_total_weight:.2e}")

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
            # Restore default device to CUDA before creating SSBroyden/LBFGS optimizer.
            # This ensures optimizer state tensors (e.g., Hessian approximation) are created
            # on the correct device, not CPU (which can happen if default was reset earlier
            # for DataLoader compatibility in time-marching windows 1+).
            torch.set_default_device(device)
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

            # DIAGNOSTIC: Full loss-term breakdown (raw → grad → weight → weighted-grad)
            # Shows exactly what the optimizer sees, to diagnose why updates are tiny.
            # ||sum|| << individual weighted grads ⇒ terms cancel (gradient conflict).
            if cfg.get('debug_prints', False) and lra_weights is not None:
                try:
                    _dbg_batch = next(iter(train_loader))
                    _dbg_params = [p for p in model.parameters() if p.requires_grad]
                    _raw_comps = loss_fn(model, _dbg_batch, return_components=True)
                    _w = lra_weights.weights
                    _raw_vals, _raw_gn, _wtd_gn = {}, {}, {}
                    _weighted_grad_flats = []
                    for _k, _v in _raw_comps.items():
                        _raw_vals[_k] = _v.item()
                        if isinstance(_v, torch.Tensor) and _v.requires_grad:
                            _grads = torch.autograd.grad(
                                _v, _dbg_params, retain_graph=True, allow_unused=True)
                            _flat = torch.cat([gg.flatten() for gg in _grads if gg is not None])
                            _raw_gn[_k] = _flat.norm().item()
                            _wk = _w.get(_k, 1.0)
                            _wtd_gn[_k] = _wk * _raw_gn[_k]
                            _weighted_grad_flats.append(_wk * _flat)
                        else:
                            _raw_gn[_k] = 0.0
                            _wtd_gn[_k] = 0.0
                    model.zero_grad(set_to_none=True)
                    # Norm of the summed weighted gradient = actual update-direction magnitude
                    _total_wg = 0.0
                    if _weighted_grad_flats:
                        _total_wg = torch.stack(_weighted_grad_flats, dim=0).sum(dim=0).norm().item()
                    _keys = list(_raw_comps.keys())
                    print("  [LossDiag] raw terms:      " +
                          ', '.join(f'{k}={_raw_vals[k]:.4e}' for k in _keys))
                    print("  [LossDiag] raw grad norms: " +
                          ', '.join(f'{k}={_raw_gn[k]:.4e}' for k in _keys))
                    print("  [LossDiag] LRA weights:    " +
                          ', '.join(f'{k}={_w.get(k, 1.0):.4f}' for k in _keys))
                    print("  [LossDiag] weighted terms: " +
                          ', '.join(f'{k}={_w.get(k, 1.0) * _raw_vals[k]:.4e}' for k in _keys))
                    print("  [LossDiag] weighted grads: " +
                          ', '.join(f'{k}={_wtd_gn[k]:.4e}' for k in _keys) +
                          f"  (||sum||={_total_wg:.4e})")
                except Exception as _e:
                    print(f"  [LossDiag] failed: {_e}")

            # DIAGNOSTIC: PirateNet alphas, causal chunks, LR
            if cfg.get('debug_prints', False):
                # PirateNet alpha cold-start check
                _net = getattr(model, 'base_model', model)
                if hasattr(_net, 'debug_state'):
                    _ds = _net.debug_state()
                    _alphas_str = ', '.join(
                        f'{a:.4f}' for a in _ds['alphas'])
                    _wn0 = (
                        _ds['block_w_norms'][0]
                        if _ds['block_w_norms'] else []
                    )
                    _wn0_str = '/'.join(f'{w:.3f}' for w in _wn0)
                    print(
                        f"  [PirateNet] alphas=[{_alphas_str}] | "
                        f"W-norms(block0)=[{_wn0_str}]"
                    )

                # Per-chunk causal breakdown
                _cs = causal_state
                if _cs is not None and 'last_weights' in _cs:
                    _w_str = ', '.join(
                        f'{w:.3f}' for w in _cs['last_weights'])
                    _cl_str = ', '.join(
                        f'{cl:.2e}'
                        for cl in _cs['last_chunk_losses']
                    )
                    _t_str = ', '.join(
                        f'{t:.3f}' for t in _cs['last_chunk_tmax'])
                    print(f"  [CausalChunks] w=[{_w_str}]")
                    print(f"  [CausalChunks] L=[{_cl_str}]")
                    print(f"  [CausalChunks] tmax=[{_t_str}]")

                # LR schedule sanity check (extended)
                _cur_lr = optimizer.param_groups[0]['lr']
                _warmup_steps = cfg.get('lr_warmup_steps', 0)
                _decay_steps = cfg.get('lr_decay_steps', 2000)
                _decay_rate = cfg.get('lr_decay_rate', 0.9)
                _base_lr = cfg.get('lr', 0.001)
                
                # Calculate expected LR
                if step_count <= _warmup_steps:
                    _phase = "warmup"
                    _expected_lr = _base_lr * (cfg.get('lr_warmup_start_factor', 0.01) + 
                                               (1 - cfg.get('lr_warmup_start_factor', 0.01)) * step_count / _warmup_steps)
                else:
                    _steps_after_warmup = step_count - _warmup_steps
                    _num_decays = _steps_after_warmup // _decay_steps
                    _expected_lr = _base_lr * (_decay_rate ** _num_decays)
                    _phase = f"decay (n={_num_decays})"
                
                _lr_match = "✓" if abs(_cur_lr - _expected_lr) / _expected_lr < 0.01 else "✗"
                print(
                    f"  [LR] lr={_cur_lr:.2e} (expected={_expected_lr:.2e} {_lr_match}) | "
                    f"step={step_count} | phase={_phase}"
                )

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

        # Patience-based early stopping on train loss. Active ONLY in Phase 3
        # (current_phase != 1; non-adaptive single-phase is current_phase 0) and only
        # from patience_start_epoch — which equals the optimizer switch epoch when a
        # second optimizer is configured, so patience watches only the second optimizer.
        # Plateau test uses a relative min-delta so a loss creeping down by a negligible
        # amount each epoch still counts as "no improvement" and eventually stops.
        if (train_loss is not None and patience_epochs > 0
                and current_phase != 1 and epoch >= patience_start_epoch):
            if train_loss < best_train_loss * (1.0 - patience_rel_delta):
                best_train_loss = train_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            # min_epochs is a grace period measured within the active window.
            if (epoch - patience_start_epoch >= min_epochs
                    and epochs_without_improvement >= patience_epochs):
                print(f"\n  [EarlyStop] No train loss improvement "
                      f">{patience_rel_delta:.1%} for {epochs_without_improvement} "
                      f"epochs (best={best_train_loss:.6f}). "
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

        # Adaptive PINN: one-shot expert spawning (M_term_tree_by_norm).
        # Spawning is only possible in Phase 1 (current_phase == 1); the spawn epoch is
        # also the Phase 1 -> Phase 3 transition, so it can never recur in Phase 3.
        # Normal trigger: epoch % spawn_every == 0.
        # After a failed spawn: also trigger spawn_retry_after epochs after the failure
        # (tracked from _spawn_last_fail_epoch). Plateau gating still applies at retries.
        _base_spawn_eligible = (is_adaptive and not spawning_complete
                                and current_phase == 1)

        _at_spawn_interval = epoch % spawn_every == 0
        _at_retry = (_spawn_retry_after is not None
                     and _spawn_last_fail_epoch >= 0
                     and epoch == _spawn_last_fail_epoch + _spawn_retry_after)
        _at_interval = _at_spawn_interval or _at_retry

        if _pretrained_force_spawn and _base_spawn_eligible:
            # Checkpoint flow: build the tree on the loaded base immediately (no
            # Phase-1 training, no interval/plateau wait), then transition to Phase 3.
            spawn_check_triggered = True
        elif _base_spawn_eligible and _spawn_require_plateau:
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

            import numpy as np
            is_copy_spawn = isinstance(model, AToELeaves)
            experts_spawned_this_step = 0

            if 'spawning_diagnostics' not in metrics:
                metrics['spawning_diagnostics'] = []

            # =============================================================
            # Dispatch on spawning_method
            # =============================================================

            if spawning_method == 'M_term_tree_by_norm':
                # One-shot: fit full tree, select top M by norm, spawn all accepted
                M = adaptive_cfg['M_experts_num']
                
                # Tree closure: AToE uses ancestors-only (additive composition);
                # ANT/AToE-Leaves use ancestors+siblings (routing/tiling).
                is_atoe_additive = isinstance(model, AToE) and not isinstance(model, AToELeaves)
                retain_siblings = not is_atoe_additive
                closure_desc = "ancestors-only (AToE)" if not retain_siblings else "ancestors+siblings"
                
                print(f"  [M-term Tree] Fitting full tree (max_depth={region_detector.max_depth}, "
                      f"min_samples_leaf={region_detector.min_samples_leaf}), selecting top M={M}...")
                print(f"  [M-term Tree] Closure mode: {closure_desc}")
                accepted_nodes, prune_depth_stats = \
                    region_detector.fit_full_tree_and_prune(
                        X=X_eval,
                        y=y_eval,
                        M=M,
                        variable_for_node_accept=variable_for_node_accept,
                        verbose=True,
                        retain_siblings=retain_siblings,
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

                # Apply smart init to newly spawned experts (glorot: Glorot hidden + zero
                # output; or parent_weights: copy hidden layers from parent).
                # copy_output depends on model type:
                #   AToE (additive): copy_output=False -> zero output, expert starts at u=0
                #   ANT/AToE-Leaves (PoU): copy_output=True -> continuous handoff
                _init_mode = problem_cfg['init']['hidden']
                _new_exp_start_idx = len(model.experts) - experts_spawned_this_step
                
                # Determine copy_output based on model type
                is_atoe_additive = isinstance(model, AToE) and not isinstance(model, AToELeaves)
                _copy_output = not is_atoe_additive  # False for AToE, True for ANT/AToE-Leaves
                
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
                            copy_output=_copy_output,
                        )
                        _par_label = 'base' if _par_idx == -1 else f'expert {_par_idx}'
                        _out_msg = "output copied" if _copy_output else "output zeroed (additive)"
                        print(f"  [ParentInit] Expert {_new_exp_idx}: hidden from {_par_label}, {_out_msg}")
                    else:
                        apply_expert_init(_new_exp, cfg)
                    apply_spectral_norm(_new_exp, cfg)

                # ── 3-phase: reinitialize base + transition to Phase 3 ──
                # The spawn epoch is also the Phase 1 -> Phase 3 transition. There is no
                # freezing: the Phase 3 optimizer is recreated over ALL params (base + every
                # spawned expert), so every parameter enters the optimizer exactly once.
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
                    # Recreate optimizer + LR scheduler with Phase 3 config over ALL params.
                    optimizer, current_optimizer_name = _create_primary_optimizer(model, active_cfg)
                    lr_scheduler = _create_lr_scheduler(optimizer, active_cfg, total_steps_p3)
                    step_count = 0
                    epochs_without_improvement = 0
                    best_train_loss = float('inf')
                    _p3_betas = active_cfg.get('soap_betas', active_cfg.get('adam_betas', '?'))
                    _p3_lr_steps = active_cfg.get('lr_decay_steps', '?')
                    _p3_warmup = active_cfg.get('lr_warmup_steps', 0)
                    print(f"  [3-Phase FIX] Phase 3 optimizer recreated: {current_optimizer_name}")
                    print(f"  [3-Phase FIX]   soap_betas={_p3_betas}, lr_decay_steps={_p3_lr_steps}, warmup={_p3_warmup} steps")
                    print(f"  [3-Phase FIX]   total params: {sum(len(pg['params']) for pg in optimizer.param_groups)}")

                metrics['optimizer_snapshots'].append({
                    'epoch': epoch,
                    'event': 'spawn',
                    'experts_spawned': experts_spawned_this_step,
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
                torch.set_default_device(None)  # Reset device context after CUDA inference
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

    # No explicit generator - uses global random state from torch.manual_seed()
    # This avoids device mismatch issues while maintaining reproducibility
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        pin_memory=False,  # Data already on device
        num_workers=0,  # Keep data on GPU
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
    # With spectral norm (nn.utils.parametrizations), live model objects cannot be
    # pickled by torch.save. In time-marching mode, cfg['_time_marching_window']
    # carries a 'prev_model' reference that would fail serialization.
    # Strip it only when spectral norm is active — the reference is transient and
    # is never read back from a checkpoint (time_marching.py manages it in memory).
    cfg_to_save = cfg
    if (cfg.get('init', {}).get('spectral_norm', False)
            and '_time_marching_window' in cfg
            and cfg['_time_marching_window'].get('prev_model') is not None):
        cfg_to_save = dict(cfg)
        tm = dict(cfg['_time_marching_window'])
        tm['prev_model'] = None
        cfg_to_save['_time_marching_window'] = tm

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer_name,
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'eval_loss': eval_loss,
        'config': cfg_to_save,
        'metrics': metrics
    }
    
    # For adaptive models, also save extended state
    if hasattr(model, 'state_dict_extended'):
        checkpoint['adaptive_state'] = model.state_dict_extended()
        checkpoint['is_adaptive'] = True

    torch.save(checkpoint, path)


def _infer_base_arch_from_state_dict(sd: Dict) -> list:
    """Best-effort base architecture from a plain FCNet state dict.

    Counts ``network.layer_{i}.weight`` tensors. Only used as a fallback when the
    checkpoint does not store its nominal ``base_architecture`` (e.g. an old vanilla
    base checkpoint). Note: with Fourier features the inferred input dim reflects the
    expanded input, so a saved nominal arch is always preferred when available.
    """
    arch = []
    i = 1
    while f'network.layer_{i}.weight' in sd:
        w = sd[f'network.layer_{i}.weight']
        if i == 1:
            arch.append(int(w.shape[1]))
        arch.append(int(w.shape[0]))
        i += 1
    return arch


def _load_pretrained_base(model: nn.Module, ckpt_path: str, cfg: Dict) -> None:
    """Load the BASE network from a checkpoint into ``model.base_model``.

    Supplies Phase 1 without training (the ``pretrained_base_checkpoint`` flow).
    Accepts either an adaptive/MoE checkpoint (takes only ``adaptive_state['base_model']``,
    ignoring its experts) or a plain base checkpoint (uses ``model_state_dict``).

    If the checkpoint's base architecture differs from the run's, the base is rebuilt
    to the checkpoint's architecture and that architecture is written back into ``cfg``
    (and ``model.config_base_architecture``) so that experts spawned later — in
    particular ``init.hidden == 'parent_weights'``, which copies the parent's layers —
    are shape-compatible with the loaded base.
    """
    from pathlib import Path as _Path
    p = _Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(
            f"pretrained_base_checkpoint not found: {ckpt_path}")
    # Trusted local checkpoint: weights_only=False (PyTorch 2.6+ defaults to True,
    # which rejects the numpy scalars stored in the saved config/metrics).
    try:
        ckpt = torch.load(p, map_location='cpu', weights_only=False)
    except TypeError:  # older torch without the weights_only kwarg
        ckpt = torch.load(p, map_location='cpu')

    adaptive_state = ckpt.get('adaptive_state') if isinstance(ckpt, dict) else None
    if adaptive_state and 'base_model' in adaptive_state:
        base_sd = adaptive_state['base_model']
        saved_arch = adaptive_state.get('base_architecture')
        saved_activation = adaptive_state.get('activation')
        saved_expert_type = (adaptive_state.get('adaptive_config') or {}).get('expert_type')
        print(f"  [PretrainedBase] Source is an adaptive/MoE checkpoint; "
              f"loading its base only (ignoring {adaptive_state.get('num_experts', '?')} experts).")
    elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        base_sd = ckpt['model_state_dict']
        _cfg_in = ckpt.get('config') or {}
        saved_arch = _cfg_in.get('base_architecture')
        saved_activation = _cfg_in.get('activation')
        saved_expert_type = (_cfg_in.get('adaptive_pinn') or {}).get('expert_type')
    else:
        raise ValueError(
            f"Could not find base weights in checkpoint {ckpt_path} "
            f"(expected 'adaptive_state.base_model' or 'model_state_dict').")

    if not saved_arch:
        saved_arch = _infer_base_arch_from_state_dict(base_sd)
        print(f"  [PretrainedBase] Checkpoint has no stored base_architecture; "
              f"inferred {saved_arch} from weights.")

    # Adopt the checkpoint's base architecture if it differs from the run's.
    if list(saved_arch) != list(model.base_architecture):
        from models.network_factory import create_network
        _old = next(model.base_model.parameters())
        device, dtype = _old.device, _old.dtype
        activation = saved_activation or getattr(model, 'activation', cfg.get('activation'))
        expert_type = saved_expert_type or cfg['adaptive_pinn'].get('expert_type', 'mlp')
        print(f"  [PretrainedBase] Adopting checkpoint base architecture: "
              f"{model.base_architecture} -> {list(saved_arch)}")
        model.base_model = create_network(
            list(saved_arch), activation, cfg, is_base=True, expert_type=expert_type
        ).to(device=device, dtype=dtype)
        model.base_architecture = list(saved_arch)
        if hasattr(model, 'config_base_architecture'):
            # Drives the architecture of experts spawned later (incl. parent_weights copy).
            model.config_base_architecture = list(saved_arch)
        cfg['base_architecture'] = list(saved_arch)
        print(f"  [PretrainedBase] Updated config base_architecture to {list(saved_arch)} "
              f"so spawned experts match the loaded base.")

    model.base_model.load_state_dict(base_sd)
    n_params = sum(q.numel() for q in model.base_model.parameters())
    print(f"  [PretrainedBase] Loaded base weights from {ckpt_path} ({n_params} params)")
    # Re-sync AToE's batched container so the forward pass sees the loaded base.
    if hasattr(model, 'batched_models'):
        model.batched_models.sync_from_models(model.base_model, model.experts)


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

