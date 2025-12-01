"""Training loop for PINN models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Dict, Callable
import json
import time

from trainer.plotting import plot_training_curves, plot_final_comparison
from trainer.utils import compute_relative_l2_error, compute_infinity_norm_error


def _create_adam_optimizer(model: nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    """Create Adam optimizer with config parameters."""
    betas = tuple(cfg.get('adam_betas', [0.9, 0.999]))
    eps = cfg.get('adam_eps', 1e-8)
    return torch.optim.Adam(
        model.parameters(),
        lr=cfg['lr'],
        betas=betas,
        eps=eps
    )


def _create_lbfgs_optimizer(model: nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    """Create LBFGS optimizer. Should be used with full-batch training."""
    return torch.optim.LBFGS(
        model.parameters(),
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

    # Load datasets
    print(f"\nLoading datasets...")
    train_data = torch.load(train_data_path)
    eval_data = torch.load(eval_data_path)

    # Move data to device
    train_data = _move_batch_to_device(train_data, device)
    eval_data = _move_batch_to_device(eval_data, device)

    print(f"  Train size: {train_data['x'].shape[0]}")
    print(f"  Eval size: {eval_data['x'].shape[0]}")

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

    # Create checkpoint directory (with architecture subdirectory like outputs)
    architecture_str = "-".join(map(str, cfg['architecture']))
    checkpoint_dir = Path("checkpoints") / cfg['problem'] / f"layers-{architecture_str}_act-{cfg['activation']}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create ncc_plots directory for periodic NCC analysis
    ncc_plots_parent = run_dir / "ncc_plots"
    ncc_plots_parent.mkdir(exist_ok=True)

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train phase
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        if current_optimizer_name == 'Adam':
            # Adam: Mini-batch training (GPU parallelized)
            for batch in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(model, batch)
                loss.backward()
                optimizer.step()
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
                # LBFGS step processes entire dataset via closure
                loss = optimizer.step(closure)
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
        should_evaluate = (epoch % print_every == 0 or epoch == 1 or epoch == epochs)
        
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
            
            for batch in train_loader:
                with torch.no_grad():
                    inputs = torch.cat([batch['x'], batch['t']], dim=1)
                    u_pred = model(inputs)
                    rel_l2 = compute_relative_l2_error(u_pred, batch['u_gt'])
                    inf_norm = compute_infinity_norm_error(u_pred, batch['u_gt'])
                    train_rel_l2 += rel_l2.item()
                    train_inf_norm += inf_norm.item()
                    n_train_batches_l2 += 1
            
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
                loss = loss_fn(model, batch)

                with torch.no_grad():
                    inputs = torch.cat([batch['x'], batch['t']], dim=1)
                    u_pred = model(inputs)
                    rel_l2 = compute_relative_l2_error(u_pred, batch['u_gt'])
                    inf_norm = compute_infinity_norm_error(u_pred, batch['u_gt'])

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

        # Periodic NCC analysis
        ncc_eval_every = cfg.get('ncc_eval_every', 0)
        if ncc_eval_every > 0 and epoch % ncc_eval_every == 0:
            print(f"\n  Running NCC analysis at epoch {epoch}...")
            _run_intermediate_ncc(model, cfg, run_dir, epoch)

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
        u_pred_eval = model(inputs_eval)

    plot_final_comparison(
        u_pred_eval.cpu().numpy(),
        eval_data['u_gt'].cpu().numpy(),
        eval_data['x'].cpu().numpy(),
        eval_data['t'].cpu().numpy(),
        training_plots_dir
    )

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
        _, visualize_evaluation, _, _, _ = get_visualization_module(cfg['problem'])
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
    return {
        'x': batch['x'].to(device),
        't': batch['t'].to(device),
        'u_gt': batch['u_gt'].to(device),
        'mask': {
            'residual': batch['mask']['residual'].to(device),
            'IC': batch['mask']['IC'].to(device),
            'BC': batch['mask']['BC'].to(device)
        }
    }


def _create_dataloader(
    data: Dict,
    batch_size: int,
    shuffle: bool
) -> DataLoader:
    """
    Create DataLoader from data dictionary.

    Args:
        data: Dictionary with 'x', 't', 'u_gt', 'mask'
        batch_size: Batch size
        shuffle: Whether to shuffle

    Returns:
        DataLoader
    """
    # Create TensorDataset
    # We need to pass all components, including masks
    dataset = TensorDataset(
        data['x'],
        data['t'],
        data['u_gt'],
        data['mask']['residual'],
        data['mask']['IC'],
        data['mask']['BC']
    )

    # Custom collate function to reconstruct dict format
    # Optimized for GPU: use tuple instead of list for better performance
    def collate_fn(batch_list):
        x_batch = torch.stack(tuple(item[0] for item in batch_list))
        t_batch = torch.stack(tuple(item[1] for item in batch_list))
        u_gt_batch = torch.stack(tuple(item[2] for item in batch_list))
        mask_res_batch = torch.stack(tuple(item[3] for item in batch_list))
        mask_ic_batch = torch.stack(tuple(item[4] for item in batch_list))
        mask_bc_batch = torch.stack(tuple(item[5] for item in batch_list))

        return {
            'x': x_batch,
            't': t_batch,
            'u_gt': u_gt_batch,
            'mask': {
                'residual': mask_res_batch,
                'IC': mask_ic_batch,
                'BC': mask_bc_batch
            }
        }

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
    torch.save(checkpoint, path)


def _run_intermediate_ncc(model, cfg, run_dir, epoch):
    """Run NCC analysis at intermediate epoch."""
    from ncc.ncc_runner import run_ncc
    
    # Get NCC data path
    ncc_data_path = Path("datasets") / cfg['problem'] / "ncc_data.pt"
    
    # Run NCC with epoch-specific output dir nested inside ncc_plots
    run_ncc(
        model=model,
        eval_data_path=str(ncc_data_path),
        cfg=cfg,
        run_dir=run_dir,
        epoch_suffix=f"_epoch_{epoch}"
    )

