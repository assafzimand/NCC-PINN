"""Training loop for PINN models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Dict, Callable
import json
import time

from trainer.plotting import plot_training_curves, plot_final_comparison
from trainer.utils import compute_relative_l2_error


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

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    # Training setup
    epochs = cfg['epochs']
    print_every = cfg['print_every']
    save_every = cfg['save_every']

    # Metrics storage
    metrics = {
        'epochs': [],
        'train_loss': [],
        'eval_loss': [],
        'train_rel_l2': [],
        'eval_rel_l2': []
    }

    best_eval_loss = float('inf')
    best_checkpoint_path = None

    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints") / cfg['problem']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train phase
        model.train()
        train_loss = 0.0
        train_rel_l2 = 0.0
        n_train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            # Compute loss
            loss = loss_fn(model, batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Compute metrics
            with torch.no_grad():
                inputs = torch.cat([batch['x'], batch['t']], dim=1)
                u_pred = model(inputs)
                rel_l2 = compute_relative_l2_error(u_pred, batch['u_gt'])

            train_loss += loss.item()
            train_rel_l2 += rel_l2.item()
            n_train_batches += 1

        train_loss /= n_train_batches
        train_rel_l2 /= n_train_batches

        # Eval phase
        model.eval()
        eval_loss = 0.0
        eval_rel_l2 = 0.0
        n_eval_batches = 0

        with torch.no_grad():
            for batch in eval_loader:
                loss = loss_fn(model, batch)

                inputs = torch.cat([batch['x'], batch['t']], dim=1)
                u_pred = model(inputs)
                rel_l2 = compute_relative_l2_error(u_pred, batch['u_gt'])

                eval_loss += loss.item()
                eval_rel_l2 += rel_l2.item()
                n_eval_batches += 1

        eval_loss /= n_eval_batches
        eval_rel_l2 /= n_eval_batches

        # Store metrics
        metrics['epochs'].append(epoch)
        metrics['train_loss'].append(train_loss)
        metrics['eval_loss'].append(eval_loss)
        metrics['train_rel_l2'].append(train_rel_l2)
        metrics['eval_rel_l2'].append(eval_rel_l2)

        # Print progress
        if epoch % print_every == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch}/{epochs}] ({elapsed:.1f}s) | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Eval Loss: {eval_loss:.6f} | "
                  f"Train Rel-L2: {train_rel_l2:.6f} | "
                  f"Eval Rel-L2: {eval_rel_l2:.6f}")

        # Save checkpoint periodically
        if epoch % save_every == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            _save_checkpoint(checkpoint_path, model, optimizer, epoch,
                           train_loss, eval_loss, cfg, metrics)
            print(f"  → Checkpoint saved: {checkpoint_path}")

        # Save best model
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_checkpoint_path = checkpoint_dir / "best_model.pt"
            _save_checkpoint(best_checkpoint_path, model, optimizer, epoch,
                           train_loss, eval_loss, cfg, metrics)

    # Save final model
    final_checkpoint_path = checkpoint_dir / "final_model.pt"
    _save_checkpoint(final_checkpoint_path, model, optimizer, epochs,
                    train_loss, eval_loss, cfg, metrics)

    print(f"\n✓ Training completed in {time.time() - start_time:.1f}s")
    print(f"  Best eval loss: {best_eval_loss:.6f}")
    print(f"  Best checkpoint: {best_checkpoint_path}")
    print(f"  Final checkpoint: {final_checkpoint_path}")

    # Plot training curves
    print(f"\nGenerating training plots...")
    training_plots_dir = run_dir / "training_plots"
    plot_training_curves(metrics, training_plots_dir)

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
    print(f"  ✓ Metrics saved to {metrics_path}")

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
        f.write(f"Best eval loss: {best_eval_loss:.6f}\n\n")
        f.write(f"Best checkpoint: {best_checkpoint_path}\n")
        f.write(f"Final checkpoint: {final_checkpoint_path}\n")
    print(f"  ✓ Summary saved to {summary_path}")

    # Save config used
    config_path = run_dir / "config_used.yaml"
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"  ✓ Config saved to {config_path}")

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
    def collate_fn(batch_list):
        x_batch = torch.stack([item[0] for item in batch_list])
        t_batch = torch.stack([item[1] for item in batch_list])
        u_gt_batch = torch.stack([item[2] for item in batch_list])
        mask_res_batch = torch.stack([item[3] for item in batch_list])
        mask_ic_batch = torch.stack([item[4] for item in batch_list])
        mask_bc_batch = torch.stack([item[5] for item in batch_list])

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
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'eval_loss': eval_loss,
        'config': cfg,
        'metrics': metrics
    }
    torch.save(checkpoint, path)

