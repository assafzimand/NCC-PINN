"""Main training orchestrator for NCC-PINN framework."""

import sys
import torch
import importlib
from pathlib import Path

from utils.io import load_config, make_run_dir
from utils.dataset_gen import generate_and_save_datasets
from models.fc_model import FCNet
from trainer.trainer import train


def main():
    """Orchestrate the complete training workflow."""
    print("=" * 60)
    print("NCC-PINN Training Orchestrator")
    print("=" * 60)

    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config()
    problem = config['problem']
    architecture = config['architecture']
    activation = config['activation']
    eval_only = config['eval_only']
    resume_from = config['resume_from']

    print(f"  Problem: {problem}")
    print(f"  Architecture: {architecture}")
    print(f"  Activation: {activation}")
    print(f"  Eval only: {eval_only}")
    print(f"  Resume from: {resume_from}")

    # Create run directory
    print("\n2. Creating run directory...")
    run_dir = make_run_dir(problem, architecture, activation)
    print(f"  Run directory: {run_dir}")

    # Save config to run directory
    import yaml
    config_path = run_dir / "config_used.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"  ✓ Config saved to {config_path}")

    # Generate datasets if missing
    print("\n3. Checking datasets...")
    dataset_dir = Path("datasets") / problem
    train_data_path = dataset_dir / "training_data.pt"
    eval_data_path = dataset_dir / "eval_data.pt"

    if not train_data_path.exists() or not eval_data_path.exists():
        print("  Datasets not found. Generating...")
        generate_and_save_datasets(config)
    else:
        print(f"  ✓ Datasets found:")
        print(f"    Train: {train_data_path}")
        print(f"    Eval: {eval_data_path}")

    # Build model
    print("\n4. Building model...")
    device = torch.device('cuda' if config['cuda'] and
                          torch.cuda.is_available() else 'cpu')
    model = FCNet(architecture, activation, config)
    print(f"  ✓ Model created: {len(model.get_layer_names())} layers")
    print(f"  Device: {device}")

    # Build loss function
    print("\n5. Building loss function...")
    loss_module = importlib.import_module(f"losses.{problem}_loss")
    loss_fn = loss_module.build_loss(**config)
    print(f"  ✓ Loss function built for {problem}")

    # Handle resume_from checkpoint
    start_epoch = 0
    optimizer = None

    if resume_from is not None:
        print(f"\n6. Loading checkpoint from: {resume_from}")
        checkpoint_path = Path(resume_from)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_from}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Model weights loaded")

        # Load optimizer state for fine-tuning
        if not eval_only:
            optimizer_state = checkpoint['optimizer_state_dict']
            start_epoch = checkpoint['epoch']
            print(f"  ✓ Optimizer state loaded (resuming from epoch {start_epoch})")
        else:
            print(f"  ✓ Model loaded for evaluation only")

        # Print checkpoint info
        if 'train_loss' in checkpoint:
            print(f"  Checkpoint info:")
            print(f"    Epoch: {checkpoint['epoch']}")
            print(f"    Train loss: {checkpoint['train_loss']:.6f}")
            print(f"    Eval loss: {checkpoint['eval_loss']:.6f}")
    else:
        print("\n6. No checkpoint to resume from")

    # Training or evaluation
    if eval_only:
        print("\n7. Evaluation-only mode")

        if resume_from is None:
            raise ValueError(
                "eval_only=True requires resume_from to be set"
            )

        # Run evaluation
        print("  Running evaluation on eval dataset...")
        model = model.to(device)
        model.eval()

        # Load eval data
        eval_data = torch.load(eval_data_path)

        # Move to device
        eval_data_device = {
            'x': eval_data['x'].to(device),
            't': eval_data['t'].to(device),
            'u_gt': eval_data['u_gt'].to(device),
            'mask': {
                'residual': eval_data['mask']['residual'].to(device),
                'IC': eval_data['mask']['IC'].to(device),
                'BC': eval_data['mask']['BC'].to(device)
            }
        }

        with torch.no_grad():
            # Compute loss
            eval_loss = loss_fn(model, eval_data_device)

            # Compute relative L2
            from trainer.utils import compute_relative_l2_error
            inputs = torch.cat([eval_data_device['x'], eval_data_device['t']],
                             dim=1)
            u_pred = model(inputs)
            eval_rel_l2 = compute_relative_l2_error(
                u_pred,
                eval_data_device['u_gt']
            )

        print(f"  ✓ Eval loss: {eval_loss.item():.6f}")
        print(f"  ✓ Eval relative L2: {eval_rel_l2.item():.6f}")

        # Save metrics
        import json
        metrics = {
            'eval_loss': eval_loss.item(),
            'eval_rel_l2': eval_rel_l2.item(),
            'checkpoint_loaded': resume_from
        }

        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  ✓ Metrics saved to {metrics_path}")

        # Generate evaluation plots
        from trainer.plotting import plot_final_comparison
        training_plots_dir = run_dir / "training_plots"

        plot_final_comparison(
            u_pred.cpu().numpy(),
            eval_data_device['u_gt'].cpu().numpy(),
            eval_data_device['x'].cpu().numpy(),
            eval_data_device['t'].cpu().numpy(),
            training_plots_dir
        )

        print(f"  ✓ Evaluation complete")

    else:
        print("\n7. Training mode")

        # Create optimizer
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
            print(f"  ✓ Optimizer created (Adam, lr={config['lr']})")
        else:
            # Reconstruct optimizer with loaded state
            optimizer_new = torch.optim.Adam(model.parameters(),
                                           lr=config['lr'])
            optimizer_new.load_state_dict(optimizer_state)
            optimizer = optimizer_new
            print(f"  ✓ Optimizer resumed")

        # Adjust epochs if resuming
        if start_epoch > 0:
            remaining_epochs = config['epochs'] - start_epoch
            print(f"  Continuing for {remaining_epochs} more epochs " +
                  f"(total: {config['epochs']})")

        # Call trainer
        checkpoint_path = train(
            model=model,
            loss_fn=loss_fn,
            train_data_path=str(train_data_path),
            eval_data_path=str(eval_data_path),
            cfg=config,
            run_dir=run_dir
        )

        print(f"\n  ✓ Training complete")
        print(f"  Best checkpoint: {checkpoint_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("✓ Run complete!")
    print("=" * 60)
    print(f"Output directory: {run_dir}")
    print(f"  - config_used.yaml")
    print(f"  - metrics.json")
    print(f"  - training_plots/")
    if not eval_only:
        print(f"  - summary.txt")
        print(f"Best checkpoint saved to: checkpoints/{problem}/")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

