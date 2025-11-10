"""NCC analysis orchestrator for trained models."""

import sys
import torch
import importlib
from pathlib import Path

from utils.io import load_config, make_run_dir
from models.fc_model import FCNet
from ncc.ncc_runner import run_ncc
from trainer.trainer import train


def main():
    """Orchestrate NCC analysis workflow."""
    print("=" * 60)
    print("NCC Analysis Orchestrator")
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

    # Setup device
    device = torch.device('cuda' if config['cuda'] and
                          torch.cuda.is_available() else 'cpu')

    # Determine checkpoint path
    checkpoint_path = None

    if not eval_only:
        # Training mode - train first, then run NCC on best checkpoint
        print("\n3. Training mode - will train then run NCC analysis")

        # Check datasets
        print("\n4. Checking datasets...")
        dataset_dir = Path("datasets") / problem
        train_data_path = dataset_dir / "training_data.pt"
        eval_data_path = dataset_dir / "eval_data.pt"

        if not train_data_path.exists() or not eval_data_path.exists():
            print("  Datasets not found. Generating...")
            from utils.dataset_gen import generate_and_save_datasets
            generate_and_save_datasets(config)
        else:
            print(f"  ✓ Datasets found:")
            print(f"    Train: {train_data_path}")
            print(f"    Eval: {eval_data_path}")

        # Build model
        print("\n5. Building model...")
        model = FCNet(architecture, activation, config)
        print(f"  ✓ Model created: {len(model.get_layer_names())} layers")

        # Build loss
        print("\n6. Building loss function...")
        loss_module = importlib.import_module(f"losses.{problem}_loss")
        loss_fn = loss_module.build_loss(**config)
        print(f"  ✓ Loss function built for {problem}")

        # Train
        print("\n7. Training...")
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

    else:
        # Eval-only mode - require resume_from
        print("\n3. Evaluation-only mode - NCC analysis only")

        if resume_from is None:
            raise ValueError(
                "eval_only=True requires resume_from to be set. "
                "Please specify the path to a trained model checkpoint."
            )

        checkpoint_path = Path(resume_from)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {resume_from}"
            )

        print(f"  ✓ Using checkpoint: {checkpoint_path}")

        # Check eval dataset
        print("\n4. Checking evaluation dataset...")
        dataset_dir = Path("datasets") / problem
        eval_data_path = dataset_dir / "eval_data.pt"

        if not eval_data_path.exists():
            print("  Evaluation data not found. Generating...")
            from utils.dataset_gen import generate_and_save_datasets
            generate_and_save_datasets(config)
        else:
            print(f"  ✓ Eval data found: {eval_data_path}")

    # Run NCC analysis
    print(f"\n{'8' if not eval_only else '5'}. Running NCC analysis...")

    # Load checkpoint
    print(f"  Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Build model
    model = FCNet(architecture, activation, config)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✓ Model weights loaded")

    # Get eval data path
    eval_data_path = Path("datasets") / problem / "eval_data.pt"

    # Run NCC
    ncc_metrics = run_ncc(
        model=model,
        eval_data_path=str(eval_data_path),
        cfg=config,
        run_dir=run_dir
    )

    # Final summary
    print("\n" + "=" * 60)
    print("✓ Complete!")
    print("=" * 60)
    print(f"Output directory: {run_dir}")
    print(f"  - config_used.yaml")
    if not eval_only:
        print(f"  - metrics.json (training)")
        print(f"  - training_plots/")
        print(f"  - summary.txt")
    print(f"  - ncc_plots/ (5 plots)")
    print(f"  - ncc_metrics.json")
    print("\nNCC Summary:")
    print(f"  Classes: {ncc_metrics['num_classes']}")
    print(f"  Layers analyzed: {ncc_metrics['layers_analyzed']}")
    print(f"  Layer accuracies:")
    for layer, acc in ncc_metrics['layer_accuracies'].items():
        print(f"    {layer}: {acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

