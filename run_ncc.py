"""NCC analysis orchestrator for trained models."""

import sys
import torch
import importlib
from pathlib import Path

from utils.io import load_config, make_run_dir
from models.fc_model import FCNet
from ncc.ncc_runner import run_ncc
from trainer.trainer import train


def run_multi_eval(checkpoints_dict, config, run_dir):
    """
    Run NCC analysis on multiple checkpoints and generate comparison plots.
    
    Args:
        checkpoints_dict: Dict of {model_name: checkpoint_path}
        config: Configuration dict
        run_dir: Directory to save results
    """
    print(f"\nRunning NCC analysis on {len(checkpoints_dict)} checkpoints...")
    
    # Storage for aggregated results
    ncc_data = {}
    
    problem = config['problem']
    ncc_data_path = Path("datasets") / problem / "ncc_data.pt"
    
    # Process each checkpoint
    for model_idx, (model_name, checkpoint_path) in enumerate(checkpoints_dict.items(), 1):
        print(f"\n[{model_idx}/{len(checkpoints_dict)}] Processing {model_name}...")
        print(f"  Checkpoint: {checkpoint_path}")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"  ERROR: Checkpoint not found, skipping...")
            continue
        
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except Exception:
            print("  Warning: Standard load failed, trying legacy mode...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            print("  Legacy checkpoint loaded")
        
        # Extract architecture and activation from checkpoint config if available
        if 'config' in checkpoint:
            checkpoint_config = checkpoint['config']
            architecture = checkpoint_config.get('architecture', config['architecture'])
            activation = checkpoint_config.get('activation', config['activation'])
            print(f"  Loaded architecture from checkpoint: {architecture}")
        elif 'architecture' in checkpoint:
            architecture = checkpoint['architecture']
            activation = checkpoint.get('activation', config['activation'])
            print(f"  Loaded architecture from checkpoint: {architecture}")
        else:
            # Fallback: use from config (might not match)
            architecture = config['architecture']
            activation = config['activation']
            print(f"  Warning: Architecture not in checkpoint, using config: {architecture}")
        
        # Build model
        model_config = config.copy()
        model_config['architecture'] = architecture
        model_config['activation'] = activation
        model = FCNet(architecture, activation, model_config)
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Remap keys for legacy checkpoints
        remapped_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('layer_') or key.startswith('output.'):
                if key.startswith('output.'):
                    layer_num = len(architecture) - 1
                    new_key = key.replace('output.', f'network.layer_{layer_num}.')
                else:
                    new_key = f'network.{key}'
                remapped_state_dict[new_key] = value
            else:
                remapped_state_dict[key] = value
        
        # Load weights
        try:
            model.load_state_dict(remapped_state_dict)
            print("  Model weights loaded")
        except RuntimeError:
            print("  Warning: Remapped keys didn't match, trying original...")
            model.load_state_dict(state_dict)
            print("  Model weights loaded")
        
        # Run NCC analysis (without saving plots, just get metrics)
        print(f"  Running NCC analysis...")
        ncc_metrics = run_ncc(
            model=model,
            eval_data_path=str(ncc_data_path),
            cfg=model_config,
            run_dir=None,  # Don't save per-model plots
            suppress_plots=True  # Add flag to suppress individual plots
        )
        
        # Store results in format expected by comparison plots
        # Structure: {model_name: {epoch: ncc_metrics}}
        ncc_data[model_name] = {
            'final': ncc_metrics
        }
        
        print(f"  Complete - Accuracy: {ncc_metrics['layer_accuracies']}")
    
    # Generate comparison plots
    print(f"\n{'='*60}")
    print("Generating comparison plots...")
    print(f"{'='*60}")
    
    from utils.comparison_plots import generate_ncc_comparison_plots_only
    generate_ncc_comparison_plots_only(run_dir, ncc_data)
    
    # Save summary
    import yaml
    summary_path = run_dir / "multi_eval_summary.yaml"
    with open(summary_path, 'w') as f:
        yaml.dump(ncc_data, f, default_flow_style=False)
    print(f"\nSummary saved to {summary_path}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("Multi-Evaluation Complete!")
    print("=" * 60)
    print(f"Output directory: {run_dir}")
    print(f"  - ncc_classification_comparison.png")
    print(f"  - ncc_compactness_comparison.png")
    print(f"  - multi_eval_summary.yaml")
    print("\nModels evaluated:")
    for model_name in ncc_data.keys():
        metrics = ncc_data[model_name]['final']
        print(f"  {model_name}:")
        print(f"    Layers: {metrics['layers_analyzed']}")
        for layer, acc in metrics['layer_accuracies'].items():
            print(f"      {layer}: {acc:.4f}")
    print("=" * 60)


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
    
    # Detect multi-eval mode
    is_multi_eval = isinstance(resume_from, dict)

    print(f"  Problem: {problem}")
    print(f"  Architecture: {architecture}")
    print(f"  Activation: {activation}")
    print(f"  Eval only: {eval_only}")
    if is_multi_eval:
        print(f"  Multi-eval mode: {len(resume_from)} checkpoints")
        for name, path in resume_from.items():
            print(f"    - {name}: {path}")
    else:
        print(f"  Resume from: {resume_from}")

    # Create run directory
    print("\n2. Creating run directory...")
    if is_multi_eval:
        # Multi-eval mode - special directory naming
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("outputs") / "multi_eval_comparison" / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Single model mode - standard naming
        run_dir = make_run_dir(problem, architecture, activation)
    print(f"  Run directory: {run_dir}")

    # Save config to run directory
    import yaml
    config_path = run_dir / "config_used.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"  Config saved to {config_path}")

    # Determine checkpoint path
    checkpoint_path = None

    if is_multi_eval:
        # Multi-eval mode - evaluate multiple checkpoints and compare
        print("\n3. Multi-evaluation mode - comparing multiple checkpoints")
        
        if not eval_only:
            raise ValueError(
                "Multi-evaluation mode (dict resume_from) requires eval_only=True. "
                "Cannot train with multiple checkpoints."
            )
        
        # Check NCC dataset
        print("\n4. Checking NCC dataset...")
        dataset_dir = Path("datasets") / problem
        ncc_data_path = dataset_dir / "ncc_data.pt"

        if not ncc_data_path.exists():
            print("  NCC data not found. Generating...")
            from utils.dataset_gen import generate_and_save_datasets
            generate_and_save_datasets(config)
        else:
            print(f"  NCC data found: {ncc_data_path}")
        
        # Run multi-eval
        run_multi_eval(resume_from, config, run_dir)
        
        # Exit early - multi-eval handles everything
        return
    
    elif not eval_only:
        # Training mode - train first, then run NCC on best checkpoint
        print("\n3. Training mode - will train then run NCC analysis")

        # Check datasets
        print("\n4. Checking datasets...")
        dataset_dir = Path("datasets") / problem
        train_data_path = dataset_dir / "training_data.pt"
        eval_data_path = dataset_dir / "eval_data.pt"
        ncc_data_path = dataset_dir / "ncc_data.pt"

        if not train_data_path.exists() or not eval_data_path.exists() or not ncc_data_path.exists():
            print("  Datasets not found. Generating...")
            from utils.dataset_gen import generate_and_save_datasets
            generate_and_save_datasets(config)
        else:
            print(f"  Datasets found:")
            print(f"    Train: {train_data_path}")
            print(f"    Eval: {eval_data_path}")
            print(f"    NCC: {ncc_data_path}")

        # Build model
        print("\n5. Building model...")
        model = FCNet(architecture, activation, config)
        print(f"  Model created: {len(model.get_layer_names())} layers")

        # Load checkpoint if resume_from is specified
        if resume_from is not None:
            print(f"\n  Loading checkpoint from: {resume_from}")
            resume_checkpoint_path = Path(resume_from)

            if not resume_checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {resume_from}")

            # Load checkpoint (with legacy support)
            try:
                checkpoint = torch.load(resume_checkpoint_path,
                                        map_location='cpu')
            except Exception:
                print("  Warning: Standard load failed, trying legacy mode...")
                checkpoint = torch.load(resume_checkpoint_path,
                                        map_location='cpu',
                                        weights_only=False)
                print("  Legacy checkpoint loaded")

            # Extract state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # Remap keys for legacy checkpoints
            remapped_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('layer_') or key.startswith('output.'):
                    if key.startswith('output.'):
                        layer_num = len(architecture) - 1
                        new_key = key.replace('output.',
                                              f'network.layer_{layer_num}.')
                    else:
                        new_key = f'network.{key}'
                    remapped_state_dict[new_key] = value
                else:
                    remapped_state_dict[key] = value

            # Load weights
            try:
                model.load_state_dict(remapped_state_dict)
                print("  Model weights loaded - continuing from checkpoint")
            except RuntimeError:
                print("  Warning: Remapped keys didn't match, trying original...")
                model.load_state_dict(state_dict)
                print("  Model weights loaded - continuing from checkpoint")

        # Build loss
        print("\n6. Building loss function...")
        loss_module = importlib.import_module(f"losses.{problem}_loss")
        loss_fn = loss_module.build_loss(**config)
        print(f"  Loss function built for {problem}")

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

        print(f"\n  Training complete")
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

        print(f"  Using checkpoint: {checkpoint_path}")

        # Check NCC dataset
        print("\n4. Checking NCC dataset...")
        dataset_dir = Path("datasets") / problem
        ncc_data_path = dataset_dir / "ncc_data.pt"

        if not ncc_data_path.exists():
            print("  NCC data not found. Generating...")
            from utils.dataset_gen import generate_and_save_datasets
            generate_and_save_datasets(config)
        else:
            print(f"  NCC data found: {ncc_data_path}")

    # Run NCC analysis
    print(f"\n{'8' if not eval_only else '5'}. Running NCC analysis...")

    # Load checkpoint
    print(f"  Loading checkpoint: {checkpoint_path}")
    try:
        # Try loading with default settings (weights_only=True in PyTorch 2.6+)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception:
        # Fallback for legacy checkpoints with custom classes
        print("  Warning: Standard load failed, trying legacy mode...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu',
                                weights_only=False)
        print("  Legacy checkpoint loaded")

    # Build model
    model = FCNet(architecture, activation, config)

    # Load model weights - handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        # Checkpoint might be just the state dict itself
        state_dict = checkpoint

    # Remap keys for legacy checkpoints with different layer naming
    # Old: layer_1.weight, layer_2.weight, ..., output.weight
    # New: network.layer_1.weight, ..., network.layer_6.weight
    remapped_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('layer_') or key.startswith('output.'):
            # Remap layer_N or output to network.layer_N or network.layer_M
            if key.startswith('output.'):
                # Output layer is the last layer in the new architecture
                layer_num = len(architecture) - 1
                new_key = key.replace('output.', f'network.layer_{layer_num}.')
            else:
                # Regular hidden layer - add network. prefix
                new_key = f'network.{key}'
            remapped_state_dict[new_key] = value
        else:
            # Already in correct format or doesn't need remapping
            remapped_state_dict[key] = value

    # Try loading with remapped keys
    try:
        model.load_state_dict(remapped_state_dict)
        print("  Model weights loaded")
    except RuntimeError:
        # If remapping didn't work, try original state dict
        print("  Warning: Remapped keys didn't match, trying original...")
        model.load_state_dict(state_dict)
        print("  Model weights loaded")

    # Get NCC data path
    ncc_data_path = Path("datasets") / problem / "ncc_data.pt"

    # Run NCC
    ncc_metrics = run_ncc(
        model=model,
        eval_data_path=str(ncc_data_path),  # Using stratified NCC dataset
        cfg=config,
        run_dir=run_dir
    )

    # In eval-only mode, also run problem-specific evaluation visualization
    if eval_only:
        print("\nGenerating problem-specific evaluation visualizations...")

        # Check if eval data exists
        eval_data_path = Path("datasets") / problem / "eval_data.pt"
        if not eval_data_path.exists():
            print("  Eval data not found. Generating...")
            from utils.dataset_gen import generate_and_save_datasets
            generate_and_save_datasets(config)

        try:
            from utils.problem_specific import get_visualization_module
            _, visualize_evaluation, _, _, _ = get_visualization_module(problem)
            visualize_evaluation(model, str(eval_data_path), run_dir, config)
        except ValueError:
            print(f"  (No custom evaluation visualization for {problem})")
        except Exception as e:
            print(f"  Warning: Could not generate evaluation visualization: {e}")

    # Final summary
    print("\n" + "=" * 60)
    print("Complete!")
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
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

