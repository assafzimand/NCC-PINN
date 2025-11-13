"""Dataset generation utilities."""

import torch
from pathlib import Path
from typing import Dict
import importlib
from utils.dataset_plotting import (
    plot_dataset,
    plot_dataset_statistics
)


def generate_and_save_datasets(config: Dict) -> None:
    """
    Generate training and evaluation datasets if they don't exist.

    Args:
        config: Configuration dictionary containing problem name,
                dataset sizes, etc.
    """
    problem = config['problem']
    cuda_available = config['cuda'] and torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')

    # Create datasets directory
    dataset_dir = Path("datasets") / problem
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_path = dataset_dir / "training_data.pt"
    eval_path = dataset_dir / "eval_data.pt"

    # Dynamically import the solver for the problem
    solver_module = importlib.import_module(f"solvers.{problem}_solver")

    # Generate training data if missing
    if not train_path.exists():
        print(f"Generating training data for {problem}...")
        train_data = solver_module.generate_dataset(
            n_residual=config['n_residual_train'],
            n_ic=config['n_initial_train'],
            n_bc=config['n_boundary_train'],
            device=device,
            config=config
        )
        torch.save(train_data, train_path)
        print(f"  ✓ Saved to {train_path}")

        # Create visualizations
        plot_path = dataset_dir / "training_data_visualization.png"
        title = f"{problem} - Training Data"
        plot_dataset(train_data, str(plot_path), title=title)

        stats_path = dataset_dir / "training_data_statistics.png"
        plot_dataset_statistics(train_data, str(stats_path))
        
        # Problem-specific visualization
        try:
            from utils.problem_specific import get_visualization_module
            visualize_dataset, _, _, _, _ = get_visualization_module(problem)
            visualize_dataset(train_data, dataset_dir, config, 'training')
        except ValueError:
            pass  # No custom visualization for this problem
    else:
        print(f"Training data already exists: {train_path}")

    # Generate evaluation data if missing
    if not eval_path.exists():
        print(f"Generating evaluation data for {problem}...")
        eval_data = solver_module.generate_dataset(
            n_residual=config['n_residual_eval'],
            n_ic=config['n_initial_eval'],
            n_bc=config['n_boundary_eval'],
            device=device,
            config=config
        )
        torch.save(eval_data, eval_path)
        print(f"  ✓ Saved to {eval_path}")

        # Create visualizations
        plot_path = dataset_dir / "eval_data_visualization.png"
        title = f"{problem} - Evaluation Data"
        plot_dataset(eval_data, str(plot_path), title=title)

        stats_path = dataset_dir / "eval_data_statistics.png"
        plot_dataset_statistics(eval_data, str(stats_path))
        
        # Problem-specific visualization
        try:
            from utils.problem_specific import get_visualization_module
            visualize_dataset, _, _, _, _ = get_visualization_module(problem)
            visualize_dataset(eval_data, dataset_dir, config, 'evaluation')
        except ValueError:
            pass  # No custom visualization for this problem
    else:
        print(f"Evaluation data already exists: {eval_path}")

    # Generate NCC data if missing (stratified)
    ncc_path = dataset_dir / "ncc_data.pt"
    if not ncc_path.exists():
        print(f"Generating stratified NCC data for {problem}...")
        
        # Generate large dataset for stratification (10x target size)
        n_large = config['n_samples_ncc'] * 10
        print(f"  Generating large dataset ({n_large} samples) for stratification...")
        large_data = solver_module.generate_dataset(
            n_residual=n_large,
            n_ic=0,  # NCC only needs residual points
            n_bc=0,
            device=device,
            config=config
        )
        
        # Determine output dimension (generic: check u_gt shape)
        output_dim = large_data['u_gt'].shape[1]
        
        # Apply uniform sampling
        print(f"  Applying uniform sampling (target: {config['n_samples_ncc']} samples)...")
        from utils.stratified_sampling import stratify_by_bins
        ncc_data = stratify_by_bins(
            large_data, 
            bins=config['bins'],
            output_dim=output_dim,
            target_size=config['n_samples_ncc'],
            device=device
        )
        
        torch.save(ncc_data, ncc_path)
        print(f"  ✓ Saved {len(ncc_data['x'])} samples to {ncc_path}")
        print(f"  ✓ All {config['bins']**output_dim} classes should be represented")
        
        # Create visualizations
        plot_path = dataset_dir / "ncc_data_visualization.png"
        title = f"{problem} - NCC Data (Stratified)"
        plot_dataset(ncc_data, str(plot_path), title=title)
        
        stats_path = dataset_dir / "ncc_data_statistics.png"
        plot_dataset_statistics(ncc_data, str(stats_path))
        
        # Problem-specific NCC visualization
        try:
            from utils.problem_specific import get_visualization_module
            _, _, visualize_ncc_dataset, _, _ = get_visualization_module(problem)
            visualize_ncc_dataset(ncc_data, dataset_dir, config, 'ncc')
        except ValueError:
            pass  # No custom NCC visualization for this problem
    else:
        print(f"NCC data already exists: {ncc_path}")


def load_dataset(
    path: str,
    device: torch.device = None
) -> Dict[str, torch.Tensor]:
    """
    Load a dataset from disk.

    Args:
        path: Path to the .pt file
        device: Device to load tensors to (if None, keeps original device)

    Returns:
        Dictionary with dataset tensors
    """
    data = torch.load(path)

    if device is not None:
        # Move all tensors to specified device
        data['x'] = data['x'].to(device)
        data['t'] = data['t'].to(device)
        data['u_gt'] = data['u_gt'].to(device)
        data['mask']['residual'] = data['mask']['residual'].to(device)
        data['mask']['IC'] = data['mask']['IC'].to(device)
        data['mask']['BC'] = data['mask']['BC'].to(device)

    return data
