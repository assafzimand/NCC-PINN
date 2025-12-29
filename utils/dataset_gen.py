"""Dataset generation utilities."""

import torch
from pathlib import Path
from typing import Dict
import importlib
from utils.dataset_plotting import (
    plot_dataset,
    plot_dataset_statistics
)


def calculate_dataset_sizes(config: Dict) -> Dict[str, int]:
    """
    Calculate dataset sizes from ratios and problem domain.
    
    Args:
        config: Configuration dictionary containing problem name,
                sampling ratios, and problem-specific domain info.
    
    Returns:
        Dictionary with calculated dataset sizes.
    """
    problem = config['problem']
    problem_cfg = config[problem]
    sampling = config['sampling']
    
    # Get dimensionality: d = spatial_dim + 1 (time)
    spatial_dim = problem_cfg['spatial_dim']
    d = spatial_dim + 1
    
    # Calculate volume V = product of all domain ranges
    spatial_domain = problem_cfg['spatial_domain']
    temporal_domain = problem_cfg['temporal_domain']
    
    V = 1.0
    for i in range(spatial_dim):
        V *= (spatial_domain[i][1] - spatial_domain[i][0])
    V *= (temporal_domain[1] - temporal_domain[0])
    
    # Calculate n_residual_train from: ratio = S^(1/d) / V^(1/d)
    # Solving: S = (ratio * V^(1/d))^d
    ratio = sampling['sample_volume_ratio']
    n_residual_train = int(round((ratio * (V ** (1/d))) ** d))
    
    # Calculate other sizes from ratios
    sizes = {
        'n_residual_train': n_residual_train,
        'n_initial_train': int(round(n_residual_train * sampling['initial_train_ratio'])),
        'n_boundary_train': int(round(n_residual_train * sampling['boundary_train_ratio'])),
        'n_residual_eval': int(round(n_residual_train * sampling['eval_train_ratio'])),
        'n_initial_eval': int(round(n_residual_train * sampling['initial_train_ratio'] * sampling['eval_train_ratio'])),
        'n_boundary_eval': int(round(n_residual_train * sampling['boundary_train_ratio'] * sampling['eval_train_ratio'])),
        'n_samples_ncc': int(round(n_residual_train * sampling['ncc_train_ratio'])),
    }
    
    # Print calculated values
    print(f"\n{'='*60}")
    print(f"Dataset Size Calculation for {problem}")
    print(f"{'='*60}")
    print(f"  Dimensionality (d): {d} ({spatial_dim} spatial + 1 time)")
    print(f"  Domain Volume (V): {V:.4f}")
    print(f"  Target Ratio (S^(1/d) / V^(1/d)): {ratio}")
    calculated_ratio = (sizes['n_residual_train'] ** (1/d)) / (V ** (1/d))
    print(f"  Calculated Ratio: {calculated_ratio:.2f}")
    print(f"\n  Dataset Sizes:")
    print(f"    n_residual_train: {sizes['n_residual_train']:,}")
    print(f"    n_initial_train:  {sizes['n_initial_train']:,}")
    print(f"    n_boundary_train: {sizes['n_boundary_train']:,}")
    print(f"    n_residual_eval:  {sizes['n_residual_eval']:,}")
    print(f"    n_initial_eval:   {sizes['n_initial_eval']:,}")
    print(f"    n_boundary_eval:  {sizes['n_boundary_eval']:,}")
    print(f"    n_samples_ncc:    {sizes['n_samples_ncc']:,}")
    print(f"{'='*60}\n")
    
    return sizes


def generate_and_save_datasets(config: Dict) -> None:
    """
    Generate training and evaluation datasets if they don't exist.

    Args:
        config: Configuration dictionary containing problem name,
                sampling ratios, etc.
    """
    problem = config['problem']
    cuda_available = config['cuda'] and torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')

    # Calculate dataset sizes from ratios
    sizes = calculate_dataset_sizes(config)

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
            n_residual=sizes['n_residual_train'],
            n_ic=sizes['n_initial_train'],
            n_bc=sizes['n_boundary_train'],
            device=device,
            config=config
        )
        torch.save(train_data, train_path)
        print(f"  Saved to {train_path}")

        # Create visualizations
        plot_path = dataset_dir / "training_data_visualization.png"
        title = f"{problem} - Training Data"
        plot_dataset(train_data, str(plot_path), title=title)

        stats_path = dataset_dir / "training_data_statistics.png"
        plot_dataset_statistics(train_data, str(stats_path))
        
        # Problem-specific visualization
        try:
            from utils.problem_specific import get_visualization_module
            viz_funcs = get_visualization_module(problem)
            visualize_dataset = viz_funcs[0]
            visualize_dataset(train_data, dataset_dir, config, 'training')
        except ValueError:
            pass  # No custom visualization for this problem
    else:
        print(f"Training data already exists: {train_path}")

    # Generate evaluation data if missing
    if not eval_path.exists():
        print(f"Generating evaluation data for {problem}...")
        eval_data = solver_module.generate_dataset(
            n_residual=sizes['n_residual_eval'],
            n_ic=sizes['n_initial_eval'],
            n_bc=sizes['n_boundary_eval'],
            device=device,
            config=config
        )
        torch.save(eval_data, eval_path)
        print(f"  Saved to {eval_path}")

        # Create visualizations
        plot_path = dataset_dir / "eval_data_visualization.png"
        title = f"{problem} - Evaluation Data"
        plot_dataset(eval_data, str(plot_path), title=title)

        stats_path = dataset_dir / "eval_data_statistics.png"
        plot_dataset_statistics(eval_data, str(stats_path))
        
        # Problem-specific visualization
        try:
            from utils.problem_specific import get_visualization_module
            viz_funcs = get_visualization_module(problem)
            visualize_dataset = viz_funcs[0]
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
        n_large = sizes['n_samples_ncc'] * 10
        print(f"  Generating large dataset ({n_large} samples) for stratification...")
        large_data = solver_module.generate_dataset(
            n_residual=n_large,
            n_ic=0,  # NCC only needs residual points
            n_bc=0,
            device=device,
            config=config
        )
        
        # Determine output dimension
        output_dim = large_data['h_gt'].shape[1]
        
        # Apply uniform sampling
        print(f"  Applying uniform sampling (target: {sizes['n_samples_ncc']} samples)...")
        from utils.stratified_sampling import stratify_by_bins
        ncc_data = stratify_by_bins(
            large_data, 
            bins=config['bins'],
            output_dim=output_dim,
            target_size=sizes['n_samples_ncc'],
            device=device
        )
        
        torch.save(ncc_data, ncc_path)
        print(f"  Saved {len(ncc_data['x'])} samples to {ncc_path}")
        print(f"  All {config['bins']**output_dim} classes should be represented")
        
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
        data['h_gt'] = data['h_gt'].to(device)
        data['mask']['residual'] = data['mask']['residual'].to(device)
        data['mask']['IC'] = data['mask']['IC'].to(device)
        data['mask']['BC'] = data['mask']['BC'].to(device)

    return data
