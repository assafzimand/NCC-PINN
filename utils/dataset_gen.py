"""Dataset generation utilities."""

import math
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
    # NOTE: We set the number of residual training samples to 20000 for faster testing
    # n_residual_train = int(round((ratio * (V ** (1/d))) ** d))
    n_residual_train = 20000
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

    # Generate frequency grid if missing
    freq_grid_path = dataset_dir / "frequency_grid.pt"
    if not freq_grid_path.exists():
        print(f"Generating frequency grid for {problem}...")
        from frequency_tracker.frequency_core import generate_frequency_grid
        
        x_grid, grid_shape, n_dims = generate_frequency_grid(config)
        N_grid = x_grid.shape[0]
        print(f"  Grid shape: {grid_shape} ({N_grid:,} total points)")
        
        # Compute h_gt on grid using solver
        print(f"  Computing ground truth on grid...")
        h_gt_grid = solver_module.evaluate_on_grid(x_grid, config)
        
        freq_data = {
            'x_grid': x_grid,           # (N_grid, d_in)
            'h_gt_grid': h_gt_grid,     # (N_grid, d_o)
            'grid_shape': list(grid_shape),  # Convert tuple to list for JSON serialization
            'n_dims': n_dims            # int
        }
        torch.save(freq_data, freq_grid_path)
        print(f"  Saved to {freq_grid_path}")
    else:
        print(f"Frequency grid already exists: {freq_grid_path}")


def _analytic_ic(problem: str, x: torch.Tensor, pc: Dict) -> torch.Tensor:
    """Return analytical IC values h(x, t=0).  Shape: (N, output_dim)."""
    if problem == 'allen_cahn':
        return x[:, 0:1] ** 2 * torch.cos(math.pi * x[:, 0:1])
    if problem == 'burgers1d':
        return -torch.sin(math.pi * x[:, 0:1])
    if problem == 'burgers2d':
        return 1.0 / (1.0 + torch.exp((x[:, 0:1] + x[:, 1:2]) / 0.2))
    if problem == 'kdv':
        return torch.cos(math.pi * x[:, 0:1])
    if problem == 'ks':
        return torch.cos(x[:, 0:1]) * (1.0 + torch.sin(x[:, 0:1]))
    if problem == 'schrodinger':
        real = 2.0 / torch.cosh(x[:, 0:1])
        imag = torch.zeros_like(real)
        return torch.cat([real, imag], dim=1)
    if problem == 'wave1d':
        return torch.sin(x[:, 0:1])
    if problem == 'fisher_kpp':
        kappa = pc.get('kappa', 25.0)
        width = math.sqrt(kappa / 6.0)
        return 1.0 / (1.0 + torch.exp(width * (x[:, 0:1] - 0.25)))
    if problem == 'conv_diff':
        return -torch.sin(math.pi * x[:, 0:1])
    raise ValueError(f"No analytic IC for problem '{problem}'")


def _analytic_bc(problem: str, x: torch.Tensor, t: torch.Tensor,
                 pc: Dict) -> torch.Tensor:
    """Return analytical BC h_gt at boundary points.  Shape: (N, output_dim).

    For problems whose loss hardcodes the target (allen_cahn, burgers1d,
    conv_diff) or uses periodic matching (kdv, ks, schrodinger), the
    returned values are never read by the loss — we fill zeros."""
    if problem == 'burgers2d':
        return 1.0 / (1.0 + torch.exp((x[:, 0:1] + x[:, 1:2] - t) / 0.2))
    if problem == 'wave1d':
        return torch.sin(x[:, 0:1]) * torch.cos(t)
    if problem == 'fisher_kpp':
        x_lo = pc['spatial_domain'][0][0]
        x_hi = pc['spatial_domain'][0][1]
        mid = (x_lo + x_hi) / 2.0
        return torch.where(x[:, 0:1] < mid,
                           torch.ones_like(x[:, 0:1]),
                           torch.zeros_like(x[:, 0:1]))
    if problem == 'schrodinger':
        return torch.zeros(x.shape[0], 2, device=x.device)
    return torch.zeros(x.shape[0], 1, device=x.device)


def regenerate_training_data(
    config: Dict,
    device: torch.device,
    resample_seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Lightweight resampling: fresh random coordinates + analytical IC/BC.

    Unlike the initial dataset generation this does **not** run any
    numerical solver — only random (x, t) sampling plus trivial
    analytical formulas for IC and BC ground truth.
    """
    problem = config['problem']
    pc = config[problem]
    spatial_dim = pc['spatial_dim']
    spatial_domain = pc['spatial_domain']
    t_min, t_max = pc['temporal_domain']
    output_dim = pc.get('output_dim', 1)

    sizes = calculate_dataset_sizes(config)
    n_res = sizes['n_residual_train']
    n_ic = sizes['n_initial_train']
    n_bc = sizes['n_boundary_train']
    N = n_res + n_ic + n_bc

    torch.manual_seed(resample_seed)

    x = torch.zeros(N, spatial_dim, device=device)
    t = torch.zeros(N, 1, device=device)
    h_gt = torch.zeros(N, output_dim, device=device)
    idx = 0

    # --- residual: random interior points (h_gt unused by PDE loss) ---
    for d in range(spatial_dim):
        lo, hi = spatial_domain[d]
        x[idx:idx + n_res, d] = torch.rand(n_res, device=device) * (hi - lo) + lo
    t[idx:idx + n_res, 0] = torch.rand(n_res, device=device) * (t_max - t_min) + t_min
    idx += n_res

    # --- IC: random x at t_min, analytical h_gt ---
    for d in range(spatial_dim):
        lo, hi = spatial_domain[d]
        x[idx:idx + n_ic, d] = torch.rand(n_ic, device=device) * (hi - lo) + lo
    t[idx:idx + n_ic, 0] = t_min
    h_gt[idx:idx + n_ic] = _analytic_ic(problem, x[idx:idx + n_ic], pc)
    idx += n_ic

    # --- BC: boundary coordinates + analytical h_gt ---
    if spatial_dim == 1:
        x_lo, x_hi = spatial_domain[0]
        n_left = n_bc // 2
        n_right = n_bc - n_left
        t_bc = torch.rand(max(n_left, n_right), device=device) * (t_max - t_min) + t_min
        x[idx:idx + n_left, 0] = x_lo
        t[idx:idx + n_left, 0] = t_bc[:n_left]
        idx += n_left
        x[idx:idx + n_right, 0] = x_hi
        t[idx:idx + n_right, 0] = t_bc[:n_right]
        idx += n_right
    else:
        x0_lo, x0_hi = spatial_domain[0]
        x1_lo, x1_hi = spatial_domain[1]
        n_per = n_bc // 4
        rem = n_bc - 4 * n_per
        for ei in range(4):
            ne = n_per + (1 if ei < rem else 0)
            if ei == 0:
                x[idx:idx + ne, 0] = x0_lo
                x[idx:idx + ne, 1] = torch.rand(ne, device=device) * (x1_hi - x1_lo) + x1_lo
            elif ei == 1:
                x[idx:idx + ne, 0] = x0_hi
                x[idx:idx + ne, 1] = torch.rand(ne, device=device) * (x1_hi - x1_lo) + x1_lo
            elif ei == 2:
                x[idx:idx + ne, 0] = torch.rand(ne, device=device) * (x0_hi - x0_lo) + x0_lo
                x[idx:idx + ne, 1] = x1_lo
            else:
                x[idx:idx + ne, 0] = torch.rand(ne, device=device) * (x0_hi - x0_lo) + x0_lo
                x[idx:idx + ne, 1] = x1_hi
            t[idx:idx + ne, 0] = torch.rand(ne, device=device) * (t_max - t_min) + t_min
            idx += ne

    bc_start = n_res + n_ic
    h_gt[bc_start:] = _analytic_bc(problem, x[bc_start:], t[bc_start:], pc)

    # --- masks ---
    mask_res = torch.zeros(N, dtype=torch.bool, device=device)
    mask_res[:n_res] = True
    mask_ic = torch.zeros(N, dtype=torch.bool, device=device)
    mask_ic[n_res:n_res + n_ic] = True
    mask_bc = torch.zeros(N, dtype=torch.bool, device=device)
    mask_bc[n_res + n_ic:] = True

    return {
        "x": x, "t": t, "h_gt": h_gt,
        "mask": {"residual": mask_res, "IC": mask_ic, "BC": mask_bc},
    }


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
