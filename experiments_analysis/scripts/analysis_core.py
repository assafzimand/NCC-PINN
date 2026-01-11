"""Core analysis utilities for consistent metric computation.

This module provides functions to compute all metrics (probe, derivatives, NCC, frequency)
using the SAME probes fitted once on training data. This ensures consistency across all
metric computations during analysis.

Key principles:
- Probes are fitted ONCE per model
- All metrics use the same probes
- Frequency analysis uses NUFFT for mathematically correct spectral analysis on scattered data
- No fallbacks - if checkpoints don't exist, analysis fails
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import re

# Import error computation functions
from trainer.utils import compute_relative_l2_error, compute_infinity_norm_error


# =============================================================================
# MODEL AND DATA LOADING
# =============================================================================

def find_model_checkpoint(run_dir: Path, problem: str, model_name: str) -> Optional[Path]:
    """Find the model checkpoint file.
    
    Checks several possible locations based on the project structure.
    
    Args:
        run_dir: The timestamped run directory
        problem: Problem name (e.g., 'schrodinger', 'wave1d')
        model_name: Model folder name
        
    Returns:
        Path to checkpoint file, or None if not found
    """
    model_dir = run_dir.parent
    
    possible_paths = [
        # Inside experiment's model folder (AWS experiment structure)
        model_dir / "checkpoints" / model_name / "best_model.pt",
        model_dir / "checkpoints" / model_name / "final_model.pt",
        # Global checkpoints folder (local development structure)
        Path("checkpoints") / problem / model_name / "best_model.pt",
        Path("checkpoints") / problem / model_name / "final_model.pt",
        # Inside run directory
        run_dir / "checkpoints" / "best_model.pt",
        run_dir / "checkpoints" / "final_model.pt",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


def parse_architecture_from_name(model_name: str) -> Tuple[List[int], str]:
    """Parse architecture and activation from model folder name.
    
    Args:
        model_name: Folder name like "schrodinger-2-140-140-140-2-tanh"
        
    Returns:
        Tuple of (architecture list, activation name)
    """
    parts = model_name.split('-')
    architecture = []
    activation = 'tanh'
    
    for part in parts[1:]:  # Skip problem name
        if part.isdigit():
            architecture.append(int(part))
        elif part in ['tanh', 'relu', 'sin', 'gelu']:
            activation = part
            break
    
    return architecture, activation


def extract_problem_from_name(model_name: str) -> Optional[str]:
    """Extract problem name from model folder name."""
    model_lower = model_name.lower()
    if 'schrodinger' in model_lower:
        return 'schrodinger'
    elif 'wave1d' in model_lower:
        return 'wave1d'
    elif 'burgers2d' in model_lower:
        return 'burgers2d'
    elif 'burgers1d' in model_lower:
        return 'burgers1d'
    return None


def get_problem_config(problem: str) -> Dict:
    """Get problem-specific configuration including domain bounds."""
    PROBLEM_CONFIGS = {
        'wave1d': {
            'spatial_dim': 1,
            'output_dim': 1,
            'spatial_domain': [[-1, 1]],
            'temporal_domain': [0, 1]
        },
        'burgers1d': {
            'spatial_dim': 1,
            'output_dim': 1,
            'spatial_domain': [[-1, 1]],
            'temporal_domain': [0, 1]
        },
        'burgers2d': {
            'spatial_dim': 2,
            'output_dim': 1,
            'spatial_domain': [[-1, 1], [-1, 1]],
            'temporal_domain': [0, 2]
        },
        'schrodinger': {
            'spatial_dim': 1,
            'output_dim': 2,
            'spatial_domain': [[-5, 5]],
            'temporal_domain': [0, np.pi / 2]
        },
    }
    return PROBLEM_CONFIGS.get(problem, {
        'spatial_dim': 1,
        'output_dim': 1,
        'spatial_domain': [[-1, 1]],
        'temporal_domain': [0, 1]
    })


def load_model_from_checkpoint(
    checkpoint_path: Path,
    architecture: List[int],
    activation: str,
    config: Dict
) -> Optional[torch.nn.Module]:
    """Load a model from checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file
        architecture: List of layer sizes
        activation: Activation function name
        config: Full configuration dict
        
    Returns:
        Loaded model or None on failure
    """
    from models.fc_model import FCNet
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        model = FCNet(architecture, activation, config)
        
        # Extract state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Try loading directly first
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
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
            model.load_state_dict(remapped_state_dict)
        
        return model
    except Exception as e:
        print(f"  Error loading model: {e}")
        return None


# Maximum grid points for analysis on CPU (to avoid memory issues)
# For 3D grids (2D spatial + 1D temporal), 128^3 = 2M points is too heavy
# Use 64^3 = 262K points instead for 3D, or 128^2 = 16K for 2D
MAX_ANALYSIS_GRID_POINTS_2D = 128 * 128  # 16,384 points for 2D grids
MAX_ANALYSIS_GRID_POINTS_3D = 64 * 64 * 64  # 262,144 points for 3D grids
MAX_ANALYSIS_GRID_PER_DIM_3D = 64  # Maximum points per dimension for 3D grids


def downsample_frequency_grid(freq_grid_data: Dict, max_per_dim: int = 64) -> Dict:
    """Downsample a frequency grid for analysis on CPU.
    
    For high-dimensional grids (3D+), computing metrics on the full grid
    can be too memory-intensive. This function creates a smaller uniform
    grid by taking every N-th point along each dimension.
    
    Args:
        freq_grid_data: Original frequency grid data dict
        max_per_dim: Maximum points per dimension for the downsampled grid
        
    Returns:
        Downsampled frequency grid data dict with same structure
    """
    grid_shape = freq_grid_data['grid_shape']
    n_dims = len(grid_shape)
    
    # Compute stride for each dimension
    strides = []
    new_shape = []
    for dim_size in grid_shape:
        stride = max(1, dim_size // max_per_dim)
        new_size = (dim_size + stride - 1) // stride  # Ceiling division
        strides.append(stride)
        new_shape.append(min(dim_size, new_size))
    
    # Reshape to grid, subsample, then flatten
    x_grid = freq_grid_data['x_grid']
    h_gt_grid = freq_grid_data['h_gt_grid']
    
    # Reshape to original grid shape
    # x_grid: (N, n_coords) -> (d0, d1, ..., n_coords)
    n_coords = x_grid.shape[1]
    x_reshaped = x_grid.reshape(*grid_shape, n_coords)
    
    # h_gt_grid: (N,) or (N, output_dim) -> (d0, d1, ...) or (d0, d1, ..., output_dim)
    if h_gt_grid.dim() == 1:
        h_reshaped = h_gt_grid.reshape(*grid_shape)
    else:
        output_dim = h_gt_grid.shape[-1]
        h_reshaped = h_gt_grid.reshape(*grid_shape, output_dim)
    
    # Create slices for subsampling
    slices = tuple(slice(0, None, stride) for stride in strides)
    
    x_subsampled = x_reshaped[slices]
    h_subsampled = h_reshaped[slices]
    
    # Get actual new shape after subsampling
    actual_new_shape = list(x_subsampled.shape[:-1])  # Exclude coord dim
    
    # Flatten back
    new_n_points = int(np.prod(actual_new_shape))
    x_flat = x_subsampled.reshape(new_n_points, n_coords)
    if h_gt_grid.dim() == 1:
        h_flat = h_subsampled.reshape(new_n_points)
    else:
        h_flat = h_subsampled.reshape(new_n_points, output_dim)
    
    return {
        'x_grid': x_flat,
        'h_gt_grid': h_flat,
        'grid_shape': actual_new_shape,
        'n_dims': n_dims,
        'original_grid_shape': list(grid_shape),  # Keep original for reference
        'downsampled': True
    }


def load_datasets(problem: str, max_grid_per_dim: int = None) -> Tuple[Dict, Dict, Dict, Dict]:
    """Load train, eval, NCC, and frequency grid datasets for a problem.
    
    Args:
        problem: Problem name
        max_grid_per_dim: Maximum points per dimension for frequency grid.
            If None, uses full grid for 2D problems, downsampled for 3D+.
            Set to a specific value to override.
        
    Returns:
        Tuple of (train_data, eval_data, ncc_data, freq_grid_data) dicts
        
    Raises:
        FileNotFoundError: If any required dataset is missing
    """
    base_path = Path('datasets') / problem
    
    train_path = base_path / 'training_data.pt'
    eval_path = base_path / 'eval_data.pt'
    ncc_path = base_path / 'ncc_data.pt'
    freq_grid_path = base_path / 'frequency_grid.pt'
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval data not found: {eval_path}")
    if not ncc_path.exists():
        raise FileNotFoundError(f"NCC data not found: {ncc_path}")
    if not freq_grid_path.exists():
        raise FileNotFoundError(f"Frequency grid data not found: {freq_grid_path}")
    
    train_data = torch.load(train_path, weights_only=False)
    eval_data = torch.load(eval_path, weights_only=False)
    ncc_data = torch.load(ncc_path, weights_only=False)
    freq_grid_data = torch.load(freq_grid_path, weights_only=False)
    
    # Diagnostic: print grid info
    grid_shape = freq_grid_data.get('grid_shape', [])
    n_points = freq_grid_data['x_grid'].shape[0]
    n_dims = len(grid_shape)
    print(f"      Frequency grid: {grid_shape} = {n_points:,} points ({n_dims}D grid)")
    
    # Auto-downsample for 3D+ grids if not already small
    if n_dims >= 3:
        # For 3D+ grids, check if we need to downsample
        if max_grid_per_dim is None:
            max_grid_per_dim = MAX_ANALYSIS_GRID_PER_DIM_3D
        
        needs_downsample = any(dim > max_grid_per_dim for dim in grid_shape)
        if needs_downsample:
            print(f"      WARNING: 3D grid too large for CPU analysis, downsampling to max {max_grid_per_dim} per dim...")
            freq_grid_data = downsample_frequency_grid(freq_grid_data, max_grid_per_dim)
            new_shape = freq_grid_data['grid_shape']
            new_points = freq_grid_data['x_grid'].shape[0]
            print(f"      Downsampled grid: {new_shape} = {new_points:,} points")
    
    return train_data, eval_data, ncc_data, freq_grid_data


# =============================================================================
# PROBE FITTING (Done ONCE per model)
# =============================================================================

def fit_probes(
    model: torch.nn.Module,
    train_data: Dict,
    device: torch.device
) -> Dict[str, torch.nn.Linear]:
    """Fit linear probes for all hidden layers using training data.
    
    This should be called ONCE per model. The returned probes are then used
    for all metric computations.
    
    Args:
        model: Loaded neural network model
        train_data: Training data dict with 'x', 't', 'h_gt' keys
        device: Device for computation
        
    Returns:
        Dict mapping layer_name -> trained Linear probe
    """
    from probes.probe_core import train_linear_probe
    
    model = model.to(device)
    model.eval()
    
    # Get hidden layer names
    all_layers = model.get_layer_names()
    hidden_layers = all_layers[:-1]  # Exclude output layer
    
    # Extract training data
    train_x = train_data['x'].to(device)
    train_t = train_data['t'].to(device)
    train_targets = train_data['h_gt'].to(device)
    
    # Ensure targets are 2D
    if train_targets.dim() == 1:
        train_targets = train_targets.unsqueeze(1)
    
    # Collect training embeddings
    handles = model.register_ncc_hooks(hidden_layers)
    with torch.no_grad():
        train_inputs = torch.cat([train_x, train_t], dim=1)
        _ = model(train_inputs)
    train_embeddings = model.activations.copy()
    model.remove_hooks()
    
    # Train probes for each layer
    probes = {}
    print("  Fitting probes for all hidden layers...")
    
    for layer_name in hidden_layers:
        train_emb = train_embeddings[layer_name]
        probe = train_linear_probe(train_emb, train_targets)
        probes[layer_name] = probe
    
    print(f"  Fitted {len(probes)} probes")
    return probes


# =============================================================================
# PROBE METRICS COMPUTATION
# =============================================================================

def compute_probe_metrics(
    probes: Dict[str, torch.nn.Linear],
    model: torch.nn.Module,
    train_data: Dict,
    eval_data: Dict,
    device: torch.device
) -> Dict:
    """Compute probe prediction metrics (rel_l2, inf_norm) on train and eval data.
    
    Args:
        probes: Dict of trained probes from fit_probes()
        model: Neural network model
        train_data: Training data dict
        eval_data: Evaluation data dict
        device: Device for computation
        
    Returns:
        Dict with 'layers_analyzed', 'train' and 'eval' metrics
    """
    from trainer.utils import compute_relative_l2_error, compute_infinity_norm_error
    
    model = model.to(device)
    model.eval()
    
    layer_names = sorted(probes.keys(), key=lambda x: int(x.split('_')[1]))
    
    def compute_metrics_on_data(data_dict):
        """Helper to compute metrics on a given dataset."""
        x = data_dict['x'].to(device)
        t = data_dict['t'].to(device)
        targets = data_dict['h_gt'].to(device)
        
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        
        # Collect embeddings
        handles = model.register_ncc_hooks(layer_names)
        with torch.no_grad():
            inputs = torch.cat([x, t], dim=1)
            _ = model(inputs)
        embeddings = model.activations.copy()
        model.remove_hooks()
        
        rel_l2_list = []
        inf_norm_list = []
        
        for layer_name in layer_names:
            emb = embeddings[layer_name]
            probe = probes[layer_name]
            
            with torch.no_grad():
                predictions = probe(emb)
            
            rel_l2 = compute_relative_l2_error(predictions, targets).item()
            inf_norm = compute_infinity_norm_error(predictions, targets).item()
            
            rel_l2_list.append(rel_l2)
            inf_norm_list.append(inf_norm)
        
        return {'rel_l2': rel_l2_list, 'inf_norm': inf_norm_list}
    
    train_metrics = compute_metrics_on_data(train_data)
    eval_metrics = compute_metrics_on_data(eval_data)
    
    return {
        'layers_analyzed': layer_names,
        'train': train_metrics,
        'eval': eval_metrics
    }


# =============================================================================
# NCC METRICS COMPUTATION
# =============================================================================

def compute_ncc_metrics(
    probes: Dict[str, torch.nn.Linear],
    model: torch.nn.Module,
    ncc_data: Dict,
    device: torch.device,
    bins: int = 10
) -> Dict:
    """Compute NCC metrics (accuracy, compactness) on NCC data.
    
    Uses the SAME probes but different data (ncc_data).
    
    Args:
        probes: Dict of trained probes from fit_probes()
        model: Neural network model
        ncc_data: NCC evaluation data dict
        device: Device for computation
        bins: Number of bins for class creation
        
    Returns:
        Dict with 'layers_analyzed', 'layer_accuracies', 'layer_margins'
    """
    from ncc.ncc_core import (
        create_class_labels_from_regression,
        compute_class_centers,
        compute_ncc_predictions,
        compute_ncc_accuracy,
        compute_margin_metrics
    )
    
    model = model.to(device)
    model.eval()
    
    layer_names = sorted(probes.keys(), key=lambda x: int(x.split('_')[1]))
    
    # Extract NCC data
    ncc_x = ncc_data['x'].to(device)
    ncc_t = ncc_data['t'].to(device)
    ncc_targets = ncc_data['h_gt'].to(device)
    
    if ncc_targets.dim() == 1:
        ncc_targets = ncc_targets.unsqueeze(1)
    
    # Create class labels from ground truth
    class_labels, class_map, bin_info = create_class_labels_from_regression(
        ncc_targets, bins, device
    )
    num_classes = bin_info['num_classes']
    
    # Collect NCC embeddings
    handles = model.register_ncc_hooks(layer_names)
    with torch.no_grad():
        ncc_inputs = torch.cat([ncc_x, ncc_t], dim=1)
        _ = model(ncc_inputs)
    ncc_embeddings = model.activations.copy()
    model.remove_hooks()
    
    # Compute NCC metrics for each layer
    layer_accuracies = {}
    layer_margins = {}
    
    for layer_name in layer_names:
        embeddings = ncc_embeddings[layer_name]
        
        # Compute centers and predictions
        centers = compute_class_centers(embeddings, class_labels, num_classes)
        predictions = compute_ncc_predictions(embeddings, centers)
        
        # Accuracy
        accuracy = compute_ncc_accuracy(predictions, class_labels)
        layer_accuracies[layer_name] = accuracy
        
        # Margins
        margins = compute_margin_metrics(embeddings, class_labels, centers, num_classes)
        layer_margins[layer_name] = {
            'mean_margin': margins['mean_margin'],
            'std_margin': margins['std_margin'],
            'fraction_positive': margins['fraction_positive']
        }
    
    return {
        'layers_analyzed': layer_names,
        'layer_accuracies': layer_accuracies,
        'layer_margins': layer_margins,
        'num_classes': num_classes,
        'bins': bins
    }


# =============================================================================
# DERIVATIVES METRICS COMPUTATION
# =============================================================================

def compute_derivatives_metrics(
    probes: Dict[str, torch.nn.Linear],
    model: torch.nn.Module,
    eval_data: Dict,
    config: Dict,
    device: torch.device
) -> Dict:
    """Compute derivatives metrics (residuals, IC/BC errors) on eval data.
    
    Uses the SAME probes for derivative computation.
    
    Args:
        probes: Dict of trained probes from fit_probes()
        model: Neural network model
        eval_data: Evaluation data dict
        config: Configuration dict with problem info
        device: Device for computation
        
    Returns:
        Dict with per-layer residual and IC/BC metrics
    """
    from derivatives_tracker.derivatives_core import track_all_layers
    
    model = model.to(device)
    model.eval()
    
    # Extract eval data
    eval_x = eval_data['x'].to(device)
    eval_t = eval_data['t'].to(device)
    
    # Track all layers
    results = track_all_layers(
        model=model,
        probes_dict=probes,
        x=eval_x,
        t=eval_t,
        device=device,
        config=config
    )
    
    # Format results for metrics dict
    layer_names = sorted(results.keys(), key=lambda x: int(x.split('_')[1]))
    
    metrics = {
        'layers_analyzed': layer_names,
        'eval': {}
    }
    
    for layer_name in layer_names:
        layer_result = results[layer_name]
        metrics['eval'][layer_name] = {
            'residual_norm': layer_result['norms']['residual_norm'],
            'residual_inf_norm': layer_result['norms']['residual_inf_norm']
        }
    
    return metrics


# =============================================================================
# NUFFT-BASED FREQUENCY METRICS
# =============================================================================

def nufft_spectrum_2d(
    values: np.ndarray,
    x_coords: np.ndarray,
    t_coords: np.ndarray,
    domain_bounds: Dict,
    n_freq: int = 64,
    n_grid_ref: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D Fourier spectrum from scattered points using NUFFT Type 1.
    
    NUFFT Type 1: non-uniform points -> uniform frequency grid
    F(k) = Σ_j f_j * exp(-i * k · x_j)
    
    Frequency calculation matches training grid-based FFT:
    sample_spacing = domain_length / n_grid_ref (typically 100)
    
    Args:
        values: (N,) function values at scattered points
        x_coords: (N,) spatial coordinates
        t_coords: (N,) temporal coordinates
        domain_bounds: Dict with 'x_min', 'x_max', 't_min', 't_max'
        n_freq: Frequency grid resolution per dimension (output size)
        n_grid_ref: Reference grid size for frequency calculation (default 100, matches training)
        
    Returns:
        F: (n_freq, n_freq) complex Fourier coefficients
        freq_x: (n_freq,) spatial frequency values
        freq_t: (n_freq,) temporal frequency values
    """
    try:
        import finufft
    except ImportError:
        raise ImportError(
            "finufft is required for NUFFT-based frequency analysis. "
            "Install with: pip install finufft"
        )
    
    N = len(values)
    
    # Normalize coordinates to [-π, π] for NUFFT
    x_min, x_max = domain_bounds['x_min'], domain_bounds['x_max']
    t_min, t_max = domain_bounds['t_min'], domain_bounds['t_max']
    
    x_norm = 2 * np.pi * (x_coords - x_min) / (x_max - x_min) - np.pi
    t_norm = 2 * np.pi * (t_coords - t_min) / (t_max - t_min) - np.pi
    
    # Ensure contiguous arrays with correct dtype
    x_norm = np.ascontiguousarray(x_norm, dtype=np.float64)
    t_norm = np.ascontiguousarray(t_norm, dtype=np.float64)
    values_complex = np.ascontiguousarray(values, dtype=np.complex128)
    
    # NUFFT Type 1: scattered points -> uniform grid
    F = finufft.nufft2d1(x_norm, t_norm, values_complex, (n_freq, n_freq), eps=1e-6)
    
    # Compute physical frequency values using reference grid size (matches training FFT)
    # This ensures consistent frequency range with training
    dx = (x_max - x_min) / n_grid_ref
    dt = (t_max - t_min) / n_grid_ref
    
    freq_x = np.fft.fftshift(np.fft.fftfreq(n_freq, d=dx))
    freq_t = np.fft.fftshift(np.fft.fftfreq(n_freq, d=dt))
    
    # Shift to center zero frequency
    F = np.fft.fftshift(F)
    
    return F, freq_x, freq_t


def nufft_spectrum_3d(
    values: np.ndarray,
    x0_coords: np.ndarray,
    x1_coords: np.ndarray,
    t_coords: np.ndarray,
    domain_bounds: Dict,
    n_freq: int = 32,
    n_grid_ref: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute 3D Fourier spectrum from scattered points using NUFFT Type 1.
    
    For 2D spatial + time problems (like burgers2d).
    
    Frequency calculation matches training grid-based FFT:
    sample_spacing = domain_length / n_grid_ref (typically 100)
    
    Args:
        values: (N,) function values
        x0_coords, x1_coords: (N,) spatial coordinates
        t_coords: (N,) temporal coordinates
        domain_bounds: Dict with bounds for each dimension
        n_freq: Frequency grid resolution per dimension (output size)
        n_grid_ref: Reference grid size for frequency calculation (default 100, matches training)
        
    Returns:
        F: (n_freq, n_freq, n_freq) complex Fourier coefficients
        freq_x0, freq_x1, freq_t: Frequency arrays
    """
    try:
        import finufft
    except ImportError:
        raise ImportError("finufft required. Install with: pip install finufft")
    
    # Normalize coordinates to [-π, π]
    x0_min, x0_max = domain_bounds['x0_min'], domain_bounds['x0_max']
    x1_min, x1_max = domain_bounds['x1_min'], domain_bounds['x1_max']
    t_min, t_max = domain_bounds['t_min'], domain_bounds['t_max']
    
    x0_norm = 2 * np.pi * (x0_coords - x0_min) / (x0_max - x0_min) - np.pi
    x1_norm = 2 * np.pi * (x1_coords - x1_min) / (x1_max - x1_min) - np.pi
    t_norm = 2 * np.pi * (t_coords - t_min) / (t_max - t_min) - np.pi
    
    x0_norm = np.ascontiguousarray(x0_norm, dtype=np.float64)
    x1_norm = np.ascontiguousarray(x1_norm, dtype=np.float64)
    t_norm = np.ascontiguousarray(t_norm, dtype=np.float64)
    values_complex = np.ascontiguousarray(values, dtype=np.complex128)
    
    # NUFFT Type 1
    F = finufft.nufft3d1(x0_norm, x1_norm, t_norm, values_complex, 
                         (n_freq, n_freq, n_freq), eps=1e-6)
    
    # Physical frequencies using reference grid size (matches training FFT)
    dx0 = (x0_max - x0_min) / n_grid_ref
    dx1 = (x1_max - x1_min) / n_grid_ref
    dt = (t_max - t_min) / n_grid_ref
    
    freq_x0 = np.fft.fftshift(np.fft.fftfreq(n_freq, d=dx0))
    freq_x1 = np.fft.fftshift(np.fft.fftfreq(n_freq, d=dx1))
    freq_t = np.fft.fftshift(np.fft.fftfreq(n_freq, d=dt))
    
    F = np.fft.fftshift(F)
    
    return F, freq_x0, freq_x1, freq_t


def compute_radial_power_spectrum(
    power: np.ndarray,
    freqs: List[np.ndarray],
    n_bins: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute radial (isotropic) power spectrum from N-D power.
    
    Args:
        power: N-D power spectrum |F|²
        freqs: List of frequency arrays per dimension
        n_bins: Number of radial bins
        
    Returns:
        k_centers: Radial frequency bin centers
        power_radial: Summed power in each radial bin
    """
    n_dims = len(freqs)
    
    # Create radial frequency grid
    mesh_freqs = np.meshgrid(*freqs, indexing='ij')
    k_magnitude = np.sqrt(sum(f**2 for f in mesh_freqs))
    
    # Bin edges
    k_max = k_magnitude.max()
    if k_max == 0:
        k_max = 1.0
    k_edges = np.linspace(0, k_max, n_bins + 1)
    k_centers = (k_edges[:-1] + k_edges[1:]) / 2
    
    # Average power in each bin (matches training computation)
    power_radial = np.zeros(n_bins)
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (k_magnitude >= k_edges[i]) & (k_magnitude <= k_edges[i + 1])
        else:
            mask = (k_magnitude >= k_edges[i]) & (k_magnitude < k_edges[i + 1])
        if mask.sum() > 0:
            power_radial[i] = power[mask].mean()  # Use mean() to match training
    
    return k_centers, power_radial


def compute_frequency_metrics_fft(
    probes: Dict[str, torch.nn.Linear],
    model: torch.nn.Module,
    freq_grid_data: Dict,
    device: torch.device,
    problem_config: Dict,
    n_bins: int = 30
) -> Dict:
    """Compute frequency metrics using standard FFT on frequency grid data.
    
    Uses frequency_grid.pt (same as training) for consistent frequency analysis.
    Matches training-time computation exactly: uses grid_shape from data file,
    standard FFT (not NUFFT), and n_bins=30.
    
    Args:
        probes: Dict of trained probes from fit_probes()
        model: Neural network model
        freq_grid_data: Frequency grid data dict (from frequency_grid.pt)
        device: Device for computation
        problem_config: Problem configuration with domain bounds
        n_bins: Number of radial frequency bins (default 30, matches training)
        
    Returns:
        Dict with:
            - k_bins: radial frequency bin centers
            - gt_radial_power: ground truth power spectrum
            - error_matrix: (n_layers, n_bins) relative spectral error per layer
            - layers_analyzed: list of layer names
    """
    from frequency_tracker.frequency_core import (
        compute_frequency_spectrum
    )
    
    model = model.to(device)
    model.eval()
    
    layer_names = sorted(probes.keys(), key=lambda x: int(x.split('_')[1]))
    n_layers = len(layer_names)
    
    # Extract grid data - grid_shape is stored in the file (e.g., (128, 128) if n_freq_grid=128)
    x_grid_tensor = freq_grid_data['x_grid'].to(device)
    h_gt_grid = freq_grid_data['h_gt_grid'].detach().cpu().numpy()
    grid_shape = tuple(freq_grid_data['grid_shape'])  # Use actual grid_shape from file
    
    # Compute sample spacings exactly as in training (frequency_core.py:249-264)
    n_grid = grid_shape[0]  # Assuming uniform grid (matches training)
    sample_spacings = []
    
    spatial_domain = problem_config['spatial_domain']
    temporal_domain = problem_config['temporal_domain']
    
    # Spatial dimensions (exactly as training: frequency_core.py:257-260)
    for dom in spatial_domain:
        domain_length = dom[1] - dom[0]
        sample_spacings.append(domain_length / n_grid)
    
    # Time dimension (exactly as training: frequency_core.py:262-264)
    time_length = temporal_domain[1] - temporal_domain[0]
    sample_spacings.append(time_length / n_grid)
    
    # Debug: print domain info
    print(f"    Domain debug: spatial_domain={spatial_domain}, temporal_domain={temporal_domain}")
    print(f"    Domain debug: grid_shape={grid_shape}, n_grid={n_grid}")
    print(f"    Domain debug: sample_spacings={sample_spacings}")
    
    # Handle multi-output ground truth
    if h_gt_grid.ndim > 1 and h_gt_grid.shape[-1] > 1:
        targets_flat = h_gt_grid.reshape(-1, h_gt_grid.shape[-1])
        targets_scalar = np.sqrt(np.sum(targets_flat ** 2, axis=1))
    else:
        targets_scalar = h_gt_grid.flatten()
    
    # Compute ground truth spectrum using standard FFT (matches training exactly)
    gt_spectrum = compute_frequency_spectrum(targets_scalar, grid_shape, sample_spacings)
    
    # Compute radial spectrum manually (matching frequency_runner.py exactly)
    gt_power = gt_spectrum['power']
    gt_freqs = gt_spectrum['freqs']
    n_dims = len(gt_freqs)
    
    # Handle multi-output: average over output dimension first
    if gt_power.ndim > n_dims:
        gt_power_avg = gt_power.mean(axis=-1)
    else:
        gt_power_avg = gt_power
    
    # Create radial frequency grid and compute k_max ONCE (matching training)
    mesh_freqs = np.meshgrid(*gt_freqs, indexing='ij')
    k_magnitude = np.sqrt(sum(f**2 for f in mesh_freqs))
    k_max = k_magnitude.max()
    if k_max == 0:
        k_max = 1.0
    
    # Compute k_max exactly as in training (no filtering, use full range)
    # This matches frequency_runner.py:185-189 and frequency_plotting.py:108-112
    
    # Debug: print frequency info
    print(f"    Frequency debug: grid_shape={grid_shape}, n_grid={n_grid}")
    print(f"    Frequency debug: sample_spacings={sample_spacings}")
    for i, freq_arr in enumerate(gt_freqs):
        print(f"    Frequency debug: dim {i}: min={freq_arr.min():.3f}, max={freq_arr.max():.3f}, shape={freq_arr.shape}")
    print(f"    Frequency debug: k_max={k_max:.3f}")
    
    k_bin_edges = np.linspace(0, k_max, n_bins + 1)
    k_bin_centers = (k_bin_edges[:-1] + k_bin_edges[1:]) / 2
    
    # Compute GT radial power using .mean() (matching training)
    gt_radial = np.zeros(n_bins)
    for bin_idx in range(n_bins):
        if bin_idx == n_bins - 1:
            mask = (k_magnitude >= k_bin_edges[bin_idx]) & (k_magnitude <= k_bin_edges[bin_idx + 1])
        else:
            mask = (k_magnitude >= k_bin_edges[bin_idx]) & (k_magnitude < k_bin_edges[bin_idx + 1])
        if mask.sum() > 0:
            gt_radial[bin_idx] = gt_power_avg[mask].mean()  # Use .mean() to match training
    
    gt_radial_safe = np.where(gt_radial > 1e-15, gt_radial, 1e-15)
    
    # Get grid embeddings for all layers
    handles = model.register_ncc_hooks(layer_names)
    with torch.no_grad():
        _ = model(x_grid_tensor)
    grid_embeddings = model.activations.copy()
    model.remove_hooks()
    
    # Compute spectral error for each layer using SAME bin edges from GT
    error_matrix = np.zeros((n_layers, n_bins))
    
    for layer_idx, layer_name in enumerate(layer_names):
        probe = probes[layer_name]
        embeddings = grid_embeddings[layer_name]
        
        with torch.no_grad():
            h_pred = probe(embeddings).cpu().numpy()
        
        # Handle multi-output
        if h_pred.ndim > 1 and h_pred.shape[1] > 1:
            h_pred_scalar = np.sqrt(np.sum(h_pred ** 2, axis=1))
        else:
            h_pred_scalar = h_pred.flatten()
        
        # Compute error spectrum using standard FFT
        error = targets_scalar - h_pred_scalar
        error_spectrum = compute_frequency_spectrum(error, grid_shape, sample_spacings)
        
        # Compute radial leftover power using SAME k_bin_edges from GT
        leftover_power = error_spectrum['power']
        leftover_freqs = error_spectrum['freqs']
        
        # Handle multi-output
        if leftover_power.ndim > len(leftover_freqs):
            leftover_avg = leftover_power.mean(axis=-1)
        else:
            leftover_avg = leftover_power
        
        # Create radial frequency grid for error (frequencies should match GT, but compute to be safe)
        mesh_freqs_l = np.meshgrid(*leftover_freqs, indexing='ij')
        k_mag_l = np.sqrt(sum(f**2 for f in mesh_freqs_l))
        
        # Use SAME bin edges from GT (not recompute k_max)
        leftover_radial = np.zeros(n_bins)
        for bin_idx in range(n_bins):
            if bin_idx == n_bins - 1:
                mask = (k_mag_l >= k_bin_edges[bin_idx]) & (k_mag_l <= k_bin_edges[bin_idx + 1])
            else:
                mask = (k_mag_l >= k_bin_edges[bin_idx]) & (k_mag_l < k_bin_edges[bin_idx + 1])
            if mask.sum() > 0:
                leftover_radial[bin_idx] = leftover_avg[mask].mean()  # Use .mean() to match training
        
        # Relative error: |FFT(error)|² / |FFT(gt)|²
        relative_error = leftover_radial / gt_radial_safe
        error_matrix[layer_idx] = relative_error
    
    # Compute model's direct output frequency error (for cross-PDE comparison)
    with torch.no_grad():
        h_pred_grid = model(x_grid_tensor).cpu().numpy()
    
    # Handle multi-output
    if h_pred_grid.ndim > 1 and h_pred_grid.shape[1] > 1:
        h_pred_scalar = np.sqrt(np.sum(h_pred_grid ** 2, axis=1))
    else:
        h_pred_scalar = h_pred_grid.flatten()
    
    # Compute model error spectrum using standard FFT
    model_error = targets_scalar - h_pred_scalar
    model_error_spectrum = compute_frequency_spectrum(model_error, grid_shape, sample_spacings)
    
    # Compute radial model error power using SAME k_bin_edges from GT
    model_leftover_power = model_error_spectrum['power']
    model_leftover_freqs = model_error_spectrum['freqs']
    
    # Handle multi-output
    if model_leftover_power.ndim > len(model_leftover_freqs):
        model_leftover_avg = model_leftover_power.mean(axis=-1)
    else:
        model_leftover_avg = model_leftover_power
    
    # Create radial frequency grid for model error
    mesh_freqs_model = np.meshgrid(*model_leftover_freqs, indexing='ij')
    k_mag_model = np.sqrt(sum(f**2 for f in mesh_freqs_model))
    
    # Use SAME bin edges from GT
    model_error_radial = np.zeros(n_bins)
    for bin_idx in range(n_bins):
        if bin_idx == n_bins - 1:
            mask = (k_mag_model >= k_bin_edges[bin_idx]) & (k_mag_model <= k_bin_edges[bin_idx + 1])
        else:
            mask = (k_mag_model >= k_bin_edges[bin_idx]) & (k_mag_model < k_bin_edges[bin_idx + 1])
        if mask.sum() > 0:
            model_error_radial[bin_idx] = model_leftover_avg[mask].mean()
    
    # Relative error: |FFT(model_error)|² / |FFT(gt)|²
    model_relative_error = model_error_radial / gt_radial_safe
    
    # Format to match expected structure
    return {
        'layers_analyzed': layer_names,
        'spectral_efficiency': {
            'k_radial_bins': k_bin_centers.tolist(),  # Use manually computed k_bin_centers
            'error_matrix': error_matrix.tolist(),  # Probe-based errors per layer
            'model_error': model_relative_error.tolist(),  # Direct model output error
            'gt_radial_power': gt_radial.tolist()
        },
        'final_layer_leftover_ratio': float(error_matrix[-1].mean()),
        'ground_truth_total_power': float(gt_radial.sum())
    }


# =============================================================================
# UNIFIED METRIC COMPUTATION
# =============================================================================

def compute_all_metrics_consistently(
    model_name: str,
    run_dir: Path,
    device: torch.device,
    bins: int = None
) -> Dict[str, Any]:
    """Compute all metrics for a model using consistent probes.
    
    This is the main entry point for analysis. It:
    1. Loads the model checkpoint (fails if not found)
    2. Fits probes ONCE on training data
    3. Computes ALL metrics using those same probes
    
    Args:
        model_name: Model folder name
        run_dir: Path to the timestamped run directory
        device: Device for computation
        bins: Number of bins for NCC classification. If None, will read from
              saved ncc_metrics.json (preferred) or config_used.yaml (fallback)
        
    Returns:
        Dict with all computed metrics
        
    Raises:
        FileNotFoundError: If checkpoint or datasets not found
        ValueError: If model cannot be loaded
    """
    print(f"  Computing metrics consistently for {model_name}...")
    
    # Extract problem from model name
    problem = extract_problem_from_name(model_name)
    if problem is None:
        raise ValueError(f"Cannot extract problem from model name: {model_name}")
    
    # Try to read problem config from config_used.yaml (matches training exactly)
    problem_config = None
    config_file = run_dir / "config_used.yaml"
    if config_file.exists():
        import yaml
        with open(config_file, 'r') as f:
            saved_config = yaml.safe_load(f)
            if problem in saved_config:
                problem_config = saved_config[problem].copy()
                print(f"    Read {problem} config from config_used.yaml")
    
    # Fallback to hardcoded config if not found
    if problem_config is None:
        problem_config = get_problem_config(problem)
        print(f"    Using default {problem} config (config_used.yaml not found)")
    
    # Find checkpoint
    checkpoint_path = find_model_checkpoint(run_dir, problem, model_name)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found for {model_name}")
    
    print(f"    Loading checkpoint: {checkpoint_path}")
    
    # Parse architecture
    architecture, activation = parse_architecture_from_name(model_name)
    if len(architecture) < 2:
        raise ValueError(f"Could not parse architecture from {model_name}")
    
    # Build config
    config = {
        'architecture': architecture,
        'activation': activation,
        'cuda': torch.cuda.is_available(),
        'problem': problem,
        problem: problem_config,
    }
    
    # Load model
    model = load_model_from_checkpoint(checkpoint_path, architecture, activation, config)
    if model is None:
        raise ValueError(f"Failed to load model from {checkpoint_path}")
    
    # Try to read bins from saved ncc_metrics.json (preferred - matches what was actually used)
    if bins is None:
        ncc_metrics_file = run_dir / "ncc_plots" / "ncc_metrics.json"
        if ncc_metrics_file.exists():
            import json
            with open(ncc_metrics_file, 'r') as f:
                saved_ncc_metrics = json.load(f)
                bins = saved_ncc_metrics.get('bins', None)
            if bins is not None:
                print(f"    Read bins={bins} from saved ncc_metrics.json")
        
        # Fallback: try config_used.yaml
        if bins is None:
            config_file = run_dir / "config_used.yaml"
            if config_file.exists():
                import yaml
                with open(config_file, 'r') as f:
                    saved_config = yaml.safe_load(f)
                    bins = saved_config.get('bins', None)
                if bins is not None:
                    print(f"    Read bins={bins} from config_used.yaml")
        
        # Final fallback: default to 5
        if bins is None:
            bins = 5
            print(f"    Using default bins={bins} (not found in saved files)")
    
    # Load datasets
    print(f"    Loading datasets for {problem}...")
    train_data, eval_data, ncc_data, freq_grid_data = load_datasets(problem)
    
    # Convert freq_grid_data to expected format for probe/derivatives metrics
    # freq_grid_data has 'x_grid' (concatenated coords) and 'h_gt_grid'
    # Need to split into 'x', 't', 'h_gt' for compatibility
    x_grid = freq_grid_data['x_grid']
    spatial_dim = x_grid.shape[1] - 1  # Last dim is time
    
    freq_grid_compat = {
        'x': x_grid[:, :spatial_dim] if spatial_dim > 1 else x_grid[:, 0:1],
        't': x_grid[:, -1:],  # Last column is time
        'h_gt': freq_grid_data['h_gt_grid'].reshape(-1, freq_grid_data['h_gt_grid'].shape[-1]) if freq_grid_data['h_gt_grid'].ndim > 1 else freq_grid_data['h_gt_grid'].reshape(-1, 1)
    }
    
    # Step 1: Fit probes ONCE
    print(f"    Fitting probes...")
    probes = fit_probes(model, train_data, device)
    
    # Step 2: Compute all metrics using same probes
    # Use freq_grid_compat for probe/derivatives/frequency metrics
    # NCC uses its own ncc_data
    print(f"    Computing probe metrics (on frequency grid)...")
    probe_metrics = compute_probe_metrics(probes, model, freq_grid_compat, freq_grid_compat, device)
    
    print(f"    Computing NCC metrics (on NCC data, bins={bins})...")
    ncc_metrics = compute_ncc_metrics(probes, model, ncc_data, device, bins=bins)
    
    print(f"    Computing derivatives metrics (on frequency grid)...")
    derivatives_metrics = compute_derivatives_metrics(probes, model, freq_grid_compat, config, device)
    
    print(f"    Computing frequency metrics (on frequency grid, standard FFT matching training)...")
    # Grid shape is already in freq_grid_data, use it directly in the function
    frequency_metrics = compute_frequency_metrics_fft(
        probes, model, freq_grid_data, device, problem_config, 
        n_bins=30  # Match training-time default
    )
    
    # Compute model L2/Linf on freq_grid for consistency
    print(f"    Computing model L2/Linf on frequency grid...")
    model.eval()
    with torch.no_grad():
        h_pred_grid = model(freq_grid_data['x_grid'].to(device)).cpu()
    h_gt_grid = freq_grid_data['h_gt_grid']
    
    # Handle shape differences
    if h_gt_grid.ndim == 1:
        h_gt_grid = h_gt_grid.reshape(-1, 1)
    if h_pred_grid.ndim == 1:
        h_pred_grid = h_pred_grid.reshape(-1, 1)
    
    freq_grid_rel_l2 = compute_relative_l2_error(h_pred_grid, h_gt_grid).item()
    freq_grid_linf = compute_infinity_norm_error(h_pred_grid, h_gt_grid).item()
    
    return {
        'probe_metrics': probe_metrics,
        'ncc_metrics': ncc_metrics,
        'derivatives_metrics': derivatives_metrics,
        'frequency_metrics': frequency_metrics,
        'probes': probes,  # Keep probes for potential further analysis
        'layers_analyzed': probe_metrics['layers_analyzed'],
        'freq_grid_rel_l2': freq_grid_rel_l2,
        'freq_grid_linf': freq_grid_linf
    }

