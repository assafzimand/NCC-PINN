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


def load_datasets(problem: str) -> Tuple[Dict, Dict, Dict, Dict]:
    """Load train, eval, NCC, and frequency grid datasets for a problem.
    
    Args:
        problem: Problem name
        
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
    n_bins: int = 32
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
    
    # Sum power in each bin
    power_radial = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (k_magnitude >= k_edges[i]) & (k_magnitude < k_edges[i + 1])
        if mask.sum() > 0:
            power_radial[i] = power[mask].sum()
    
    return k_centers, power_radial


def compute_frequency_metrics_nufft(
    probes: Dict[str, torch.nn.Linear],
    model: torch.nn.Module,
    freq_grid_data: Dict,
    device: torch.device,
    problem_config: Dict,
    n_freq: int = 64,
    n_bins: int = 32,
    n_grid_ref: int = 100
) -> Dict:
    """Compute frequency metrics using NUFFT on frequency grid data.
    
    Uses frequency_grid.pt (same as training) for consistent frequency analysis.
    Frequency calculation matches training: sample_spacing = domain_length / n_grid_ref
    
    Args:
        probes: Dict of trained probes from fit_probes()
        model: Neural network model
        freq_grid_data: Frequency grid data dict (from frequency_grid.pt)
        device: Device for computation
        problem_config: Problem configuration with domain bounds
        n_freq: NUFFT frequency grid resolution (output size)
        n_bins: Number of radial frequency bins
        n_grid_ref: Reference grid size for frequency calculation (default 100, matches training)
        
    Returns:
        Dict with:
            - k_bins: radial frequency bin centers
            - gt_radial_power: ground truth power spectrum
            - error_matrix: (n_layers, n_bins) relative spectral error per layer
            - layers_analyzed: list of layer names
    """
    model = model.to(device)
    model.eval()
    
    layer_names = sorted(probes.keys(), key=lambda x: int(x.split('_')[1]))
    n_layers = len(layer_names)
    
    # Extract grid data (detach to avoid requires_grad issues)
    # freq_grid_data has x_grid (already concatenated coords) and h_gt_grid
    x_grid_tensor = freq_grid_data['x_grid'].to(device)
    h_gt_grid = freq_grid_data['h_gt_grid'].detach().cpu().numpy()
    grid_shape = tuple(freq_grid_data['grid_shape'])
    
    # Flatten grid for coordinate extraction
    # x_grid has shape (N_grid_total, n_dims) where n_dims = spatial_dim + 1 (time)
    spatial_dim = x_grid_tensor.shape[1] - 1  # Last dimension is time
    
    if spatial_dim == 1:
        eval_x = x_grid_tensor[:, 0].detach().cpu().numpy()
        eval_t = x_grid_tensor[:, 1].detach().cpu().numpy()
    elif spatial_dim == 2:
        eval_x = x_grid_tensor[:, :2].detach().cpu().numpy()  # (N, 2)
        eval_t = x_grid_tensor[:, 2].detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported spatial dimension: {spatial_dim}")
    
    # Handle multi-output ground truth
    if h_gt_grid.ndim > 1 and h_gt_grid.shape[-1] > 1:
        # Flatten spatial dimensions, keep output dim
        targets_flat = h_gt_grid.reshape(-1, h_gt_grid.shape[-1])
        targets_scalar = np.sqrt(np.sum(targets_flat ** 2, axis=1))
    else:
        eval_targets = h_gt_grid
        targets_scalar = h_gt_grid.flatten()
    
    # Compute domain bounds from problem config
    spatial_domain = problem_config['spatial_domain']
    temporal_domain = problem_config['temporal_domain']
    
    # Compute ground truth spectrum
    if spatial_dim == 1:
        eval_x_flat = eval_x.reshape(-1)
        eval_t_flat = eval_t.reshape(-1)
        
        domain_bounds = {
            'x_min': spatial_domain[0][0], 'x_max': spatial_domain[0][1],
            't_min': temporal_domain[0], 't_max': temporal_domain[1]
        }
        
        F_gt, freq_x, freq_t = nufft_spectrum_2d(
            targets_scalar, eval_x_flat, eval_t_flat, domain_bounds, n_freq, n_grid_ref
        )
        power_gt = np.abs(F_gt) ** 2
        k_bins, gt_radial = compute_radial_power_spectrum(power_gt, [freq_x, freq_t], n_bins)
        
    elif spatial_dim == 2:
        x0 = eval_x[:, 0]
        x1 = eval_x[:, 1]
        eval_t_flat = eval_t.reshape(-1)
        
        domain_bounds = {
            'x0_min': spatial_domain[0][0], 'x0_max': spatial_domain[0][1],
            'x1_min': spatial_domain[1][0], 'x1_max': spatial_domain[1][1],
            't_min': temporal_domain[0], 't_max': temporal_domain[1]
        }
        
        # Use smaller n_freq for 3D to manage memory
        n_freq_3d = min(n_freq, 32)
        
        F_gt, freq_x0, freq_x1, freq_t = nufft_spectrum_3d(
            targets_scalar, x0, x1, eval_t_flat, domain_bounds, n_freq_3d, n_grid_ref
        )
        power_gt = np.abs(F_gt) ** 2
        k_bins, gt_radial = compute_radial_power_spectrum(
            power_gt, [freq_x0, freq_x1, freq_t], n_bins
        )
    else:
        raise ValueError(f"Unsupported spatial dimension: {spatial_dim}")
    
    gt_radial_safe = np.where(gt_radial > 1e-15, gt_radial, 1e-15)
    
    # Get grid embeddings for all layers using x_grid
    handles = model.register_ncc_hooks(layer_names)
    with torch.no_grad():
        _ = model(x_grid_tensor)
    grid_embeddings = model.activations.copy()
    model.remove_hooks()
    
    # Compute spectral error for each layer
    error_matrix = np.zeros((n_layers, n_bins))
    
    for layer_idx, layer_name in enumerate(layer_names):
        probe = probes[layer_name]
        grid_emb = grid_embeddings[layer_name]
        
        with torch.no_grad():
            predictions = probe(grid_emb).detach().cpu().numpy()
        
        # Handle multi-output
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            preds_scalar = np.sqrt(np.sum(predictions ** 2, axis=1))
        else:
            preds_scalar = predictions.flatten()
        
        # Compute error spectrum (leftover = gt - pred)
        error = targets_scalar - preds_scalar
        
        if spatial_dim == 1:
            F_error, _, _ = nufft_spectrum_2d(error, eval_x_flat, eval_t_flat, domain_bounds, n_freq, n_grid_ref)
            power_error = np.abs(F_error) ** 2
            _, error_radial = compute_radial_power_spectrum(power_error, [freq_x, freq_t], n_bins)
        else:
            F_error, _, _, _ = nufft_spectrum_3d(error, x0, x1, eval_t_flat, domain_bounds, n_freq_3d, n_grid_ref)
            power_error = np.abs(F_error) ** 2
            _, error_radial = compute_radial_power_spectrum(
                power_error, [freq_x0, freq_x1, freq_t], n_bins
            )
        
        # Store relative error
        error_matrix[layer_idx] = error_radial / gt_radial_safe
    
    return {
        'layers_analyzed': layer_names,
        'k_bins': k_bins.tolist(),
        'gt_radial_power': gt_radial.tolist(),
        'spectral_efficiency': {
            'k_radial_bins': k_bins.tolist(),
            'error_matrix': error_matrix.tolist(),
            'gt_radial_power': gt_radial.tolist()
        }
    }


# =============================================================================
# UNIFIED METRIC COMPUTATION
# =============================================================================

def compute_all_metrics_consistently(
    model_name: str,
    run_dir: Path,
    device: torch.device,
    ncc_bins: int = 10,
    n_freq: int = 64
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
        ncc_bins: Number of bins for NCC classification
        n_freq: NUFFT frequency grid resolution
        
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
    
    problem_config = get_problem_config(problem)
    
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
    # Use freq_grid_compat for probe/derivatives for consistency with frequency analysis
    print(f"    Computing probe metrics (on frequency grid)...")
    probe_metrics = compute_probe_metrics(probes, model, train_data, freq_grid_compat, device)
    
    print(f"    Computing NCC metrics (on NCC data)...")
    ncc_metrics = compute_ncc_metrics(probes, model, ncc_data, device, bins=ncc_bins)
    
    print(f"    Computing derivatives metrics (on frequency grid)...")
    derivatives_metrics = compute_derivatives_metrics(probes, model, freq_grid_compat, config, device)
    
    print(f"    Computing frequency metrics (on frequency grid, grid-based FFT)...")
    # Extract grid shape to determine n_grid_ref for frequency calculation
    grid_shape = tuple(freq_grid_data.get('grid_shape', [100, 100]))
    n_grid_ref = grid_shape[0]  # Typically 100 for all problems
    
    frequency_metrics = compute_frequency_metrics_nufft(
        probes, model, freq_grid_data, device, problem_config, 
        n_freq=n_freq, n_grid_ref=n_grid_ref
    )
    
    return {
        'probe_metrics': probe_metrics,
        'ncc_metrics': ncc_metrics,
        'derivatives_metrics': derivatives_metrics,
        'frequency_metrics': frequency_metrics,
        'probes': probes,  # Keep probes for potential further analysis
        'layers_analyzed': probe_metrics['layers_analyzed']
    }

