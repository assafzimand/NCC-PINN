"""Combined Frequency-Monotonicity Analysis Script.

Correlates non-monotonic violations with frequency domain behavior:
1. Shows which frequencies improved/degraded at violation layers
2. Overlays GT frequency markers on change heatmaps to find spatial correlations
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import sys
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from models.fc_model import FCNet
from probes.probe_core import train_linear_probe, compute_probe_predictions
from frequency_tracker.frequency_core import compute_frequency_spectrum, compute_radial_spectrum


# =============================================================================
# CONFIGURATION
# =============================================================================

DEGRADATION_THRESHOLD = 0.1  # 10% relative degradation threshold

# Marker styles for frequency bins (5 bins: very low -> very high)
FREQ_MARKERS = [
    {'marker': 'o', 'label': 'Very Low (0-20%)', 'color': '#2ecc71'},
    {'marker': 's', 'label': 'Low (20-40%)', 'color': '#3498db'},
    {'marker': '^', 'label': 'Medium (40-60%)', 'color': '#f1c40f'},
    {'marker': 'D', 'label': 'High (60-80%)', 'color': '#e67e22'},
    {'marker': '*', 'label': 'Very High (80-100%)', 'color': '#e74c3c'},
]

# Metrics that have change heatmaps for overlay analysis
METRICS_WITH_HEATMAPS = [
    'probe_rel_l2',
    'probe_linf',
    'ncc_accuracy',
    'residual_l2',
    'residual_linf',
]

# Full metrics config (subset of analyze_capacity_experiment.py)
METRICS_CONFIG = {
    'ncc_accuracy': {
        'direction': 'increase',
        'source': 'ncc_metrics',
        'display_name': 'NCC Accuracy',
        'extract_fn': lambda m: [m['layer_accuracies'][layer] for layer in m['layers_analyzed']],
    },
    'probe_rel_l2': {
        'direction': 'decrease',
        'source': 'probe_metrics',
        'display_name': 'Probe Rel-L2 (eval)',
        'extract_fn': lambda m: m['eval']['rel_l2'],
    },
    'probe_linf': {
        'direction': 'decrease',
        'source': 'probe_metrics',
        'display_name': 'Probe L-inf (eval)',
        'extract_fn': lambda m: m['eval']['inf_norm'],
    },
    'residual_l2': {
        'direction': 'decrease',
        'source': 'derivatives_metrics',
        'display_name': 'Residual L2 (eval)',
        'extract_fn': lambda m: [
            m['eval'][layer]['residual_norm'] 
            for layer in m['layers_analyzed']
        ],
    },
    'residual_linf': {
        'direction': 'decrease',
        'source': 'derivatives_metrics',
        'display_name': 'Residual L-inf (eval)',
        'extract_fn': lambda m: [
            m['eval'][layer]['residual_inf_norm'] 
            for layer in m['layers_analyzed']
        ],
    },
}


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def find_model_run_dir(model_dir: Path) -> Optional[Path]:
    """Find the most recent timestamped run directory for a model."""
    run_dirs = [d for d in model_dir.iterdir() 
                if d.is_dir() and re.match(r'\d{8}_\d{6}', d.name)]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda x: x.stat().st_mtime)


def load_json_file(path: Path) -> Optional[Dict]:
    """Load JSON file if it exists."""
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None


def load_all_model_data(experiment_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load all relevant data for analysis.
    
    Returns:
        Dict mapping model_name -> {
            'model_name': str,
            'run_dir': Path,
            'ncc_metrics': dict,
            'probe_metrics': dict,
            'derivatives_metrics': dict,
            'frequency_metrics': dict,
            'layers_analyzed': list
        }
    """
    models_data = {}
    experiment_path = Path(experiment_path)
    
    for model_dir in experiment_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # Skip non-model directories
        if model_name.startswith('.') or model_name.endswith('.csv') or model_name.endswith('.yaml'):
            continue
        
        run_dir = find_model_run_dir(model_dir)
        if run_dir is None:
            continue
        
        # Load all metrics files
        ncc_metrics = load_json_file(run_dir / "ncc_plots" / "ncc_metrics.json")
        probe_metrics = load_json_file(run_dir / "probe_plots" / "probe_metrics.json")
        derivatives_metrics = load_json_file(run_dir / "derivatives_plots" / "derivatives_metrics.json")
        frequency_metrics = load_json_file(run_dir / "frequency_plots" / "frequency_metrics.json")
        
        # Skip models without frequency metrics (required for this analysis)
        if frequency_metrics is None:
            print(f"  Warning: {model_name} has no frequency_metrics.json, skipping")
            continue
        
        layers_analyzed = frequency_metrics.get('layers_analyzed', [])
        
        models_data[model_name] = {
            'model_name': model_name,
            'run_dir': run_dir,
            'ncc_metrics': ncc_metrics,
            'probe_metrics': probe_metrics,
            'derivatives_metrics': derivatives_metrics,
            'frequency_metrics': frequency_metrics,
            'layers_analyzed': layers_analyzed,
        }
    
    return models_data


# =============================================================================
# NON-MONOTONIC DETECTION
# =============================================================================

def check_monotonicity(
    values: List[float],
    direction: str,
    threshold: float = DEGRADATION_THRESHOLD
) -> List[Dict[str, Any]]:
    """Check if values are monotonic, return violation info.
    
    Returns:
        List of violations: {layer_num, prev_value, current_value, degradation_rel}
    """
    violations = []
    
    for i in range(1, len(values)):
        prev_val = values[i - 1]
        curr_val = values[i]
        
        if np.isnan(prev_val) or np.isnan(curr_val):
            continue
        
        is_violation = False
        if direction == 'increase':
            if curr_val < prev_val:
                is_violation = True
                degradation_abs = prev_val - curr_val
        else:
            if curr_val > prev_val:
                is_violation = True
                degradation_abs = curr_val - prev_val
        
        if is_violation:
            if abs(prev_val) > 1e-10:
                degradation_rel = degradation_abs / abs(prev_val)
            else:
                degradation_rel = float('inf') if degradation_abs > 0 else 0
            
            if degradation_rel > threshold:
                violations.append({
                    'layer_num': i + 1,  # 1-based layer number
                    'layer_idx': i,      # 0-based index (position in spectral_efficiency)
                    'prev_value': prev_val,
                    'current_value': curr_val,
                    'degradation_rel': degradation_rel,
                })
    
    return violations


def detect_violations_per_metric(
    models_data: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, List[Dict]]]:
    """Detect violations for each metric and model.
    
    Returns:
        Dict[metric_name][model_name] -> list of violations
    """
    violations = {metric: {} for metric in METRICS_CONFIG.keys()}
    
    for model_name, data in models_data.items():
        for metric_name, config in METRICS_CONFIG.items():
            source = config['source']
            direction = config['direction']
            extract_fn = config['extract_fn']
            
            metrics = data.get(source)
            if metrics is None:
                continue
            
            try:
                values = extract_fn(metrics)
            except (KeyError, TypeError, IndexError):
                continue
            
            if len(values) < 2:
                continue
            
            model_violations = check_monotonicity(values, direction)
            if model_violations:
                violations[metric_name][model_name] = model_violations
    
    return violations


# =============================================================================
# FREQUENCY ERROR REDUCTION FUNCTIONS
# =============================================================================

def compute_spectral_errors_from_predictions(
    predictions: Dict[str, np.ndarray],
    targets: np.ndarray,
    coordinates: np.ndarray,
    problem_config: Dict,
    n_bins: int = 32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute spectral errors from probe predictions on eval data.
    
    This computes the frequency-domain error for each layer using FFT,
    allowing us to see which frequencies have errors at each layer.
    
    Args:
        predictions: Dict {layer_name: predictions (N, d_o)}
        targets: Ground truth (N, d_o) 
        coordinates: (N, d) spatial+temporal coordinates
        problem_config: Problem config with domain info
        n_bins: Number of radial frequency bins
        
    Returns:
        k_bins: Radial frequency bin centers
        error_matrix: (n_layers, n_bins) spectral error for each layer
        gt_radial: (n_bins,) ground truth radial power
    """
    layer_names = sorted(predictions.keys(), key=lambda x: int(x.split('_')[1]))
    n_layers = len(layer_names)
    
    # Determine grid shape from coordinates
    # For irregular data, we need to interpolate to a regular grid for FFT
    n_dims = coordinates.shape[1]
    
    # Create a regular grid for FFT
    n_grid = 64  # Grid resolution for FFT
    
    # Get domain bounds from coordinates
    domain_mins = coordinates.min(axis=0)
    domain_maxs = coordinates.max(axis=0)
    
    # Create regular grid
    grids = [np.linspace(domain_mins[i], domain_maxs[i], n_grid) for i in range(n_dims)]
    mesh = np.meshgrid(*grids, indexing='ij')
    grid_points = np.stack([m.flatten() for m in mesh], axis=1)  # (N_grid, n_dims)
    
    # Compute sample spacings for physical frequencies
    sample_spacings = [(domain_maxs[i] - domain_mins[i]) / n_grid for i in range(n_dims)]
    grid_shape = tuple([n_grid] * n_dims)
    
    # Interpolate ground truth to regular grid
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    
    if targets.ndim == 1:
        targets_flat = targets
    else:
        # For multi-output, use magnitude
        if targets.shape[1] > 1:
            targets_flat = np.sqrt(np.sum(targets ** 2, axis=1))
        else:
            targets_flat = targets.flatten()
    
    # Create interpolator
    try:
        interp_gt = LinearNDInterpolator(coordinates, targets_flat)
        gt_on_grid = interp_gt(grid_points)
        # Fill NaN with nearest neighbor
        nan_mask = np.isnan(gt_on_grid)
        if nan_mask.any():
            interp_nearest = NearestNDInterpolator(coordinates, targets_flat)
            gt_on_grid[nan_mask] = interp_nearest(grid_points[nan_mask])
    except Exception:
        # Fallback to nearest neighbor
        interp_nearest = NearestNDInterpolator(coordinates, targets_flat)
        gt_on_grid = interp_nearest(grid_points)
    
    # Compute ground truth spectrum
    gt_spectrum = compute_frequency_spectrum(gt_on_grid, grid_shape, sample_spacings)
    k_bins, gt_radial = compute_radial_spectrum(gt_spectrum['power'], gt_spectrum['freqs'], n_bins)
    gt_radial_safe = np.where(gt_radial > 1e-15, gt_radial, 1e-15)
    
    # Compute spectral error for each layer
    error_matrix = np.zeros((n_layers, n_bins))
    
    for layer_idx, layer_name in enumerate(layer_names):
        preds = predictions[layer_name]
        
        # Handle multi-output
        if preds.ndim > 1 and preds.shape[1] > 1:
            preds_flat = np.sqrt(np.sum(preds ** 2, axis=1))
        else:
            preds_flat = preds.flatten()
        
        # Interpolate predictions to regular grid
        try:
            interp_pred = LinearNDInterpolator(coordinates, preds_flat)
            pred_on_grid = interp_pred(grid_points)
            nan_mask = np.isnan(pred_on_grid)
            if nan_mask.any():
                interp_nearest = NearestNDInterpolator(coordinates, preds_flat)
                pred_on_grid[nan_mask] = interp_nearest(grid_points[nan_mask])
        except Exception:
            interp_nearest = NearestNDInterpolator(coordinates, preds_flat)
            pred_on_grid = interp_nearest(grid_points)
        
        # Compute error spectrum (leftover = gt - pred)
        error = gt_on_grid - pred_on_grid
        error_spectrum = compute_frequency_spectrum(error, grid_shape, sample_spacings)
        _, error_radial = compute_radial_spectrum(error_spectrum['power'], error_spectrum['freqs'], n_bins)
        
        # Store relative error
        error_matrix[layer_idx] = error_radial / gt_radial_safe
    
    return k_bins, error_matrix, gt_radial


def get_frequency_reduction_at_layer(
    freq_metrics: Dict,
    layer_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Get frequency error reduction E_{i-1} - E_i at a specific layer transition.
    
    Args:
        freq_metrics: frequency_metrics.json data OR computed spectral data
        layer_idx: 0-based index of the layer (error at layer_idx vs layer_idx-1)
        
    Returns:
        (k_bins, error_reduction) where positive = improvement
    """
    spectral_eff = freq_metrics.get('spectral_efficiency', {})
    k_bins = np.array(spectral_eff.get('k_radial_bins', []))
    error_matrix = np.array(spectral_eff.get('error_matrix', []))
    
    if len(error_matrix) == 0 or layer_idx <= 0 or layer_idx >= len(error_matrix):
        return np.array([]), np.array([])
    
    # Error reduction: prev_error - current_error (positive = improvement)
    prev_error = error_matrix[layer_idx - 1]
    curr_error = error_matrix[layer_idx]
    error_reduction = prev_error - curr_error
    
    return k_bins, error_reduction


def get_frequency_reduction_from_matrix(
    k_bins: np.ndarray,
    error_matrix: np.ndarray,
    layer_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Get frequency error reduction from pre-computed error matrix.
    
    Args:
        k_bins: Radial frequency bin centers
        error_matrix: (n_layers, n_bins) spectral error matrix
        layer_idx: 0-based index of the layer
        
    Returns:
        (k_bins, error_reduction) where positive = improvement
    """
    if len(error_matrix) == 0 or layer_idx <= 0 or layer_idx >= len(error_matrix):
        return np.array([]), np.array([])
    
    prev_error = error_matrix[layer_idx - 1]
    curr_error = error_matrix[layer_idx]
    error_reduction = prev_error - curr_error
    
    return k_bins, error_reduction


def plot_freq_reduction_from_eval(
    model_name: str,
    k_bins: np.ndarray,
    error_matrix: np.ndarray,
    violation_layers: List[int],
    output_dir: Path
):
    """Generate per-model frequency reduction plot using eval data computed spectra.
    
    Args:
        model_name: Name of the model
        k_bins: Radial frequency bin centers
        error_matrix: (n_layers, n_bins) spectral error matrix computed from eval data
        violation_layers: List of layer indices (0-based) where violations occurred
        output_dir: Directory to save plot
    """
    if not violation_layers or len(k_bins) == 0:
        return
    
    n_violations = len(violation_layers)
    fig, axes = plt.subplots(1, n_violations, figsize=(5 * n_violations, 4), squeeze=False)
    
    for idx, layer_idx in enumerate(violation_layers):
        ax = axes[0, idx]
        
        k_bins_layer, error_reduction = get_frequency_reduction_from_matrix(
            k_bins, error_matrix, layer_idx
        )
        
        if len(error_reduction) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Plot error reduction
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Fill positive (improvement) in green, negative (degradation) in red
        ax.fill_between(k_bins_layer, 0, error_reduction, 
                       where=(error_reduction >= 0), color='#2ecc71', alpha=0.5, label='Improved')
        ax.fill_between(k_bins_layer, 0, error_reduction,
                       where=(error_reduction < 0), color='#e74c3c', alpha=0.5, label='Degraded')
        ax.plot(k_bins_layer, error_reduction, 'k-', linewidth=1.5)
        
        ax.set_xlabel('Radial Frequency |k| (Hz)')
        ax.set_ylabel('Error Reduction (E_{i-1} - E_i)')
        # Use 1-based layer numbering for display
        ax.set_title(f'Layer {layer_idx} → {layer_idx + 1}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{model_name}\nFrequency Error Reduction at Violation Layers (from eval data)', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace('/', '_').replace('\\', '_')
    save_path = output_dir / f'{safe_name}_freq_reduction.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_freq_reduction_per_model(
    model_name: str,
    freq_metrics: Dict,
    violation_layers: List[int],
    output_dir: Path
):
    """Generate per-model frequency reduction plot for violation layers.
    
    DEPRECATED: This uses frequency_metrics.json from grid data.
    Use plot_freq_reduction_from_eval for accurate results on eval data.
    
    Args:
        model_name: Name of the model
        freq_metrics: frequency_metrics.json data
        violation_layers: List of layer indices (0-based) where violations occurred
        output_dir: Directory to save plot
    """
    if not violation_layers:
        return
    
    spectral_eff = freq_metrics.get('spectral_efficiency', {})
    k_bins = np.array(spectral_eff.get('k_radial_bins', []))
    gt_radial = np.array(spectral_eff.get('gt_radial_power', []))
    
    if len(k_bins) == 0:
        return
    
    n_violations = len(violation_layers)
    fig, axes = plt.subplots(1, n_violations, figsize=(5 * n_violations, 4), squeeze=False)
    
    for idx, layer_idx in enumerate(violation_layers):
        ax = axes[0, idx]
        
        k_bins_layer, error_reduction = get_frequency_reduction_at_layer(freq_metrics, layer_idx)
        
        if len(error_reduction) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Plot error reduction
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Fill positive (improvement) in green, negative (degradation) in red
        ax.fill_between(k_bins_layer, 0, error_reduction, 
                       where=(error_reduction >= 0), color='#2ecc71', alpha=0.5, label='Improved')
        ax.fill_between(k_bins_layer, 0, error_reduction,
                       where=(error_reduction < 0), color='#e74c3c', alpha=0.5, label='Degraded')
        ax.plot(k_bins_layer, error_reduction, 'k-', linewidth=1.5)
        
        ax.set_xlabel('Radial Frequency |k| (Hz)')
        ax.set_ylabel('Error Reduction (E_{i-1} - E_i)')
        ax.set_title(f'Layer {layer_idx} → {layer_idx + 1}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{model_name}\nFrequency Error Reduction at Violation Layers', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace('/', '_').replace('\\', '_')
    save_path = output_dir / f'{safe_name}_freq_reduction.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_aggregated_freq_reduction(
    models_data: Dict[str, Dict[str, Any]],
    violations: Dict[str, List[Dict]],
    output_dir: Path
):
    """Plot aggregated frequency reduction across all violating models.
    
    Args:
        models_data: All model data
        violations: {model_name: [violation dicts]} for this metric
        output_dir: Directory to save plot
    """
    if not violations:
        return
    
    # Collect all error reductions
    all_reductions = []
    k_bins_ref = None
    
    for model_name, model_violations in violations.items():
        freq_metrics = models_data[model_name].get('frequency_metrics')
        if freq_metrics is None:
            continue
        
        for v in model_violations:
            layer_idx = v['layer_idx']
            k_bins, error_reduction = get_frequency_reduction_at_layer(freq_metrics, layer_idx)
            
            if len(error_reduction) > 0:
                if k_bins_ref is None:
                    k_bins_ref = k_bins
                all_reductions.append(error_reduction)
    
    if not all_reductions or k_bins_ref is None:
        return
    
    # Compute statistics
    all_reductions = np.array(all_reductions)
    mean_reduction = np.mean(all_reductions, axis=0)
    std_reduction = np.std(all_reductions, axis=0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Mean with confidence band
    ax.fill_between(k_bins_ref, mean_reduction - std_reduction, mean_reduction + std_reduction,
                   alpha=0.3, color='#3498db', label='±1 std')
    ax.plot(k_bins_ref, mean_reduction, 'b-', linewidth=2, label='Mean')
    
    # Color the zero crossing regions
    ax.fill_between(k_bins_ref, 0, mean_reduction,
                   where=(mean_reduction >= 0), color='#2ecc71', alpha=0.2)
    ax.fill_between(k_bins_ref, 0, mean_reduction,
                   where=(mean_reduction < 0), color='#e74c3c', alpha=0.2)
    
    ax.set_xlabel('Radial Frequency |k| (Hz)', fontsize=11)
    ax.set_ylabel('Mean Error Reduction (E_{i-1} - E_i)', fontsize=11)
    ax.set_title(f'Aggregated Frequency Error Reduction\n({len(all_reductions)} violations from {len(violations)} models)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add annotation for problem frequencies
    if np.any(mean_reduction < 0):
        problem_freqs = k_bins_ref[mean_reduction < 0]
        if len(problem_freqs) > 0:
            ax.axvspan(problem_freqs.min(), problem_freqs.max(), 
                      alpha=0.1, color='red', label='Problem range')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / 'aggregated_freq_reduction.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_aggregated_freq_reduction_v2(
    models_data: Dict[str, Dict[str, Any]],
    violations: Dict[str, List[Dict]],
    eval_spectral_data: Dict[str, Dict],
    output_dir: Path
):
    """Plot aggregated frequency reduction using eval-computed spectral data.
    
    This version prioritizes eval-computed spectral data for accuracy,
    falling back to frequency_metrics.json only if eval data unavailable.
    
    Args:
        models_data: All model data
        violations: {model_name: [violation dicts]} for this metric
        eval_spectral_data: {model_name: {'k_bins': array, 'error_matrix': array}}
        output_dir: Directory to save plot
    """
    if not violations:
        return
    
    # Collect all error reductions
    all_reductions = []
    k_bins_ref = None
    
    for model_name, model_violations in violations.items():
        # Try eval-computed spectral data first
        if model_name in eval_spectral_data:
            k_bins = eval_spectral_data[model_name]['k_bins']
            error_matrix = eval_spectral_data[model_name]['error_matrix']
            
            for v in model_violations:
                layer_idx = v['layer_idx']
                if layer_idx > 0 and layer_idx < len(error_matrix):
                    k_bins_layer, error_reduction = get_frequency_reduction_from_matrix(
                        k_bins, error_matrix, layer_idx
                    )
                    if len(error_reduction) > 0:
                        if k_bins_ref is None:
                            k_bins_ref = k_bins_layer
                        all_reductions.append(error_reduction)
        else:
            # Fallback to frequency_metrics.json
            freq_metrics = models_data[model_name].get('frequency_metrics')
            if freq_metrics is None:
                continue
            
            for v in model_violations:
                layer_idx = v['layer_idx']
                k_bins, error_reduction = get_frequency_reduction_at_layer(freq_metrics, layer_idx)
                
                if len(error_reduction) > 0:
                    if k_bins_ref is None:
                        k_bins_ref = k_bins
                    all_reductions.append(error_reduction)
    
    if not all_reductions or k_bins_ref is None:
        return
    
    # Compute statistics
    all_reductions = np.array(all_reductions)
    mean_reduction = np.mean(all_reductions, axis=0)
    std_reduction = np.std(all_reductions, axis=0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Mean with confidence band
    ax.fill_between(k_bins_ref, mean_reduction - std_reduction, mean_reduction + std_reduction,
                   alpha=0.3, color='#3498db', label='±1 std')
    ax.plot(k_bins_ref, mean_reduction, 'b-', linewidth=2, label='Mean')
    
    # Color the zero crossing regions
    ax.fill_between(k_bins_ref, 0, mean_reduction,
                   where=(mean_reduction >= 0), color='#2ecc71', alpha=0.2)
    ax.fill_between(k_bins_ref, 0, mean_reduction,
                   where=(mean_reduction < 0), color='#e74c3c', alpha=0.2)
    
    ax.set_xlabel('Radial Frequency |k| (Hz)', fontsize=11)
    ax.set_ylabel('Mean Error Reduction (E_{i-1} - E_i)', fontsize=11)
    n_from_eval = len(eval_spectral_data)
    ax.set_title(f'Aggregated Frequency Error Reduction (from eval data)\n'
                f'({len(all_reductions)} violations from {len(violations)} models, '
                f'{n_from_eval} computed from eval)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add annotation for problem frequencies
    if np.any(mean_reduction < 0):
        problem_freqs = k_bins_ref[mean_reduction < 0]
        if len(problem_freqs) > 0:
            ax.axvspan(problem_freqs.min(), problem_freqs.max(), 
                      alpha=0.1, color='red', label='Problem range')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / 'aggregated_freq_reduction.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# SPATIAL-FREQUENCY OVERLAY FUNCTIONS
# =============================================================================

def compute_local_dominant_frequency(
    h_gt: np.ndarray,
    coordinates: np.ndarray,
    window_size: int = 8,
    n_bins: int = 5
) -> np.ndarray:
    """Compute dominant frequency bin at each spatial point using gradient magnitude.
    
    Uses gradient magnitude as a proxy for local high-frequency content.
    This is faster than windowed FFT and works well for identifying
    regions with high vs low frequency content.
    
    Args:
        h_gt: Ground truth values (N,) or (N, output_dim)
        coordinates: (N, d) spatial+temporal coordinates
        window_size: Not used in gradient method, kept for API compatibility
        n_bins: Number of frequency bins
        
    Returns:
        (N,) array of frequency bin indices (0 = very low, n_bins-1 = very high)
    """
    from scipy.spatial import KDTree
    
    if h_gt.ndim > 1:
        h_gt = np.linalg.norm(h_gt, axis=1)  # Use magnitude for multi-output
    
    N = len(h_gt)
    n_dims = coordinates.shape[1]
    
    # Build KD-tree for neighbor queries
    tree = KDTree(coordinates)
    
    # Estimate local gradient magnitude at each point
    gradient_magnitudes = np.zeros(N)
    k_neighbors = min(2 * n_dims + 1, N)  # Enough neighbors to estimate gradient
    
    for i in range(N):
        # Query nearest neighbors
        distances, indices = tree.query(coordinates[i], k=k_neighbors)
        
        if len(indices) < 2:
            continue
        
        # Estimate gradient as max value difference / distance
        neighbor_values = h_gt[indices[1:]]  # Exclude self
        neighbor_dists = distances[1:]
        
        # Avoid division by zero
        valid_mask = neighbor_dists > 1e-10
        if not np.any(valid_mask):
            continue
        
        value_diffs = np.abs(neighbor_values - h_gt[i])
        gradients = value_diffs[valid_mask] / neighbor_dists[valid_mask]
        gradient_magnitudes[i] = np.max(gradients)
    
    # Bin gradient magnitudes into frequency categories
    # Higher gradient = higher frequency content
    percentiles = np.percentile(gradient_magnitudes[gradient_magnitudes > 0], 
                                np.linspace(0, 100, n_bins + 1))
    
    freq_bins = np.digitize(gradient_magnitudes, percentiles[1:-1])
    freq_bins = np.clip(freq_bins, 0, n_bins - 1)
    
    return freq_bins


def plot_spatial_freq_overlay(
    change_ratios: np.ndarray,
    freq_bins: np.ndarray,
    coordinates: np.ndarray,
    layer_transition: str,
    metric_name: str,
    output_path: Path,
    subsample_factor: int = 15
):
    """Create overlay plot: change heatmap with frequency markers.
    
    Args:
        change_ratios: (N,) error change ratios (< 1 = improved, > 1 = degraded)
        freq_bins: (N,) frequency bin index for each point
        coordinates: (N, d) spatial+temporal coordinates
        layer_transition: String like "Layer 2 → 3"
        metric_name: Name of the metric
        output_path: Path to save the plot
        subsample_factor: Subsample markers to avoid clutter
    """
    from scipy.interpolate import griddata
    
    n_dims = coordinates.shape[1]
    
    # Create figure
    if n_dims == 2:  # 1D spatial + time
        fig, ax = plt.subplots(figsize=(12, 8))
        x_coord = coordinates[:, 0]
        t_coord = coordinates[:, 1]
        
        # Create regular grid for continuous heatmap
        x_min, x_max = x_coord.min(), x_coord.max()
        t_min, t_max = t_coord.min(), t_coord.max()
        grid_x, grid_t = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(t_min, t_max, 200)
        )
        
        # Interpolate change_ratios to grid
        points = np.column_stack([x_coord, t_coord])
        grid_change = griddata(points, change_ratios, (grid_x, grid_t),
                              method='linear', fill_value=1.0)
        
        # Background: continuous change heatmap (matching probe_error_change_heatmaps)
        norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=3.0)
        im = ax.contourf(grid_x, grid_t, grid_change, levels=50,
                        cmap='RdYlGn_r', norm=norm, alpha=1.0)
        
        # Overlay: frequency markers (subsampled) - black symbols with thin outlines
        subsample_idx = np.arange(0, len(coordinates), subsample_factor)
        
        for bin_idx, marker_style in enumerate(FREQ_MARKERS):
            mask = (freq_bins[subsample_idx] == bin_idx)
            if np.any(mask):
                ax.scatter(x_coord[subsample_idx][mask], t_coord[subsample_idx][mask],
                          marker=marker_style['marker'], facecolors='none',
                          edgecolors='black', linewidths=0.6, s=25, alpha=0.8,
                          label=marker_style['label'], zorder=10)
        
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Error Change Ratio\n(<1: improved, >1: degraded)')
        
    elif n_dims == 3:  # 2D spatial + time - use time slices
        x0_coord = coordinates[:, 0]
        x1_coord = coordinates[:, 1]
        t_coord = coordinates[:, 2]
        
        # Time slices for 2D spatial visualization
        TIME_SLICES_2D = [0.0, 0.5, 1.0, 1.5, 2.0]
        n_slices = len(TIME_SLICES_2D)
        
        fig, axes = plt.subplots(1, n_slices, figsize=(4 * n_slices, 4))
        if n_slices == 1:
            axes = [axes]
        
        norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=3.0)
        
        # Create regular grid for interpolation
        n_grid = 50
        x0_grid_lin = np.linspace(x0_coord.min(), x0_coord.max(), n_grid)
        x1_grid_lin = np.linspace(x1_coord.min(), x1_coord.max(), n_grid)
        X0_grid, X1_grid = np.meshgrid(x0_grid_lin, x1_grid_lin)
        
        for slice_idx, t_slice in enumerate(TIME_SLICES_2D):
            ax = axes[slice_idx]
            
            # Find points near this time slice
            t_tolerance = 0.1
            mask = np.abs(t_coord - t_slice) < t_tolerance
            
            if not np.any(mask):
                ax.text(0.5, 0.5, f'No data at t={t_slice}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue
            
            x0_slice = x0_coord[mask]
            x1_slice = x1_coord[mask]
            change_slice = change_ratios[mask]
            freq_bins_slice = freq_bins[mask]
            
            # Interpolate change ratios to grid
            change_grid = griddata((x0_slice, x1_slice), change_slice, 
                                  (X0_grid, X1_grid), method='linear', fill_value=1.0)
            
            # Background: continuous heatmap
            im = ax.contourf(X0_grid, X1_grid, change_grid, levels=50,
                           cmap='RdYlGn_r', norm=norm, alpha=1.0)
            
            # Overlay: frequency markers (subsampled)
            subsample_idx = np.arange(0, len(x0_slice), subsample_factor)
            
            for bin_idx, marker_style in enumerate(FREQ_MARKERS):
                bin_mask = (freq_bins_slice[subsample_idx] == bin_idx)
                if np.any(bin_mask):
                    ax.scatter(x0_slice[subsample_idx][bin_mask],
                              x1_slice[subsample_idx][bin_mask],
                              marker=marker_style['marker'], facecolors='none',
                              edgecolors='black', linewidths=0.6, s=20, alpha=0.8,
                              label=marker_style['label'] if slice_idx == 0 else '',
                              zorder=10)
            
            ax.set_xlabel('x0')
            ax.set_ylabel('x1')
            ax.set_title(f't = {t_slice}')
            ax.set_aspect('equal')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.05)
        cbar.set_label('Error Change Ratio\n(<1: improved, >1: degraded)')
        
        # Add legend on first subplot
        if n_slices > 0:
            axes[0].legend(loc='upper left', fontsize=8, title='GT Frequency')
    
    else:
        print(f"  Warning: Unsupported dimension {n_dims} for overlay plot")
        return
    
    ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=9, title='GT Frequency')
    ax.set_title(f'{metric_name}: {layer_transition}\nChange Heatmap with GT Frequency Overlay',
                fontsize=12)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def extract_problem_from_name(model_name: str) -> Optional[str]:
    """Extract problem name from model folder name."""
    model_lower = model_name.lower()
    if 'schrodinger' in model_lower:
        return 'schrodinger'
    elif 'wave1d' in model_lower:
        return 'wave1d'
    elif 'burgers1d' in model_lower:
        return 'burgers1d'
    elif 'burgers2d' in model_lower:
        return 'burgers2d'
    return None


def find_model_checkpoint(run_dir: Path, problem: str, model_name: str) -> Optional[Path]:
    """Find the model checkpoint file.
    
    Checks several possible locations based on the project structure.
    """
    # Get model_dir (parent of run_dir, which is the timestamped directory)
    model_dir = run_dir.parent
    
    # Possible checkpoint locations (in order of priority):
    # 1. Inside model folder: model_dir/checkpoints/model_name/best_model.pt
    # 2. Global checkpoints folder: checkpoints/problem/model_name/best_model.pt
    # 3. Inside run directory: run_dir/checkpoints/best_model.pt
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


def load_model_from_checkpoint(checkpoint_path: Path, architecture: List[int], 
                               activation: str, config: Dict) -> Optional[torch.nn.Module]:
    """Load a model from checkpoint file."""
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
        print(f"  Warning: Failed to load model: {e}")
        return None


def compute_per_point_probe_predictions(
    model: torch.nn.Module,
    train_data: Dict,
    eval_data: Dict,
    device: torch.device
) -> Dict[str, np.ndarray]:
    """Compute per-point probe predictions for all layers.
    
    Returns:
        Dictionary {layer_name: predictions (N, output_dim)}
    """
    model = model.to(device)
    model.eval()
    
    # Get hidden layer names
    all_layers = model.get_layer_names()
    hidden_layers = all_layers[:-1]  # Exclude output layer
    
    # Extract training data
    train_x = train_data['x'].to(device)
    train_t = train_data['t'].to(device)
    train_targets = train_data['h_gt'].to(device)
    
    # Extract eval data
    eval_x = eval_data['x'].to(device)
    eval_t = eval_data['t'].to(device)
    eval_targets = eval_data['h_gt'].to(device)
    
    # Collect training embeddings
    handles = model.register_ncc_hooks(hidden_layers)
    with torch.no_grad():
        train_inputs = torch.cat([train_x, train_t], dim=1)
        _ = model(train_inputs)
    train_embeddings = model.activations.copy()
    model.remove_hooks()
    
    # Collect eval embeddings
    _ = model.register_ncc_hooks(hidden_layers)
    with torch.no_grad():
        eval_inputs = torch.cat([eval_x, eval_t], dim=1)
        _ = model(eval_inputs)
    eval_embeddings = model.activations.copy()
    model.remove_hooks()
    
    # Train probes and get predictions
    predictions = {}
    for layer_name in hidden_layers:
        train_emb = train_embeddings[layer_name]
        eval_emb = eval_embeddings[layer_name]
        
        # Train probe on training data
        probe = train_linear_probe(train_emb, train_targets)
        
        # Get predictions on eval data
        eval_preds = compute_probe_predictions(probe, eval_emb)
        predictions[layer_name] = eval_preds.cpu().numpy()
    
    return predictions, eval_targets.cpu().numpy()


def generate_overlay_for_model(
    model_name: str,
    model_data: Dict[str, Any],
    violation_layers: List[int],
    metric_name: str,
    output_dir: Path
):
    """Generate spatial-frequency overlay plots for a model's violation layers.
    
    This loads the model checkpoint and computes per-point probe predictions
    to generate accurate change ratio heatmaps.
    """
    run_dir = model_data['run_dir']
    
    # Extract problem from model name
    problem = extract_problem_from_name(model_name)
    if problem is None:
        print(f"  Warning: Cannot extract problem from {model_name}, skipping overlay")
        return
    
    # Load evaluation and training data
    eval_data_path = Path('datasets') / problem / 'eval_data.pt'
    train_data_path = Path('datasets') / problem / 'training_data.pt'
    
    if not eval_data_path.exists() or not train_data_path.exists():
        print(f"  Warning: Dataset not found for {problem}, skipping overlay")
        return
    
    eval_data = torch.load(eval_data_path, weights_only=False)
    train_data = torch.load(train_data_path, weights_only=False)
    
    eval_x = eval_data['x'].numpy()
    eval_t = eval_data['t'].numpy()
    h_gt = eval_data['h_gt'].numpy()
    
    # Combine coordinates
    if eval_x.ndim == 1:
        eval_x_coords = eval_x.reshape(-1, 1)
    else:
        eval_x_coords = eval_x
    if eval_t.ndim == 1:
        eval_t_coords = eval_t.reshape(-1, 1)
    else:
        eval_t_coords = eval_t
    coordinates = np.hstack([eval_x_coords, eval_t_coords])
    
    # Compute frequency bins for GT
    freq_bins = compute_local_dominant_frequency(h_gt, coordinates)
    
    # Find model checkpoint
    checkpoint_path = find_model_checkpoint(run_dir, problem, model_name)
    if checkpoint_path is None:
        print(f"  Warning: No checkpoint found for {model_name}, using aggregate errors")
        # Fallback to aggregate errors (uniform heatmap)
        _generate_overlay_with_aggregate_errors(
            model_name, model_data, violation_layers, metric_name, 
            output_dir, coordinates, freq_bins, h_gt
        )
        return
    
    print(f"  Loading model from {checkpoint_path}")
    
    # Parse architecture from model name
    parts = model_name.split('-')
    architecture = []
    activation = 'tanh'
    for part in parts[1:]:
        if part.isdigit():
            architecture.append(int(part))
        elif part in ['tanh', 'relu', 'sin', 'gelu']:
            activation = part
            break
    
    if len(architecture) < 2:
        print(f"  Warning: Could not parse architecture from {model_name}")
        return
    
    # Determine spatial_dim and output_dim based on problem
    PROBLEM_CONFIGS = {
        'wave1d': {'spatial_dim': 1, 'output_dim': 1},
        'burgers1d': {'spatial_dim': 1, 'output_dim': 1},
        'burgers2d': {'spatial_dim': 2, 'output_dim': 1},
        'schrodinger': {'spatial_dim': 1, 'output_dim': 2},
    }
    
    problem_config = PROBLEM_CONFIGS.get(problem, {'spatial_dim': 1, 'output_dim': 1})
    
    # Create config matching FCNet requirements
    config = {
        'architecture': architecture,
        'activation': activation,
        'cuda': torch.cuda.is_available(),
        'problem': problem,
        problem: problem_config,  # FCNet requires config[problem]['spatial_dim'] etc.
    }
    
    # Load model
    model = load_model_from_checkpoint(checkpoint_path, architecture, activation, config)
    if model is None:
        print(f"  Warning: Failed to load model, using aggregate errors")
        _generate_overlay_with_aggregate_errors(
            model_name, model_data, violation_layers, metric_name,
            output_dir, coordinates, freq_bins, h_gt
        )
        return
    
    # Compute per-point probe predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Computing per-point probe predictions on {device}...")
    
    try:
        predictions, targets = compute_per_point_probe_predictions(
            model, train_data, eval_data, device
        )
    except Exception as e:
        print(f"  Warning: Failed to compute predictions: {e}")
        _generate_overlay_with_aggregate_errors(
            model_name, model_data, violation_layers, metric_name,
            output_dir, coordinates, freq_bins, h_gt
        )
        return
    
    # Get sorted layer names
    layer_names = sorted(predictions.keys(), key=lambda x: int(x.split('_')[1]))
    
    # For each violation layer, compute per-point change ratios and plot
    for layer_idx in violation_layers:
        if layer_idx <= 0 or layer_idx >= len(layer_names):
            continue
        
        prev_layer = layer_names[layer_idx - 1]
        curr_layer = layer_names[layer_idx]
        
        prev_preds = predictions[prev_layer]
        curr_preds = predictions[curr_layer]
        
        # Compute per-point errors (using magnitude for multi-output)
        if prev_preds.ndim > 1 and prev_preds.shape[1] > 1:
            prev_error = np.sqrt(np.sum((prev_preds - targets) ** 2, axis=1))
            curr_error = np.sqrt(np.sum((curr_preds - targets) ** 2, axis=1))
        else:
            prev_error = np.abs(prev_preds.flatten() - targets.flatten())
            curr_error = np.abs(curr_preds.flatten() - targets.flatten())
        
        # Compute per-point change ratio: curr_error / prev_error
        eps = 1e-10
        change_ratios = curr_error / (prev_error + eps)
        change_ratios = np.clip(change_ratios, 0, 5)  # Cap extreme values
        
        layer_transition = f"Layer {layer_idx} → {layer_idx + 1}"
        output_path = output_dir / f"{model_name.replace('/', '_')}_overlay_layer{layer_idx + 1}.png"
        
        plot_spatial_freq_overlay(
            change_ratios=change_ratios,
            freq_bins=freq_bins,
            coordinates=coordinates,
            layer_transition=layer_transition,
            metric_name=metric_name,
            output_path=output_path
        )


def _generate_overlay_with_aggregate_errors(
    model_name: str,
    model_data: Dict[str, Any],
    violation_layers: List[int],
    metric_name: str,
    output_dir: Path,
    coordinates: np.ndarray,
    freq_bins: np.ndarray,
    h_gt: np.ndarray
):
    """Fallback: Generate overlay using aggregate errors (uniform heatmap).
    
    Used when model checkpoint is not available.
    """
    probe_metrics = model_data.get('probe_metrics')
    if probe_metrics is None:
        print(f"  Warning: No probe metrics for {model_name}, skipping overlay")
        return
    
    eval_errors = probe_metrics.get('eval', {}).get('rel_l2', [])
    if len(eval_errors) < 2:
        print(f"  Warning: Not enough probe layers for {model_name}, skipping overlay")
        return
    
    for layer_idx in violation_layers:
        if layer_idx <= 0 or layer_idx >= len(eval_errors):
            continue
        
        prev_error = eval_errors[layer_idx - 1]
        curr_error = eval_errors[layer_idx]
        
        change_ratio = curr_error / (prev_error + 1e-10)
        change_ratios = np.full(len(coordinates), change_ratio)
        
        layer_transition = f"Layer {layer_idx} → {layer_idx + 1} (aggregate)"
        output_path = output_dir / f"{model_name.replace('/', '_')}_overlay_layer{layer_idx + 1}.png"
        
        plot_spatial_freq_overlay(
            change_ratios=change_ratios,
            freq_bins=freq_bins,
            coordinates=coordinates,
            layer_transition=layer_transition,
            metric_name=metric_name,
            output_path=output_path
        )


def generate_freq_reduction_from_eval(
    model_name: str,
    model_data: Dict[str, Any],
    violation_layers: List[int],
    output_dir: Path
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Generate frequency reduction plot using spectral errors from eval data.
    
    This ensures the frequency analysis is on the same data where violations were detected.
    
    Args:
        model_name: Name of the model
        model_data: Model data dict including run_dir
        violation_layers: List of layer indices where violations occurred
        output_dir: Directory to save the plot
        
    Returns:
        (k_bins, error_matrix) if successful, None otherwise
    """
    run_dir = model_data['run_dir']
    
    # Extract problem from model name
    problem = extract_problem_from_name(model_name)
    if problem is None:
        print(f"    Warning: Cannot extract problem from {model_name}, falling back to JSON metrics")
        return None
    
    # Load datasets
    eval_data_path = Path('datasets') / problem / 'eval_data.pt'
    train_data_path = Path('datasets') / problem / 'training_data.pt'
    
    if not eval_data_path.exists() or not train_data_path.exists():
        print(f"    Warning: Dataset not found for {problem}, falling back to JSON metrics")
        return None
    
    eval_data = torch.load(eval_data_path, weights_only=False)
    train_data = torch.load(train_data_path, weights_only=False)
    
    eval_x = eval_data['x'].numpy()
    eval_t = eval_data['t'].numpy()
    h_gt = eval_data['h_gt'].numpy()
    
    # Combine coordinates
    if eval_x.ndim == 1:
        eval_x_coords = eval_x.reshape(-1, 1)
    else:
        eval_x_coords = eval_x
    if eval_t.ndim == 1:
        eval_t_coords = eval_t.reshape(-1, 1)
    else:
        eval_t_coords = eval_t
    coordinates = np.hstack([eval_x_coords, eval_t_coords])
    
    # Find model checkpoint
    checkpoint_path = find_model_checkpoint(run_dir, problem, model_name)
    if checkpoint_path is None:
        print(f"    Warning: No checkpoint found for {model_name}, falling back to JSON metrics")
        return None
    
    # Parse architecture from model name
    parts = model_name.split('-')
    architecture = []
    activation = 'tanh'
    for part in parts[1:]:
        if part.isdigit():
            architecture.append(int(part))
        elif part in ['tanh', 'relu', 'sin', 'gelu']:
            activation = part
            break
    
    if len(architecture) < 2:
        print(f"    Warning: Could not parse architecture from {model_name}")
        return None
    
    # Problem-specific config
    PROBLEM_CONFIGS = {
        'wave1d': {'spatial_dim': 1, 'output_dim': 1},
        'burgers1d': {'spatial_dim': 1, 'output_dim': 1},
        'burgers2d': {'spatial_dim': 2, 'output_dim': 1},
        'schrodinger': {'spatial_dim': 1, 'output_dim': 2},
    }
    
    problem_config = PROBLEM_CONFIGS.get(problem, {'spatial_dim': 1, 'output_dim': 1})
    
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
        print(f"    Warning: Failed to load model, falling back to JSON metrics")
        return None
    
    # Compute per-point probe predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        predictions, targets = compute_per_point_probe_predictions(
            model, train_data, eval_data, device
        )
    except Exception as e:
        print(f"    Warning: Failed to compute predictions: {e}")
        return None
    
    # Compute spectral errors from predictions on eval data
    print(f"    Computing spectral errors from eval data...")
    try:
        k_bins, error_matrix, gt_radial = compute_spectral_errors_from_predictions(
            predictions=predictions,
            targets=targets,
            coordinates=coordinates,
            problem_config=problem_config
        )
    except Exception as e:
        print(f"    Warning: Failed to compute spectral errors: {e}")
        return None
    
    # Generate the plot
    plot_freq_reduction_from_eval(
        model_name=model_name,
        k_bins=k_bins,
        error_matrix=error_matrix,
        violation_layers=violation_layers,
        output_dir=output_dir
    )
    
    return k_bins, error_matrix


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def parse_model_architecture(model_name: str) -> Tuple[int, int]:
    """Extract num_layers and approximate weight count from model name.
    
    Returns:
        (num_layers, num_parameters)
    """
    parts = model_name.split('-')
    architecture = []
    for part in parts:
        if part.isdigit():
            architecture.append(int(part))
    
    if len(architecture) < 2:
        return 0, 0
    
    num_layers = len(architecture) - 2  # Exclude input and output
    
    # Calculate parameters
    total_params = 0
    for i in range(len(architecture) - 1):
        total_params += architecture[i] * architecture[i + 1] + architecture[i + 1]
    
    return num_layers, total_params


def generate_summary_table(
    all_violations: Dict[str, Dict[str, List[Dict]]],
    models_data: Dict[str, Dict[str, Any]],
    output_dir: Path
):
    """Create summary table of frequency-violation correlations.
    
    Shows per metric:
    - Number of models with violations
    - Total number of violations
    - Layers/weights of violating models
    - Average frequency where metric decreased
    """
    metrics_list = list(METRICS_CONFIG.keys())
    
    summary_data = []
    for metric_name in metrics_list:
        violations = all_violations.get(metric_name, {})
        n_models = len(violations)
        total_violations = sum(len(v) for v in violations.values())
        
        # Collect layers and weights of violating models
        model_info_list = []
        mean_neg_freqs = []
        
        for model_name, model_violations in violations.items():
            # Get architecture info
            num_layers, num_params = parse_model_architecture(model_name)
            if num_layers > 0:
                # Format as "Xlyr/Yk" 
                weight_str = f"{num_params // 1000}k" if num_params >= 1000 else str(num_params)
                model_info_list.append(f"{num_layers}lyr/{weight_str}")
            
            # Compute average decreasing frequency
            freq_metrics = models_data.get(model_name, {}).get('frequency_metrics')
            if freq_metrics is None:
                continue
            
            for v in model_violations:
                k_bins, error_reduction = get_frequency_reduction_at_layer(freq_metrics, v['layer_idx'])
                if len(error_reduction) > 0 and len(k_bins) > 0:
                    # Find frequencies where error increased (negative reduction)
                    neg_mask = error_reduction < 0
                    if np.any(neg_mask):
                        neg_freqs = k_bins[neg_mask]
                        mean_neg_freqs.append(np.mean(neg_freqs))
        
        avg_decreasing_freq = f"{np.mean(mean_neg_freqs):.2f}" if mean_neg_freqs else "N/A"
        models_str = ", ".join(model_info_list) if model_info_list else "N/A"
        
        summary_data.append({
            'Metric': METRICS_CONFIG[metric_name]['display_name'],
            '# Models': n_models,
            '# Violations': total_violations,
            'Violating Models': models_str[:40] + ('...' if len(models_str) > 40 else ''),
            'Avg Decreasing Freq': avg_decreasing_freq,
        })
    
    # Create table plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    if summary_data:
        columns = list(summary_data[0].keys())
        cell_text = [[str(row[col]) for col in columns] for row in summary_data]
        
        table = ax.table(
            cellText=cell_text,
            colLabels=columns,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style header
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#2c3e50')
                cell.set_text_props(color='white', fontweight='bold')
    
    ax.set_title('Frequency-Monotonicity Analysis Summary', fontsize=14, fontweight='bold', pad=20)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / 'summary.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Summary table saved to {save_path}")


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def get_latest_analysis_index(experiment_name: str, analysis_base_dir: Path) -> Optional[str]:
    """Get the latest existing analysis index for an experiment."""
    exp_dir = analysis_base_dir / experiment_name
    if not exp_dir.exists():
        return None
    
    existing_indices = []
    for item in exp_dir.iterdir():
        if item.is_dir() and item.name.startswith("analysis_"):
            try:
                idx = int(item.name.split("_")[1])
                existing_indices.append(idx)
            except (ValueError, IndexError):
                pass
    
    if not existing_indices:
        return None
    
    return f"analysis_{max(existing_indices)}"


def main(experiment_dir: str):
    """Run complete frequency-monotonicity analysis."""
    experiment_path = Path(experiment_dir)
    
    if not experiment_path.exists():
        print(f"Error: Experiment directory not found: {experiment_path}")
        return
    
    experiment_name = experiment_path.name
    
    print("=" * 70)
    print("Combined Frequency-Monotonicity Analysis")
    print("=" * 70)
    print(f"Experiment: {experiment_name}")
    
    # Determine output directory - use existing analysis folder structure
    analysis_base_dir = Path(__file__).parent.parent / "analysis"
    latest_index = get_latest_analysis_index(experiment_name, analysis_base_dir)
    
    if latest_index is None:
        # No existing analysis - create analysis_1
        latest_index = "analysis_1"
        print(f"  No existing analysis found, creating {latest_index}")
    
    output_base = analysis_base_dir / experiment_name / latest_index / "monotonous_freq_analysis"
    output_base.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_base}")
    
    # Step 1: Load all model data
    print("\nStep 1: Loading model data...")
    models_data = load_all_model_data(experiment_path)
    print(f"  Loaded {len(models_data)} models with frequency metrics")
    
    if not models_data:
        print("No models found with frequency metrics. Exiting.")
        return
    
    # Step 2: Detect violations per metric
    print("\nStep 2: Detecting non-monotonic violations...")
    all_violations = detect_violations_per_metric(models_data)
    
    total_violations = sum(len(v) for metric_violations in all_violations.values() 
                          for v in metric_violations.values())
    print(f"  Total violations found: {total_violations}")
    
    # Step 3: Generate per-metric analysis
    print("\nStep 3: Generating per-metric analysis...")
    
    for metric_name, config in METRICS_CONFIG.items():
        violations = all_violations.get(metric_name, {})
        
        if not violations:
            print(f"  {metric_name}: No violations, skipping")
            continue
        
        print(f"  {metric_name}: {len(violations)} models with violations")
        
        metric_dir = output_base / metric_name
        per_model_dir = metric_dir / "per_model"
        overlay_dir = metric_dir / "spatial_freq_overlay"
        
        # Generate per-model frequency reduction plots from EVAL DATA
        # This ensures consistency with where violations were detected
        eval_spectral_data = {}  # Store computed spectral data for aggregation
        
        for model_name, model_violations in violations.items():
            violation_layers = [v['layer_idx'] for v in model_violations]
            
            print(f"    Processing {model_name}...")
            
            # Try to compute from eval data first (accurate)
            result = generate_freq_reduction_from_eval(
                model_name=model_name,
                model_data=models_data[model_name],
                violation_layers=violation_layers,
                output_dir=per_model_dir
            )
            
            if result is not None:
                k_bins, error_matrix = result
                eval_spectral_data[model_name] = {
                    'k_bins': k_bins,
                    'error_matrix': error_matrix
                }
            else:
                # Fallback to frequency_metrics.json (from grid, may differ from eval)
                print(f"    Falling back to frequency_metrics.json for {model_name}")
                freq_metrics = models_data[model_name].get('frequency_metrics')
                if freq_metrics:
                    plot_freq_reduction_per_model(
                        model_name=model_name,
                        freq_metrics=freq_metrics,
                        violation_layers=violation_layers,
                        output_dir=per_model_dir
                    )
        
        # Generate aggregated plot using eval-computed spectral data where available
        plot_aggregated_freq_reduction_v2(
            models_data=models_data,
            violations=violations,
            eval_spectral_data=eval_spectral_data,
            output_dir=metric_dir
        )
        
        # Generate overlay plots (only for metrics with spatial change heatmaps)
        if metric_name in METRICS_WITH_HEATMAPS:
            for model_name, model_violations in violations.items():
                violation_layers = [v['layer_idx'] for v in model_violations]
                generate_overlay_for_model(
                    model_name=model_name,
                    model_data=models_data[model_name],
                    violation_layers=violation_layers,
                    metric_name=config['display_name'],
                    output_dir=overlay_dir
                )
    
    # Step 4: Generate summary table
    print("\nStep 4: Generating summary table...")
    generate_summary_table(all_violations, models_data, output_base)
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print(f"Results saved to: {output_base}")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_frequency_monotonicity.py <experiment_path>")
        print("Example: python analyze_frequency_monotonicity.py outputs/experiments/wave1d_capacity_20251213")
        sys.exit(1)
    
    experiment_path = sys.argv[1]
    main(experiment_path)

