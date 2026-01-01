"""Frequency tracker orchestrator."""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict

from frequency_tracker.frequency_core import (
    analyze_all_layers_frequency,
    compute_frequency_spectrum,
    compute_binned_frequency_errors
)
from frequency_tracker.frequency_plotting import generate_all_frequency_plots


def run_frequency_tracker(
    model: torch.nn.Module,
    train_data_path: str,
    eval_data_path: str,
    cfg: Dict,
    run_dir: Path,
    epoch_suffix: str = ""
) -> Dict:
    """
    Run complete frequency analysis.
    
    This function:
    1. First runs probe analysis to train linear probes for each layer
    2. Loads the pre-generated frequency grid
    3. Evaluates probes on the grid
    4. Computes FFT spectra (cumulative, added, leftover)
    5. Generates visualizations
    
    Args:
        model: Trained neural network model
        train_data_path: Path to training_data.pt
        eval_data_path: Path to eval_data.pt
        cfg: Configuration dictionary
        run_dir: Output directory for this run
        epoch_suffix: Optional suffix for epoch-specific analysis (e.g., "_epoch_2")
        
    Returns:
        Dictionary with frequency metrics summary
    """
    print("\n" + "=" * 60)
    print("Frequency Tracker Analysis")
    print("=" * 60)
    
    device = torch.device('cuda' if cfg['cuda'] and 
                          torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = model.to(device)
    model.eval()
    
    # Step 1: Load pre-generated frequency grid
    print("\n" + "=" * 60)
    print("Step 1: Loading Frequency Grid")
    print("=" * 60)
    
    problem = cfg['problem']
    dataset_dir = Path("datasets") / problem
    freq_grid_path = dataset_dir / "frequency_grid.pt"
    
    if not freq_grid_path.exists():
        raise FileNotFoundError(
            f"Frequency grid not found: {freq_grid_path}\n"
            "Run dataset generation first (generate_and_save_datasets)"
        )
    
    freq_data = torch.load(freq_grid_path)
    grid_shape = tuple(freq_data['grid_shape'])
    n_dims = freq_data['n_dims']
    N_grid = freq_data['x_grid'].shape[0]
    
    print(f"  Grid shape: {grid_shape}")
    print(f"  Grid points: {N_grid:,}")
    print(f"  Dimensions: {n_dims}")
    
    # Step 2: Train probes to get probe objects (needed for frequency analysis)
    print("\n" + "=" * 60)
    print("Step 2: Training Linear Probes")
    print("=" * 60)
    
    # Load training data for probe training
    train_data = torch.load(train_data_path)
    
    train_x = train_data['x'].to(device)
    train_t = train_data['t'].to(device)
    train_targets = train_data['h_gt'].to(device)
    
    # Get all layer names (exclude output layer)
    all_layers = model.get_layer_names()
    hidden_layers = all_layers[:-1]
    print(f"  Analyzing layers: {hidden_layers}")
    
    # Register hooks to get embeddings (without gradients for probe training)
    handles = model.register_ncc_hooks(hidden_layers, keep_gradients=False)
    
    with torch.no_grad():
        train_inputs = torch.cat([train_x, train_t], dim=1)
        _ = model(train_inputs)
    
    train_embeddings = model.activations.copy()
    model.remove_hooks()
    
    # Train probes
    from probes.probe_core import train_linear_probe
    
    probes_dict = {}
    print("  Training probes...")
    for layer_name in hidden_layers:
        embeddings = train_embeddings[layer_name]
        probe = train_linear_probe(embeddings, train_targets)
        probes_dict[layer_name] = probe
        print(f"    {layer_name}: probe trained ({embeddings.shape[1]} -> {train_targets.shape[1]})")
    
    # Step 3: Analyze frequency content for all layers
    print("\n" + "=" * 60)
    print("Step 3: Computing Frequency Spectra")
    print("=" * 60)
    
    freq_results, sample_spacings = analyze_all_layers_frequency(
        model=model,
        probes_dict=probes_dict,
        freq_data=freq_data,
        device=device,
        config=cfg
    )
    
    # Compute ground truth spectrum for comparison (with same physical frequency units)
    h_gt_grid = freq_data['h_gt_grid'].cpu().numpy()
    if h_gt_grid.shape[1] == 1:
        h_gt_flat = h_gt_grid.flatten()
    else:
        h_gt_flat = h_gt_grid
    h_gt_spectrum = compute_frequency_spectrum(h_gt_flat, grid_shape, sample_spacings)
    
    # Step 4: Generate visualizations
    print("\n" + "=" * 60)
    print("Step 4: Generating Visualizations")
    print("=" * 60)
    
    if epoch_suffix:
        # Periodic frequency: save inside frequency_plots/frequency_plots_epoch_X/
        freq_plots_dir = run_dir / "frequency_plots" / f"frequency_plots{epoch_suffix}"
    else:
        # Final frequency: save directly in frequency_plots/
        freq_plots_dir = run_dir / "frequency_plots"
    
    freq_plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Get problem info for plotting
    problem_cfg = cfg[problem]
    spatial_dim = problem_cfg.get('spatial_dim', 1)
    
    generate_all_frequency_plots(
        freq_results=freq_results,
        h_gt_spectrum=h_gt_spectrum,
        save_dir=freq_plots_dir,
        config=cfg
    )
    
    # Step 5: Build and save metrics summary
    print("\n" + "=" * 60)
    print("Step 5: Saving Metrics")
    print("=" * 60)
    
    # Compute binned error matrix for spectral learning efficiency
    n_bins = 20
    error_matrix_list = []
    k_radial_ref = None
    
    for layer_name in hidden_layers:
        layer_data = freq_results[layer_name]
        leftover = layer_data['leftover']
        
        # Compute binned errors for radial spectrum
        binned_errors = compute_binned_frequency_errors(
            power_pred=leftover['power'],
            power_gt=h_gt_spectrum['power'],
            freqs=leftover['freqs'],
            spatial_dim=spatial_dim,
            n_bins=n_bins,
            is_leftover=True
        )
        
        k_radial, mean_error = binned_errors['radial']
        if k_radial_ref is None:
            k_radial_ref = k_radial.tolist()
        error_matrix_list.append(mean_error.tolist())
    
    metrics_summary = {
        'layers_analyzed': hidden_layers,
        'num_layers': len(hidden_layers),
        'grid_shape': list(grid_shape),
        'n_grid_points': N_grid,
        'layer_metrics': {},
        'spectral_efficiency': {
            'k_radial_bins': k_radial_ref,
            'error_matrix': error_matrix_list  # List of lists: [layer][freq_bin]
        }
    }
    
    # Collect per-layer metrics
    for layer_name in hidden_layers:
        layer_data = freq_results[layer_name]
        
        # Total power metrics
        cumulative_power = float(layer_data['cumulative']['power'].sum())
        leftover_power = float(layer_data['leftover']['power'].sum())
        added_power = float(layer_data['added']['power'].sum())
        
        # Radial spectrum metrics
        k_radial, power_radial = layer_data['radial']
        k_peak = float(k_radial[np.argmax(power_radial)])
        
        metrics_summary['layer_metrics'][layer_name] = {
            'cumulative_total_power': cumulative_power,
            'leftover_total_power': leftover_power,
            'added_total_power': added_power,
            'radial_peak_frequency': k_peak,
            'leftover_ratio': leftover_power / (cumulative_power + 1e-10)
        }
    
    # Add final layer summary
    final_layer = hidden_layers[-1]
    metrics_summary['final_layer_cumulative_power'] = metrics_summary['layer_metrics'][final_layer]['cumulative_total_power']
    metrics_summary['final_layer_leftover_power'] = metrics_summary['layer_metrics'][final_layer]['leftover_total_power']
    metrics_summary['final_layer_leftover_ratio'] = metrics_summary['layer_metrics'][final_layer]['leftover_ratio']
    
    # Ground truth total power
    metrics_summary['ground_truth_total_power'] = float(h_gt_spectrum['power'].sum())
    
    # Save metrics
    metrics_path = freq_plots_dir / "frequency_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"  Frequency metrics saved to {metrics_path}")
    
    # Step 7: Summary
    print("\n" + "=" * 60)
    print("Frequency Tracker Analysis Complete")
    print("=" * 60)
    print(f"Results saved to: {freq_plots_dir}")
    print("  Generated plots:")
    print("    - radial_spectrum_cumulative.png")
    print("    - radial_spectrum_added.png")
    print("    - radial_spectrum_leftover.png")
    print("    - marginal_spectra_*.png")
    print(f"Metrics: {metrics_path}")
    print("\nFinal Layer Summary:")
    print(f"  Cumulative Power: {metrics_summary['final_layer_cumulative_power']:.2e}")
    print(f"  Leftover Power: {metrics_summary['final_layer_leftover_power']:.2e}")
    print(f"  Leftover Ratio: {metrics_summary['final_layer_leftover_ratio']:.4f}")
    print("=" * 60)
    
    return metrics_summary

