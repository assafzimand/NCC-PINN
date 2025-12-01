"""Derivatives tracker orchestrator."""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict

from derivatives_tracker.derivatives_core import track_all_layers
from derivatives_tracker.derivatives_plotting import generate_all_derivative_plots
from probes.probe_runner import run_probes


def run_derivatives_tracker(
    model: torch.nn.Module,
    train_data_path: str,
    eval_data_path: str,
    cfg: Dict,
    run_dir: Path
) -> Dict:
    """
    Run complete derivatives tracking analysis.
    
    This function:
    1. First runs probe analysis to train linear probes for each layer
    2. Uses those probes to project each layer to 2D
    3. Computes derivatives and residual terms via autograd
    4. Generates comprehensive visualizations
    
    Args:
        model: Trained neural network model
        train_data_path: Path to training_data.pt
        eval_data_path: Path to eval_data.pt
        cfg: Configuration dictionary
        run_dir: Output directory for this run
        
    Returns:
        Dictionary with derivatives metrics summary
    """
    print("\n" + "=" * 60)
    print("Derivatives Tracker Analysis")
    print("=" * 60)
    
    device = torch.device('cuda' if cfg['cuda'] and 
                          torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = model.to(device)
    model.eval()
    
    # Step 1: Run probe analysis to get trained probes
    print("\n" + "=" * 60)
    print("Step 1: Training Linear Probes")
    print("=" * 60)
    
    probe_results = run_probes(
        model=model,
        train_data_path=train_data_path,
        eval_data_path=eval_data_path,
        cfg=cfg,
        run_dir=run_dir
    )
    
    # Extract trained probes from probe_results
    # We need to load probe_results from the actual probe_runner output
    # since run_probes returns a summary, not the full results
    probe_plots_dir = run_dir / "probe_plots"
    
    # Load data
    print("\n" + "=" * 60)
    print("Step 2: Loading Data for Derivatives")
    print("=" * 60)
    
    train_data = torch.load(train_data_path)
    eval_data = torch.load(eval_data_path)
    
    train_x = train_data['x'].to(device)
    train_t = train_data['t'].to(device)
    
    eval_x = eval_data['x'].to(device)
    eval_t = eval_data['t'].to(device)
    
    N_train = train_x.shape[0]
    N_eval = eval_x.shape[0]
    print(f"  Train samples: {N_train}")
    print(f"  Eval samples: {N_eval}")
    
    # Step 3: Re-train probes to get probe objects (needed for derivatives)
    # We need the actual probe objects, so we'll extract embeddings and train again
    print("\n" + "=" * 60)
    print("Step 3: Extracting Probes for Derivatives")
    print("=" * 60)
    
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
    
    # Train probes to get probe objects
    from probes.probe_core import train_linear_probe
    
    train_targets = train_data['u_gt'].to(device)
    probes_dict = {}
    
    print("  Training probes...")
    for layer_name in hidden_layers:
        embeddings = train_embeddings[layer_name]
        probe = train_linear_probe(embeddings, train_targets)
        probes_dict[layer_name] = probe
        print(f"    {layer_name}: probe trained ({embeddings.shape[1]} → 2)")
    
    # Step 4: Track derivatives for train and eval datasets
    print("\n" + "=" * 60)
    print("Step 4: Computing Derivatives and Residuals")
    print("=" * 60)
    
    print("\nProcessing training data...")
    train_derivatives = track_all_layers(
        model=model,
        probes_dict=probes_dict,
        x=train_x,
        t=train_t,
        device=device
    )
    
    print("\nProcessing evaluation data...")
    eval_derivatives = track_all_layers(
        model=model,
        probes_dict=probes_dict,
        x=eval_x,
        t=eval_t,
        device=device
    )
    
    # Step 5: Generate visualizations (use eval data)
    print("\n" + "=" * 60)
    print("Step 5: Generating Visualizations")
    print("=" * 60)
    
    derivatives_plots_dir = run_dir / "derivatives_plots"
    derivatives_plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy for plotting
    eval_x_np = eval_x.cpu().numpy()
    eval_t_np = eval_t.cpu().numpy()
    
    generate_all_derivative_plots(
        derivatives_results=eval_derivatives,
        x=eval_x_np,
        t=eval_t_np,
        save_dir=derivatives_plots_dir
    )
    
    # Step 6: Build and save metrics summary
    print("\n" + "=" * 60)
    print("Step 6: Saving Metrics")
    print("=" * 60)
    
    metrics_summary = {
        'layers_analyzed': hidden_layers,
        'num_layers': len(hidden_layers),
        'train': {},
        'eval': {},
        'layer_norms': {}
    }
    
    # Collect norms for each layer
    for layer_name in sorted(train_derivatives.keys()):
        metrics_summary['train'][layer_name] = train_derivatives[layer_name]['norms']
        metrics_summary['eval'][layer_name] = eval_derivatives[layer_name]['norms']
        metrics_summary['layer_norms'][layer_name] = {
            'train_residual': train_derivatives[layer_name]['norms']['residual_norm'],
            'eval_residual': eval_derivatives[layer_name]['norms']['residual_norm']
        }
    
    # Add final layer residual for easy comparison
    final_layer = hidden_layers[-1]
    metrics_summary['final_layer_train_residual'] = train_derivatives[final_layer]['norms']['residual_norm']
    metrics_summary['final_layer_eval_residual'] = eval_derivatives[final_layer]['norms']['residual_norm']
    
    # Save metrics
    metrics_path = derivatives_plots_dir / "derivatives_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"  Derivatives metrics saved to {metrics_path}")
    
    # Step 7: Summary
    print("\n" + "=" * 60)
    print("Derivatives Tracker Analysis Complete")
    print("=" * 60)
    print(f"Results saved to: {derivatives_plots_dir}")
    print("  Generated plots:")
    print("    - residual_evolution.png")
    print("    - term_magnitudes.png")
    print("    - real_imag_components.png")
    print("    - derivative_heatmaps_*.png (for selected layers)")
    print("    - residual_heatmaps_*.png (for selected layers)")
    print("    - residual_balance.png")
    print(f"Metrics: {metrics_path}")
    print("\nFinal Layer Residuals:")
    print(f"  Train: {metrics_summary['final_layer_train_residual']:.6e}")
    print(f"  Eval:  {metrics_summary['final_layer_eval_residual']:.6e}")
    print("=" * 60)
    
    return metrics_summary

