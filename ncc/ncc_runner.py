"""NCC runner for analyzing trained models."""

import torch
import json
from pathlib import Path
from typing import Dict

from ncc.ncc_core import compute_all_ncc_metrics
from ncc.ncc_plotting import generate_all_ncc_plots


def run_ncc(
    model: torch.nn.Module,
    eval_data_path: str,
    cfg: Dict,
    run_dir: Path
) -> Dict:
    """
    Run complete NCC analysis on a trained model.

    Args:
        model: Trained neural network model
        eval_data_path: Path to NCC data (.pt file) - stratified dataset
        cfg: Configuration dictionary
        run_dir: Output directory for this run

    Returns:
        Dictionary with NCC metrics summary
    """
    print("\n" + "=" * 60)
    print("NCC Analysis")
    print("=" * 60)

    # Setup device
    device = torch.device('cuda' if cfg['cuda'] and
                          torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load NCC data (stratified dataset)
    print(f"\nLoading NCC data from: {eval_data_path}")
    eval_data = torch.load(eval_data_path)

    # Move data to device
    x = eval_data['x'].to(device)
    t = eval_data['t'].to(device)
    u_gt = eval_data['u_gt'].to(device)

    N = x.shape[0]
    print(f"  NCC samples: {N}")

    # Get bins parameter
    bins = cfg['bins']
    print(f"  Bins per dimension: {bins}")

    # Get all layer names (exclude output layer for NCC analysis)
    all_layers = model.get_layer_names()
    hidden_layers = all_layers[:-1]  # Exclude output layer
    print(f"  Analyzing layers: {hidden_layers}")

    # Register hooks to capture activations
    print("\nRegistering hooks...")
    handles = model.register_ncc_hooks(hidden_layers)
    print(f"  ✓ Hooks registered on {len(handles)} layers")

    # Forward pass to collect activations
    print("\nRunning forward pass to collect activations...")
    with torch.no_grad():
        inputs = torch.cat([x, t], dim=1)
        u_pred = model(inputs)

    # Get activations from model
    embeddings_dict = model.activations.copy()
    print(f"  ✓ Collected activations from {len(embeddings_dict)} layers")

    # Verify activations
    for layer_name, activations in embeddings_dict.items():
        print(f"    {layer_name}: {activations.shape}")

    # Clean up hooks
    model.remove_hooks()

    # Compute NCC metrics
    print("\nComputing NCC metrics...")
    ncc_results = compute_all_ncc_metrics(
        embeddings_dict=embeddings_dict,
        ground_truth_outputs=u_gt,
        bins=bins,
        device=device
    )

    num_classes = ncc_results['bin_info']['num_classes']
    print(f"  ✓ Created {num_classes} classes from regression outputs")

    # Print metrics summary
    print("\n  Per-layer NCC accuracy:")
    for layer_name, metrics in ncc_results['layer_metrics'].items():
        acc = metrics['accuracy']
        print(f"    {layer_name}: {acc:.4f}")

    # Generate plots
    ncc_plots_dir = run_dir / "ncc_plots"
    print(f"\nGenerating NCC plots...")
    generate_all_ncc_plots(ncc_results, ncc_plots_dir)

    # Save metrics summary as JSON
    metrics_summary = {
        'bins': bins,
        'num_classes': num_classes,
        'num_samples': N,
        'layers_analyzed': hidden_layers,
        'layer_accuracies': {
            layer_name: metrics['accuracy']
            for layer_name, metrics in ncc_results['layer_metrics'].items()
        },
        'layer_compactness': {
            layer_name: {
                'intra_class': metrics['compactness']['intra_class_dist'],
                'inter_class_mean': metrics['compactness']['inter_class_mean'],
                'inter_class_std': metrics['compactness']['inter_class_std']
            }
            for layer_name, metrics in ncc_results['layer_metrics'].items()
        },
        'layer_margins': {
            layer_name: {
                'mean_margin': metrics['margins']['mean_margin'],
                'std_margin': metrics['margins']['std_margin'],
                'fraction_positive': metrics['margins']['fraction_positive']
            }
            for layer_name, metrics in ncc_results['layer_metrics'].items()
        }
    }

    ncc_metrics_path = run_dir / "ncc_metrics.json"
    with open(ncc_metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"  ✓ NCC metrics saved to {ncc_metrics_path}")

    print("\n" + "=" * 60)
    print("✓ NCC Analysis Complete")
    print("=" * 60)
    print(f"Results saved to: {ncc_plots_dir}")
    print("  - ncc_layer_accuracy.png")
    print("  - ncc_compactness.png")
    print("  - ncc_center_geometry.png")
    print("  - ncc_margin.png")
    print("  - ncc_confusions.png")
    print(f"Metrics: {ncc_metrics_path}")
    print("=" * 60)

    return metrics_summary

