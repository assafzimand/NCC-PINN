"""Regenerate comparison plots for experiment batches.

Supports two directory structures:
  1. Multi-PDE batch: root → PDE dirs → timestamp dirs (each with a different model)
  2. Single batch:    root → architecture dirs → timestamp dirs
"""

import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from typing import Dict

import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils.comparison_plots import (
    generate_ncc_classification_plot,
    generate_ncc_compactness_plot,
    generate_probe_comparison_plots,
    generate_derivatives_comparison_plots,
    generate_frequency_coverage_comparison,
    plot_spectral_learning_efficiency_comparison
)

_TIMESTAMP_RE = re.compile(r'\d{8}_\d{6}$')


def _is_timestamp_dir(d: Path) -> bool:
    return d.is_dir() and bool(_TIMESTAMP_RE.match(d.name))


def _get_model_name(ts_dir: Path) -> str:
    """Extract model name from config_used.yaml."""
    config_file = ts_dir / "config_used.yaml"
    if config_file.exists():
        try:
            import yaml
            with open(config_file) as f:
                cfg = yaml.safe_load(f)
            return cfg.get('model', ts_dir.name)
        except Exception:
            pass
    return ts_dir.name


def _build_run_name(ts_dir: Path) -> str:
    """Build a descriptive experiment name from a timestamp run directory.

    Reads config_used.yaml and extracts key training parameters to
    differentiate runs of the same architecture.
    Falls back to the timestamp folder name if config is unavailable.
    """
    config_file = ts_dir / "config_used.yaml"
    if not config_file.exists():
        return ts_dir.name

    try:
        import yaml
        with open(config_file) as f:
            cfg = yaml.safe_load(f)

        parts = []

        # Training params
        parts.append(f"ep{cfg.get('epochs', '?')}")
        lr = cfg.get('lr', None)
        if lr is not None:
            parts.append(f"lr{lr}")

        # Adaptive params (if present)
        adaptive = cfg.get('adaptive_pinn', {})
        if adaptive.get('enabled', False):
            spawn = adaptive.get('spawn_every_epochs')
            if spawn is not None:
                parts.append(f"sp{spawn}")
            problem_cfg = cfg.get(cfg.get('problem', ''), {})
            wt = problem_cfg.get('wavelet_threshold')
            if wt is not None:
                parts.append(f"wt{wt}")
            if adaptive.get('only_leaves', False):
                parts.append("leaves")

        # Optimizer switch
        switch = cfg.get('optimizer_switch_fraction')
        if switch is not None:
            parts.append(f"sw{switch}")

        name = "_".join(str(p) for p in parts)
        # Append short timestamp to guarantee uniqueness
        name += f"_{ts_dir.name[-6:]}"
        return name

    except Exception:
        return ts_dir.name


def _generate_training_results_plot(parent_dir, df):
    """Generate training and results comparison table (copied from run_experiments.py)."""
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np

    fig = plt.figure(figsize=(16, 6))
    ax3 = fig.add_subplot(111)
    ax3.axis('off')

    # Create colored table
    table_data = []
    col_labels = ['Experiment', 'Train Loss', 'Eval Loss', 'Train Rel-L2', 'Train Inf',
                  'Eval Rel-L2', 'Eval Inf', 'NCC Final Acc', 'Margin SNR',
                  'Deriv Train Res', 'Deriv Eval Res']

    for _, row in df.iterrows():
        row_data = [
            row['experiment'],
            f"{row['final_train_loss']:.6f}",
            f"{row['final_eval_loss']:.6f}",
            f"{row['final_train_rel_l2']:.6f}",
            f"{row['final_train_inf_norm']:.6f}",
            f"{row['final_eval_rel_l2']:.6f}",
            f"{row['final_eval_inf_norm']:.6f}",
            f"{row['ncc_final_accuracy']:.6f}",
            f"{row['margin_snr']:.2f}"
        ]
        # Add derivatives if available
        if 'deriv_final_train_residual' in row and not pd.isna(row['deriv_final_train_residual']):
            row_data.append(f"{row['deriv_final_train_residual']:.2e}")
        else:
            row_data.append("N/A")
        if 'deriv_final_eval_residual' in row and not pd.isna(row['deriv_final_eval_residual']):
            row_data.append(f"{row['deriv_final_eval_residual']:.2e}")
        else:
            row_data.append("N/A")
        table_data.append(row_data)

    table = ax3.table(cellText=table_data, colLabels=col_labels,
                     cellLoc='center', loc='center',
                     bbox=[0.05, 0.1, 0.9, 0.8])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)

    # Create green-to-red colormap
    cmap = LinearSegmentedColormap.from_list('GreenRed', ['#2ecc71', '#f1c40f', '#e74c3c'])

    # Color coding for each column
    num_cols = len(col_labels)
    for col_idx in range(1, min(num_cols, len(df.columns) + 1)):
        # Check if this column exists in the dataframe
        if col_idx >= len(df.columns):
            continue

        col_name = df.columns[col_idx]
        values = df[col_name].values

        # Skip if all NaN
        if pd.isna(values).all():
            continue

        # For losses/errors/residuals, lower is better; for accuracy and margin SNR, higher is better
        if col_idx == 7 or col_idx == 8:  # NCC accuracy and Margin SNR - higher is better
            norm_values = 1 - (values - values.min()) / (values.max() - values.min() + 1e-10)
        else:  # Losses, errors, residuals - lower is better
            norm_values = (values - values.min()) / (values.max() - values.min() + 1e-10)

        for row_idx, norm_val in enumerate(norm_values):
            if not pd.isna(norm_val):
                cell = table[(row_idx + 1, col_idx)]
                color = cmap(norm_val)
                cell.set_facecolor(color)
                cell.set_alpha(0.7)

    # Style header
    for col_idx in range(num_cols):
        cell = table[(0, col_idx)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')

    fig.suptitle('Training and Results Comparison', fontsize=16, fontweight='bold', y=0.92)

    plt.savefig(parent_dir / "training_and_results_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Training and results comparison saved to training_and_results_comparison.png")


def generate_comparison_for_batch(batch_dir: Path, label: str = None):
    """Generate comparison plots for a single experiment batch.

    Handles two layouts:
      Flat   – batch_dir contains timestamp dirs directly (e.g. per-PDE dir
               where each timestamp is a different model).
      Nested – batch_dir contains architecture dirs, each with timestamp subdirs.
    """
    display_name = label or batch_dir.name
    print(f"\n{'='*70}")
    print(f"Processing: {display_name}")
    print(f"{'='*70}\n")

    child_dirs = sorted([d for d in batch_dir.iterdir()
                         if d.is_dir() and d.name != 'checkpoints'])

    if not child_dirs:
        print(f"  No subdirectories found in {batch_dir}")
        return

    # --- Detect flat structure (timestamps directly under batch_dir) ---
    direct_ts_dirs = [d for d in child_dirs
                      if _is_timestamp_dir(d) and (d / 'metrics.json').exists()]

    results = {}
    if direct_ts_dirs:
        print(f"  Found {len(direct_ts_dirs)} experiment runs (flat / per-PDE structure)")
        for ts_dir in direct_ts_dirs:
            exp_name = _get_model_name(ts_dir)
            results[exp_name] = ts_dir
    else:
        # --- Nested structure (architecture dirs → timestamp subdirs) ---
        model_dirs = child_dirs
        print(f"  Found {len(model_dirs)} architecture dirs: {[d.name for d in model_dirs]}")

        for model_dir in model_dirs:
            timestamp_dirs = sorted(
                [d for d in model_dir.iterdir()
                 if d.is_dir() and d.name != 'checkpoints']
            )
            if not timestamp_dirs:
                results[model_dir.name] = model_dir
                continue

            if len(model_dirs) == 1 and len(timestamp_dirs) > 1:
                print(f"  Single architecture with {len(timestamp_dirs)} runs — comparing all")
                for ts_dir in timestamp_dirs:
                    exp_name = _build_run_name(ts_dir)
                    results[exp_name] = ts_dir
            else:
                results[model_dir.name] = timestamp_dirs[-1]

    # Collect training metrics (same logic as run_experiments.py)
    metrics_data = []
    ncc_data = {}
    probe_data = {}
    derivatives_data = {}
    frequency_data = {}
    expert_regions_data = {}

    for exp_name, result_path in results.items():
        if result_path is None:
            continue

        # Load training metrics
        metrics_file = result_path / "metrics.json"
        if not metrics_file.exists():
            print(f"  Warning: No metrics.json found for {exp_name}")
            continue

        with open(metrics_file) as f:
            train_metrics = json.load(f)

        # Collect NCC metrics
        ncc_plots_dir = result_path / "ncc_plots"
        ncc_epochs = {}

        if ncc_plots_dir.exists():
            final_ncc_file = ncc_plots_dir / "ncc_metrics.json"
            if final_ncc_file.exists():
                with open(final_ncc_file) as f:
                    ncc_epochs['final'] = json.load(f)

            # Load periodic NCCs
            for subdir in ncc_plots_dir.iterdir():
                if subdir.is_dir() and subdir.name.startswith("ncc_plots_epoch_"):
                    epoch_num = int(subdir.name.split("_")[-1])
                    epoch_file = subdir / "ncc_metrics.json"
                    if epoch_file.exists():
                        with open(epoch_file) as f:
                            ncc_epochs[epoch_num] = json.load(f)

        # If no NCC data, use defaults
        if not ncc_epochs:
            print(f"  Warning: No NCC data found for {exp_name}, using defaults")
            final_ncc = None
        else:
            final_ncc = ncc_epochs.get('final', list(ncc_epochs.values())[-1])

        # Load probe metrics
        probe_file = result_path / "probe_plots" / "probe_metrics.json"
        if probe_file.exists():
            with open(probe_file) as f:
                probe_metrics = json.load(f)
                probe_data[exp_name] = probe_metrics

        # Load derivatives metrics
        deriv_file = result_path / "derivatives_plots" / "derivatives_metrics.json"
        deriv_metrics = None
        if deriv_file.exists():
            with open(deriv_file) as f:
                deriv_metrics = json.load(f)
                derivatives_data[exp_name] = deriv_metrics

        # Load frequency metrics
        freq_file = result_path / "frequency_plots" / "frequency_metrics.json"
        if freq_file.exists():
            with open(freq_file) as f:
                freq_metrics = json.load(f)
                frequency_data[exp_name] = freq_metrics

        # Load expert regions for adaptive PINN
        expert_regions_file = result_path / "adaptive_plots" / "expert_regions.json"
        if expert_regions_file.exists():
            try:
                from adaptive.indicators import RegionDescriptor
                from adaptive.visualization import load_regions_metadata
                regions = load_regions_metadata(expert_regions_file)
                if regions:
                    expert_regions_data[exp_name] = regions
            except Exception as e:
                print(f"  Warning: Could not load expert regions for {exp_name}: {e}")

        # Extract margin SNR if NCC data available
        if final_ncc:
            final_layer = list(final_ncc['layer_accuracies'].keys())[-1]
            margin_mean = final_ncc['layer_margins'][final_layer]['mean_margin']
            margin_std = final_ncc['layer_margins'][final_layer]['std_margin']
            margin_snr = margin_mean / margin_std if margin_std > 0 else 0
            ncc_accuracy = final_ncc['layer_accuracies'][final_layer]
        else:
            margin_snr = float('nan')
            ncc_accuracy = float('nan')

        # Build metrics row
        metrics_row = {
            'experiment': exp_name,
            'final_train_loss': train_metrics['train_loss'][-1],
            'final_eval_loss': train_metrics['eval_loss'][-1],
            'final_train_rel_l2': train_metrics['train_rel_l2'][-1],
            'final_train_inf_norm': train_metrics['train_inf_norm'][-1],
            'final_eval_rel_l2': train_metrics['eval_rel_l2'][-1],
            'final_eval_inf_norm': train_metrics['eval_inf_norm'][-1],
            'ncc_final_accuracy': ncc_accuracy,
            'margin_snr': margin_snr
        }

        # Add derivatives if available
        if deriv_metrics:
            metrics_row['deriv_final_train_residual'] = deriv_metrics['final_layer_train_residual']
            metrics_row['deriv_final_eval_residual'] = deriv_metrics['final_layer_eval_residual']

        metrics_data.append(metrics_row)
        if ncc_epochs:
            ncc_data[exp_name] = ncc_epochs

    if not metrics_data:
        print(f"  No valid results to compare for batch {batch_dir.name}")
        return

    # Create comparison table
    df = pd.DataFrame(metrics_data)
    df.to_csv(batch_dir / "comparison_summary.csv", index=False)
    print(f"  Comparison table saved to comparison_summary.csv")

    # Generate plots
    _generate_training_results_plot(batch_dir, df)

    if ncc_data:
        generate_ncc_classification_plot(batch_dir, ncc_data)
        generate_ncc_compactness_plot(batch_dir, ncc_data)

    if probe_data:
        generate_probe_comparison_plots(batch_dir, probe_data)

    if derivatives_data:
        generate_derivatives_comparison_plots(batch_dir, derivatives_data)

    if frequency_data:
        generate_frequency_coverage_comparison(batch_dir, frequency_data)
        plot_spectral_learning_efficiency_comparison(frequency_data, batch_dir)

    if expert_regions_data:
        print(f"  Generating expert regions comparison ({len(expert_regions_data)} experiments)...")
        try:
            from adaptive.visualization import (
                plot_expert_regions_comparison, prepare_ground_truth_grid
            )

            # Get domain bounds from first experiment's config
            first_result_path = list(results.values())[0]
            if first_result_path is not None:
                config_file = first_result_path / "config_used.yaml"
                if config_file.exists():
                    import yaml
                    with open(config_file) as f:
                        exp_config = yaml.safe_load(f)
                    problem = exp_config.get('problem', 'burgers1d')
                    problem_config = exp_config.get(problem, {})
                    spatial_domain = problem_config.get('spatial_domain', [[-1, 1]])
                    temporal_domain = problem_config.get('temporal_domain', [0, 1])

                    # Build domain bounds
                    if len(spatial_domain) == 1:
                        domain_bounds = {
                            'lower': [spatial_domain[0][0], temporal_domain[0]],
                            'upper': [spatial_domain[0][1], temporal_domain[1]]
                        }
                        problem_type = '2d'
                    else:
                        domain_bounds = {
                            'lower': [spatial_domain[0][0], spatial_domain[1][0], temporal_domain[0]],
                            'upper': [spatial_domain[0][1], spatial_domain[1][1], temporal_domain[1]]
                        }
                        problem_type = '3d'

                    # Load eval data for ground truth
                    gt_grid, gt_x, gt_t = None, None, None
                    if problem_type == '2d':
                        eval_data_path = Path("datasets") / problem / "eval_data.pt"
                        if eval_data_path.exists():
                            try:
                                eval_data = torch.load(eval_data_path, map_location='cpu')
                                gt_grid, gt_x, gt_t = prepare_ground_truth_grid(
                                    eval_data, domain_bounds
                                )
                            except Exception as e:
                                print(f"  Warning: Could not load ground truth: {e}")

                    plot_expert_regions_comparison(
                        experiment_regions=expert_regions_data,
                        domain_bounds=domain_bounds,
                        output_path=batch_dir / "expert_regions_comparison.png",
                        problem_type=problem_type,
                        ground_truth=gt_grid,
                        grid_x=gt_x,
                        grid_t=gt_t
                    )
        except Exception as e:
            print(f"  Error generating expert regions comparison: {e}")

    print(f"\n  [OK] Comparison plots saved to {batch_dir}")


def _detect_structure(target_path: Path):
    """Detect the directory layout.

    Returns one of:
      'multi_pde'   – target has PDE child dirs, each with timestamp subdirs
      'single_batch'– target is a single batch (arch dirs → timestamp subdirs,
                       or flat timestamps)
      'multi_batch' – target contains multiple independent batch dirs
    """
    child_dirs = [d for d in target_path.iterdir() if d.is_dir()]
    if not child_dirs:
        return 'multi_batch'

    # Multi-PDE: each child dir has ≥2 timestamp subdirs with metrics.json
    pde_like = 0
    for cd in child_dirs:
        ts_dirs = [d for d in cd.iterdir()
                   if _is_timestamp_dir(d) and (d / 'metrics.json').exists()]
        if len(ts_dirs) >= 2:
            pde_like += 1
    if pde_like >= 2:
        return 'multi_pde'

    # Single batch: any child (or grandchild) has metrics.json
    for cd in child_dirs:
        if (cd / 'metrics.json').exists():
            return 'single_batch'
        for sub in cd.iterdir():
            if sub.is_dir() and (sub / 'metrics.json').exists():
                return 'single_batch'

    return 'multi_batch'


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        target_path = Path(sys.argv[1])
    else:
        target_path = Path("outputs/experiments/AToE-New")

    if not target_path.exists():
        print(f"Error: Directory not found: {target_path}")
        return

    structure = _detect_structure(target_path)
    print(f"Detected structure: {structure}")

    if structure == 'multi_pde':
        pde_dirs = sorted([d for d in target_path.iterdir() if d.is_dir()])
        print(f"Found {len(pde_dirs)} PDE group(s):")
        for pd_dir in pde_dirs:
            pde_label = pd_dir.name.split('-')[0]
            print(f"  - {pde_label} ({pd_dir.name})")

        for pd_dir in pde_dirs:
            pde_label = pd_dir.name.split('-')[0]
            try:
                generate_comparison_for_batch(
                    pd_dir, label=f"{pde_label} ({pd_dir.name})")
            except Exception as e:
                print(f"\nError processing {pd_dir.name}: {e}")
                import traceback
                traceback.print_exc()

    elif structure == 'single_batch':
        generate_comparison_for_batch(target_path)

    else:
        batch_dirs = sorted([d for d in target_path.iterdir() if d.is_dir()])
        if not batch_dirs:
            print(f"No experiment batches found in {target_path}")
            return
        print(f"Found {len(batch_dirs)} experiment batch(es):")
        for bd in batch_dirs:
            print(f"  - {bd.name}")
        for bd in batch_dirs:
            try:
                generate_comparison_for_batch(bd)
            except Exception as e:
                print(f"\nError processing {bd.name}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*70}")
    print("Done! All comparison plots regenerated.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
