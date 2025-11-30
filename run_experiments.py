"""Automated experiment runner for architecture search."""

import yaml
import shutil
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import sys


def load_experiment_plan(plan_path="experiments_plan.yaml"):
    """Load experiment plan from YAML file."""
    with open(plan_path, 'r') as f:
        return yaml.safe_load(f)


def create_experiment_dir(plan):
    """Create parent experiment directory."""
    base = plan['base_config']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create descriptive folder name
    exp_name = plan['experiment_name']
    params = f"lr{base['lr']}_ep{base['epochs']}_bs{base['batch_size']}_bins{base['bins']}"
    
    parent_dir = Path("outputs") / "experiments" / f"{exp_name}_{params}_{timestamp}"
    parent_dir.mkdir(parents=True, exist_ok=True)
    
    return parent_dir


def run_single_experiment(exp_config, base_config, exp_name, parent_dir):
    """Run one experiment."""
    print(f"\n{'='*70}")
    print(f"Running Experiment: {exp_name}")
    print(f"Architecture: {exp_config['architecture']}")
    print(f"{'='*70}\n")
    
    # Merge configs
    config = {**base_config, **exp_config}
    
    # Generate architecture-based folder name (same format as make_run_dir)
    layers_str = "-".join(map(str, config['architecture']))
    arch_folder_name = f"{config['problem']}_layers-{layers_str}_act-{config['activation']}"
    exp_output_dir = parent_dir / arch_folder_name
    
    # Temporarily update config.yaml
    config_backup_path = Path('config/config.yaml.backup')
    shutil.copy('config/config.yaml', config_backup_path)
    
    # Temporarily update outputs path in the environment
    original_outputs_path = Path("outputs")
    temp_outputs_path = exp_output_dir / "temp_outputs"
    
    try:
        with open('config/config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Run training + NCC (with real-time output)
        result = subprocess.run([sys.executable, 'run_ncc.py'])
        
        if result.returncode != 0:
            print(f"\nERROR in {exp_name} NCC: Process exited with code {result.returncode}")
            return None
        
        # Run probes analysis
        print(f"\n{'='*70}")
        print(f"Running Probe Analysis for: {exp_name}")
        print(f"{'='*70}\n")
        result_probes = subprocess.run([sys.executable, 'run_probes.py'])
        
        if result_probes.returncode != 0:
            print(f"\nWARNING in {exp_name} Probes: Process exited with code {result_probes.returncode}")
            # Don't return None - probes are optional, continue with NCC results
        
        # Move outputs to experiment directory
        # Find latest architecture directory matching this experiment
        outputs_root = Path("outputs")
        if outputs_root.exists():
            # Build pattern to match architecture folder
            layers_str = "-".join(map(str, config['architecture']))
            arch_folder_name = f"{config['problem']}_layers-{layers_str}_act-{config['activation']}"
            arch_dir = outputs_root / arch_folder_name
            
            if arch_dir.exists():
                # Find the TWO LATEST timestamp directories (one from NCC, one from probes)
                timestamp_dirs = sorted(
                    [d for d in arch_dir.glob("*/") if d.is_dir()], 
                    key=lambda x: x.stat().st_mtime
                )
                
                if len(timestamp_dirs) >= 2:
                    # Get the two most recent directories (NCC and Probes)
                    ncc_dir = timestamp_dirs[-2]  # Second to last (NCC ran first)
                    probe_dir = timestamp_dirs[-1]  # Last (Probes ran second)
                    
                    # Create experiment output directory
                    exp_output_dir.mkdir(parents=True, exist_ok=True)
                    dest_dir = exp_output_dir / ncc_dir.name
                    
                    # Move NCC results
                    if dest_dir.exists():
                        shutil.rmtree(dest_dir)
                    shutil.move(str(ncc_dir), str(dest_dir))
                    
                    # Merge probe results into the same directory
                    probe_plots_src = probe_dir / "probe_plots"
                    if probe_plots_src.exists():
                        probe_plots_dest = dest_dir / "probe_plots"
                        if probe_plots_dest.exists():
                            shutil.rmtree(probe_plots_dest)
                        shutil.move(str(probe_plots_src), str(probe_plots_dest))
                    
                    # Clean up the probe directory (we've moved what we need)
                    if probe_dir.exists():
                        shutil.rmtree(probe_dir)
                    
                    # Also move corresponding checkpoints to experiment folder
                    checkpoints_root = Path("checkpoints") / config['problem']
                    if checkpoints_root.exists():
                        checkpoint_pattern = f"layers-{layers_str}_act-{config['activation']}"
                        checkpoint_dirs = list(checkpoints_root.glob(checkpoint_pattern))
                        
                        if checkpoint_dirs:
                            checkpoint_dir = checkpoint_dirs[0]
                            # Create checkpoints folder in experiment directory
                            exp_checkpoints_dir = exp_output_dir / "checkpoints"
                            exp_checkpoints_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Move checkpoint directory
                            dest_checkpoint = exp_checkpoints_dir / checkpoint_dir.name
                            if dest_checkpoint.exists():
                                shutil.rmtree(dest_checkpoint)
                            shutil.move(str(checkpoint_dir), str(dest_checkpoint))
                    
                    # Return the moved timestamp directory
                    return dest_dir
        
        return None
        
    finally:
        # Restore original config
        if config_backup_path.exists():
            shutil.move(str(config_backup_path), 'config/config.yaml')


def generate_comparison_report(parent_dir, results):
    """Generate comparison plots and tables."""
    print(f"\n{'='*70}")
    print("Generating Comparison Report")
    print(f"{'='*70}\n")
    
    # Collect training metrics
    metrics_data = []
    ncc_data = {}  # Store all NCC data for periodic plots
    probe_data = {}  # Store all probe data for comparison plots
    
    for exp_name, result_path in results.items():
        if result_path is None:
            continue
            
        # Load training metrics
        metrics_file = result_path / "metrics.json"
        if not metrics_file.exists():
            continue
            
        with open(metrics_file) as f:
            train_metrics = json.load(f)
        
        # Collect NCC metrics from all epochs (periodic + final)
        ncc_plots_dir = result_path / "ncc_plots"
        ncc_epochs = {}
        
        if ncc_plots_dir.exists():
            # Load final NCC
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
        
        if not ncc_epochs:
            continue
        
        # Store for table
        final_ncc = ncc_epochs.get('final', list(ncc_epochs.values())[-1])
        
        # Load probe metrics
        probe_file = result_path / "probe_plots" / "probe_metrics.json"
        probe_metrics = None
        if probe_file.exists():
            with open(probe_file) as f:
                probe_metrics = json.load(f)
                probe_data[exp_name] = probe_metrics
        
        # Extract margin SNR for final layer
        final_layer = list(final_ncc['layer_accuracies'].keys())[-1]
        margin_mean = final_ncc['layer_margins'][final_layer]['mean_margin']
        margin_std = final_ncc['layer_margins'][final_layer]['std_margin']
        margin_snr = margin_mean / margin_std if margin_std > 0 else 0
        
        # Build metrics data row
        metrics_row = {
            'experiment': exp_name,
            'final_train_loss': train_metrics['train_loss'][-1],
            'final_eval_loss': train_metrics['eval_loss'][-1],
            'final_train_rel_l2': train_metrics['train_rel_l2'][-1],
            'final_train_inf_norm': train_metrics['train_inf_norm'][-1],
            'final_eval_rel_l2': train_metrics['eval_rel_l2'][-1],
            'final_eval_inf_norm': train_metrics['eval_inf_norm'][-1],
            'ncc_final_accuracy': final_ncc['layer_accuracies'][list(final_ncc['layer_accuracies'].keys())[-1]],
            'margin_snr': margin_snr
        }
        
        # Add probe metrics if available (last layer probe)
        if probe_metrics:
            metrics_row['probe_final_train_rel_l2'] = probe_metrics['train']['rel_l2'][-1]
            metrics_row['probe_final_eval_rel_l2'] = probe_metrics['eval']['rel_l2'][-1]
        
        metrics_data.append(metrics_row)
        
        # Store for NCC plots
        ncc_data[exp_name] = ncc_epochs
    
    if not metrics_data:
        print("  No valid results to compare.")
        return
    
    # Create comparison table
    df = pd.DataFrame(metrics_data)
    df.to_csv(parent_dir / "comparison_summary.csv", index=False)
    print(f"  Comparison table saved to comparison_summary.csv")
    
    # Generate the three plots
    _generate_training_results_plot(parent_dir, df)
    
    # Use shared comparison plot functions
    from utils.comparison_plots import generate_ncc_classification_plot, generate_ncc_compactness_plot
    generate_ncc_classification_plot(parent_dir, ncc_data)
    generate_ncc_compactness_plot(parent_dir, ncc_data)
    
    # Generate probe comparison plots if probe data available
    if probe_data:
        from utils.comparison_plots import generate_probe_comparison_plots
        generate_probe_comparison_plots(parent_dir, probe_data)
    
    print(f"\nComparison report saved to {parent_dir}")


def _generate_training_results_plot(parent_dir, df):
    """Generate training and results comparison plot."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.35)
    
    # Top row: Eval Loss and Rel-L2
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(df['experiment'], df['final_eval_loss'])
    ax1.set_title('Final Evaluation Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Experiment')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(df['experiment'], df['final_eval_rel_l2'])
    ax2.set_title('Final Evaluation Rel-L2 Error', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Experiment')
    ax2.set_ylabel('Rel-L2 Error')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Bottom: Colored table
    ax3 = fig.add_subplot(gs[1:, :])
    ax3.axis('off')
    
    # Create colored table
    table_data = []
    col_labels = ['Experiment', 'Train Loss', 'Eval Loss', 'Train Rel-L2', 'Train Inf', 'Eval Rel-L2', 'Eval Inf', 'NCC Final Acc', 'Margin SNR']
    
    for _, row in df.iterrows():
        table_data.append([
            row['experiment'],
            f"{row['final_train_loss']:.6f}",
            f"{row['final_eval_loss']:.6f}",
            f"{row['final_train_rel_l2']:.6f}",
            f"{row['final_train_inf_norm']:.6f}",
            f"{row['final_eval_rel_l2']:.6f}",
            f"{row['final_eval_inf_norm']:.6f}",
            f"{row['ncc_final_accuracy']:.6f}",
            f"{row['margin_snr']:.2f}"
        ])
    
    table = ax3.table(cellText=table_data, colLabels=col_labels,
                     cellLoc='center', loc='center',
                     bbox=[0.05, 0.1, 0.9, 0.8])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)
    
    # Color cells per column (green=best, red=worst)
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    
    # Create green-to-red colormap
    cmap = LinearSegmentedColormap.from_list('GreenRed', ['#2ecc71', '#f1c40f', '#e74c3c'])
    
    for col_idx in range(1, 9):  # Skip experiment name column (now 9 columns total)
        values = df.iloc[:, col_idx].values
        
        # For losses/errors, lower is better; for accuracy and margin SNR, higher is better
        if col_idx == 7 or col_idx == 8:  # NCC accuracy and Margin SNR - higher is better
            norm_values = 1 - (values - values.min()) / (values.max() - values.min() + 1e-10)
        else:  # Losses and errors - lower is better
            norm_values = (values - values.min()) / (values.max() - values.min() + 1e-10)
        
        for row_idx, norm_val in enumerate(norm_values):
            cell = table[(row_idx + 1, col_idx)]
            color = cmap(norm_val)
            cell.set_facecolor(color)
            cell.set_alpha(0.7)
    
    # Style header
    for col_idx in range(9):
        cell = table[(0, col_idx)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')
    
    fig.suptitle('Training and Results Comparison', fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(parent_dir / "training_and_results_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Training and results comparison saved to training_and_results_comparison.png")




def main():
    """Main experiment runner."""
    print("="*70)
    print("NCC-PINN: Automated Experiments")
    print("="*70)
    
    # Load experiment plan
    plan = load_experiment_plan()
    parent_dir = create_experiment_dir(plan)
    
    # Save experiment plan
    shutil.copy("experiments_plan.yaml", parent_dir / "experiments_plan.yaml")
    
    print(f"\nExperiment Directory: {parent_dir}")
    print(f"Total Experiments: {len(plan['experiments'])}\n")
    
    results = {}
    
    # Run each experiment
    for i, exp in enumerate(plan['experiments'], 1):
        print(f"\n[{i}/{len(plan['experiments'])}]")
        result = run_single_experiment(
            exp, 
            plan['base_config'],
            exp['name'],
            parent_dir
        )
        results[exp['name']] = result
    
    # Generate comparison report
    generate_comparison_report(parent_dir, results)
    
    print(f"\n{'='*70}")
    print("All Experiments Complete!")
    print(f"{'='*70}")
    print(f"Results saved to: {parent_dir}")


if __name__ == "__main__":
    main()

