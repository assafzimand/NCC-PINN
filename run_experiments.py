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
            print(f"\nERROR in {exp_name}: Process exited with code {result.returncode}")
            return None
        
        # Move outputs to experiment directory
        # Find latest architecture directory matching this experiment
        outputs_root = Path("outputs")
        if outputs_root.exists():
            # Build glob pattern to match architecture folder
            arch_pattern = f"{config['problem']}_layers-*_act-{config['activation']}"
            arch_dirs = sorted(outputs_root.glob(arch_pattern), 
                             key=lambda x: x.stat().st_mtime)
            
            if arch_dirs:
                # Get the most recent architecture directory
                latest_arch_dir = arch_dirs[-1]
                
                # Move entire architecture directory to experiment folder
                if exp_output_dir.exists():
                    shutil.rmtree(exp_output_dir)  # Remove if exists (shouldn't happen)
                shutil.move(str(latest_arch_dir), str(exp_output_dir))
                
                # Also move corresponding checkpoints to experiment folder
                checkpoints_root = Path("checkpoints") / config['problem']
                if checkpoints_root.exists():
                    layers_str = "-".join(map(str, config['architecture']))
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
                
                # Find the timestamp directory inside the moved architecture folder
                # Exclude the checkpoints directory we just created
                timestamp_dirs = sorted(
                    [d for d in exp_output_dir.glob("*/") if d.name != "checkpoints"], 
                    key=lambda x: x.stat().st_mtime
                )
                if timestamp_dirs:
                    return timestamp_dirs[-1]
        
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
        metrics_data.append({
            'experiment': exp_name,
            'final_train_loss': train_metrics['train_loss'][-1],
            'final_eval_loss': train_metrics['eval_loss'][-1],
            'final_train_rel_l2': train_metrics['train_rel_l2'][-1],
            'final_eval_rel_l2': train_metrics['eval_rel_l2'][-1],
            'ncc_final_accuracy': final_ncc['layer_accuracies'][list(final_ncc['layer_accuracies'].keys())[-1]]
        })
        
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
    _generate_ncc_classification_plot(parent_dir, ncc_data)
    _generate_ncc_compactness_plot(parent_dir, ncc_data)
    
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
    col_labels = ['Experiment', 'Train Loss', 'Eval Loss', 'Train Rel-L2', 'Eval Rel-L2', 'NCC Final Acc']
    
    for _, row in df.iterrows():
        table_data.append([
            row['experiment'],
            f"{row['final_train_loss']:.6f}",
            f"{row['final_eval_loss']:.6f}",
            f"{row['final_train_rel_l2']:.6f}",
            f"{row['final_eval_rel_l2']:.6f}",
            f"{row['ncc_final_accuracy']:.6f}"
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
    
    for col_idx in range(1, 6):  # Skip experiment name column
        values = df.iloc[:, col_idx].values
        
        # For losses/errors, lower is better; for accuracy, higher is better
        if col_idx == 5:  # NCC accuracy - higher is better
            norm_values = 1 - (values - values.min()) / (values.max() - values.min() + 1e-10)
        else:  # Losses and errors - lower is better
            norm_values = (values - values.min()) / (values.max() - values.min() + 1e-10)
        
        for row_idx, norm_val in enumerate(norm_values):
            cell = table[(row_idx + 1, col_idx)]
            color = cmap(norm_val)
            cell.set_facecolor(color)
            cell.set_alpha(0.7)
    
    # Style header
    for col_idx in range(6):
        cell = table[(0, col_idx)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')
    
    fig.suptitle('Training and Results Comparison', fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(parent_dir / "training_and_results_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Training and results comparison saved to training_and_results_comparison.png")


def _generate_ncc_classification_plot(parent_dir, ncc_data):
    """Generate NCC classification accuracy comparison across layers and epochs."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define color families for models
    color_families = [
        ['#ffcccc', '#ff9999', '#ff6666', '#ff3333', '#cc0000'],  # Reds
        ['#cce5ff', '#99ccff', '#66b3ff', '#3399ff', '#0066cc'],  # Blues
        ['#ccffcc', '#99ff99', '#66ff66', '#33cc33', '#009900'],  # Greens
        ['#ffe5cc', '#ffcc99', '#ffb366', '#ff9933', '#cc6600'],  # Oranges
        ['#e5ccff', '#cc99ff', '#b366ff', '#9933ff', '#6600cc'],  # Purples
    ]
    
    model_names = list(ncc_data.keys())
    
    for model_idx, (model_name, epochs_data) in enumerate(ncc_data.items()):
        color_family = color_families[model_idx % len(color_families)]
        
        # Sort epochs (numeric first, then 'final')
        sorted_epochs = sorted([e for e in epochs_data.keys() if isinstance(e, int)])
        if 'final' in epochs_data:
            sorted_epochs.append('final')
        
        for epoch_idx, epoch_key in enumerate(sorted_epochs):
            ncc_metrics = epochs_data[epoch_key]
            layers = ncc_metrics['layers_analyzed']
            accuracies = [ncc_metrics['layer_accuracies'][layer] for layer in layers]
            
            # Darker color for later epochs
            color_idx = min(epoch_idx, len(color_family) - 1)
            color = color_family[color_idx]
            
            epoch_label = f"Epoch {epoch_key}" if isinstance(epoch_key, int) else "Final"
            label = f"{model_name}" if epoch_idx == 0 else None
            
            ax.plot(layers, accuracies, marker='o', color=color, 
                   linewidth=2, markersize=6, alpha=0.8, label=label)
    
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('NCC Classification Accuracy Comparison\n(Darker shades = later epochs)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(parent_dir / "ncc_classification_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  NCC classification comparison saved to ncc_classification_comparison.png")


def _generate_ncc_compactness_plot(parent_dir, ncc_data):
    """Generate NCC compactness (margin) comparison across layers and epochs."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define color families for models
    color_families = [
        ['#ffcccc', '#ff9999', '#ff6666', '#ff3333', '#cc0000'],  # Reds
        ['#cce5ff', '#99ccff', '#66b3ff', '#3399ff', '#0066cc'],  # Blues
        ['#ccffcc', '#99ff99', '#66ff66', '#33cc33', '#009900'],  # Greens
        ['#ffe5cc', '#ffcc99', '#ffb366', '#ff9933', '#cc6600'],  # Oranges
        ['#e5ccff', '#cc99ff', '#b366ff', '#9933ff', '#6600cc'],  # Purples
    ]
    
    model_names = list(ncc_data.keys())
    
    for model_idx, (model_name, epochs_data) in enumerate(ncc_data.items()):
        color_family = color_families[model_idx % len(color_families)]
        
        # Sort epochs (numeric first, then 'final')
        sorted_epochs = sorted([e for e in epochs_data.keys() if isinstance(e, int)])
        if 'final' in epochs_data:
            sorted_epochs.append('final')
        
        for epoch_idx, epoch_key in enumerate(sorted_epochs):
            ncc_metrics = epochs_data[epoch_key]
            layers = ncc_metrics['layers_analyzed']
            margins = [ncc_metrics['layer_margins'][layer]['mean_margin'] for layer in layers]
            
            # Darker color for later epochs
            color_idx = min(epoch_idx, len(color_family) - 1)
            color = color_family[color_idx]
            
            epoch_label = f"Epoch {epoch_key}" if isinstance(epoch_key, int) else "Final"
            label = f"{model_name}" if epoch_idx == 0 else None
            
            ax.plot(layers, margins, marker='o', color=color, 
                   linewidth=2, markersize=6, alpha=0.8, label=label)
    
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Margin', fontsize=12, fontweight='bold')
    ax.set_title('NCC Compactness (Mean Margin) Comparison\n(Darker shades = later epochs)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(parent_dir / "ncc_compactness_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  NCC compactness comparison saved to ncc_compactness_comparison.png")


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

