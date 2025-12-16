"""Analysis script for capacity experiments.

This script analyzes experiment folders containing multiple model runs,
identifies non-monotonic metrics, generates comparison plots, and provides
detailed statistics on model performance.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import shutil
import re
from collections import defaultdict
import yaml

# Import comparison plot functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.comparison_plots import (
    generate_ncc_classification_plot,
    generate_ncc_compactness_plot,
    generate_probe_comparison_plots,
    generate_derivatives_comparison_plots
)


# Constants
DEGRADATION_THRESHOLD = 0.10  # 10% relative degradation threshold


def parse_model_name(model_name: str) -> List[int]:
    """Extract architecture from folder name.
    
    Args:
        model_name: Folder name like "schrodinger-2-140-140-140-2-tanh"
        
    Returns:
        List of layer sizes [2, 140, 140, 140, 2]
    """
    # Extract numbers from the model name
    parts = model_name.split('-')
    architecture = []
    for part in parts:
        if part.isdigit():
            architecture.append(int(part))
    return architecture


def calculate_num_parameters(architecture: List[int]) -> int:
    """Calculate total trainable parameters from architecture.
    
    Args:
        architecture: List of layer sizes [input, h1, h2, ..., output]
        
    Returns:
        Total number of trainable parameters
    """
    total = 0
    for i in range(len(architecture) - 1):
        # Weights: prev_layer_size * current_layer_size
        total += architecture[i] * architecture[i + 1]
        # Biases: current_layer_size
        total += architecture[i + 1]
    return total


def check_monotonicity(
    metric_values: List[float],
    direction: str = 'improving',
    threshold: float = DEGRADATION_THRESHOLD
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Check if metric is monotonic with a degradation threshold.
    
    Args:
        metric_values: List of metric values across layers
        direction: 'improving' (higher is better) or 'degrading' (lower is better)
        threshold: Relative degradation threshold (default 0.10 = 10%)
        
    Returns:
        Tuple of (is_monotonic, list of violations)
        Each violation dict contains:
        - layer_idx: Index where violation occurred
        - prev_value: Previous layer value
        - current_value: Current layer value
        - degradation_abs: Absolute degradation
        - degradation_rel: Relative degradation percentage
        - passed_threshold: Whether degradation > threshold (10% by default)
    """
    violations = []
    
    for i in range(1, len(metric_values)):
        prev_val = metric_values[i - 1]
        curr_val = metric_values[i]
        
        if direction == 'improving':
            # Higher is better - check if current < previous
            if curr_val < prev_val:
                degradation_abs = prev_val - curr_val
                degradation_rel = degradation_abs / prev_val if prev_val != 0 else 0
                # Check if degradation exceeds threshold (10% by default)
                passed_threshold = degradation_rel > threshold
                
                violations.append({
                    'layer_idx': i,
                    'prev_value': prev_val,
                    'current_value': curr_val,
                    'degradation_abs': degradation_abs,
                    'degradation_rel': degradation_rel,
                    'passed_threshold': passed_threshold
                })
        else:  # degrading - lower is better
            # Check if current > previous
            if curr_val > prev_val:
                degradation_abs = curr_val - prev_val
                degradation_rel = degradation_abs / prev_val if prev_val != 0 else 0
                # Check if degradation exceeds threshold (10% by default)
                passed_threshold = degradation_rel > threshold
                
                violations.append({
                    'layer_idx': i,
                    'prev_value': prev_val,
                    'current_value': curr_val,
                    'degradation_abs': degradation_abs,
                    'degradation_rel': degradation_rel,
                    'passed_threshold': passed_threshold
                })
    
    is_monotonic = len(violations) == 0
    return is_monotonic, violations


def load_experiment_plan(experiment_path: Path) -> Optional[Dict[str, Any]]:
    """Load experiment plan YAML to map experiment names to architectures.
    
    Args:
        experiment_path: Path to experiment folder
        
    Returns:
        Dict mapping experiment name -> architecture, or None if not found
    """
    plan_file = experiment_path / "experiments_plan.yaml"
    if not plan_file.exists():
        return None
    
    with open(plan_file, 'r') as f:
        plan = yaml.safe_load(f)
    
    # Create mapping from experiment name to architecture
    name_to_arch = {}
    if 'experiments' in plan:
        for exp in plan['experiments']:
            if 'name' in exp and 'architecture' in exp:
                name_to_arch[exp['name']] = exp['architecture']
    
    return name_to_arch


def load_all_model_metrics(experiment_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load all metrics from all models in experiment.
    
    Args:
        experiment_path: Path to experiment folder
        
    Returns:
        Dict mapping model_name -> {
            'model_name': str,
            'architecture': List[int],
            'num_layers': int,
            'num_parameters': int,
            'run_dir': Path,
            'metrics': dict from metrics.json,
            'ncc_metrics': dict from ncc_metrics.json,
            'probe_metrics': dict from probe_metrics.json,
            'derivatives_metrics': dict from derivatives_metrics.json,
            'eval_rel_l2': float,
            'eval_linf': float
        }
    """
    models_data = {}
    experiment_path = Path(experiment_path)
    
    # Find all model folders (directories that match pattern)
    for model_dir in experiment_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # Skip comparison files and other non-model directories
        if model_name.startswith('.') or model_name in ['comparison_summary.csv', 'experiments_plan.yaml']:
            continue
        
        # Find the timestamped run directory (most recent one)
        run_dirs = [d for d in model_dir.iterdir() if d.is_dir() and re.match(r'\d{8}_\d{6}', d.name)]
        if not run_dirs:
            continue
        
        # Get most recent run directory
        run_dir = max(run_dirs, key=lambda x: x.stat().st_mtime)
        
        # Load metrics.json
        metrics_file = run_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Extract eval metrics
        eval_rel_l2 = metrics.get('eval_rel_l2', [])
        eval_linf = metrics.get('eval_inf_norm', [])
        
        final_eval_rel_l2 = eval_rel_l2[-1] if eval_rel_l2 else None
        final_eval_linf = eval_linf[-1] if eval_linf else None
        
        # Try to get from comparison_summary.csv as fallback
        if final_eval_rel_l2 is None or final_eval_linf is None:
            comparison_file = experiment_path / "comparison_summary.csv"
            if comparison_file.exists():
                # Match by architecture - find row with matching architecture
                # We'll match by finding the architecture in the experiments_plan.yaml
                # For now, try to match by model folder name pattern
                # The comparison_summary has experiment names, not model folder names
                # We'll use the metrics.json values if available, otherwise skip
                pass
        
        # Load NCC metrics
        ncc_metrics = None
        ncc_file = run_dir / "ncc_plots" / "ncc_metrics.json"
        if ncc_file.exists():
            with open(ncc_file, 'r') as f:
                ncc_metrics = json.load(f)
        
        # Load probe metrics
        probe_metrics = None
        probe_file = run_dir / "probe_plots" / "probe_metrics.json"
        if probe_file.exists():
            with open(probe_file, 'r') as f:
                probe_metrics = json.load(f)
        
        # Load derivatives metrics
        derivatives_metrics = None
        deriv_file = run_dir / "derivatives_plots" / "derivatives_metrics.json"
        if deriv_file.exists():
            with open(deriv_file, 'r') as f:
                derivatives_metrics = json.load(f)
        
        # Parse architecture
        architecture = parse_model_name(model_name)
        num_layers = len(architecture) - 2  # Exclude input and output layers
        num_parameters = calculate_num_parameters(architecture)
        
        models_data[model_name] = {
            'model_name': model_name,
            'architecture': architecture,
            'num_layers': num_layers,
            'num_parameters': num_parameters,
            'run_dir': run_dir,
            'metrics': metrics,
            'ncc_metrics': ncc_metrics,
            'probe_metrics': probe_metrics,
            'derivatives_metrics': derivatives_metrics,
            'eval_rel_l2': final_eval_rel_l2,
            'eval_linf': final_eval_linf
        }
    
    return models_data


def extract_weight_from_experiment_name(exp_name: str) -> Optional[str]:
    """Extract weight count from experiment name (e.g., "40k" from "3lyr_40k_Wn_flat").
    
    Args:
        exp_name: Experiment name like "3lyr_40k_Wn_flat"
        
    Returns:
        Weight string like "40k", or None if not found
    """
    # Pattern: number followed by 'k' (e.g., "40k", "30k")
    match = re.search(r'(\d+k)', exp_name)
    if match:
        return match.group(1)
    return None


def group_models_by_attribute(
    models_data: Dict[str, Dict[str, Any]],
    attribute: str,
    experiment_plan: Optional[Dict[str, List[int]]] = None
) -> Dict[Any, List[str]]:
    """Group models by attribute (layer count or weight count).
    
    Args:
        models_data: Dict from load_all_model_metrics
        attribute: 'num_layers' or 'num_parameters'
        experiment_plan: Optional mapping from experiment name to architecture
        
    Returns:
        Dict mapping attribute_value -> list of model names
    """
    groups = defaultdict(list)
    
    if attribute == 'num_parameters' and experiment_plan:
        # For weight-based grouping, use experiment plan to match architectures
        # and extract weight count from experiment name (e.g., "40k" from "3lyr_40k_Wn_flat")
        
        # Create mapping: architecture -> weight_count from experiment name
        arch_to_weight = {}
        for exp_name, arch in experiment_plan.items():
            weight_str = extract_weight_from_experiment_name(exp_name)
            if weight_str:
                # Use tuple of architecture as key (to handle multiple experiments with same arch)
                arch_tuple = tuple(arch)
                if arch_tuple not in arch_to_weight:
                    arch_to_weight[arch_tuple] = weight_str
        
        # Now match model folder names to experiment names via architecture
        for model_name, data in models_data.items():
            model_arch = data['architecture']
            arch_tuple = tuple(model_arch)
            
            # Find matching weight count from experiment plan
            weight_count = arch_to_weight.get(arch_tuple)
            if weight_count:
                groups[weight_count].append(model_name)
            else:
                # Fallback: use calculated parameters if no match found
                groups[data['num_parameters']].append(model_name)
    else:
        # For layer-based grouping, use the attribute directly
        for model_name, data in models_data.items():
            key = data[attribute]
            groups[key].append(model_name)
    
    return dict(groups)


def detect_non_monotonic_metrics(models_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detect non-monotonic metrics across all models.
    
    Args:
        models_data: Dict from load_all_model_metrics
        
    Returns:
        List of non-monotonic violations, each with:
        - model_name: str
        - metric_type: str (NCC, Probe, Derivatives_Residual, etc.)
        - layer_idx: int
        - layer_name: str
        - prev_value: float
        - current_value: float
        - degradation_abs: float
        - degradation_rel: float
        - passed_threshold: bool
    """
    violations = []
    
    for model_name, data in models_data.items():
        # Check NCC metrics
        if data['ncc_metrics']:
            ncc_metrics = data['ncc_metrics']
            layers = ncc_metrics.get('layers_analyzed', [])
            
            # Check layer accuracies (should increase)
            if 'layer_accuracies' in ncc_metrics:
                accuracies = [ncc_metrics['layer_accuracies'][layer] for layer in layers]
                is_mono, viols = check_monotonicity(accuracies, direction='improving')
                for v in viols:
                    violations.append({
                        'model_name': model_name,
                        'metric_type': 'NCC',
                        'layer_idx': v['layer_idx'],
                        'layer_name': layers[v['layer_idx']],
                        'prev_value': v['prev_value'],
                        'current_value': v['current_value'],
                        'degradation_abs': v['degradation_abs'],
                        'degradation_rel': v['degradation_rel'],
                        'passed_threshold': v['passed_threshold']
                    })
        
        # Check Probe metrics
        if data['probe_metrics']:
            probe_metrics = data['probe_metrics']
            layers = probe_metrics.get('layers_analyzed', [])
            
            # Check train rel_l2 (should decrease)
            if 'train' in probe_metrics and 'rel_l2' in probe_metrics['train']:
                rel_l2_values = probe_metrics['train']['rel_l2']
                is_mono, viols = check_monotonicity(rel_l2_values, direction='degrading')
                for v in viols:
                    violations.append({
                        'model_name': model_name,
                        'metric_type': 'Probe',
                        'layer_idx': v['layer_idx'],
                        'layer_name': layers[v['layer_idx']] if v['layer_idx'] < len(layers) else f"layer_{v['layer_idx']+1}",
                        'prev_value': v['prev_value'],
                        'current_value': v['current_value'],
                        'degradation_abs': v['degradation_abs'],
                        'degradation_rel': v['degradation_rel'],
                        'passed_threshold': v['passed_threshold']
                    })
            
            # Check eval rel_l2 (should decrease)
            if 'eval' in probe_metrics and 'rel_l2' in probe_metrics['eval']:
                rel_l2_values = probe_metrics['eval']['rel_l2']
                is_mono, viols = check_monotonicity(rel_l2_values, direction='degrading')
                for v in viols:
                    violations.append({
                        'model_name': model_name,
                        'metric_type': 'Probe',
                        'layer_idx': v['layer_idx'],
                        'layer_name': layers[v['layer_idx']] if v['layer_idx'] < len(layers) else f"layer_{v['layer_idx']+1}",
                        'prev_value': v['prev_value'],
                        'current_value': v['current_value'],
                        'degradation_abs': v['degradation_abs'],
                        'degradation_rel': v['degradation_rel'],
                        'passed_threshold': v['passed_threshold'],
                        'split': 'eval'  # Mark as eval split
                    })
        
        # Check Derivatives metrics - 4 categories
        if data['derivatives_metrics']:
            deriv_metrics = data['derivatives_metrics']
            layers = deriv_metrics.get('layers_analyzed', [])
            
            # 1. Residual
            if 'train' in deriv_metrics:
                residual_values_train = []
                residual_values_eval = []
                for layer in layers:
                    if layer in deriv_metrics['train']:
                        residual_values_train.append(deriv_metrics['train'][layer].get('residual_norm', np.nan))
                    if layer in deriv_metrics['eval']:
                        residual_values_eval.append(deriv_metrics['eval'][layer].get('residual_norm', np.nan))
                
                # Check train residual (should decrease)
                if len(residual_values_train) > 1:
                    is_mono, viols = check_monotonicity(residual_values_train, direction='degrading')
                    for v in viols:
                        violations.append({
                            'model_name': model_name,
                            'metric_type': 'Derivatives_Residual',
                            'layer_idx': v['layer_idx'],
                            'layer_name': layers[v['layer_idx']] if v['layer_idx'] < len(layers) else f"layer_{v['layer_idx']+1}",
                            'prev_value': v['prev_value'],
                            'current_value': v['current_value'],
                            'degradation_abs': v['degradation_abs'],
                            'degradation_rel': v['degradation_rel'],
                            'passed_threshold': v['passed_threshold'],
                            'split': 'train'
                        })
                
                # Check eval residual (should decrease)
                if len(residual_values_eval) > 1:
                    is_mono, viols = check_monotonicity(residual_values_eval, direction='degrading')
                    for v in viols:
                        violations.append({
                            'model_name': model_name,
                            'metric_type': 'Derivatives_Residual',
                            'layer_idx': v['layer_idx'],
                            'layer_name': layers[v['layer_idx']] if v['layer_idx'] < len(layers) else f"layer_{v['layer_idx']+1}",
                            'prev_value': v['prev_value'],
                            'current_value': v['current_value'],
                            'degradation_abs': v['degradation_abs'],
                            'degradation_rel': v['degradation_rel'],
                            'passed_threshold': v['passed_threshold'],
                            'split': 'eval'
                        })
            
            # 2. IC (Initial Condition)
            if 'ic' in deriv_metrics:
                for split in ['train', 'eval']:
                    if split in deriv_metrics['ic']:
                        ic_l2_values = []
                        ic_linf_values = []
                        for layer in layers:
                            if layer in deriv_metrics['ic'][split]:
                                ic_l2_values.append(deriv_metrics['ic'][split][layer].get('l2', np.nan))
                                ic_linf_values.append(deriv_metrics['ic'][split][layer].get('linf', np.nan))
                        
                        # Check L2 (should decrease)
                        if len(ic_l2_values) > 1:
                            is_mono, viols = check_monotonicity(ic_l2_values, direction='degrading')
                            for v in viols:
                                violations.append({
                                    'model_name': model_name,
                                    'metric_type': 'Derivatives_IC',
                                    'layer_idx': v['layer_idx'],
                                    'layer_name': layers[v['layer_idx']] if v['layer_idx'] < len(layers) else f"layer_{v['layer_idx']+1}",
                                    'prev_value': v['prev_value'],
                                    'current_value': v['current_value'],
                                    'degradation_abs': v['degradation_abs'],
                                    'degradation_rel': v['degradation_rel'],
                                    'passed_threshold': v['passed_threshold'],
                                    'split': split,
                                    'norm': 'l2'
                                })
            
            # 3. BC Values
            if 'bc_value' in deriv_metrics:
                for split in ['train', 'eval']:
                    if split in deriv_metrics['bc_value']:
                        bc_value_l2_values = []
                        bc_value_linf_values = []
                        for layer in layers:
                            if layer in deriv_metrics['bc_value'][split]:
                                bc_value_l2_values.append(deriv_metrics['bc_value'][split][layer].get('l2', np.nan))
                                bc_value_linf_values.append(deriv_metrics['bc_value'][split][layer].get('linf', np.nan))
                        
                        # Check L2 (should decrease)
                        if len(bc_value_l2_values) > 1:
                            is_mono, viols = check_monotonicity(bc_value_l2_values, direction='degrading')
                            for v in viols:
                                violations.append({
                                    'model_name': model_name,
                                    'metric_type': 'Derivatives_BC_Value',
                                    'layer_idx': v['layer_idx'],
                                    'layer_name': layers[v['layer_idx']] if v['layer_idx'] < len(layers) else f"layer_{v['layer_idx']+1}",
                                    'prev_value': v['prev_value'],
                                    'current_value': v['current_value'],
                                    'degradation_abs': v['degradation_abs'],
                                    'degradation_rel': v['degradation_rel'],
                                    'passed_threshold': v['passed_threshold'],
                                    'split': split,
                                    'norm': 'l2'
                                })
            
            # 4. BC Derivatives
            if 'bc_derivative' in deriv_metrics:
                for split in ['train', 'eval']:
                    if split in deriv_metrics['bc_derivative']:
                        bc_deriv_l2_values = []
                        bc_deriv_linf_values = []
                        for layer in layers:
                            if layer in deriv_metrics['bc_derivative'][split]:
                                bc_deriv_l2_values.append(deriv_metrics['bc_derivative'][split][layer].get('l2', np.nan))
                                bc_deriv_linf_values.append(deriv_metrics['bc_derivative'][split][layer].get('linf', np.nan))
                        
                        # Check L2 (should decrease)
                        if len(bc_deriv_l2_values) > 1:
                            is_mono, viols = check_monotonicity(bc_deriv_l2_values, direction='degrading')
                            for v in viols:
                                violations.append({
                                    'model_name': model_name,
                                    'metric_type': 'Derivatives_BC_Derivative',
                                    'layer_idx': v['layer_idx'],
                                    'layer_name': layers[v['layer_idx']] if v['layer_idx'] < len(layers) else f"layer_{v['layer_idx']+1}",
                                    'prev_value': v['prev_value'],
                                    'current_value': v['current_value'],
                                    'degradation_abs': v['degradation_abs'],
                                    'degradation_rel': v['degradation_rel'],
                                    'passed_threshold': v['passed_threshold'],
                                    'split': split,
                                    'norm': 'l2'
                                })
    
    return violations


def copy_change_heatmaps(
    violations: List[Dict[str, Any]],
    models_data: Dict[str, Dict[str, Any]],
    non_mono_dir: Path
):
    """Copy change heatmaps for non-monotonic models.
    
    Args:
        violations: List of violations from detect_non_monotonic_metrics
        models_data: Dict from load_all_model_metrics
        non_mono_dir: Non-monotonic comparisons directory (heatmaps go inside)
    """
    heatmap_dir = non_mono_dir / "non_monotonic_heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    
    # Group violations by model and metric type, collect layer info
    model_violations = defaultdict(lambda: defaultdict(list))
    for v in violations:
        if v['passed_threshold']:
            model_name = v['model_name']
            metric_type = v['metric_type']
            layer_name = v.get('layer_name', f"layer_{v['layer_idx']+1}")
            model_violations[model_name][metric_type].append(layer_name)
    
    # Copy heatmaps
    for model_name, metric_dict in model_violations.items():
        if model_name not in models_data:
            continue
        
        run_dir = models_data[model_name]['run_dir']
        
        for metric_type, layers in metric_dict.items():
            # Create folder name with layer info: model_name_layer_X
            layer_str = "_".join(sorted(set(layers)))  # e.g., "layer_2" or "layer_1_layer_3"
            folder_name = f"{model_name}_{layer_str}"
            model_heatmap_dir = heatmap_dir / folder_name / metric_type
            model_heatmap_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy based on metric type
            if metric_type == 'NCC':
                src1 = run_dir / "ncc_plots" / "ncc_classification_accuracy_changes.png"
                src2 = run_dir / "ncc_plots" / "ncc_classification_input_space_accuracy_changes.png"
                if src1.exists():
                    shutil.copy2(src1, model_heatmap_dir / src1.name)
                if src2.exists():
                    shutil.copy2(src2, model_heatmap_dir / src2.name)
            
            elif metric_type == 'Probe':
                src = run_dir / "probe_plots" / "probe_error_change_heatmaps.png"
                if src.exists():
                    shutil.copy2(src, model_heatmap_dir / src.name)
            
            elif metric_type == 'Derivatives_Residual':
                src = run_dir / "derivatives_plots" / "residual_change_heatmaps.png"
                if src.exists():
                    shutil.copy2(src, model_heatmap_dir / src.name)
            
            # Derivatives_IC, Derivatives_BC_Value, Derivatives_BC_Derivative have no change heatmaps
            # Skip copying for them


def generate_rankings(models_data: Dict[str, Dict[str, Any]], output_dir: Path):
    """Generate model rankings by eval_rel_L2 and eval_L_inf as images with tables.
    
    Args:
        models_data: Dict from load_all_model_metrics
        output_dir: Output directory for analysis
    """
    import matplotlib.pyplot as plt
    
    # Prepare data for ranking
    ranking_data = []
    for model_name, data in models_data.items():
        ranking_data.append({
            'model_name': model_name,
            'num_layers': data['num_layers'],
            'num_parameters': data['num_parameters'],
            'eval_rel_l2': data['eval_rel_l2'],
            'eval_linf': data['eval_linf']
        })
    
    df = pd.DataFrame(ranking_data)
    
    # Sort by eval_rel_l2 (best = lowest)
    df_rel_l2 = df.sort_values('eval_rel_l2').reset_index(drop=True)
    df_rel_l2['rank'] = range(1, len(df_rel_l2) + 1)
    df_rel_l2 = df_rel_l2[['rank', 'model_name', 'num_layers', 'num_parameters', 'eval_rel_l2', 'eval_linf']]
    
    # Sort by eval_linf (best = lowest)
    df_linf = df.sort_values('eval_linf').reset_index(drop=True)
    df_linf['rank'] = range(1, len(df_linf) + 1)
    df_linf = df_linf[['rank', 'model_name', 'num_layers', 'num_parameters', 'eval_rel_l2', 'eval_linf']]
    
    # Save CSV files
    df_rel_l2.to_csv(output_dir / "model_ranking_by_eval_rel_l2.csv", index=False)
    df_linf.to_csv(output_dir / "model_ranking_by_eval_linf.csv", index=False)
    
    # Create table images
    def create_ranking_table_image(df_ranked, title, save_path):
        fig, ax = plt.subplots(figsize=(16, max(8, len(df_ranked) * 0.4)))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        for _, row in df_ranked.iterrows():
            table_data.append([
                str(int(row['rank'])),
                row['model_name'],
                str(int(row['num_layers'])),
                f"{int(row['num_parameters']):,}",
                f"{row['eval_rel_l2']:.6f}",
                f"{row['eval_linf']:.6f}"
            ])
        
        table = ax.table(
            cellText=table_data,
            colLabels=['Rank', 'Model Name', 'Layers', 'Parameters', 'Eval Rel-L2', 'Eval L∞'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(6):
            cell = table[(0, i)]
            cell.set_facecolor('#34495e')
            cell.set_text_props(weight='bold', color='white')
        
        # Color code best/worst rows
        for i in range(1, len(table_data) + 1):
            if i == 1:  # Best
                for j in range(6):
                    table[(i, j)].set_facecolor('#d4edda')
            elif i == len(table_data):  # Worst
                for j in range(6):
                    table[(i, j)].set_facecolor('#f8d7da')
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Generate table images
    create_ranking_table_image(
        df_rel_l2,
        'Model Ranking by Eval Rel-L2 Error (Lower is Better)',
        output_dir / "model_ranking_by_eval_rel_l2.png"
    )
    
    create_ranking_table_image(
        df_linf,
        'Model Ranking by Eval L∞ Error (Lower is Better)',
        output_dir / "model_ranking_by_eval_linf.png"
    )
    
    print(f"  Rankings saved to {output_dir}")


def organize_non_monotonic_by_metric(
    violations: List[Dict[str, Any]],
    models_data: Dict[str, Dict[str, Any]],
    non_mono_dir: Path,
    experiment_plan: Optional[Dict[str, List[int]]] = None
):
    """Organize non-monotonic models by metric type.
    
    For each metric type that has violations:
    - Create a folder for that metric
    - Generate a comparison plot showing all models that violated it
    - Copy relevant heatmaps to that metric folder
    
    Args:
        violations: List of violations from detect_non_monotonic_metrics
        models_data: Dict of all models data (to ensure we have all model data)
        non_mono_dir: Non-monotonic comparisons directory
        experiment_plan: Optional mapping from experiment name to architecture
    """
    # Group violations by metric type
    metric_violations = defaultdict(list)
    for v in violations:
        if v['passed_threshold']:
            metric_type = v['metric_type']
            metric_violations[metric_type].append(v)
    
    # Process each metric type
    for metric_type, metric_viols in metric_violations.items():
        # Get unique models for this metric type
        models_for_metric = set(v['model_name'] for v in metric_viols)
        
        if not models_for_metric:
            continue
        
        # Create folder for this metric type
        metric_dir = non_mono_dir / metric_type
        metric_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"    Processing {metric_type}: {len(models_for_metric)} models")
        
        # Prepare data for plotting (only models that violated this metric)
        # Use models_data to ensure we have all model data available
        metric_models_data = {k: v for k, v in models_data.items() if k in models_for_metric}
        
        # Debug: Check if we're missing models
        missing_models = models_for_metric - set(metric_models_data.keys())
        if missing_models:
            print(f"      Warning: {len(missing_models)} models in violations but not in models_data: {missing_models}")
        
        # Generate comparison plot for this metric
        if metric_type == 'NCC':
            ncc_data = {}
            for model_name, data in metric_models_data.items():
                if data.get('ncc_metrics'):  # Use .get() to be safer
                    ncc_data[model_name] = {'final': data['ncc_metrics']}
            if ncc_data:
                generate_ncc_classification_plot(metric_dir, ncc_data)
                generate_ncc_compactness_plot(metric_dir, ncc_data)
            else:
                print(f"      Warning: No NCC data found for {len(metric_models_data)} models")
        
        elif metric_type == 'Probe':
            probe_data = {}
            for model_name, data in metric_models_data.items():
                if data.get('probe_metrics'):  # Use .get() to be safer
                    probe_data[model_name] = data['probe_metrics']
            if probe_data:
                generate_probe_comparison_plots(metric_dir, probe_data)
            else:
                print(f"      Warning: No Probe data found for {len(metric_models_data)} models")
        
        elif metric_type.startswith('Derivatives_'):
            derivatives_data = {}
            for model_name, data in metric_models_data.items():
                if data.get('derivatives_metrics'):  # Use .get() to be safer
                    derivatives_data[model_name] = data['derivatives_metrics']
            if derivatives_data:
                generate_derivatives_comparison_plots(metric_dir, derivatives_data)
            else:
                print(f"      Warning: No Derivatives data found for {len(metric_models_data)} models")
        
        # Copy relevant heatmaps to this metric folder
        copy_heatmaps_for_metric(metric_viols, metric_models_data, metric_dir)


def copy_heatmaps_for_metric(
    metric_violations: List[Dict[str, Any]],
    models_data: Dict[str, Dict[str, Any]],
    metric_dir: Path
):
    """Copy heatmaps for models that violated a specific metric.
    
    Args:
        metric_violations: List of violations for this metric type
        models_data: Dict of model data
        metric_dir: Directory for this metric type
    """
    heatmap_dir = metric_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by model and collect layer info
    model_layers = defaultdict(set)
    for v in metric_violations:
        model_name = v['model_name']
        layer_name = v.get('layer_name', f"layer_{v['layer_idx']+1}")
        model_layers[model_name].add(layer_name)
    
    # Copy heatmaps for each model
    for model_name, layers in model_layers.items():
        if model_name not in models_data:
            continue
        
        run_dir = models_data[model_name]['run_dir']
        metric_type = metric_violations[0]['metric_type']  # All same metric type
        
        # Create model folder with layer info
        layer_str = "_".join(sorted(layers))
        model_heatmap_dir = heatmap_dir / f"{model_name}_{layer_str}"
        model_heatmap_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy based on metric type
        if metric_type == 'NCC':
            src1 = run_dir / "ncc_plots" / "ncc_classification_accuracy_changes.png"
            src2 = run_dir / "ncc_plots" / "ncc_classification_input_space_accuracy_changes.png"
            if src1.exists():
                shutil.copy2(src1, model_heatmap_dir / src1.name)
            if src2.exists():
                shutil.copy2(src2, model_heatmap_dir / src2.name)
        
        elif metric_type == 'Probe':
            src = run_dir / "probe_plots" / "probe_error_change_heatmaps.png"
            if src.exists():
                shutil.copy2(src, model_heatmap_dir / src.name)
        
        elif metric_type == 'Derivatives_Residual':
            src = run_dir / "derivatives_plots" / "residual_change_heatmaps.png"
            if src.exists():
                shutil.copy2(src, model_heatmap_dir / src.name)
        
        # Derivatives_IC, Derivatives_BC_Value, Derivatives_BC_Derivative have no heatmaps
        # Skip copying for them


def generate_non_monotonic_ranking(
    violations: List[Dict[str, Any]],
    models_data: Dict[str, Dict[str, Any]],
    output_dir: Path
):
    """Generate ranking table for non-monotonic models by worst degradation.
    
    Args:
        violations: List of violations from detect_non_monotonic_metrics
        models_data: Dict from load_all_model_metrics
        output_dir: Output directory for analysis
    """
    import matplotlib.pyplot as plt
    
    # Filter to only violations that passed threshold
    passed_violations = [v for v in violations if v['passed_threshold']]
    
    if not passed_violations:
        return
    
    # For each model, find the worst degradation
    model_worst_degradation = {}
    for v in passed_violations:
        model_name = v['model_name']
        if model_name not in models_data:
            continue
        
        degradation_rel = v['degradation_rel']
        
        if model_name not in model_worst_degradation:
            model_worst_degradation[model_name] = {
                'worst_degradation_rel': degradation_rel,
                'worst_degradation_abs': v['degradation_abs'],
                'worst_layer_idx': v['layer_idx'],
                'worst_layer_name': v.get('layer_name', f"layer_{v['layer_idx']+1}"),
                'worst_metric_type': v['metric_type'],
                'violation_count': 1
            }
        else:
            # Update if this violation is worse
            if degradation_rel > model_worst_degradation[model_name]['worst_degradation_rel']:
                model_worst_degradation[model_name]['worst_degradation_rel'] = degradation_rel
                model_worst_degradation[model_name]['worst_degradation_abs'] = v['degradation_abs']
                model_worst_degradation[model_name]['worst_layer_idx'] = v['layer_idx']
                model_worst_degradation[model_name]['worst_layer_name'] = v.get('layer_name', f"layer_{v['layer_idx']+1}")
                model_worst_degradation[model_name]['worst_metric_type'] = v['metric_type']
            model_worst_degradation[model_name]['violation_count'] += 1
    
    # Prepare data for ranking
    ranking_data = []
    for model_name, degradation_info in model_worst_degradation.items():
        data = models_data[model_name]
        ranking_data.append({
            'model_name': model_name,
            'num_layers': data['num_layers'],
            'num_parameters': data['num_parameters'],
            'worst_degradation_rel': degradation_info['worst_degradation_rel'],
            'worst_degradation_abs': degradation_info['worst_degradation_abs'],
            'worst_layer': degradation_info['worst_layer_name'],
            'worst_metric_type': degradation_info['worst_metric_type'],
            'violation_count': degradation_info['violation_count']
        })
    
    df = pd.DataFrame(ranking_data)
    
    # Sort by worst degradation (biggest = rank 1, worst)
    df_ranked = df.sort_values('worst_degradation_rel', ascending=False).reset_index(drop=True)
    df_ranked['rank'] = range(1, len(df_ranked) + 1)
    df_ranked = df_ranked[['rank', 'model_name', 'num_layers', 'num_parameters', 
                          'worst_degradation_rel', 'worst_degradation_abs', 
                          'worst_layer', 'worst_metric_type', 'violation_count']]
    
    # Save CSV
    df_ranked.to_csv(output_dir / "non_monotonic_ranking.csv", index=False)
    
    # Create table image
    fig, ax = plt.subplots(figsize=(18, max(8, len(df_ranked) * 0.4)))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for _, row in df_ranked.iterrows():
        table_data.append([
            str(int(row['rank'])),
            row['model_name'],
            str(int(row['num_layers'])),
            f"{int(row['num_parameters']):,}",
            f"{row['worst_degradation_rel']*100:.2f}%",
            f"{row['worst_degradation_abs']:.6f}",
            row['worst_layer'],
            row['worst_metric_type'],
            str(int(row['violation_count']))
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Rank', 'Model Name', 'Layers', 'Parameters', 'Worst Degradation (%)', 
                  'Worst Degradation (abs)', 'Worst Layer', 'Metric Type', 'Violations'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(9):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')
    
    # Color code worst/best rows (rank 1 = worst degradation = red, last = best = green)
    for i in range(1, len(table_data) + 1):
        if i == 1:  # Worst degradation
            for j in range(9):
                table[(i, j)].set_facecolor('#f8d7da')  # Light red
        elif i == len(table_data):  # Best (least degradation)
            for j in range(9):
                table[(i, j)].set_facecolor('#d4edda')  # Light green
    
    plt.title('Non-Monotonic Models Ranking by Worst Degradation (Rank 1 = Worst)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_dir / "non_monotonic_ranking.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Non-monotonic ranking saved to {output_dir}")


def generate_comparison_plots(
    models_data: Dict[str, Dict[str, Any]],
    output_dir: Path,
    group_by: str = 'layers',
    experiment_plan: Optional[Dict[str, List[int]]] = None,
    add_weights_to_labels: bool = False
):
    """Generate comparison plots for similar models.
    
    Args:
        models_data: Dict from load_all_model_metrics
        output_dir: Output directory for analysis
        group_by: 'layers' or 'weights'
        experiment_plan: Optional mapping from experiment name to architecture
        add_weights_to_labels: If True, add weight count to model names in legends
    """
    if group_by == 'layers':
        groups = group_models_by_attribute(models_data, 'num_layers', experiment_plan)
        base_dir = output_dir / "comparisons_by_layers"
    else:
        groups = group_models_by_attribute(models_data, 'num_parameters', experiment_plan)
        base_dir = output_dir / "comparisons_by_weights"
    
    base_dir.mkdir(parents=True, exist_ok=True)
    
    for group_key, model_names in groups.items():
        if len(model_names) < 2:
            continue  # Need at least 2 models for comparison
        
        if group_by == 'weights':
            # For weights, use the weight string directly (e.g., "40k")
            group_dir = base_dir / f"{group_key}_params_comparison"
        else:
            group_dir = base_dir / f"{group_key}_layers_comparison"
        group_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for comparison plots
        ncc_data = {}
        probe_data = {}
        derivatives_data = {}
        
        # Create mapping for label modification if needed
        label_map = {}
        if add_weights_to_labels:
            for model_name in model_names:
                data = models_data[model_name]
                weight_str = f"({data['num_parameters']:,} params)"
                label_map[model_name] = f"{model_name} {weight_str}"
        
        for model_name in model_names:
            data = models_data[model_name]
            
            # Use modified label if needed
            display_name = label_map.get(model_name, model_name)
            
            # NCC data
            if data['ncc_metrics']:
                ncc_data[display_name] = {'final': data['ncc_metrics']}
            
            # Probe data
            if data['probe_metrics']:
                probe_data[display_name] = data['probe_metrics']
            
            # Derivatives data
            if data['derivatives_metrics']:
                derivatives_data[display_name] = data['derivatives_metrics']
        
        # Generate plots
        if ncc_data:
            generate_ncc_classification_plot(group_dir, ncc_data)
            generate_ncc_compactness_plot(group_dir, ncc_data)
        
        if probe_data:
            generate_probe_comparison_plots(group_dir, probe_data)
        
        if derivatives_data:
            generate_derivatives_comparison_plots(group_dir, derivatives_data)
        
        print(f"  Comparison plots saved to {group_dir}")


def generate_non_monotonic_summary_plots(
    violations: List[Dict[str, Any]],
    models_data: Dict[str, Dict[str, Any]],
    output_dir: Path
):
    """Generate summary plots for non-monotonic models.
    
    Args:
        violations: List of violations from detect_non_monotonic_metrics
        models_data: Dict from load_all_model_metrics
        output_dir: Output directory for analysis
    """
    import matplotlib.pyplot as plt
    
    # Filter to only violations that passed threshold
    passed_violations = [v for v in violations if v['passed_threshold']]
    
    if not passed_violations:
        return
    
    summary_dir = output_dir / "non_monotonic_summary_plots"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    plot_data = []
    for v in passed_violations:
        model_name = v['model_name']
        if model_name not in models_data:
            continue
        data = models_data[model_name]
        plot_data.append({
            'num_layers': data['num_layers'],
            'num_parameters': data['num_parameters'],
            'layer_idx': v['layer_idx'],
            'metric_type': v['metric_type']
        })
    
    df = pd.DataFrame(plot_data)
    
    # Plot 1: Distribution by number of layers
    fig, ax = plt.subplots(figsize=(10, 6))
    layer_counts = df['num_layers'].value_counts().sort_index()
    ax.bar(layer_counts.index.astype(str), layer_counts.values, color='#3498db', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Layers', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Non-Monotonic Violations', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Non-Monotonic Violations by Layer Count', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, (layer, count) in enumerate(layer_counts.items()):
        ax.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(summary_dir / "distribution_by_layers.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Distribution by weights (group into bins)
    fig, ax = plt.subplots(figsize=(12, 6))
    param_bins = [0, 10000, 20000, 30000, 40000, 50000, float('inf')]
    param_labels = ['<10k', '10k-20k', '20k-30k', '30k-40k', '40k-50k', '50k+']
    df['param_bin'] = pd.cut(df['num_parameters'], bins=param_bins, labels=param_labels)
    weight_counts = df['param_bin'].value_counts().reindex(param_labels, fill_value=0)
    ax.bar(range(len(weight_counts)), weight_counts.values, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(weight_counts)))
    ax.set_xticklabels(weight_counts.index, rotation=45, ha='right')
    ax.set_xlabel('Number of Parameters', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Non-Monotonic Violations', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Non-Monotonic Violations by Parameter Count', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, count in enumerate(weight_counts.values):
        if count > 0:
            ax.text(i, count, str(int(count)), ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(summary_dir / "distribution_by_weights.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Which layer the violation appears in
    fig, ax = plt.subplots(figsize=(10, 6))
    layer_idx_counts = df['layer_idx'].value_counts().sort_index()
    ax.bar(layer_idx_counts.index.astype(str), layer_idx_counts.values, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Layer Index (where violation occurred)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Non-Monotonic Violations', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Non-Monotonic Violations by Layer Position', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for layer_idx, count in layer_idx_counts.items():
        ax.text(layer_idx, count, str(count), ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(summary_dir / "distribution_by_layer_position.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Which metric type the violation appears in
    fig, ax = plt.subplots(figsize=(12, 6))
    metric_counts = df['metric_type'].value_counts()
    ax.bar(range(len(metric_counts)), metric_counts.values, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(metric_counts)))
    ax.set_xticklabels(metric_counts.index, rotation=45, ha='right')
    ax.set_xlabel('Metric Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Non-Monotonic Violations', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Non-Monotonic Violations by Metric Type', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, count in enumerate(metric_counts.values):
        ax.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(summary_dir / "distribution_by_metric_type.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Summary plots saved to {summary_dir}")


def calculate_non_monotonic_statistics(
    violations: List[Dict[str, Any]],
    models_data: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """Calculate statistics on non-monotonic violations.
    
    Args:
        violations: List of violations from detect_non_monotonic_metrics
        models_data: Dict from load_all_model_metrics
        
    Returns:
        DataFrame with statistics grouped by metric type, num_layers, num_parameters, and overall
    """
    # Filter to only violations that passed threshold
    passed_violations = [v for v in violations if v['passed_threshold']]
    
    if not passed_violations:
        return pd.DataFrame()
    
    # Prepare data for statistics
    stats_data = []
    for v in passed_violations:
        model_name = v['model_name']
        if model_name not in models_data:
            continue
        
        data = models_data[model_name]
        stats_data.append({
            'metric_type': v['metric_type'],
            'num_layers': data['num_layers'],
            'num_parameters': data['num_parameters'],
            'layer_idx': v['layer_idx'],
            'degradation_abs': v['degradation_abs'],
            'degradation_rel': v['degradation_rel']
        })
    
    df = pd.DataFrame(stats_data)
    
    # Calculate statistics grouped by different attributes
    results = []
    
    # Overall statistics
    if len(df) > 0:
        results.append({
            'group_type': 'Overall',
            'group_value': 'All',
            'count': len(df),
            'mean_degradation_abs': df['degradation_abs'].mean(),
            'std_degradation_abs': df['degradation_abs'].std(),
            'mean_degradation_rel': df['degradation_rel'].mean(),
            'std_degradation_rel': df['degradation_rel'].std(),
            'mean_layer_idx': df['layer_idx'].mean(),
            'std_layer_idx': df['layer_idx'].std()
        })
    
    # By metric type
    for metric_type in df['metric_type'].unique():
        subset = df[df['metric_type'] == metric_type]
        if len(subset) > 0:
            results.append({
                'group_type': 'Metric Type',
                'group_value': metric_type,
                'count': len(subset),
                'mean_degradation_abs': subset['degradation_abs'].mean(),
                'std_degradation_abs': subset['degradation_abs'].std(),
                'mean_degradation_rel': subset['degradation_rel'].mean(),
                'std_degradation_rel': subset['degradation_rel'].std(),
                'mean_layer_idx': subset['layer_idx'].mean(),
                'std_layer_idx': subset['layer_idx'].std()
            })
    
    # By number of layers
    for num_layers in df['num_layers'].unique():
        subset = df[df['num_layers'] == num_layers]
        if len(subset) > 0:
            results.append({
                'group_type': 'Num Layers',
                'group_value': str(num_layers),
                'count': len(subset),
                'mean_degradation_abs': subset['degradation_abs'].mean(),
                'std_degradation_abs': subset['degradation_abs'].std(),
                'mean_degradation_rel': subset['degradation_rel'].mean(),
                'std_degradation_rel': subset['degradation_rel'].std(),
                'mean_layer_idx': subset['layer_idx'].mean(),
                'std_layer_idx': subset['layer_idx'].std()
            })
    
    # By number of parameters (group into bins)
    if 'num_parameters' in df.columns:
        param_bins = [0, 10000, 20000, 30000, 40000, 50000, float('inf')]
        param_labels = ['<10k', '10k-20k', '20k-30k', '30k-40k', '40k-50k', '50k+']
        df['param_bin'] = pd.cut(df['num_parameters'], bins=param_bins, labels=param_labels)
        
        for bin_label in param_labels:
            subset = df[df['param_bin'] == bin_label]
            if len(subset) > 0:
                results.append({
                    'group_type': 'Num Parameters',
                    'group_value': str(bin_label),
                    'count': len(subset),
                    'mean_degradation_abs': subset['degradation_abs'].mean(),
                    'std_degradation_abs': subset['degradation_abs'].std(),
                    'mean_degradation_rel': subset['degradation_rel'].mean(),
                    'std_degradation_rel': subset['degradation_rel'].std(),
                    'mean_layer_idx': subset['layer_idx'].mean(),
                    'std_layer_idx': subset['layer_idx'].std()
                })
    
    return pd.DataFrame(results)


def get_next_analysis_index(experiment_name: str, analysis_base_dir: Path) -> str:
    """Get the next running index for analysis output.
    
    Args:
        experiment_name: Name of the experiment
        analysis_base_dir: Base directory for analysis outputs
        
    Returns:
        Next index as string (e.g., "analysis_1", "analysis_2", ...)
    """
    exp_dir = analysis_base_dir / experiment_name
    if not exp_dir.exists():
        return "analysis_1"
    
    # Find existing indices
    existing_indices = []
    for item in exp_dir.iterdir():
        if item.is_dir() and item.name.startswith("analysis_"):
            try:
                idx = int(item.name.split("_")[1])
                existing_indices.append(idx)
            except (ValueError, IndexError):
                pass
    
    if not existing_indices:
        return "analysis_1"
    
    next_idx = max(existing_indices) + 1
    return f"analysis_{next_idx}"


def main(experiment_path: str):
    """Main analysis function.
    
    Args:
        experiment_path: Path to experiment folder
    """
    experiment_path = Path(experiment_path)
    
    # Check if it's a capacity experiment
    if 'capacity' not in experiment_path.name.lower():
        print(f"Warning: Experiment name '{experiment_path.name}' doesn't contain 'capacity'")
        print("Proceeding anyway...")
    
    experiment_name = experiment_path.name
    
    # Get output directory with running index
    analysis_base_dir = Path(__file__).parent.parent / "analysis"
    index = get_next_analysis_index(experiment_name, analysis_base_dir)
    output_dir = analysis_base_dir / experiment_name / index
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Capacity Experiment Analysis")
    print("=" * 70)
    print(f"Experiment: {experiment_name}")
    print(f"Output directory: {output_dir}")
    print(f"Analysis index: {index}")
    print()
    
    # Load experiment plan for weight-based grouping
    experiment_plan = load_experiment_plan(experiment_path)
    
    # Load all model metrics
    print("Loading model metrics...")
    models_data = load_all_model_metrics(experiment_path)
    print(f"  Loaded {len(models_data)} models")
    print()
    
    # A. Generate rankings
    print("A. Generating model rankings...")
    generate_rankings(models_data, output_dir)
    print()
    
    # B. Generate comparison plots
    print("B. Generating comparison plots...")
    generate_comparison_plots(models_data, output_dir, group_by='layers', experiment_plan=experiment_plan)
    generate_comparison_plots(models_data, output_dir, group_by='weights', experiment_plan=experiment_plan)
    
    # Don't generate top-level comparison plots in root (only in subdirectories)
    print()
    
    # C. Detect non-monotonic metrics
    print("C. Detecting non-monotonic metrics...")
    violations = detect_non_monotonic_metrics(models_data)
    
    # Count violations (with pure monotone, all violations pass threshold)
    total_violations = len(violations)
    
    print(f"  Total violations detected: {total_violations}")
    passed_count = sum(1 for v in violations if v['passed_threshold'])
    print(f"  Violations passing 10% threshold: {passed_count}")
    
    # Save violations
    with open(output_dir / "non_monotonic_models.json", 'w') as f:
        json.dump(violations, f, indent=2)
    print("  Saved violations to non_monotonic_models.json")
    
    # Generate ranking table for non-monotonic models in root folder
    non_mono_models = set(v['model_name'] for v in violations if v['passed_threshold'])
    if non_mono_models:
        print("  Generating non-monotonic ranking table in root...")
        generate_non_monotonic_ranking(violations, models_data, output_dir)
    print()
    
    # D. Generate comparison plots for non-monotonic models ONLY
    print("D. Generating comparison plots for non-monotonic models...")
    # Get unique model names that have violations (all violations count in pure monotone)
    non_mono_models = set(v['model_name'] for v in violations if v['passed_threshold'])
    
    print(f"  Found {len(non_mono_models)} unique non-monotonic models")
    print(f"  Models: {sorted(non_mono_models)}")
    
    if non_mono_models:
        # Filter to ONLY non-monotonic models
        non_mono_data = {k: v for k, v in models_data.items() if k in non_mono_models}
        non_mono_dir = output_dir / "non_monotonic_comparisons"
        non_mono_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate ranking table for non-monotonic models (by worst degradation)
        print("  Generating ranking table for non-monotonic models...")
        generate_non_monotonic_ranking(violations, models_data, non_mono_dir)
        
        # Organize by metric type
        print("  Organizing non-monotonic models by metric type...")
        organize_non_monotonic_by_metric(violations, models_data, non_mono_dir, experiment_plan)
        
        print(f"  Comparison plots saved to {non_mono_dir}")
    else:
        print("  No non-monotonic models found")
    print()
    
    # F. Calculate statistics and generate summary plots
    print("F. Calculating non-monotonic statistics and generating summary plots...")
    stats_df = calculate_non_monotonic_statistics(violations, models_data)
    
    if len(stats_df) > 0:
        stats_df.to_csv(output_dir / "non_monotonic_statistics.csv", index=False)
        
        # Generate summary text
        summary_lines = []
        summary_lines.append("Non-Monotonic (Pure Monotone) Statistics Summary")
        summary_lines.append("=" * 70)
        summary_lines.append(f"\nTotal violations: {total_violations}")
        summary_lines.append("\nStatistics by Group:\n")
        
        for _, row in stats_df.iterrows():
            summary_lines.append(f"{row['group_type']}: {row['group_value']}")
            summary_lines.append(f"  Count: {row['count']}")
            summary_lines.append(f"  Mean degradation (abs): {row['mean_degradation_abs']:.6f} ± {row['std_degradation_abs']:.6f}")
            summary_lines.append(f"  Mean degradation (rel): {row['mean_degradation_rel']*100:.2f}% ± {row['std_degradation_rel']*100:.2f}%")
            summary_lines.append(f"  Mean layer position: {row['mean_layer_idx']:.2f} ± {row['std_layer_idx']:.2f}")
            summary_lines.append("")
        
        with open(output_dir / "non_monotonic_statistics_summary.txt", 'w') as f:
            f.write("\n".join(summary_lines))
        
        # Generate summary plots
        generate_non_monotonic_summary_plots(violations, models_data, output_dir)
        
        print("  Statistics saved to non_monotonic_statistics.csv")
        print("  Summary saved to non_monotonic_statistics_summary.txt")
        print("  Summary plots saved to non_monotonic_summary_plots/")
    else:
        print("  No violations found")
    print()
    
    print("=" * 70)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_capacity_experiment.py <experiment_path>")
        print("Example: python analyze_capacity_experiment.py outputs/experiments/Schrodinger_Capacity_20251213_175504")
        sys.exit(1)
    
    experiment_path = sys.argv[1]
    main(experiment_path)

