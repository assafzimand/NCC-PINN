"""Analysis script for capacity experiments.

This script analyzes experiment folders containing multiple model runs,
identifies non-monotonic metrics, generates comparison plots, and provides
detailed statistics on model performance.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil
import re
from collections import defaultdict
import yaml
import matplotlib.pyplot as plt


def _safe_log_scale(ax, values_list):
    """Set log scale on y-axis only if all data has positive values.
    
    Returns:
        bool: True if log scale was applied, False if linear scale is used.
    """
    all_values = []
    for v in values_list:
        if isinstance(v, (list, np.ndarray)):
            all_values.extend(np.array(v).flatten())
        else:
            all_values.append(v)
    all_values = np.array(all_values)
    # Filter out NaN values for the check
    valid_values = all_values[~np.isnan(all_values)]
    if len(valid_values) > 0 and np.all(valid_values > 0):
        ax.set_yscale('log')
        return True
    return False

# Import comparison plot functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.comparison_plots import (
    generate_ncc_classification_plot,
    generate_ncc_compactness_plot,
    generate_probe_comparison_plots,
    generate_derivatives_comparison_plots
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEGRADATION_THRESHOLD = 0.1  # 5% relative degradation threshold

# Metrics configuration: defines all 7 metrics to check for monotonicity
# Each metric specifies:
#   - direction: 'increase' (higher is better) or 'decrease' (lower is better)
#   - source: which metrics file to read from ('ncc_metrics', 'probe_metrics', 'derivatives_metrics')
#   - folder: which folder contains the plots ('ncc_plots', 'probe_plots', 'derivatives_plots')
#   - extract_fn: function to extract layer values from metrics dict
METRICS_CONFIG = {
    'ncc_accuracy': {
        'direction': 'increase',
        'source': 'ncc_metrics',
        'folder': 'ncc_plots',
        'display_name': 'NCC Accuracy',
        'extract_fn': lambda m: [m['layer_accuracies'][layer] for layer in m['layers_analyzed']],
    },
    'ncc_compactness': {
        'direction': 'increase',
        'source': 'ncc_metrics',
        'folder': 'ncc_plots',
        'display_name': 'NCC Compactness (Margin SNR)',
        'extract_fn': lambda m: [
            m['layer_margins'][layer]['mean_margin'] / m['layer_margins'][layer]['std_margin']
            if m['layer_margins'][layer]['std_margin'] > 0 else 0
            for layer in m['layers_analyzed']
        ],
    },
    'probe_rel_l2': {
        'direction': 'decrease',
        'source': 'probe_metrics',
        'folder': 'probe_plots',
        'display_name': 'Probe Rel-L2 (eval)',
        'extract_fn': lambda m: m['eval']['rel_l2'],
    },
    'probe_linf': {
        'direction': 'decrease',
        'source': 'probe_metrics',
        'folder': 'probe_plots',
        'display_name': 'Probe L-inf (eval)',
        'extract_fn': lambda m: m['eval']['inf_norm'],
    },
    'residual_l2': {
        'direction': 'decrease',
        'source': 'derivatives_metrics',
        'folder': 'derivatives_plots',
        'display_name': 'Residual L2 (eval)',
        'extract_fn': lambda m: [
            m['eval'][layer]['residual_norm'] 
            for layer in m['layers_analyzed']
        ],
    },
    'residual_linf': {
        'direction': 'decrease',
        'source': 'derivatives_metrics',
        'folder': 'derivatives_plots',
        'display_name': 'Residual L-inf (eval)',
        'extract_fn': lambda m: [
            m['eval'][layer]['residual_inf_norm'] 
            for layer in m['layers_analyzed']
        ],
    },
    'ic_l2': {
        'direction': 'decrease',
        'source': 'derivatives_metrics',
        'folder': 'derivatives_plots',
        'display_name': 'IC L2 (eval)',
        'extract_fn': lambda m: [
            m['ic']['eval'][layer]['l2'] 
            for layer in m['layers_analyzed']
        ],
    },
    'ic_linf': {
        'direction': 'decrease',
        'source': 'derivatives_metrics',
        'folder': 'derivatives_plots',
        'display_name': 'IC L-inf (eval)',
        'extract_fn': lambda m: [
            m['ic']['eval'][layer]['linf'] 
            for layer in m['layers_analyzed']
        ],
    },
    'bc_value_l2': {
        'direction': 'decrease',
        'source': 'derivatives_metrics',
        'folder': 'derivatives_plots',
        'display_name': 'BC Value L2 (eval)',
        'extract_fn': lambda m: [
            m['bc_value']['eval'][layer]['l2'] 
            for layer in m['layers_analyzed']
        ],
    },
    'bc_value_linf': {
        'direction': 'decrease',
        'source': 'derivatives_metrics',
        'folder': 'derivatives_plots',
        'display_name': 'BC Value L-inf (eval)',
        'extract_fn': lambda m: [
            m['bc_value']['eval'][layer]['linf'] 
            for layer in m['layers_analyzed']
        ],
    },
    'bc_derivative_l2': {
        'direction': 'decrease',
        'source': 'derivatives_metrics',
        'folder': 'derivatives_plots',
        'display_name': 'BC Derivative L2 (eval)',
        'extract_fn': lambda m: [
            m['bc_derivative']['eval'][layer]['l2'] 
            for layer in m['layers_analyzed']
        ],
    },
    'bc_derivative_linf': {
        'direction': 'decrease',
        'source': 'derivatives_metrics',
        'folder': 'derivatives_plots',
        'display_name': 'BC Derivative L-inf (eval)',
        'extract_fn': lambda m: [
            m['bc_derivative']['eval'][layer]['linf'] 
            for layer in m['layers_analyzed']
        ],
    },
}

# Problem-specific BC metrics configuration
# Maps problem name -> which BC metrics are included in the loss function
PROBLEM_BC_CONFIG = {
    'schrodinger': {
        'bc_value': True,       # Periodic: h(-5,t) = h(5,t)
        'bc_derivative': True,  # Periodic: h_x(-5,t) = h_x(5,t)
    },
    'wave1d': {
        'bc_value': True,       # Dirichlet-like: match analytical boundary
        'bc_derivative': False,  # Not enforced in loss
    },
    'burgers1d': {
        'bc_value': True,       # Dirichlet: h(t,-1) = h(t,1) = 0
        'bc_derivative': False,  # Not enforced in loss
    },
    'burgers2d': {
        'bc_value': True,       # Dirichlet: match analytical solution at boundaries
        'bc_derivative': False,  # Not enforced in loss
    },
}

# Default for unknown problems (conservative: include all)
DEFAULT_BC_CONFIG = {'bc_value': True, 'bc_derivative': True}


def get_problem_from_model_name(model_name: str) -> Optional[str]:
    """Extract problem type from model name."""
    model_lower = model_name.lower()
    if 'schrodinger' in model_lower:
        return 'schrodinger'
    elif 'wave1d' in model_lower:
        return 'wave1d'
    elif 'burgers1d' in model_lower:
        return 'burgers1d'
    return None


def get_active_metrics(problem_name: Optional[str]) -> List[str]:
    """Get list of metric names that are active for a given problem.
    
    Filters out BC metrics that are not part of the problem's loss function.
    """
    bc_config = PROBLEM_BC_CONFIG.get(problem_name, DEFAULT_BC_CONFIG)
    
    active_metrics = []
    for metric_name in METRICS_CONFIG.keys():
        # Filter BC metrics based on problem config
        if metric_name.startswith('bc_value') and not bc_config.get('bc_value', True):
            continue
        if metric_name.startswith('bc_derivative') and not bc_config.get('bc_derivative', True):
            continue
        active_metrics.append(metric_name)
    
    return active_metrics


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def parse_model_name(model_name: str) -> List[int]:
    """Extract architecture from folder name.
    
    Args:
        model_name: Folder name like "schrodinger-2-140-140-140-2-tanh"
        
    Returns:
        List of layer sizes [2, 140, 140, 140, 2]
    """
    parts = model_name.split('-')
    architecture = []
    for part in parts:
        if part.isdigit():
            architecture.append(int(part))
    return architecture


def calculate_num_parameters(architecture: List[int]) -> int:
    """Calculate total trainable parameters from architecture."""
    total = 0
    for i in range(len(architecture) - 1):
        total += architecture[i] * architecture[i + 1]  # Weights
        total += architecture[i + 1]  # Biases
    return total


def extract_weight_from_experiment_name(exp_name: str) -> Optional[str]:
    """Extract weight count from experiment name (e.g., "40k" from "3lyr_40k_Wn_flat")."""
    match = re.search(r'(\d+k)', exp_name)
    if match:
        return match.group(1)
    return None


# =============================================================================
# LOADING FUNCTIONS
# =============================================================================

def load_experiment_plan(experiment_path: Path) -> Optional[Dict[str, Any]]:
    """Load experiment plan YAML to map experiment names to architectures."""
    plan_file = experiment_path / "experiments_plan.yaml"
    if not plan_file.exists():
        return None
    
    with open(plan_file, 'r') as f:
        plan = yaml.safe_load(f)
    
    name_to_arch = {}
    if 'experiments' in plan:
        for exp in plan['experiments']:
            if 'name' in exp and 'architecture' in exp:
                name_to_arch[exp['name']] = exp['architecture']
    
    return name_to_arch


def load_all_model_metrics(experiment_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load all metrics from all models in experiment.
    
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
    
    for model_dir in experiment_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # Skip non-model directories
        if model_name.startswith('.') or model_name in ['comparison_summary.csv', 'experiments_plan.yaml']:
            continue
        
        # Find the timestamped run directory (most recent one)
        run_dirs = [d for d in model_dir.iterdir() if d.is_dir() and re.match(r'\d{8}_\d{6}', d.name)]
        if not run_dirs:
            continue
        
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
        
        # Load frequency metrics
        frequency_metrics = None
        freq_file = run_dir / "frequency_plots" / "frequency_metrics.json"
        if freq_file.exists():
            with open(freq_file, 'r') as f:
                frequency_metrics = json.load(f)
        
        # Parse architecture
        architecture = parse_model_name(model_name)
        num_layers = len(architecture) - 2  # Exclude input and output layers
        num_parameters = calculate_num_parameters(architecture)
        
        # Determine problem type from model name
        problem_name = get_problem_from_model_name(model_name)
        
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
            'frequency_metrics': frequency_metrics,
            'eval_rel_l2': final_eval_rel_l2,
            'eval_linf': final_eval_linf,
            'problem_name': problem_name
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
    """Check if values are monotonic with a degradation threshold.
    
    Args:
        values: List of metric values across layers (index 0 = layer 1, etc.)
        direction: 'increase' (higher is better) or 'decrease' (lower is better)
        threshold: Relative degradation threshold (default 0.10 = 10%)
        
    Returns:
        List of violations, each containing layer_num (2+), prev_value, current_value, 
        degradation_abs, degradation_rel
        
    Note: layer_num starts at 2 because layer 1 has no previous layer to compare to.
    A violation at layer_num=2 means layer 2 is worse than layer 1.
    """
    violations = []
    
    for i in range(1, len(values)):
        prev_val = values[i - 1]
        curr_val = values[i]
        
        # Skip if values are NaN
        if np.isnan(prev_val) or np.isnan(curr_val):
            continue
        
        is_violation = False
        if direction == 'increase':
            # Higher is better - violation if current < previous
            if curr_val < prev_val:
                is_violation = True
                degradation_abs = prev_val - curr_val
        else:  # decrease
            # Lower is better - violation if current > previous
            if curr_val > prev_val:
                is_violation = True
                degradation_abs = curr_val - prev_val
        
        if is_violation:
            # Calculate relative degradation (relative to previous value)
            if abs(prev_val) > 1e-10:
                degradation_rel = degradation_abs / abs(prev_val)
            else:
                degradation_rel = float('inf') if degradation_abs > 0 else 0
            
            # Only record if exceeds threshold
            if degradation_rel > threshold:
                # layer_num = i + 1 (convert 0-based index to 1-based layer number)
                # i=1 means comparing values[1] to values[0], so layer 2 is worse than layer 1
                violations.append({
                    'layer_num': i + 1,  # Layer number (2, 3, 4, ...) - never 1
                    'prev_value': prev_val,
                    'current_value': curr_val,
                    'degradation_abs': degradation_abs,
                    'degradation_rel': degradation_rel,
                })
    
    return violations


def detect_non_monotonic_metrics(models_data: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Detect non-monotonic metrics across all models using METRICS_CONFIG.
    
    Only checks metrics that are relevant to each problem's loss function.
    BC metrics are filtered based on PROBLEM_BC_CONFIG.
    
    Returns:
        Dict mapping metric_name -> list of violations, where each violation contains:
        - model_name: str
        - layer_idx: int (1-based, where violation occurred)
        - prev_value: float
        - current_value: float
        - degradation_abs: float
        - degradation_rel: float
    """
    all_violations = defaultdict(list)
    
    for model_name, data in models_data.items():
        # Get problem-specific BC config
        problem_name = data.get('problem_name')
        bc_config = PROBLEM_BC_CONFIG.get(problem_name, DEFAULT_BC_CONFIG)
        
        for metric_name, config in METRICS_CONFIG.items():
            # Skip BC metrics not relevant to this problem's loss function
            if metric_name.startswith('bc_value') and not bc_config.get('bc_value', True):
                continue
            if metric_name.startswith('bc_derivative') and not bc_config.get('bc_derivative', True):
                continue
            
            source = config['source']
            direction = config['direction']
            extract_fn = config['extract_fn']
            
            # Get the metrics data
            metrics = data.get(source)
            if metrics is None:
                continue
            
            # Extract values
            try:
                values = extract_fn(metrics)
            except (KeyError, TypeError, IndexError):
                # Metric not available for this model
                continue
            
            if len(values) < 2:
                continue
            
            # Check monotonicity
            violations = check_monotonicity(values, direction)
            
            for v in violations:
                all_violations[metric_name].append({
                    'model_name': model_name,
                    'layer_num': v['layer_num'],  # Layer number (2+), never 1
                    'prev_value': v['prev_value'],
                    'current_value': v['current_value'],
                    'degradation_abs': v['degradation_abs'],
                    'degradation_rel': v['degradation_rel'],
                })
    
    return dict(all_violations)


def get_all_metrics_violated_by_model(
    all_violations: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[str]]:
    """Get list of metrics violated by each model.
    
    Returns:
        Dict mapping model_name -> list of metric names violated
    """
    model_metrics = defaultdict(set)
    for metric_name, violations in all_violations.items():
        for v in violations:
            model_metrics[v['model_name']].add(metric_name)
    return {k: sorted(list(v)) for k, v in model_metrics.items()}


# =============================================================================
# RANKING AND TABLE GENERATION
# =============================================================================

def generate_rankings(models_data: Dict[str, Dict[str, Any]], output_dir: Path):
    """Generate model rankings by eval_rel_L2 and eval_L_inf as images with tables."""
    ranking_data = []
    for model_name, data in models_data.items():
        ranking_data.append({
            'rank': 0,
            'model_name': model_name,
            'num_layers': str(data['num_layers']),
            'num_parameters': f"{data['num_parameters']:,}",
            'eval_rel_l2': f"{data['eval_rel_l2']:.6f}",
            'eval_linf': f"{data['eval_linf']:.6f}",
            '_sort_rel_l2': data['eval_rel_l2'],
            '_sort_linf': data['eval_linf']
        })
    
    # Sort by eval_rel_l2 (best = lowest)
    ranking_data_l2 = sorted(ranking_data, key=lambda x: x['_sort_rel_l2'])
    for i, row in enumerate(ranking_data_l2):
        row['rank'] = str(i + 1)
    df_rel_l2 = pd.DataFrame(ranking_data_l2)
    
    # Sort by eval_linf (best = lowest)
    ranking_data_linf = sorted(ranking_data, key=lambda x: x['_sort_linf'])
    for i, row in enumerate(ranking_data_linf):
        row['rank'] = str(i + 1)
    df_linf = pd.DataFrame(ranking_data_linf)
    
    # Save CSV files
    cols = ['rank', 'model_name', 'num_layers', 'num_parameters', 'eval_rel_l2', 'eval_linf']
    df_rel_l2[cols].to_csv(output_dir / "model_ranking_by_eval_rel_l2.csv", index=False)
    df_linf[cols].to_csv(output_dir / "model_ranking_by_eval_linf.csv", index=False)
    
    # Create table images using the shared function
    col_labels = ['Rank', 'Model Name', 'Layers', 'Parameters', 'Eval Rel-L2', 'Eval L-inf']
    
    create_table_image(
        df_rel_l2, cols, col_labels,
        'Model Ranking by Eval Rel-L2 Error (Lower is Better)',
        output_dir / "model_ranking_by_eval_rel_l2.png"
    )
    
    create_table_image(
        df_linf, cols, col_labels,
        'Model Ranking by Eval L-inf Error (Lower is Better)',
        output_dir / "model_ranking_by_eval_linf.png"
    )
    
    print(f"  Rankings saved to {output_dir}")


def create_table_image(
    df: pd.DataFrame,
    columns: List[str],
    col_labels: List[str],
    title: str,
    save_path: Path,
    col_widths: Optional[List[float]] = None
):
    """Create a table image with auto-sized columns for model names.
    
    Args:
        df: DataFrame with data
        columns: Column names to include
        col_labels: Display labels for columns
        title: Title for the table
        save_path: Where to save the image
        col_widths: Optional list of relative column widths
    """
    # Calculate figure width based on longest model name
    max_model_len = max(len(str(row)) for row in df['model_name']) if 'model_name' in df.columns else 30
    fig_width = max(22, 12 + max_model_len * 0.15)
    
    fig, ax = plt.subplots(figsize=(fig_width, max(6, len(df) * 0.5 + 2)))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = df[columns].values.tolist()
    
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    # Set column widths if provided
    if col_widths:
        for i, width in enumerate(col_widths):
            for row in range(len(table_data) + 1):
                table[(row, i)].set_width(width)
    
    # Style header
    for i in range(len(col_labels)):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')
    
    # Color code best/worst rows
    if len(table_data) > 0:
        for j in range(len(col_labels)):
            table[(1, j)].set_facecolor('#f8d7da')  # First row (worst)
        if len(table_data) > 1:
            for j in range(len(col_labels)):
                table[(len(table_data), j)].set_facecolor('#d4edda')  # Last row (best)
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_per_metric_ranking_table(
    metric_name: str,
    violations: List[Dict[str, Any]],
    models_data: Dict[str, Dict[str, Any]],
    all_violations: Dict[str, List[Dict[str, Any]]],
    output_dir: Path
):
    """Generate ranking table for a specific metric's violations."""
    if not violations:
        return
    
    # Get other metrics violated by each model
    model_other_metrics = get_all_metrics_violated_by_model(all_violations)
    
    # Aggregate violations by model: collect all violating layers and find worst
    model_info = {}
    for v in violations:
        model_name = v['model_name']
        if model_name not in model_info:
            model_info[model_name] = {
                'worst_degradation_rel': v['degradation_rel'],
                'violating_layers': set()
            }
        else:
            if v['degradation_rel'] > model_info[model_name]['worst_degradation_rel']:
                model_info[model_name]['worst_degradation_rel'] = v['degradation_rel']
        model_info[model_name]['violating_layers'].add(v['layer_num'])
    
    # Build ranking data
    ranking_data = []
    for model_name, info in model_info.items():
        if model_name not in models_data:
            continue
        data = models_data[model_name]
        other_metrics = [m for m in model_other_metrics.get(model_name, []) if m != metric_name]
        other_str = ', '.join(other_metrics) if other_metrics else 'None'
        layers_sorted = sorted(info['violating_layers'])
        layers_str = ', '.join(str(l) for l in layers_sorted)
        ranking_data.append({
            'model_name': model_name,
            'num_layers': str(data['num_layers']),
            'num_parameters': f"{data['num_parameters']:,}",
            'non_monotone_layers': layers_str,
            'worst_degradation_pct': f"{info['worst_degradation_rel'] * 100:.1f}%",
            'other_metrics_violated': other_str[:60] + ('...' if len(other_str) > 60 else ''),
        })
    
    # Sort by worst degradation (biggest first)
    ranking_data.sort(key=lambda x: float(x['worst_degradation_pct'].rstrip('%')), reverse=True)
    
    df = pd.DataFrame(ranking_data)
    
    # Save CSV
    df.to_csv(output_dir / "ranking_table.csv", index=False)
    
    # Create image (no rank column)
    display_name = METRICS_CONFIG[metric_name]['display_name']
    columns = ['model_name', 'num_layers', 'num_parameters', 'non_monotone_layers',
               'worst_degradation_pct', 'other_metrics_violated']
    col_labels = ['Model Name', 'Layers', 'Params', 'Violating Layer(s)',
                  'Worst Degradation', 'Other Metrics Violated']
    
    create_table_image(
        df, columns, col_labels,
        f'Non-Monotonic Models: {display_name}\n(Sorted by Worst Degradation)',
        output_dir / "ranking_table.png"
    )


def generate_overall_non_monotonic_ranking(
    all_violations: Dict[str, List[Dict[str, Any]]],
    models_data: Dict[str, Dict[str, Any]],
    output_dir: Path
):
    """Generate overall ranking table for all non-monotonic models."""
    if not all_violations:
        return
    
    # Get all metrics violated by each model
    model_other_metrics = get_all_metrics_violated_by_model(all_violations)
    
    # Collect all info per model: worst violation and all violating layers
    model_info = {}
    for metric_name, violations in all_violations.items():
        for v in violations:
            model_name = v['model_name']
            if model_name not in model_info:
                model_info[model_name] = {
                    'worst_metric': metric_name,
                    'worst_degradation_rel': v['degradation_rel'],
                    'violating_layers': set()
                }
            else:
                if v['degradation_rel'] > model_info[model_name]['worst_degradation_rel']:
                    model_info[model_name]['worst_metric'] = metric_name
                    model_info[model_name]['worst_degradation_rel'] = v['degradation_rel']
            # Collect all violating layer numbers (2+, never 1)
            model_info[model_name]['violating_layers'].add(v['layer_num'])
    
    # Build ranking data
    ranking_data = []
    for model_name, info in model_info.items():
        if model_name not in models_data:
            continue
        data = models_data[model_name]
        all_metrics = model_other_metrics.get(model_name, [])
        # Format violating layers as comma-separated list
        layers_sorted = sorted(info['violating_layers'])
        layers_str = ', '.join(str(l) for l in layers_sorted)
        ranking_data.append({
            'model_name': model_name,
            'num_layers': str(data['num_layers']),
            'num_parameters': f"{data['num_parameters']:,}",
            'worst_metric': METRICS_CONFIG[info['worst_metric']]['display_name'],
            'non_monotone_layers': layers_str,
            'worst_degradation_pct': f"{info['worst_degradation_rel'] * 100:.1f}%",
            'metrics_violated': str(len(all_metrics)),
        })
    
    # Sort by worst degradation (biggest first)
    ranking_data.sort(key=lambda x: float(x['worst_degradation_pct'].rstrip('%')), reverse=True)
    
    df = pd.DataFrame(ranking_data)
    
    # Save CSV
    df.to_csv(output_dir / "all_models_ranking.csv", index=False)
    
    # Create image (no rank column)
    columns = ['model_name', 'num_layers', 'num_parameters', 'worst_metric',
               'non_monotone_layers', 'worst_degradation_pct', 'metrics_violated']
    col_labels = ['Model Name', 'Layers', 'Params', 'Worst Metric',
                  'Violating Layer(s)', 'Worst Degradation', 'Metrics Violated']
    
    create_table_image(
        df, columns, col_labels,
        'All Non-Monotonic Models\n(Sorted by Worst Degradation)',
        output_dir / "all_models_ranking.png"
    )


# =============================================================================
# COPYING METRIC FOLDERS
# =============================================================================

def copy_metric_folders(
    metric_name: str,
    violations: List[Dict[str, Any]],
    models_data: Dict[str, Dict[str, Any]],
    output_dir: Path
):
    """Copy entire metric folder for each violating model.
    
    Args:
        metric_name: Name of the metric
        violations: List of violations for this metric
        models_data: All model data
        output_dir: Where to copy folders to
    """
    if not violations:
        return
    
    folder_name = METRICS_CONFIG[metric_name]['folder']
    model_folders_dir = output_dir / "model_folders"
    model_folders_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique models
    violating_models = set(v['model_name'] for v in violations)
    
    for model_name in violating_models:
        if model_name not in models_data:
            continue
        
        run_dir = models_data[model_name]['run_dir']
        src = run_dir / folder_name
        
        if not src.exists():
            print(f"      Warning: Source folder not found: {src}")
            continue
        
        dst = model_folders_dir / model_name
        
        # Copy entire folder
        try:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        except Exception as e:
            print(f"      Warning: Failed to copy {src} to {dst}: {e}")


# =============================================================================
# COMPARISON PLOTS
# =============================================================================

def generate_metric_comparison_plots(
    metric_name: str,
    violations: List[Dict[str, Any]],
    models_data: Dict[str, Dict[str, Any]],
    output_dir: Path
):
    """Generate comparison plots for models that violated a specific metric.
    
    Only generates plots relevant to the specific metric type.
    
    Args:
        metric_name: Name of the metric
        violations: List of violations for this metric
        models_data: All model data
        output_dir: Where to save plots
    """
    if not violations:
        return
    
    # Get unique violating models
    violating_models = set(v['model_name'] for v in violations)
    metric_models_data = {k: v for k, v in models_data.items() if k in violating_models}
    
    if not metric_models_data:
        return
    
    # Determine which specific plot(s) to generate based on metric name
    if metric_name == 'ncc_accuracy':
        ncc_data = {}
        for model_name, data in metric_models_data.items():
            if data.get('ncc_metrics'):
                ncc_data[model_name] = {'final': data['ncc_metrics']}
        if ncc_data:
            generate_ncc_classification_plot(output_dir, ncc_data)
    
    elif metric_name == 'ncc_compactness':
        ncc_data = {}
        for model_name, data in metric_models_data.items():
            if data.get('ncc_metrics'):
                ncc_data[model_name] = {'final': data['ncc_metrics']}
        if ncc_data:
            generate_ncc_compactness_plot(output_dir, ncc_data)
    
    elif metric_name.startswith('probe'):
        # Generate probe comparison plots
        probe_data = {}
        for model_name, data in metric_models_data.items():
            if data.get('probe_metrics'):
                probe_data[model_name] = data['probe_metrics']
        if probe_data:
            generate_probe_comparison_plots(output_dir, probe_data)
    
    elif metric_name.startswith('residual'):
        # Generate only residual comparison plot
        derivatives_data = {}
        for model_name, data in metric_models_data.items():
            if data.get('derivatives_metrics'):
                derivatives_data[model_name] = data['derivatives_metrics']
        if derivatives_data:
            _generate_single_derivatives_plot(output_dir, derivatives_data, 'residual')
    
    elif metric_name.startswith('ic'):
        # Generate only IC comparison plot
        derivatives_data = {}
        for model_name, data in metric_models_data.items():
            if data.get('derivatives_metrics'):
                derivatives_data[model_name] = data['derivatives_metrics']
        if derivatives_data:
            _generate_single_derivatives_plot(output_dir, derivatives_data, 'ic')
    
    elif metric_name.startswith('bc_value'):
        # Generate only BC value comparison plot
        derivatives_data = {}
        for model_name, data in metric_models_data.items():
            if data.get('derivatives_metrics'):
                derivatives_data[model_name] = data['derivatives_metrics']
        if derivatives_data:
            _generate_single_derivatives_plot(output_dir, derivatives_data, 'bc_value')
    
    elif metric_name.startswith('bc_derivative'):
        # Generate only BC derivative comparison plot
        derivatives_data = {}
        for model_name, data in metric_models_data.items():
            if data.get('derivatives_metrics'):
                derivatives_data[model_name] = data['derivatives_metrics']
        if derivatives_data:
            _generate_single_derivatives_plot(output_dir, derivatives_data, 'bc_derivative')


def _generate_single_derivatives_plot(
    output_dir: Path,
    derivatives_data: Dict[str, Dict[str, Any]],
    plot_type: str
):
    """Generate a single derivatives comparison plot.
    
    Args:
        output_dir: Where to save the plot
        derivatives_data: Derivatives metrics data
        plot_type: One of 'residual', 'ic', 'bc_value', 'bc_derivative'
    """
    from utils.comparison_plots import _build_color_map, _long_path
    
    exp_names = sorted(derivatives_data.keys())
    color_map = _build_color_map(exp_names)
    max_layers = max(len(data['layers_analyzed']) for data in derivatives_data.values())
    
    # Config for each plot type
    plot_configs = {
        'residual': {
            'section_key': None,
            'key_map': {'l2': 'residual_norm', 'linf': 'residual_inf_norm'},
            'title': 'Residual Comparison',
            'filename': 'derivatives_residual_comparison.png'
        },
        'ic': {
            'section_key': 'ic',
            'key_map': {'l2': 'l2', 'linf': 'linf'},
            'title': 'IC Comparison',
            'filename': 'derivatives_ic_comparison.png'
        },
        'bc_value': {
            'section_key': 'bc_value',
            'key_map': {'l2': 'l2', 'linf': 'linf'},
            'title': 'BC Value Comparison',
            'filename': 'derivatives_bc_value_comparison.png'
        },
        'bc_derivative': {
            'section_key': 'bc_derivative',
            'key_map': {'l2': 'l2', 'linf': 'linf'},
            'title': 'BC Derivative Comparison',
            'filename': 'derivatives_bc_derivative_comparison.png'
        }
    }
    
    cfg = plot_configs[plot_type]
    section_key = cfg['section_key']
    key_map = cfg['key_map']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    panel_cfg = [
        (axes[0, 0], 'train', 'l2', 'Train L2'),
        (axes[0, 1], 'eval', 'l2', 'Eval L2'),
        (axes[1, 0], 'train', 'linf', 'Train L-inf'),
        (axes[1, 1], 'eval', 'linf', 'Eval L-inf'),
    ]
    
    for panel_idx, (ax, split, metric_id, panel_title) in enumerate(panel_cfg):
        all_panel_values = []
        for exp_name in exp_names:
            exp_data = derivatives_data[exp_name]
            layers = exp_data['layers_analyzed']
            layer_indices = list(range(1, len(layers) + 1))
            values = []
            metric_key = key_map[metric_id]
            for layer_name in layers:
                if section_key is None:
                    layer_metrics = exp_data.get(split, {}).get(layer_name, {})
                else:
                    layer_metrics = (
                        exp_data.get(section_key, {})
                        .get(split, {})
                        .get(layer_name, {})
                    )
                values.append(layer_metrics.get(metric_key, np.nan))
            
            ax.plot(
                layer_indices,
                values,
                marker='o',
                linewidth=2,
                markersize=6,
                label=exp_name,
                color=color_map[exp_name]
            )
            all_panel_values.extend(values)
        
        ax.set_xlabel('Layer', fontsize=11)
        ax.set_ylabel('Mean Norm', fontsize=11)
        ax.set_xticks(range(1, max_layers + 1))
        is_log = _safe_log_scale(ax, [all_panel_values])
        scale_str = "[log]" if is_log else "[linear]"
        ax.set_title(f'{panel_title} {scale_str}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if panel_idx == 0:
            ax.legend(fontsize=9, loc='best')
    
    scale_label = "(Log Scale)" if is_log else "(Linear Scale)"
    plt.suptitle(f"{cfg['title']} {scale_label}", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    save_path = _long_path(Path(output_dir) / cfg['filename'])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    {cfg['title']} saved")


def group_models_by_attribute(
    models_data: Dict[str, Dict[str, Any]],
    attribute: str,
    experiment_plan: Optional[Dict[str, List[int]]] = None
) -> Dict[Any, List[str]]:
    """Group models by attribute (layer count or weight count)."""
    groups = defaultdict(list)
    
    if attribute == 'num_parameters' and experiment_plan:
        # Create mapping: architecture -> weight_count from experiment name
        arch_to_weight = {}
        for exp_name, arch in experiment_plan.items():
            weight_str = extract_weight_from_experiment_name(exp_name)
            if weight_str:
                arch_tuple = tuple(arch)
                if arch_tuple not in arch_to_weight:
                    arch_to_weight[arch_tuple] = weight_str
        
        for model_name, data in models_data.items():
            model_arch = data['architecture']
            arch_tuple = tuple(model_arch)
            weight_count = arch_to_weight.get(arch_tuple)
            if weight_count:
                groups[weight_count].append(model_name)
            else:
                groups[data['num_parameters']].append(model_name)
    else:
        for model_name, data in models_data.items():
            key = data[attribute]
            groups[key].append(model_name)
    
    return dict(groups)


def generate_comparison_plots(
    models_data: Dict[str, Dict[str, Any]],
    output_dir: Path,
    group_by: str = 'layers',
    experiment_plan: Optional[Dict[str, List[int]]] = None,
    add_weights_to_labels: bool = False
):
    """Generate comparison plots for similar models grouped by layers or weights."""
    if group_by == 'layers':
        groups = group_models_by_attribute(models_data, 'num_layers', experiment_plan)
        base_dir = output_dir / "comparisons_by_layers"
    else:
        groups = group_models_by_attribute(models_data, 'num_parameters', experiment_plan)
        base_dir = output_dir / "comparisons_by_weights"
    
    base_dir.mkdir(parents=True, exist_ok=True)
    
    for group_key, model_names in groups.items():
        if len(model_names) < 2:
            continue
        
        if group_by == 'weights':
            group_dir = base_dir / f"{group_key}_params_comparison"
        else:
            group_dir = base_dir / f"{group_key}_layers_comparison"
        group_dir.mkdir(parents=True, exist_ok=True)
        
        ncc_data = {}
        probe_data = {}
        derivatives_data = {}
        
        label_map = {}
        if add_weights_to_labels:
            for model_name in model_names:
                data = models_data[model_name]
                weight_str = f"({data['num_parameters']:,} params)"
                label_map[model_name] = f"{model_name} {weight_str}"
        
        for model_name in model_names:
            data = models_data[model_name]
            display_name = label_map.get(model_name, model_name)
            
            if data['ncc_metrics']:
                ncc_data[display_name] = {'final': data['ncc_metrics']}
            if data['probe_metrics']:
                probe_data[display_name] = data['probe_metrics']
            if data['derivatives_metrics']:
                derivatives_data[display_name] = data['derivatives_metrics']
        
        if ncc_data:
            generate_ncc_classification_plot(group_dir, ncc_data)
            generate_ncc_compactness_plot(group_dir, ncc_data)
        if probe_data:
            generate_probe_comparison_plots(group_dir, probe_data)
        if derivatives_data:
            generate_derivatives_comparison_plots(group_dir, derivatives_data)
        
        # Generate frequency coverage comparison
        frequency_data = {}
        for model_name in model_names:
            data = models_data[model_name]
            display_name = label_map.get(model_name, model_name)
            if data.get('frequency_metrics'):
                frequency_data[display_name] = data['frequency_metrics']
        
        if frequency_data:
            print(f"  Generating frequency comparison plots for {len(frequency_data)} models...")
            from frequency_tracker.frequency_plotting import plot_spectral_learning_efficiency_comparison
            
            generate_frequency_coverage_comparison(group_dir, frequency_data)
            plot_spectral_learning_efficiency_comparison(frequency_data, group_dir)
        else:
            print(f"  No frequency metrics found for models in this group")
        
        print(f"  Comparison plots saved to {group_dir}")


def generate_frequency_coverage_comparison(
    output_dir: Path,
    frequency_data: Dict[str, Dict]
) -> None:
    """
    Generate frequency coverage comparison heatmap.
    
    X-axis: Model name
    Y-axis: Frequency band |k| (binned)
    Color: Final error in that frequency band (after all layers)
    
    Args:
        output_dir: Directory to save plot
        frequency_data: Dict mapping model_name -> frequency_metrics dict
    """
    from matplotlib.colors import LogNorm
    
    # Collect data for all models
    model_names = list(frequency_data.keys())
    if not model_names:
        return
    
    # Get frequency bins from first model (all should have same structure)
    first_model = frequency_data[model_names[0]]
    final_layer = first_model['layers_analyzed'][-1]
    final_layer_metrics = first_model['layer_metrics'][final_layer]
    
    # We need to load the actual frequency results to get binned errors
    # For now, we'll use the radial leftover power as a proxy
    # This requires loading the actual frequency_plots data
    
    # Try to load frequency results from saved files
    n_bins = 20
    error_matrix = []
    k_radial_ref = None
    
    for model_name in model_names:
        metrics = frequency_data[model_name]
        
        # Get final layer leftover power ratio per frequency bin
        # We'll need to reconstruct this from the metrics or load the actual results
        # For now, let's create a simplified version using the total leftover power
        
        # Check if we can load the actual frequency results
        # This would require the frequency_plots directory structure
        # For now, we'll create a placeholder that shows the concept
        
        # Get final leftover ratio (overall, not per frequency)
        final_layer = metrics['layers_analyzed'][-1]
        final_leftover_ratio = metrics['layer_metrics'][final_layer].get('leftover_ratio', 1.0)
        
        # Create a simple representation: assume uniform error across frequencies
        # In a full implementation, we'd load the actual binned frequency data
        if k_radial_ref is None:
            # Create frequency bins (0 to max |k|)
            k_max = 0.5  # Typical max for normalized frequencies
            k_radial_ref = np.linspace(0, k_max, n_bins)
        
        # For now, use the overall leftover ratio as a constant across frequencies
        # This is a simplified version - full implementation would load actual binned data
        error_row = np.full(n_bins, final_leftover_ratio)
        error_matrix.append(error_row)
    
    error_matrix = np.array(error_matrix).T  # Transpose: rows = freq bins, cols = models
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(10, len(model_names) * 1.5), 8))
    
    im = ax.imshow(error_matrix, aspect='auto', cmap='viridis_r', 
                   interpolation='bilinear', origin='lower')
    
    # Set ticks
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    
    # Y-axis: frequency bins
    y_ticks = np.linspace(0, n_bins - 1, min(10, n_bins))
    y_tick_labels = [f'{k_radial_ref[int(i)]:.2f}' for i in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    ax.set_ylabel('Radial Frequency |k|', fontsize=12, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Remaining Error (lower = better coverage)', 
                   fontsize=11, rotation=270, labelpad=20)
    
    # Try log scale for colorbar
    if np.all(error_matrix > 0):
        im.set_norm(LogNorm(vmin=error_matrix[error_matrix > 0].min(), 
                           vmax=error_matrix.max()))
        cbar.set_label('Remaining Error (log scale) [log]', 
                       fontsize=11, rotation=270, labelpad=20)
    
    ax.set_title('Frequency Coverage Comparison\n(Final Error by Frequency Band)', 
                 fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    save_path = output_dir / 'frequency_coverage_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Frequency coverage comparison saved to {save_path}")


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def generate_non_monotonic_summary(
    all_violations: Dict[str, List[Dict[str, Any]]],
    models_data: Dict[str, Dict[str, Any]],
    output_dir: Path,
    experiment_plan: Optional[Dict[str, List[int]]] = None
):
    """Generate summary statistics and plots for all non-monotonic violations.
    
    Files are saved directly to output_dir (non_monotonic_comparisons).
    
    Args:
        all_violations: Dict mapping metric_name -> list of violations
        models_data: Dict mapping model_name -> model data
        output_dir: Directory to save summary files
        experiment_plan: Optional dict mapping experiment names to architectures
    """
    
    # Count total violations
    total_violations = sum(len(v) for v in all_violations.values())
    total_models = len(set(
        v['model_name'] 
        for violations in all_violations.values() 
        for v in violations
    ))
    
    if total_violations == 0:
        print("  No violations found - skipping summary generation")
        return
    
    # Output directly to the provided directory (non_monotonic_comparisons)
    summary_dir = output_dir
    
    # Prepare data for plots
    plot_data = []
    for metric_name, violations in all_violations.items():
        for v in violations:
            model_name = v['model_name']
            if model_name not in models_data:
                continue
            data = models_data[model_name]
            plot_data.append({
                'metric': metric_name,
                'model_name': model_name,
                'num_layers': data['num_layers'],
                'num_parameters': data['num_parameters'],
                'layer_num': v['layer_num'],  # Layer number (2+), never 1
                'degradation_rel': v['degradation_rel'],
            })
    
    df = pd.DataFrame(plot_data)
    
    # Save summary CSV
    df.to_csv(summary_dir / "all_violations.csv", index=False)
    
    # Generate overall ranking table for all non-monotonic models
    generate_overall_non_monotonic_ranking(all_violations, models_data, summary_dir)
    
    # Plot 1: Violations by metric (show only ACTIVE metrics for this problem)
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Determine problem from first model (assume all models are same problem)
    first_model_data = next(iter(models_data.values()), {})
    problem_name = first_model_data.get('problem_name')
    active_metrics = get_active_metrics(problem_name)
    
    # Create counts for only ACTIVE metrics
    metric_counts_dict = df['metric'].value_counts().to_dict() if len(df) > 0 else {}
    all_counts = [metric_counts_dict.get(m, 0) for m in active_metrics]
    all_labels = [METRICS_CONFIG[m]['display_name'] for m in active_metrics]
    
    x_pos = range(len(active_metrics))
    ax.bar(x_pos, all_counts, color='#3498db', alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Violations', fontsize=12, fontweight='bold')
    title = 'Non-Monotonic Violations by Metric Type'
    if problem_name:
        title += f' ({problem_name})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, count in enumerate(all_counts):
        ax.text(i, count + 0.1, str(count), ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(summary_dir / "violations_by_metric.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Violations by layer count (depth)
    fig, ax = plt.subplots(figsize=(10, 6))
    layer_counts = df['num_layers'].value_counts().sort_index()
    x_positions = list(range(len(layer_counts)))
    ax.bar(x_positions, layer_counts.values, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(x) for x in layer_counts.index])
    ax.set_xlabel('Number of Hidden Layers', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Violations', fontsize=12, fontweight='bold')
    ax.set_title('Non-Monotonic Violations by Model Depth', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, count in enumerate(layer_counts.values):
        ax.text(i, count + 0.1, str(count), ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(summary_dir / "violations_by_depth.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Violations by layer position (layers 2-7, layer 1 can NEVER have violations)
    fig, ax = plt.subplots(figsize=(10, 6))
    # Layer numbers start at 2 (layer 1 has no previous layer to compare to)
    max_layer_num = 7
    layer_numbers = list(range(2, max_layer_num + 1))  # [2, 3, 4, 5, 6, 7]
    layer_num_counts = df['layer_num'].value_counts().to_dict() if len(df) > 0 else {}
    counts_by_layer = [layer_num_counts.get(layer, 0) for layer in layer_numbers]
    
    x_positions = list(range(len(layer_numbers)))
    ax.bar(x_positions, counts_by_layer, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(l) for l in layer_numbers])
    ax.set_xlabel('Layer Number (where violation occurred)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Violations', fontsize=12, fontweight='bold')
    ax.set_title('Non-Monotonic Violations by Layer Position\n(Layer N worse than Layer N-1)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, count in enumerate(counts_by_layer):
        ax.text(i, count + 0.1, str(count), ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(summary_dir / "violations_by_layer_position.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Violations by weights (parameter count bins)
    # Extract weight labels dynamically from experiment plan using architecture mapping
    fig, ax = plt.subplots(figsize=(12, 6))
    
    weight_labels = []
    arch_to_weight = {}
    
    if experiment_plan:
        # Build mapping: architecture tuple -> weight label (like group_models_by_attribute)
        for exp_name, arch in experiment_plan.items():
            weight_str = extract_weight_from_experiment_name(exp_name)
            if weight_str:
                arch_tuple = tuple(arch)
                if arch_tuple not in arch_to_weight:
                    arch_to_weight[arch_tuple] = weight_str
                if weight_str not in weight_labels:
                    weight_labels.append(weight_str)
    
    # Sort weight labels numerically (e.g., ['5k', '10k', '20k', '30k'])
    def weight_to_num(w):
        return int(w.replace('k', '')) * 1000
    
    if weight_labels and arch_to_weight:
        weight_labels = sorted(weight_labels, key=weight_to_num)
        
        # Map each model to its weight category via architecture
        df_copy = df.copy()
        
        def get_weight_for_model(model_name):
            if model_name in models_data:
                arch = models_data[model_name].get('architecture', [])
                return arch_to_weight.get(tuple(arch))
            return None
        
        df_copy['weight_label'] = df_copy['model_name'].apply(get_weight_for_model)
        weight_counts = df_copy['weight_label'].value_counts().reindex(weight_labels, fill_value=0)
    else:
        # Fallback to hardcoded bins if no experiment plan
        param_bins = [0, 15000, 25000, 35000, 45000, float('inf')]
        weight_labels = ['10k', '20k', '30k', '40k', '50k+']
        df_copy = df.copy()
        df_copy['param_bin'] = pd.cut(df_copy['num_parameters'], bins=param_bins, labels=weight_labels)
        weight_counts = df_copy['param_bin'].value_counts().reindex(weight_labels, fill_value=0)
    
    x_positions = list(range(len(weight_labels)))
    ax.bar(x_positions, weight_counts.values, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(weight_labels)
    ax.set_xlabel('Parameter Count (approx)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Violations', fontsize=12, fontweight='bold')
    ax.set_title('Non-Monotonic Violations by Model Size (Weights)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, count in enumerate(weight_counts.values):
        ax.text(i, count + 0.1, str(int(count)), ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(summary_dir / "violations_by_weights.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Write summary text
    summary_text = [
        "Non-Monotonic Analysis Summary",
        "=" * 60,
        f"Problem: {problem_name or 'unknown'}",
        f"Total violations: {total_violations}",
        f"Total models with violations: {total_models}",
        f"Total models analyzed: {len(models_data)}",
        "",
    ]
    
    # Add info about which BC metrics were excluded
    bc_config = PROBLEM_BC_CONFIG.get(problem_name, DEFAULT_BC_CONFIG)
    excluded_metrics = []
    if not bc_config.get('bc_value', True):
        excluded_metrics.extend(['bc_value_l2', 'bc_value_linf'])
    if not bc_config.get('bc_derivative', True):
        excluded_metrics.extend(['bc_derivative_l2', 'bc_derivative_linf'])
    
    if excluded_metrics:
        summary_text.append(f"Excluded BC metrics (not in loss): {', '.join(excluded_metrics)}")
        summary_text.append("")
    
    summary_text.append("Violations by metric:")
    for metric_name in sorted(all_violations.keys()):
        count = len(all_violations[metric_name])
        display_name = METRICS_CONFIG[metric_name]['display_name']
        models = set(v['model_name'] for v in all_violations[metric_name])
        summary_text.append(f"  {display_name}: {count} violations in {len(models)} models")
    
    with open(summary_dir / "summary.txt", 'w') as f:
        f.write('\n'.join(summary_text))
    
    print(f"  Summary saved to {summary_dir}")


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def get_next_analysis_index(experiment_name: str, analysis_base_dir: Path) -> str:
    """Get the next running index for analysis output."""
    exp_dir = analysis_base_dir / experiment_name
    if not exp_dir.exists():
        return "analysis_1"
    
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
    
    return f"analysis_{max(existing_indices) + 1}"


def main(experiment_path: str):
    """Main analysis function."""
    experiment_path = Path(experiment_path)
    
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
    
    # Determine problem type and show BC config
    if models_data:
        first_model = next(iter(models_data.values()))
        problem_name = first_model.get('problem_name')
        if problem_name:
            bc_config = PROBLEM_BC_CONFIG.get(problem_name, DEFAULT_BC_CONFIG)
            print(f"  Problem type: {problem_name}")
            print(f"  BC metrics active: bc_value={bc_config.get('bc_value', True)}, "
                  f"bc_derivative={bc_config.get('bc_derivative', True)}")
        else:
            print("  Problem type: unknown (all BC metrics will be checked)")
    print()
    
    # A. Generate rankings
    print("A. Generating model rankings...")
    generate_rankings(models_data, output_dir)
    print()
    
    # B. Generate comparison plots by layers and weights
    print("B. Generating comparison plots...")
    generate_comparison_plots(models_data, output_dir, group_by='layers', experiment_plan=experiment_plan)
    generate_comparison_plots(models_data, output_dir, group_by='weights', experiment_plan=experiment_plan)
    print()
    
    # C. Detect non-monotonic metrics
    print("C. Detecting non-monotonic metrics (>10% threshold)...")
    all_violations = detect_non_monotonic_metrics(models_data)
    
    total_violations = sum(len(v) for v in all_violations.values())
    total_models = len(set(
        v['model_name'] 
        for violations in all_violations.values() 
        for v in violations
    ))
    
    print(f"  Total violations: {total_violations}")
    print(f"  Models with violations: {total_models}")
    print(f"  Metrics with violations: {len(all_violations)}")
    
    # Save all violations to JSON
    violations_serializable = {
        k: v for k, v in all_violations.items()
    }
    with open(output_dir / "non_monotonic_violations.json", 'w') as f:
        json.dump(violations_serializable, f, indent=2)
    print("  Saved violations to non_monotonic_violations.json")
    print()
    
    # D. Process each metric with violations
    print("D. Processing non-monotonic metrics...")
    non_mono_dir = output_dir / "non_monotonic_comparisons"
    non_mono_dir.mkdir(parents=True, exist_ok=True)
    
    for metric_name, violations in all_violations.items():
        if not violations:
            continue
        
        display_name = METRICS_CONFIG[metric_name]['display_name']
        violating_models = set(v['model_name'] for v in violations)
        print(f"  Processing {display_name}: {len(violations)} violations in {len(violating_models)} models")
        
        # Create metric folder
        metric_dir = non_mono_dir / metric_name
        metric_dir.mkdir(parents=True, exist_ok=True)
        
        # A. Generate ranking table for this metric
        generate_per_metric_ranking_table(metric_name, violations, models_data, all_violations, metric_dir)
        
        # B. Generate comparison plots for violating models
        generate_metric_comparison_plots(metric_name, violations, models_data, metric_dir)
        
        # C. Copy entire metric folders for violating models
        copy_metric_folders(metric_name, violations, models_data, metric_dir)
    
    print()
    
    # E. Generate summary statistics (directly in non_monotonic_comparisons folder)
    print("E. Generating summary statistics...")
    generate_non_monotonic_summary(all_violations, models_data, non_mono_dir, experiment_plan)
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
