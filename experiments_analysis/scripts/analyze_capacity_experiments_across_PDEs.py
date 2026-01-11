"""Cross-PDE analysis script for capacity experiments.

This script analyzes multiple capacity experiment folders (one per PDE) and generates
comparative analysis including:
- PDE statistics table with per-dimension sampling density
- Best model summary per PDE
- Rank comparisons by layers and weights (for both Rel-L2 and L-inf)
- Non-monotonic violation plots with per-PDE coloring
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import re
from collections import defaultdict
import yaml
import matplotlib.pyplot as plt
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch

# Import from existing analyze_capacity_experiment.py
from experiments_analysis.scripts.analyze_capacity_experiment import (
    METRICS_CONFIG,
    PROBLEM_BC_CONFIG,
    DEFAULT_BC_CONFIG,
    parse_model_name,
    calculate_num_parameters,
    extract_weight_from_experiment_name,
    check_monotonicity,
    get_problem_from_model_name,
    get_active_metrics,
)

# Import from analysis_core for consistent metric computation
from experiments_analysis.scripts.analysis_core import compute_all_metrics_consistently


# =============================================================================
# PDE COLORS
# =============================================================================

PDE_COLORS = {
    'schrodinger': '#e74c3c',   # Red
    'wave1d': '#3498db',        # Blue
    'burgers1d': '#2ecc71',     # Green
    'burgers2d': '#9b59b6',     # Purple
}


# =============================================================================
# PDE ANALYTICAL SOLUTIONS
# =============================================================================

PDE_FORMULAS = {
    'schrodinger': {
        'name': 'Nonlinear Schrödinger',
        'pde': r'$i \, h_t + \frac{1}{2} h_{xx} + |h|^2 h = 0$',
        'domain': r'$x \in [-5, 5], \quad t \in [0, \frac{\pi}{2}]$',
        'ic': r'$h(x, 0) = 2 \, \mathrm{sech}(x)$',
        'bc': r'Periodic: $h(-5, t) = h(5, t)$',
        'solution': r'$h(x,t) = \frac{4[\cosh(3x) + 3e^{4it}\cosh(x)]}{\cosh(4x) + 4\cosh(2x) + 3\cos(4t)}$',
        'solution_note': 'N=2 Soliton (analytical)',
        'has_analytical': True,
    },
    'wave1d': {
        'name': '1D Wave Equation',
        'pde': r'$h_{tt} - h_{xx} = 0$',
        'domain': r'$x \in [-5, 5], \quad t \in [0, 2\pi]$',
        'ic': r'$h(x, 0) = \sin(x), \quad h_t(x, 0) = 0$',
        'bc': r'Dirichlet: $h(\pm 5, t) = \sin(\pm 5) \cos(t)$',
        'solution': r'$h(x, t) = \sin(x) \cos(t)$',
        'solution_note': 'Standing wave (analytical)',
        'has_analytical': True,
    },
    'burgers1d': {
        'name': '1D Viscous Burgers',
        'pde': r'$h_t + h \, h_x - \frac{\nu}{\pi} h_{xx} = 0$',
        'domain': r'$x \in [-1, 1], \quad t \in [0, 1]$',
        'ic': r'$h(x, 0) = -\sin(\pi x)$',
        'bc': r'Dirichlet: $h(-1, t) = h(1, t) = 0$',
        'solution': r'$h = \frac{4\pi\nu \sum_{n=1}^{\infty} n \, I_n(\frac{1}{2\pi\nu}) e^{-n^2\pi^2\nu t} \sin(n\pi x)}{I_0(\frac{1}{2\pi\nu}) + 2\sum_{n=1}^{\infty} I_n(\frac{1}{2\pi\nu}) e^{-n^2\pi^2\nu t} \cos(n\pi x)}$',
        'solution_note': 'Cole-Hopf (Bessel series)',
        'has_analytical': True,
    },
    'burgers2d': {
        'name': '2D Viscous Burgers',
        'pde': r'$h_t + h (h_{x_0} + h_{x_1}) - \nu (h_{x_0 x_0} + h_{x_1 x_1}) = 0$',
        'domain': r'$(x_0, x_1) \in [0,1]^2, \quad t \in [0, 2]$',
        'ic': r'$h(x_0, x_1, 0) = \frac{1}{1 + e^{(x_0 + x_1)/0.2}}$',
        'bc': r'Dirichlet from analytical solution',
        'solution': r'$h(x_0, x_1, t) = \frac{1}{1 + e^{(x_0 + x_1 - t)/0.2}}$',
        'solution_note': 'Sigmoid traveling wave (analytical)',
        'has_analytical': True,
    },
}


def generate_pde_formulas_figure(pde_names: List[str], output_dir: Path):
    """Generate a figure showing PDE formulas, IC, BC, and analytical solutions."""
    
    # Filter to only PDEs we have formulas for
    pde_names = [p for p in pde_names if p in PDE_FORMULAS]
    
    if not pde_names:
        return
    
    # Enable LaTeX rendering
    plt.rcParams['text.usetex'] = False  # Use mathtext instead of full LaTeX
    plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern font
    
    n_pdes = len(pde_names)
    fig_height = 2.5 * n_pdes + 1
    
    fig, axes = plt.subplots(n_pdes, 1, figsize=(14, fig_height))
    if n_pdes == 1:
        axes = [axes]
    
    for idx, pde_name in enumerate(pde_names):
        ax = axes[idx]
        ax.axis('off')
        
        formulas = PDE_FORMULAS[pde_name]
        color = PDE_COLORS.get(pde_name, '#333333')
        
        # Build text content
        y_pos = 0.95
        line_height = 0.14
        
        # Title
        ax.text(0.5, y_pos, formulas['name'], fontsize=14, fontweight='bold',
                ha='center', va='top', color=color, transform=ax.transAxes)
        y_pos -= line_height * 0.8
        
        # PDE
        ax.text(0.02, y_pos, 'PDE:', fontsize=11, fontweight='bold',
                ha='left', va='top', transform=ax.transAxes)
        ax.text(0.12, y_pos, formulas['pde'], fontsize=12,
                ha='left', va='top', transform=ax.transAxes)
        y_pos -= line_height
        
        # Domain
        ax.text(0.02, y_pos, 'Domain:', fontsize=11, fontweight='bold',
                ha='left', va='top', transform=ax.transAxes)
        ax.text(0.12, y_pos, formulas['domain'], fontsize=11,
                ha='left', va='top', transform=ax.transAxes)
        y_pos -= line_height
        
        # IC
        ax.text(0.02, y_pos, 'IC:', fontsize=11, fontweight='bold',
                ha='left', va='top', transform=ax.transAxes)
        ax.text(0.12, y_pos, formulas['ic'], fontsize=11,
                ha='left', va='top', transform=ax.transAxes)
        y_pos -= line_height
        
        # BC
        ax.text(0.02, y_pos, 'BC:', fontsize=11, fontweight='bold',
                ha='left', va='top', transform=ax.transAxes)
        ax.text(0.12, y_pos, formulas['bc'], fontsize=11,
                ha='left', va='top', transform=ax.transAxes)
        y_pos -= line_height
        
        # Solution (use smaller font for long formulas)
        ax.text(0.02, y_pos, 'Solution:', fontsize=11, fontweight='bold',
                ha='left', va='top', transform=ax.transAxes)
        sol_fontsize = 10 if len(formulas['solution']) > 60 else 12
        ax.text(0.12, y_pos, formulas['solution'], fontsize=sol_fontsize,
                ha='left', va='top', color='black', transform=ax.transAxes)
        
        # Add note about solution method
        ax.text(0.98, y_pos, f"({formulas['solution_note']})", fontsize=9,
                ha='right', va='top', style='italic', color='gray', transform=ax.transAxes)
        
        # Add border
        rect = plt.Rectangle((0.0, 0.0), 1.0, 1.0, fill=False, 
                             edgecolor=color, linewidth=2, transform=ax.transAxes)
        ax.add_patch(rect)
    
    plt.suptitle('PDE Definitions and Analytical Solutions', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / "pde_formulas.png", dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    # Reset rcParams
    plt.rcParams['text.usetex'] = False


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_pde_from_path(experiment_path: Path) -> str:
    """Extract PDE name from experiment folder name."""
    folder_name = experiment_path.name.lower()
    if 'schrodinger' in folder_name:
        return 'schrodinger'
    elif 'wave1d' in folder_name:
        return 'wave1d'
    elif 'burgers2d' in folder_name:
        return 'burgers2d'
    elif 'burgers1d' in folder_name:
        return 'burgers1d'
    # Try to extract from first part of name
    parts = folder_name.split('_')
    if parts:
        return parts[0]
    return 'unknown'


def load_experiment_config(experiment_path: Path) -> Dict[str, Any]:
    """Load full experiment config from experiments_plan.yaml."""
    plan_file = experiment_path / "experiments_plan.yaml"
    if not plan_file.exists():
        return {}
    
    with open(plan_file, 'r') as f:
        return yaml.safe_load(f)


def load_experiment_plan(experiment_path: Path) -> Optional[Dict[str, Any]]:
    """Load experiment plan YAML to map experiment names to architectures.
    
    Returns dict with 'name_to_arch' and 'arch_to_weight_label'.
    """
    plan_file = experiment_path / "experiments_plan.yaml"
    if not plan_file.exists():
        return None
    
    with open(plan_file, 'r') as f:
        plan = yaml.safe_load(f)
    
    name_to_arch = {}
    arch_to_weight_label = {}
    
    if 'experiments' in plan:
        for exp in plan['experiments']:
            if 'name' in exp and 'architecture' in exp:
                exp_name = exp['name']
                arch = exp['architecture']
                name_to_arch[exp_name] = arch
                
                weight_label = extract_weight_from_experiment_name(exp_name)
                if weight_label:
                    arch_to_weight_label[tuple(arch)] = weight_label
    
    return {
        'name_to_arch': name_to_arch,
        'arch_to_weight_label': arch_to_weight_label
    }


def load_all_model_metrics(
    experiment_path: Path,
    experiment_plan: Optional[Dict[str, Any]] = None,
    device: torch.device = None
) -> Dict[str, Dict[str, Any]]:
    """Load experiment structure and compute all metrics consistently.
    
    This function mirrors analyze_capacity_experiment.py's approach:
    1. Scans the experiment folder to find model directories
    2. For each model, loads the checkpoint and datasets
    3. Fits probes ONCE per model
    4. Computes ALL metrics using those same probes (including frequency metrics)
    
    Args:
        experiment_path: Path to experiment folder
        experiment_plan: Optional experiment plan dict
        device: Device for computation (auto-detected if None)
    
    Returns:
        Dict mapping model_name -> model data with all computed metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models_data = {}
    experiment_path = Path(experiment_path)
    
    arch_to_weight_label = {}
    if experiment_plan and 'arch_to_weight_label' in experiment_plan:
        arch_to_weight_label = experiment_plan['arch_to_weight_label']
    
    # First pass: collect all model directories
    model_dirs_to_process = []
    
    for model_dir in experiment_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        if model_name.startswith('.') or model_name in ['comparison_summary.csv', 'experiments_plan.yaml']:
            continue
        
        # Skip image files and other non-model directories
        if model_name.endswith('.png') or model_name.endswith('.csv'):
            continue
        
        run_dirs = [d for d in model_dir.iterdir() if d.is_dir() and re.match(r'\d{8}_\d{6}', d.name)]
        if not run_dirs:
            continue
        
        run_dir = max(run_dirs, key=lambda x: x.stat().st_mtime)
        
        metrics_file = run_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        
        model_dirs_to_process.append((model_name, run_dir, metrics_file))
    
    print(f"  Found {len(model_dirs_to_process)} models to analyze")
    
    # Second pass: compute metrics consistently for each model
    for idx, (model_name, run_dir, metrics_file) in enumerate(model_dirs_to_process, 1):
        print(f"\n  [{idx}/{len(model_dirs_to_process)}] Processing {model_name}...")
        
        # Load training metrics from JSON
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Parse architecture
        architecture = parse_model_name(model_name)
        num_layers = len(architecture) - 2  # Exclude input and output layers
        num_parameters = calculate_num_parameters(architecture)
        weight_label = arch_to_weight_label.get(tuple(architecture))
        problem_name = get_problem_from_model_name(model_name)
        
        # Compute all metrics consistently using analysis_core (on freq_grid)
        try:
            computed_metrics = compute_all_metrics_consistently(
                model_name=model_name,
                run_dir=run_dir,
                device=device
            )
            
            ncc_metrics = computed_metrics['ncc_metrics']
            probe_metrics = computed_metrics['probe_metrics']
            derivatives_metrics = computed_metrics['derivatives_metrics']
            frequency_metrics = computed_metrics['frequency_metrics']
            
            # Compute L2/Linf on freq_grid for consistency with frequency analysis
            # This is computed during compute_all_metrics_consistently and stored in computed_metrics
            final_eval_rel_l2 = computed_metrics.get('freq_grid_rel_l2')
            final_eval_linf = computed_metrics.get('freq_grid_linf')
            
        except FileNotFoundError as e:
            print(f"    ERROR: {e}")
            print(f"    Skipping {model_name} - checkpoint or dataset not found")
            continue
        except Exception as e:
            print(f"    ERROR computing metrics: {e}")
            print(f"    Skipping {model_name}")
            continue
        
        models_data[model_name] = {
            'model_name': model_name,
            'architecture': architecture,
            'num_layers': num_layers,
            'num_parameters': num_parameters,
            'weight_label': weight_label,
            'run_dir': run_dir,
            'metrics': metrics,
            'ncc_metrics': ncc_metrics,
            'probe_metrics': probe_metrics,
            'derivatives_metrics': derivatives_metrics,
            'frequency_metrics': frequency_metrics,
            'probes': computed_metrics.get('probes'),
            'eval_rel_l2': final_eval_rel_l2,
            'eval_linf': final_eval_linf,
            'problem_name': problem_name
        }
    
    return models_data


def detect_non_monotonic_metrics(models_data: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Detect non-monotonic metrics across all models."""
    all_violations = defaultdict(list)
    
    for model_name, data in models_data.items():
        problem_name = data.get('problem_name')
        bc_config = PROBLEM_BC_CONFIG.get(problem_name, DEFAULT_BC_CONFIG)
        
        for metric_name, config in METRICS_CONFIG.items():
            if metric_name.startswith('bc_value') and not bc_config.get('bc_value', True):
                continue
            if metric_name.startswith('bc_derivative') and not bc_config.get('bc_derivative', True):
                continue
            
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
            
            violations = check_monotonicity(values, direction)
            
            for v in violations:
                all_violations[metric_name].append({
                    'model_name': model_name,
                    'layer_num': v['layer_num'],
                    'prev_value': v['prev_value'],
                    'current_value': v['current_value'],
                    'degradation_abs': v['degradation_abs'],
                    'degradation_rel': v['degradation_rel'],
                })
    
    return dict(all_violations)


# =============================================================================
# PDE STATISTICS EXTRACTION
# =============================================================================

def get_derivative_order(deriv_name: str) -> int:
    """Get the derivative order from a derivative name.
    
    Examples:
        h_xx -> 2, h_t -> 1, h_x0x0 -> 2, h_tt -> 2
    """
    # Remove h_ prefix
    after_h = deriv_name.replace('h_', '').replace('h', '')
    # Count derivative symbols (x, t, x0, x1, etc.)
    # For h_xx: after_h = 'xx' -> matches ['x', 'x'] -> 2
    # For h_x0x0: after_h = 'x0x0' -> matches ['x0', 'x0'] -> 2
    # For h_t: after_h = 't' -> matches ['t'] -> 1
    matches = re.findall(r'x\d*|t', after_h)
    return len(matches)


def get_nonlinearity_rank(term_metadata: Dict) -> int:
    """Get nonlinearity rank from term metadata.
    
    Returns:
        0: Linear (empty metadata)
        2: Quadratic (convection term h*h_x)
        3: Cubic (|h|²h term)
    """
    if not term_metadata:
        return 0
    
    for term_key in term_metadata.keys():
        if 'cubic' in term_key.lower():
            return 3
        if 'convection' in term_key.lower():
            return 2
    
    return 1  # Default nonlinear


def extract_pde_stats(pde_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract PDE statistics from config and residual module."""
    base_config = config.get('base_config', config)
    pde_config = base_config.get(pde_name, {})
    
    # Get dimensions
    spatial_dim = pde_config.get('spatial_dim', 1)
    d = spatial_dim + 1  # Total dimensions (spatial + time)
    
    # Get domain volume
    spatial_domain = pde_config.get('spatial_domain', [[0, 1]])
    temporal_domain = pde_config.get('temporal_domain', [0, 1])
    
    spatial_vol = 1.0
    for domain_range in spatial_domain:
        spatial_vol *= (domain_range[1] - domain_range[0])
    temporal_vol = temporal_domain[1] - temporal_domain[0]
    total_vol = spatial_vol * temporal_vol
    
    # Get training samples
    n_residual = base_config.get('n_residual_train', 0)
    n_initial = base_config.get('n_initial_train', 0)
    n_boundary = base_config.get('n_boundary_train', 0)
    total_samples = n_residual + n_initial + n_boundary
    
    # Calculate per-dimension density: S^(1/d) / V^(1/d)
    if total_vol > 0 and total_samples > 0:
        per_dim_density = (total_samples ** (1/d)) / (total_vol ** (1/d))
    else:
        per_dim_density = 0
    
    # Get derivative order and nonlinearity from residual module
    deriv_order = 2  # Default
    nonlin_rank = 0  # Default (linear)
    
    try:
        if pde_name == 'schrodinger':
            from derivatives_tracker.residuals.schrodinger_residual import get_relevant_derivatives, get_term_metadata
        elif pde_name == 'wave1d':
            from derivatives_tracker.residuals.wave1d_residual import get_relevant_derivatives, get_term_metadata
        elif pde_name == 'burgers1d':
            from derivatives_tracker.residuals.burgers1d_residual import get_relevant_derivatives, get_term_metadata
        elif pde_name == 'burgers2d':
            from derivatives_tracker.residuals.burgers2d_residual import get_relevant_derivatives, get_term_metadata
        else:
            get_relevant_derivatives = None
            get_term_metadata = None
        
        if get_relevant_derivatives:
            derivs = get_relevant_derivatives()
            deriv_order = max(get_derivative_order(d) for d in derivs)
        
        if get_term_metadata:
            term_meta = get_term_metadata()
            nonlin_rank = get_nonlinearity_rank(term_meta)
    except ImportError:
        pass
    
    return {
        'pde': pde_name,
        'dims': d,
        'deriv_order': deriv_order,
        'nonlin_rank': nonlin_rank,
        'volume': total_vol,
        'samples': total_samples,
        'per_dim_density': per_dim_density
    }


# =============================================================================
# TABLE GENERATION
# =============================================================================

def create_table_image(
    df: pd.DataFrame,
    columns: List[str],
    col_labels: List[str],
    title: str,
    save_path: Path,
):
    """Create a table image with auto-sized columns."""
    max_model_len = max(len(str(row)) for row in df[columns[0]]) if len(df) > 0 else 30
    fig_width = max(18, 10 + max_model_len * 0.12)
    
    fig, ax = plt.subplots(figsize=(fig_width, max(4, len(df) * 0.5 + 2)))
    ax.axis('tight')
    ax.axis('off')
    
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
    
    # Style header
    for i in range(len(col_labels)):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# SECTION 1: PDE STATISTICS TABLE
# =============================================================================

def generate_pde_stats_table(pde_data: Dict[str, Dict], output_dir: Path):
    """Generate PDE statistics summary table and formulas figure."""
    stats_list = []
    
    for pde_name, data in pde_data.items():
        stats = extract_pde_stats(pde_name, data['config'])
        stats_list.append(stats)
    
    df = pd.DataFrame(stats_list)
    
    # Format columns
    df['volume_fmt'] = df['volume'].apply(lambda x: f"{x:.2f}")
    df['samples_fmt'] = df['samples'].apply(lambda x: f"{int(x):,}")
    df['density_fmt'] = df['per_dim_density'].apply(lambda x: f"{x:.2f}")
    
    # Save CSV
    df.to_csv(output_dir / "pde_stats_summary.csv", index=False)
    
    # Create stats table image
    columns = ['pde', 'dims', 'deriv_order', 'nonlin_rank', 'volume_fmt', 'samples_fmt', 'density_fmt']
    col_labels = ['PDE', 'Dims (d)', 'Deriv Order', 'Nonlin Rank', 'Volume (V)', 'Samples (S)', 'S^(1/d) / V^(1/d)']
    
    create_table_image(
        df, columns, col_labels,
        'PDE Statistics Summary',
        output_dir / "pde_stats_summary.png"
    )
    
    # Generate PDE formulas figure
    pde_names = list(pde_data.keys())
    generate_pde_formulas_figure(pde_names, output_dir)
    
    print(f"  PDE stats table and formulas saved to {output_dir}")


# =============================================================================
# SECTION 2: BEST MODEL SUMMARY
# =============================================================================

def generate_best_models_summary(pde_data: Dict[str, Dict], output_dir: Path):
    """Generate best model summary tables per PDE (by Rel-L2 and by L-inf)."""
    
    # === Best by Rel-L2 ===
    best_models_l2 = []
    for pde_name, data in pde_data.items():
        models = data['models']
        best_model = None
        best_val = float('inf')
        
        for model_name, model_data in models.items():
            val = model_data.get('eval_rel_l2')
            if val is not None and val < best_val:
                best_val = val
                best_model = model_data
        
        if best_model:
            best_models_l2.append({
                'pde': pde_name,
                'model_name': best_model['model_name'],
                'layers': best_model['num_layers'],
                'weights': best_model.get('weight_label') or f"{best_model['num_parameters']:,}",
                'eval_rel_l2': f"{best_model['eval_rel_l2']:.6f}",
                'eval_linf': f"{best_model['eval_linf']:.6f}" if best_model.get('eval_linf') else 'N/A'
            })
    
    df_l2 = pd.DataFrame(best_models_l2)
    df_l2.to_csv(output_dir / "best_models_summary_rel_l2.csv", index=False)
    
    columns = ['pde', 'model_name', 'layers', 'weights', 'eval_rel_l2', 'eval_linf']
    col_labels = ['PDE', 'Best Model', 'Layers', 'Weights', 'Eval Rel-L2', 'Eval L-inf']
    
    create_table_image(
        df_l2, columns, col_labels,
        'Best Model Summary (by Eval Rel-L2)',
        output_dir / "best_models_summary_rel_l2.png"
    )
    
    # === Best by L-inf ===
    best_models_linf = []
    for pde_name, data in pde_data.items():
        models = data['models']
        best_model = None
        best_val = float('inf')
        
        for model_name, model_data in models.items():
            val = model_data.get('eval_linf')
            if val is not None and val < best_val:
                best_val = val
                best_model = model_data
        
        if best_model:
            best_models_linf.append({
                'pde': pde_name,
                'model_name': best_model['model_name'],
                'layers': best_model['num_layers'],
                'weights': best_model.get('weight_label') or f"{best_model['num_parameters']:,}",
                'eval_rel_l2': f"{best_model['eval_rel_l2']:.6f}" if best_model.get('eval_rel_l2') else 'N/A',
                'eval_linf': f"{best_model['eval_linf']:.6f}"
            })
    
    df_linf = pd.DataFrame(best_models_linf)
    df_linf.to_csv(output_dir / "best_models_summary_linf.csv", index=False)
    
    create_table_image(
        df_linf, columns, col_labels,
        'Best Model Summary (by Eval L-inf)',
        output_dir / "best_models_summary_linf.png"
    )
    
    print(f"  Best models summary (Rel-L2 and L-inf) saved to {output_dir}")


# =============================================================================
# SECTION 3: RANK COMPARISONS
# =============================================================================

def compute_ranks_per_pde(
    models: Dict[str, Dict],
    metric: str = 'eval_rel_l2'
) -> Dict[str, int]:
    """Compute ranks for all models within a PDE by a given metric.
    
    Returns dict mapping model_name -> rank (1 = best).
    """
    # Filter models with valid metric values
    valid_models = [(name, data[metric]) for name, data in models.items() 
                    if data.get(metric) is not None]
    
    # Sort by metric (lower is better for both L2 and L-inf)
    sorted_models = sorted(valid_models, key=lambda x: x[1])
    
    # Assign ranks
    ranks = {}
    for rank, (name, _) in enumerate(sorted_models, 1):
        ranks[name] = rank
    
    return ranks


def get_model_key(model_data: Dict, key_type: str) -> Optional[str]:
    """Get a key for grouping models (layers or weights)."""
    if key_type == 'layers':
        return str(model_data['num_layers'])
    elif key_type == 'weights':
        return model_data.get('weight_label')
    return None


def generate_rank_comparison_table(
    pde_data: Dict[str, Dict],
    group_by: str,  # 'layers' or 'weights'
    group_value: str,  # e.g., '3' for 3 layers or '10k' for 10k weights
    metric: str,  # 'eval_rel_l2' or 'eval_linf'
    output_dir: Path
):
    """Generate a rank comparison table for a specific group."""
    # Compute ranks for each PDE
    pde_ranks = {}
    for pde_name, data in pde_data.items():
        pde_ranks[pde_name] = compute_ranks_per_pde(data['models'], metric)
    
    # Collect models matching the group criteria
    rows = []
    other_key = 'weights' if group_by == 'layers' else 'layers'
    
    # Get all unique values for the other key
    other_values = set()
    for pde_name, data in pde_data.items():
        for model_name, model_data in data['models'].items():
            if get_model_key(model_data, group_by) == group_value:
                other_val = get_model_key(model_data, other_key)
                if other_val:
                    other_values.add(other_val)
    
    # Sort other values
    if other_key == 'weights':
        def weight_sort_key(w):
            try:
                return int(w.replace('k', '')) * 1000
            except:
                return 0
        other_values = sorted(other_values, key=weight_sort_key)
    else:
        other_values = sorted(other_values, key=lambda x: int(x) if x.isdigit() else 0)
    
    for other_val in other_values:
        row = {other_key.capitalize(): other_val}
        
        for pde_name in sorted(pde_data.keys()):
            data = pde_data[pde_name]
            rank = None
            
            for model_name, model_data in data['models'].items():
                if (get_model_key(model_data, group_by) == group_value and
                    get_model_key(model_data, other_key) == other_val):
                    rank = pde_ranks[pde_name].get(model_name)
                    break
            
            row[pde_name.capitalize()] = str(rank) if rank else '-'
        
        rows.append(row)
    
    if not rows:
        return
    
    df = pd.DataFrame(rows)
    
    # Create image
    columns = list(df.columns)
    col_labels = columns.copy()
    
    metric_display = 'Rel-L2' if metric == 'eval_rel_l2' else 'L-inf'
    title = f"Rank Comparison: {group_value} {group_by.capitalize()} (by Eval {metric_display})"
    filename = f"{group_value}_{group_by}_rank_comparison_{metric.replace('eval_', '')}.png"
    
    create_table_image(df, columns, col_labels, title, output_dir / filename)


def generate_mean_rank_chart(
    pde_data: Dict[str, Dict],
    group_by: str,  # 'layers' or 'weights'
    metric: str,  # 'eval_rel_l2' or 'eval_linf'
    output_dir: Path
):
    """Generate a bar chart showing mean rank per group value per PDE."""
    # Compute ranks for each PDE
    pde_ranks = {}
    for pde_name, data in pde_data.items():
        pde_ranks[pde_name] = compute_ranks_per_pde(data['models'], metric)
    
    # Collect mean ranks per group value per PDE
    group_values = set()
    for pde_name, data in pde_data.items():
        for model_data in data['models'].values():
            gv = get_model_key(model_data, group_by)
            if gv:
                group_values.add(gv)
    
    # Sort group values
    if group_by == 'weights':
        def weight_sort_key(w):
            try:
                return int(w.replace('k', '')) * 1000
            except:
                return 0
        group_values = sorted(group_values, key=weight_sort_key)
    else:
        group_values = sorted(group_values, key=lambda x: int(x) if x.isdigit() else 0)
    
    # Calculate mean ranks
    pde_names = sorted(pde_data.keys())
    mean_ranks = {pde: [] for pde in pde_names}
    
    for gv in group_values:
        for pde_name in pde_names:
            data = pde_data[pde_name]
            ranks = []
            for model_name, model_data in data['models'].items():
                if get_model_key(model_data, group_by) == gv:
                    rank = pde_ranks[pde_name].get(model_name)
                    if rank:
                        ranks.append(rank)
            
            mean_rank = np.mean(ranks) if ranks else np.nan
            mean_ranks[pde_name].append(mean_rank)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(group_values))
    width = 0.8 / len(pde_names)
    
    for i, pde_name in enumerate(pde_names):
        offset = (i - len(pde_names)/2 + 0.5) * width
        color = PDE_COLORS.get(pde_name, f'C{i}')
        bars = ax.bar(x + offset, mean_ranks[pde_name], width, 
                      label=pde_name.capitalize(), color=color, alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, mean_ranks[pde_name]):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel(group_by.capitalize(), fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Rank (lower is better)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(group_values)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    metric_display = 'Rel-L2' if metric == 'eval_rel_l2' else 'L-inf'
    ax.set_title(f'Mean Rank by {group_by.capitalize()} (by Eval {metric_display})', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    filename = f"{group_by}_mean_rank_comparison_{metric.replace('eval_', '')}.png"
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()


def generate_rank_comparisons(pde_data: Dict[str, Dict], output_dir: Path):
    """Generate all rank comparison tables and charts."""
    
    # Get all unique layer counts and weight categories
    layer_counts = set()
    weight_categories = set()
    
    for pde_name, data in pde_data.items():
        for model_data in data['models'].values():
            layer_counts.add(str(model_data['num_layers']))
            wl = model_data.get('weight_label')
            if wl:
                weight_categories.add(wl)
    
    layer_counts = sorted(layer_counts, key=lambda x: int(x) if x.isdigit() else 0)
    
    def weight_sort_key(w):
        try:
            return int(w.replace('k', '')) * 1000
        except:
            return 0
    weight_categories = sorted(weight_categories, key=weight_sort_key)
    
    # Create output directories
    layers_dir = output_dir / "comparisons_by_layers"
    weights_dir = output_dir / "comparisons_by_weights"
    layers_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate tables and charts for each metric
    for metric in ['eval_rel_l2', 'eval_linf']:
        # By layers
        for layer_count in layer_counts:
            generate_rank_comparison_table(pde_data, 'layers', layer_count, metric, layers_dir)
        generate_mean_rank_chart(pde_data, 'layers', metric, layers_dir)
        
        # By weights
        for weight_cat in weight_categories:
            generate_rank_comparison_table(pde_data, 'weights', weight_cat, metric, weights_dir)
        generate_mean_rank_chart(pde_data, 'weights', metric, weights_dir)
    
    print(f"  Rank comparisons saved to {output_dir}")


# =============================================================================
# SECTION 4: NON-MONOTONIC ANALYSIS
# =============================================================================

def generate_cross_pde_violation_plots(pde_data: Dict[str, Dict], output_dir: Path):
    """Generate non-monotonic violation plots with per-PDE coloring."""
    
    violations_dir = output_dir / "non_monotonic_analysis"
    violations_dir.mkdir(parents=True, exist_ok=True)
    
    pde_names = sorted(pde_data.keys())
    
    # Collect violations per PDE
    pde_violations = {}
    for pde_name in pde_names:
        pde_violations[pde_name] = pde_data[pde_name]['violations']
    
    # 4a. Violations by metric
    _plot_violations_by_metric(pde_violations, pde_names, violations_dir)
    
    # 4b. Violations by depth (layer count)
    _plot_violations_by_depth(pde_data, pde_names, violations_dir)
    
    # 4c. Violations by layer position
    _plot_violations_by_layer_position(pde_violations, pde_names, violations_dir)
    
    # 4d. Violations by weights
    _plot_violations_by_weights(pde_data, pde_names, violations_dir)
    
    print(f"  Non-monotonic plots saved to {violations_dir}")


def _plot_violations_by_metric(pde_violations: Dict, pde_names: List[str], output_dir: Path):
    """Plot violations by metric type across PDEs."""
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Get all metrics that have violations
    all_metrics = set()
    for violations in pde_violations.values():
        all_metrics.update(violations.keys())
    
    # Use display names
    metric_order = [m for m in METRICS_CONFIG.keys() if m in all_metrics]
    metric_labels = [METRICS_CONFIG[m]['display_name'] for m in metric_order]
    
    x = np.arange(len(metric_order))
    width = 0.8 / len(pde_names)
    
    for i, pde_name in enumerate(pde_names):
        violations = pde_violations[pde_name]
        counts = [len(violations.get(m, [])) for m in metric_order]
        offset = (i - len(pde_names)/2 + 0.5) * width
        color = PDE_COLORS.get(pde_name, f'C{i}')
        ax.bar(x + offset, counts, width, label=pde_name.capitalize(), 
               color=color, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Violations', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Non-Monotonic Violations by Metric Type (All PDEs)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "violations_by_metric_all_PDEs.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_violations_by_depth(pde_data: Dict, pde_names: List[str], output_dir: Path):
    """Plot violations by model depth (layer count) across PDEs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get all layer counts
    layer_counts = set()
    for data in pde_data.values():
        for model_data in data['models'].values():
            layer_counts.add(model_data['num_layers'])
    layer_counts = sorted(layer_counts)
    
    x = np.arange(len(layer_counts))
    width = 0.8 / len(pde_names)
    
    for i, pde_name in enumerate(pde_names):
        violations = pde_data[pde_name]['violations']
        models = pde_data[pde_name]['models']
        
        # Count violations per layer count
        counts = []
        for lc in layer_counts:
            count = 0
            for metric_violations in violations.values():
                for v in metric_violations:
                    model_name = v['model_name']
                    if model_name in models and models[model_name]['num_layers'] == lc:
                        count += 1
            counts.append(count)
        
        offset = (i - len(pde_names)/2 + 0.5) * width
        color = PDE_COLORS.get(pde_name, f'C{i}')
        ax.bar(x + offset, counts, width, label=pde_name.capitalize(),
               color=color, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Number of Hidden Layers', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Violations', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(lc) for lc in layer_counts])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Non-Monotonic Violations by Model Depth (All PDEs)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "violations_by_depth_all_PDEs.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_violations_by_layer_position(pde_violations: Dict, pde_names: List[str], output_dir: Path):
    """Plot violations by layer position across PDEs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Layer positions start at 2 (layer 2 worse than layer 1)
    layer_positions = list(range(2, 8))
    
    x = np.arange(len(layer_positions))
    width = 0.8 / len(pde_names)
    
    for i, pde_name in enumerate(pde_names):
        violations = pde_violations[pde_name]
        
        counts = []
        for lp in layer_positions:
            count = 0
            for metric_violations in violations.values():
                for v in metric_violations:
                    if v['layer_num'] == lp:
                        count += 1
            counts.append(count)
        
        offset = (i - len(pde_names)/2 + 0.5) * width
        color = PDE_COLORS.get(pde_name, f'C{i}')
        ax.bar(x + offset, counts, width, label=pde_name.capitalize(),
               color=color, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Layer Number (where violation occurred)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Violations', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(lp) for lp in layer_positions])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Non-Monotonic Violations by Layer Position (All PDEs)\n(Layer N worse than Layer N-1)',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "violations_by_layer_position_all_PDEs.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_violations_by_weights(pde_data: Dict, pde_names: List[str], output_dir: Path):
    """Plot violations by weight category across PDEs."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get all weight categories
    weight_categories = set()
    for data in pde_data.values():
        for model_data in data['models'].values():
            wl = model_data.get('weight_label')
            if wl:
                weight_categories.add(wl)
    
    def weight_sort_key(w):
        try:
            return int(w.replace('k', '')) * 1000
        except:
            return 0
    weight_categories = sorted(weight_categories, key=weight_sort_key)
    
    if not weight_categories:
        print("    No weight labels found, skipping weight violations plot")
        plt.close()
        return
    
    x = np.arange(len(weight_categories))
    width = 0.8 / len(pde_names)
    
    for i, pde_name in enumerate(pde_names):
        violations = pde_data[pde_name]['violations']
        models = pde_data[pde_name]['models']
        
        counts = []
        for wc in weight_categories:
            count = 0
            for metric_violations in violations.values():
                for v in metric_violations:
                    model_name = v['model_name']
                    if model_name in models and models[model_name].get('weight_label') == wc:
                        count += 1
            counts.append(count)
        
        offset = (i - len(pde_names)/2 + 0.5) * width
        color = PDE_COLORS.get(pde_name, f'C{i}')
        ax.bar(x + offset, counts, width, label=pde_name.capitalize(),
               color=color, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Weight Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Violations', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(weight_categories)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Non-Monotonic Violations by Model Size (All PDEs)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "violations_by_weights_all_PDEs.png", dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# SECTION 5: FREQUENCY DOMAIN ANALYSIS
# =============================================================================

def get_frequency_data_for_pde(pde_data: Dict) -> Dict[str, Any]:
    """Extract frequency data from all models in a PDE experiment.
    
    Returns dict with:
    - gt_radial_power: Ground truth radial power spectrum (from any model)
    - k_radial_bins: Frequency bins
    - model_errors: Dict[model_name -> final layer error array]
    - model_error_matrices: Dict[model_name -> full error matrix [layers x freq]]
    """
    result = {
        'gt_radial_power': None,
        'k_radial_bins': None,
        'model_errors': {},
        'model_error_matrices': {},
        'model_layer_counts': {},
        'model_weight_labels': {}
    }
    
    for model_name, model_data in pde_data['models'].items():
        freq = model_data.get('frequency_metrics')
        if not freq:
            continue
        
        spectral = freq.get('spectral_efficiency', {})
        if not spectral:
            continue
        
        # Get GT spectrum (same for all models of same PDE)
        if result['gt_radial_power'] is None:
            result['gt_radial_power'] = np.array(spectral.get('gt_radial_power', []))
            result['k_radial_bins'] = np.array(spectral.get('k_radial_bins', []))
        
        error_matrix = spectral.get('error_matrix', [])
        if error_matrix:
            result['model_errors'][model_name] = np.array(error_matrix[-1])  # Final layer
            result['model_error_matrices'][model_name] = np.array(error_matrix)
            result['model_layer_counts'][model_name] = model_data['num_layers']
            result['model_weight_labels'][model_name] = model_data.get('weight_label')
    
    return result


def generate_gt_frequency_heatmap(pde_data: Dict[str, Dict], output_dir: Path):
    """Generate GT frequency content heatmap (rows=PDEs, cols=frequency bins)."""
    freq_dir = output_dir / "frequency_analysis"
    freq_dir.mkdir(parents=True, exist_ok=True)
    
    pde_names = sorted(pde_data.keys())
    
    # Collect GT spectra
    gt_spectra = {}
    k_bins = None
    
    for pde_name in pde_names:
        freq_data = get_frequency_data_for_pde(pde_data[pde_name])
        if freq_data['gt_radial_power'] is not None and len(freq_data['gt_radial_power']) > 0:
            gt_spectra[pde_name] = freq_data['gt_radial_power']
            if k_bins is None:
                k_bins = freq_data['k_radial_bins']
    
    if not gt_spectra:
        print("  No GT frequency data found - skipping GT heatmap")
        return
    
    # Normalize each spectrum to sum to 1 for fair comparison
    normalized_spectra = {}
    for pde_name, spectrum in gt_spectra.items():
        total = spectrum.sum()
        if total > 0:
            normalized_spectra[pde_name] = spectrum / total
        else:
            normalized_spectra[pde_name] = spectrum
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 4 + len(pde_names) * 0.5))
    
    # Build matrix
    matrix = np.array([normalized_spectra[pde] for pde in pde_names])
    
    im = ax.imshow(matrix, aspect='auto', cmap='viridis', interpolation='bilinear')
    
    # Labels
    ax.set_yticks(range(len(pde_names)))
    ax.set_yticklabels([p.capitalize() for p in pde_names], fontsize=11)
    
    # X-axis: frequency bins
    n_bins = len(k_bins) if k_bins is not None else matrix.shape[1]
    n_ticks = min(10, n_bins)
    tick_positions = np.linspace(0, n_bins - 1, n_ticks, dtype=int)
    if k_bins is not None:
        tick_labels = [f'{k_bins[i]:.1f}' for i in tick_positions]
    else:
        tick_labels = [str(i) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('Radial Frequency |k| (Hz)', fontsize=11, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Power', fontsize=10)
    
    ax.set_title('Ground Truth Frequency Content by PDE\n(Normalized Power Distribution)', 
                 fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(freq_dir / "gt_frequency_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  GT frequency heatmap saved")


def generate_frequency_error_heatmaps(pde_data: Dict[str, Dict], output_dir: Path):
    """Generate frequency error heatmaps - overall and grouped by layers/weights."""
    freq_dir = output_dir / "frequency_analysis"
    freq_dir.mkdir(parents=True, exist_ok=True)
    
    pde_names = sorted(pde_data.keys())
    
    # Collect all frequency data
    all_freq_data = {}
    k_bins = None
    
    for pde_name in pde_names:
        freq_data = get_frequency_data_for_pde(pde_data[pde_name])
        if freq_data['model_errors']:
            all_freq_data[pde_name] = freq_data
            if k_bins is None and freq_data['k_radial_bins'] is not None:
                k_bins = freq_data['k_radial_bins']
    
    if not all_freq_data:
        print("  No frequency error data found - skipping error heatmaps")
        return
    
    # 2a. Overall mean (all models per PDE)
    _generate_error_heatmap_overall(all_freq_data, pde_names, k_bins, freq_dir)
    
    # Get all unique layer counts and weight categories
    all_layer_counts = set()
    all_weight_cats = set()
    for pde_name, freq_data in all_freq_data.items():
        all_layer_counts.update(freq_data['model_layer_counts'].values())
        all_weight_cats.update([w for w in freq_data['model_weight_labels'].values() if w])
    
    all_layer_counts = sorted(all_layer_counts)
    all_weight_cats = sorted(all_weight_cats, key=lambda w: int(w.replace('k', '')) * 1000 if w else 0)
    
    # 2b. Grouped by layers
    if len(all_layer_counts) > 1:
        _generate_error_heatmap_grouped(all_freq_data, pde_names, k_bins, 
                                        all_layer_counts, 'layers', freq_dir)
    
    # 2c. Grouped by weights
    if len(all_weight_cats) > 1:
        _generate_error_heatmap_grouped(all_freq_data, pde_names, k_bins,
                                        all_weight_cats, 'weights', freq_dir)


def _generate_error_heatmap_overall(all_freq_data: Dict, pde_names: List[str], 
                                     k_bins: np.ndarray, output_dir: Path):
    """Generate overall frequency error heatmap (mean across all models)."""
    fig, ax = plt.subplots(figsize=(14, 4 + len(pde_names) * 0.5))
    
    # Build matrix: mean error per PDE
    rows = []
    valid_pdes = []
    for pde_name in pde_names:
        if pde_name not in all_freq_data:
            continue
        freq_data = all_freq_data[pde_name]
        if not freq_data['model_errors']:
            continue
        
        # Mean across all models
        errors = list(freq_data['model_errors'].values())
        mean_error = np.mean(errors, axis=0)
        rows.append(mean_error)
        valid_pdes.append(pde_name)
    
    if not rows:
        plt.close()
        return
    
    matrix = np.array(rows)
    
    # Use log scale for better visibility
    matrix_log = np.log10(matrix + 1e-10)
    
    im = ax.imshow(matrix_log, aspect='auto', cmap='Reds', interpolation='bilinear')
    
    ax.set_yticks(range(len(valid_pdes)))
    ax.set_yticklabels([p.capitalize() for p in valid_pdes], fontsize=11)
    
    n_bins = matrix.shape[1]
    n_ticks = min(10, n_bins)
    tick_positions = np.linspace(0, n_bins - 1, n_ticks, dtype=int)
    if k_bins is not None and len(k_bins) > 0:
        tick_labels = [f'{k_bins[i]:.1f}' for i in tick_positions if i < len(k_bins)]
    else:
        tick_labels = [str(i) for i in tick_positions]
    ax.set_xticks(tick_positions[:len(tick_labels)])
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('Radial Frequency |k| (Hz)', fontsize=11, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Log10 Relative Error', fontsize=10)
    
    ax.set_title('Frequency Learning Difficulty by PDE\n(Mean Error Across All Models) [log scale]',
                 fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "frequency_error_heatmap_overall.png", dpi=150, bbox_inches='tight')
    plt.close()


def _generate_error_heatmap_grouped(all_freq_data: Dict, pde_names: List[str],
                                     k_bins: np.ndarray, group_values: List,
                                     group_by: str, output_dir: Path):
    """Generate side-by-side error heatmaps grouped by layers or weights."""
    n_groups = len(group_values)
    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 4 + len(pde_names) * 0.4))
    if n_groups == 1:
        axes = [axes]
    
    for idx, group_val in enumerate(group_values):
        ax = axes[idx]
        
        # Build matrix for this group
        rows = []
        valid_pdes = []
        
        for pde_name in pde_names:
            if pde_name not in all_freq_data:
                continue
            freq_data = all_freq_data[pde_name]
            
            # Filter models by group
            group_errors = []
            for model_name, error in freq_data['model_errors'].items():
                if group_by == 'layers':
                    if freq_data['model_layer_counts'].get(model_name) == group_val:
                        group_errors.append(error)
                else:  # weights
                    if freq_data['model_weight_labels'].get(model_name) == group_val:
                        group_errors.append(error)
            
            if group_errors:
                mean_error = np.mean(group_errors, axis=0)
                rows.append(mean_error)
                valid_pdes.append(pde_name)
        
        if not rows:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
            ax.set_title(f'{group_val} {group_by.capitalize()}', fontsize=11, fontweight='bold')
            ax.axis('off')
            continue
        
        matrix = np.array(rows)
        matrix_log = np.log10(matrix + 1e-10)
        
        im = ax.imshow(matrix_log, aspect='auto', cmap='Reds', interpolation='bilinear')
        
        ax.set_yticks(range(len(valid_pdes)))
        ax.set_yticklabels([p.capitalize() for p in valid_pdes], fontsize=9)
        
        n_bins = matrix.shape[1]
        n_ticks = min(6, n_bins)
        tick_positions = np.linspace(0, n_bins - 1, n_ticks, dtype=int)
        if k_bins is not None and len(k_bins) > 0:
            tick_labels = [f'{k_bins[i]:.1f}' for i in tick_positions if i < len(k_bins)]
        else:
            tick_labels = [str(i) for i in tick_positions]
        ax.set_xticks(tick_positions[:len(tick_labels)])
        ax.set_xticklabels(tick_labels, fontsize=8)
        
        if idx == 0:
            ax.set_ylabel('PDE', fontsize=10, fontweight='bold')
        ax.set_xlabel('|k| (Hz)', fontsize=9)
        ax.set_title(f'{group_val} {group_by.capitalize()}', fontsize=11, fontweight='bold')
    
    # Add shared colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Log10 Error', fontsize=10)
    
    fig.suptitle(f'Frequency Error by {group_by.capitalize()} [log scale]', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.savefig(output_dir / f"frequency_error_by_{group_by}.png", dpi=150, bbox_inches='tight')
    plt.close()


def generate_layerwise_frequency_heatmaps(pde_data: Dict[str, Dict], output_dir: Path):
    """Generate layer-wise frequency error heatmaps grouped by layers/weights."""
    freq_dir = output_dir / "frequency_analysis"
    freq_dir.mkdir(parents=True, exist_ok=True)
    
    pde_names = sorted(pde_data.keys())
    
    # Collect all frequency data
    all_freq_data = {}
    k_bins = None
    
    for pde_name in pde_names:
        freq_data = get_frequency_data_for_pde(pde_data[pde_name])
        if freq_data['model_error_matrices']:
            all_freq_data[pde_name] = freq_data
            if k_bins is None and freq_data['k_radial_bins'] is not None:
                k_bins = freq_data['k_radial_bins']
    
    if not all_freq_data:
        print("  No layerwise frequency data found - skipping")
        return
    
    # Get all unique layer counts and weight categories
    all_layer_counts = set()
    all_weight_cats = set()
    for pde_name, freq_data in all_freq_data.items():
        all_layer_counts.update(freq_data['model_layer_counts'].values())
        all_weight_cats.update([w for w in freq_data['model_weight_labels'].values() if w])
    
    all_layer_counts = sorted(all_layer_counts)
    all_weight_cats = sorted(all_weight_cats, key=lambda w: int(w.replace('k', '')) * 1000 if w else 0)
    
    # 3a. By layer count - show layer-by-layer progression
    if len(all_layer_counts) > 1:
        _generate_layerwise_by_group(all_freq_data, pde_names, k_bins,
                                      all_layer_counts, 'layers', freq_dir)
    
    # 3b. By weights
    if len(all_weight_cats) > 1:
        _generate_layerwise_by_group(all_freq_data, pde_names, k_bins,
                                      all_weight_cats, 'weights', freq_dir)


def _generate_layerwise_by_group(all_freq_data: Dict, pde_names: List[str],
                                  k_bins: np.ndarray, group_values: List,
                                  group_by: str, output_dir: Path):
    """Generate layer-wise heatmaps for each group value."""
    n_groups = len(group_values)
    
    for group_val in group_values:
        # Collect all error matrices for this group across PDEs
        all_matrices = []
        max_layers = 0
        
        for pde_name in pde_names:
            if pde_name not in all_freq_data:
                continue
            freq_data = all_freq_data[pde_name]
            
            for model_name, error_matrix in freq_data['model_error_matrices'].items():
                if group_by == 'layers':
                    if freq_data['model_layer_counts'].get(model_name) == group_val:
                        all_matrices.append(error_matrix)
                        max_layers = max(max_layers, error_matrix.shape[0])
                else:
                    if freq_data['model_weight_labels'].get(model_name) == group_val:
                        all_matrices.append(error_matrix)
                        max_layers = max(max_layers, error_matrix.shape[0])
        
        if not all_matrices or max_layers == 0:
            continue
        
        # Pad matrices to same size and compute mean
        n_freq = all_matrices[0].shape[1]
        padded = np.full((len(all_matrices), max_layers, n_freq), np.nan)
        for i, mat in enumerate(all_matrices):
            padded[i, :mat.shape[0], :] = mat
        
        mean_matrix = np.nanmean(padded, axis=0)
        
        # Transpose matrix to swap axes: now [freq x layers]
        mean_matrix_T = mean_matrix.T
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(4 + max_layers * 1.2, 8))
        
        matrix_log = np.log10(mean_matrix_T + 1e-10)
        im = ax.imshow(matrix_log, aspect='auto', cmap='Reds', interpolation='bilinear')
        
        # X-axis: Layers
        ax.set_xticks(range(max_layers))
        ax.set_xticklabels([f'Layer {i+1}' for i in range(max_layers)], fontsize=10, rotation=45, ha='right')
        ax.set_xlabel('Layer', fontsize=11, fontweight='bold')
        
        # Y-axis: Frequency bins
        n_bins = n_freq
        n_ticks = min(10, n_bins)
        tick_positions = np.linspace(0, n_bins - 1, n_ticks, dtype=int)
        if k_bins is not None and len(k_bins) > 0:
            tick_labels = [f'{k_bins[i]:.1f}' for i in tick_positions if i < len(k_bins)]
        else:
            tick_labels = [str(i) for i in tick_positions]
        ax.set_yticks(tick_positions[:len(tick_labels)])
        ax.set_yticklabels(tick_labels)
        ax.set_ylabel('Radial Frequency |k| (Hz)', fontsize=11, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Log10 Relative Error', fontsize=10)
        
        ax.set_title(f'Layer-wise Frequency Learning: {group_val} {group_by.capitalize()}\n'
                     f'(Mean Across {len(all_matrices)} Models, All PDEs) [log scale]',
                     fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"layerwise_frequency_{group_by}_{group_val}.png", 
                    dpi=150, bbox_inches='tight')
        plt.close()


def generate_learning_progression_comparison(pde_data: Dict[str, Dict], output_dir: Path):
    """Generate layer-wise learning progression comparison (probe vs frequency)."""
    freq_dir = output_dir / "frequency_analysis"
    freq_dir.mkdir(parents=True, exist_ok=True)
    
    pde_names = sorted(pde_data.keys())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for pde_name in pde_names:
        # Get best model (by eval_rel_l2)
        models = pde_data[pde_name]['models']
        best_model = None
        best_val = float('inf')
        
        for model_name, model_data in models.items():
            val = model_data.get('eval_rel_l2')
            if val is not None and val < best_val:
                best_val = val
                best_model = model_data
        
        if not best_model:
            continue
        
        color = PDE_COLORS.get(pde_name, 'gray')
        
        # Probe error per layer
        probe = best_model.get('probe_metrics')
        if probe and 'eval' in probe:
            rel_l2 = probe['eval'].get('rel_l2', [])
            if rel_l2:
                layers = list(range(1, len(rel_l2) + 1))
                ax1.plot(layers, rel_l2, 'o-', color=color, label=pde_name.capitalize(),
                        linewidth=2, markersize=6)
        
        # Frequency leftover ratio per layer
        freq = best_model.get('frequency_metrics')
        if freq and 'layer_metrics' in freq:
            layer_metrics = freq['layer_metrics']
            if layer_metrics:
                layers_analyzed = list(layer_metrics.keys())
                leftover_ratios = [layer_metrics[l].get('leftover_ratio', 0) for l in layers_analyzed]
                layer_nums = list(range(1, len(leftover_ratios) + 1))
                ax2.plot(layer_nums, leftover_ratios, 's-', color=color, 
                        label=pde_name.capitalize(), linewidth=2, markersize=6)
    
    # Configure probe plot
    ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probe Rel-L2 Error', fontsize=12, fontweight='bold')
    ax1.set_title('Probe Error per Layer\n(Best Model per PDE)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Configure frequency plot
    ax2.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency Leftover Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('Frequency Leftover Ratio per Layer\n(Best Model per PDE)', 
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(freq_dir / "learning_progression_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Learning progression comparison saved")


def get_frequency_reduction_at_layer(
    freq_metrics: Dict,
    layer_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Get frequency error change ratio E_i / E_{i-1} at a specific layer transition.
    
    Args:
        freq_metrics: frequency_metrics data (computed on frequency grid)
        layer_idx: 0-based index of the layer (error at layer_idx vs layer_idx-1)
        
    Returns:
        (k_bins, error_ratio) where:
        - < 1 = improvement (gets greener as it approaches 0)
        - 1 = no change
        - > 1 = degradation (gets redder as it increases)
    """
    spectral_eff = freq_metrics.get('spectral_efficiency', {})
    k_bins = np.array(spectral_eff.get('k_radial_bins', []))
    error_matrix = np.array(spectral_eff.get('error_matrix', []))
    
    if len(error_matrix) == 0 or layer_idx <= 0 or layer_idx >= len(error_matrix):
        return np.array([]), np.array([])
    
    # Error ratio: curr_error / prev_error (similar to change_heatmaps)
    prev_error = error_matrix[layer_idx - 1]
    curr_error = error_matrix[layer_idx]
    
    # Avoid division by zero
    eps = 1e-10
    error_ratio = curr_error / (prev_error + eps)
    
    return k_bins, error_ratio


def generate_per_metric_aggregated_reduction(pde_data: Dict[str, Dict], output_dir: Path):
    """Generate per-metric aggregated frequency reduction plots across all PDEs.
    
    Creates one image per metric. Each image has:
    - Row 1: GT frequency power spectrum for each PDE (columns)
    - Row 2: Aggregated frequency error reduction for each PDE (columns)
    """
    freq_dir = output_dir / "frequency_analysis"
    freq_dir.mkdir(parents=True, exist_ok=True)
    
    pde_names = sorted(pde_data.keys())
    n_pdes = len(pde_names)
    
    # Define which metrics to generate plots for
    metrics_to_plot = ['probe_rel_l2', 'probe_linf', 'ncc_accuracy', 'derivatives_l2', 'derivatives_linf']
    
    for metric_name in metrics_to_plot:
        if metric_name not in METRICS_CONFIG:
            continue
        
        # Collect data per PDE
        pde_gt_data = {}  # {pde_name: (k_bins, gt_radial_power)}
        pde_aggregated_ratios = {}  # {pde_name: (k_bins, mean_ratio, std_ratio, n_violations)}
        
        for pde_name in pde_names:
            models = pde_data[pde_name]['models']
            violations = pde_data[pde_name]['violations']
            
            # Get violations for this metric
            metric_violations = violations.get(metric_name, [])
            if not metric_violations:
                continue
            
            # Group violations by model
            violations_by_model = defaultdict(list)
            for v in metric_violations:
                violations_by_model[v['model_name']].append(v)
            
            # Collect all error ratios and GT from this PDE
            all_ratios = []
            k_bins_ref = None
            gt_radial_ref = None
            
            for model_name, model_violations in violations_by_model.items():
                if model_name not in models:
                    continue
                    
                freq_metrics = models[model_name].get('frequency_metrics')
                if freq_metrics is None:
                    continue
                
                # Get GT spectrum (same for all models in PDE)
                if gt_radial_ref is None:
                    spectral_eff = freq_metrics.get('spectral_efficiency', {})
                    k_bins_ref = np.array(spectral_eff.get('k_radial_bins', []))
                    gt_radial_ref = np.array(spectral_eff.get('gt_radial_power', []))
                
                for v in model_violations:
                    layer_idx = v.get('layer_idx', v.get('layer_num', 1) - 1)
                    k_bins, error_ratio = get_frequency_reduction_at_layer(freq_metrics, layer_idx)
                    
                    if len(error_ratio) > 0:
                        all_ratios.append(error_ratio)
            
            if all_ratios and k_bins_ref is not None and len(k_bins_ref) > 0:
                all_ratios = np.array(all_ratios)
                mean_ratio = np.mean(all_ratios, axis=0)
                std_ratio = np.std(all_ratios, axis=0)
                
                pde_gt_data[pde_name] = (k_bins_ref, gt_radial_ref)
                pde_aggregated_ratios[pde_name] = (k_bins_ref, mean_ratio, std_ratio, len(all_ratios))
        
        # Skip if no data for any PDE
        if not pde_aggregated_ratios:
            continue
        
        # Create figure: 2 rows x n_pdes columns
        valid_pdes = [p for p in pde_names if p in pde_aggregated_ratios]
        n_valid = len(valid_pdes)
        
        if n_valid == 0:
            continue
        
        fig, axes = plt.subplots(2, n_valid, figsize=(5 * n_valid, 8), squeeze=False)
        
        for col_idx, pde_name in enumerate(valid_pdes):
            color = PDE_COLORS.get(pde_name, '#333333')
            
            # Row 1: GT power spectrum
            ax_gt = axes[0, col_idx]
            if pde_name in pde_gt_data:
                k_bins, gt_power = pde_gt_data[pde_name]
                ax_gt.fill_between(k_bins, 0, gt_power, alpha=0.4, color=color)
                ax_gt.plot(k_bins, gt_power, color=color, linewidth=2, label='GT Power')
                ax_gt.set_xlabel('|k| (Hz)', fontsize=10)
                if col_idx == 0:
                    ax_gt.set_ylabel('Power |FFT|²', fontsize=10)
                if np.all(gt_power > 0):
                    ax_gt.set_yscale('log')
                ax_gt.set_title(f'{pde_name.capitalize()}', fontsize=11, fontweight='bold')
                ax_gt.grid(True, alpha=0.3)
            
            # Row 2: Aggregated frequency reduction
            ax_err = axes[1, col_idx]
            if pde_name in pde_aggregated_ratios:
                k_bins, mean_ratio, std_ratio, n_violations = pde_aggregated_ratios[pde_name]
                
                # Reference line at y=1
                ax_err.axhline(y=1, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
                
                # Mean with confidence band
                ax_err.fill_between(k_bins, mean_ratio - std_ratio, mean_ratio + std_ratio,
                                   alpha=0.3, color=color)
                ax_err.plot(k_bins, mean_ratio, color=color, linewidth=2, label='Mean')
                
                # Color regions: < 1 = improvement (green), > 1 = degradation (red)
                ax_err.fill_between(k_bins, 1, mean_ratio,
                                   where=(mean_ratio < 1), color='#2ecc71', alpha=0.2)
                ax_err.fill_between(k_bins, 1, mean_ratio,
                                   where=(mean_ratio > 1), color='#e74c3c', alpha=0.2)
                
                ax_err.set_xlabel('|k| (Hz)', fontsize=10)
                if col_idx == 0:
                    ax_err.set_ylabel('Error Ratio (E_i / E_{i-1})', fontsize=10)
                ax_err.set_title(f'n={n_violations} violations', fontsize=10)
                ax_err.grid(True, alpha=0.3)
        
        # Format metric name for title
        metric_display = metric_name.replace('_', ' ').title()
        fig.suptitle(f'Aggregated Frequency Reduction - {metric_display}', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        save_path = freq_dir / f"aggregated_freq_reduction_{metric_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Aggregated frequency reduction for {metric_name} saved")


def get_weight_label_from_params(num_parameters: int) -> str:
    """Convert number of parameters to a weight label bucket (5k, 10k, 20k, 30k, 40k).
    
    Uses nearest bucket based on actual parameter count.
    """
    buckets = [5000, 10000, 20000, 30000, 40000]
    bucket_labels = ['5k', '10k', '20k', '30k', '40k']
    
    # Find closest bucket
    min_diff = float('inf')
    closest_label = bucket_labels[-1]
    
    for bucket, label in zip(buckets, bucket_labels):
        diff = abs(num_parameters - bucket)
        if diff < min_diff:
            min_diff = diff
            closest_label = label
    
    return closest_label


def get_model_identity(model_data: Dict) -> Tuple[int, str]:
    """Get model identity as (num_layers, weight_label) for cross-PDE comparison.
    
    Models with same layers and weights are considered the same model
    even if they have different input/output dimensions.
    
    IMPORTANT: Uses calculated num_parameters to determine weight bucket,
    not the name-based weight_label, to avoid mismatches.
    """
    num_layers = model_data.get('num_layers', 0)
    num_parameters = model_data.get('num_parameters', 0)
    
    # Calculate weight label from actual parameters
    weight_label = get_weight_label_from_params(num_parameters)
    
    return (num_layers, weight_label)


def generate_cross_pde_frequency_coverage(pde_data: Dict[str, Dict], output_dir: Path):
    """Generate frequency coverage comparison across all PDEs with rank table.
    
    Creates a figure with:
    - Top row: Frequency coverage plots for each PDE (side by side, log scale)
    - Bottom: Rank table showing each model's rank (1-12) in each PDE by L2 and L-inf
    
    Same color = same model identity (same layers + weights across PDEs)
    """
    freq_dir = output_dir / "frequency_analysis"
    freq_dir.mkdir(parents=True, exist_ok=True)
    
    pde_names = sorted(pde_data.keys())
    n_pdes = len(pde_names)
    
    if n_pdes == 0:
        return
    
    # Collect all unique model identities across all PDEs
    # Debug: print mapping to verify
    print("  Model identity mapping (based on calculated params):")
    all_model_identities = set()
    for pde_name in pde_names:
        models = pde_data[pde_name]['models']
        for model_name, model_data in models.items():
            identity = get_model_identity(model_data)
            num_params = model_data.get('num_parameters', 0)
            old_label = model_data.get('weight_label', 'N/A')
            if identity[1]:  # Only if weight_label exists
                all_model_identities.add(identity)
                print(f"    {model_name}: {num_params:,} params -> {identity[0]}L-{identity[1]} (name had: {old_label})")
    
    # Sort identities by layers first, then weight
    def sort_key_layers_first(identity):
        layers, weight = identity
        # Convert weight like "5k" to int 5000
        weight_num = int(weight.replace('k', '')) * 1000 if weight else 0
        return (layers, weight_num)
    
    # Sort identities by weight first, then layers
    def sort_key_weights_first(identity):
        layers, weight = identity
        # Convert weight like "5k" to int 5000
        weight_num = int(weight.replace('k', '')) * 1000 if weight else 0
        return (weight_num, layers)
    
    sorted_identities = sorted(all_model_identities, key=sort_key_layers_first)
    sorted_identities_by_weight = sorted(all_model_identities, key=sort_key_weights_first)
    n_models = len(sorted_identities)
    
    if n_models == 0:
        print("  No model identities found - skipping frequency coverage")
        return
    
    # Assign colors to each model identity
    identity_colors = {}
    colormap = plt.cm.tab20(np.linspace(0, 1, max(n_models, 1)))
    print("\n=== COLOR ASSIGNMENT DEBUG ===")
    print(f"Total models: {n_models}")
    print("Colors assigned by layers→weights sort order:")
    for idx, identity in enumerate(sorted_identities):
        identity_colors[identity] = colormap[idx % len(colormap)]
        color_rgb = identity_colors[identity][:3]
        print(f"  Index {idx}: {identity[0]}L-{identity[1]} -> RGB({color_rgb[0]:.3f}, {color_rgb[1]:.3f}, {color_rgb[2]:.3f})")
    
    # Create identity labels for legend/table
    identity_labels = {}
    for identity in sorted_identities:
        layers, weight = identity
        identity_labels[identity] = f"{layers}L-{weight}"
    
    # Collect frequency data and ranks for each PDE
    pde_freq_coverage = {}  # {pde_name: {identity: (k_bins, final_error)}}
    pde_ranks_l2 = {}  # {pde_name: {identity: rank}}
    pde_ranks_linf = {}  # {pde_name: {identity: rank}}
    
    for pde_name in pde_names:
        print(f"\n=== Processing PDE: {pde_name} ===")
        models = pde_data[pde_name]['models']
        freq_coverage = {}
        model_l2 = []  # [(identity, l2_error)]
        model_linf = []  # [(identity, linf_error)]
        
        for model_name, model_data in models.items():
            identity = get_model_identity(model_data)
            if identity not in sorted_identities:
                continue
            
            l2 = model_data.get('eval_rel_l2')
            linf = model_data.get('eval_linf')
            
            # Get frequency metrics
            freq = model_data.get('frequency_metrics')
            has_freq = False
            freq_error_range = "N/A"
            if freq and 'spectral_efficiency' in freq:
                spectral = freq['spectral_efficiency']
                k_bins = np.array(spectral.get('k_radial_bins', []))
                model_error = np.array(spectral.get('model_error', []))  # Use model's direct output, not probe
                
                if len(model_error) > 0 and len(k_bins) > 0:
                    freq_coverage[identity] = (k_bins, model_error)
                    has_freq = True
                    freq_error_range = f"[{model_error.min():.2e}, {model_error.max():.2e}]"
            
            l2_str = f"{l2:.6f}" if l2 is not None else 'N/A'
            color_rgb = identity_colors[identity][:3]
            print(f"    {model_name}")
            print(f"      -> Identity: {identity[0]}L-{identity[1]}")
            print(f"      -> Color: RGB({color_rgb[0]:.3f}, {color_rgb[1]:.3f}, {color_rgb[2]:.3f})")
            print(f"      -> L2: {l2_str}")
            print(f"      -> Has freq data: {has_freq}, Error range: {freq_error_range}")
            
            # Get L2 and L-inf for ranking
            if l2 is not None:
                model_l2.append((identity, l2))
            if linf is not None:
                model_linf.append((identity, linf))
        
        pde_freq_coverage[pde_name] = freq_coverage
        
        # Compute ranks (1 = best, lower error)
        model_l2.sort(key=lambda x: x[1])
        model_linf.sort(key=lambda x: x[1])
        
        pde_ranks_l2[pde_name] = {identity: rank + 1 for rank, (identity, _) in enumerate(model_l2)}
        pde_ranks_linf[pde_name] = {identity: rank + 1 for rank, (identity, _) in enumerate(model_linf)}
        
        # Print ranking with colors
        print(f"\n  === L2 Ranking for {pde_name} ===")
        for rank, (identity, l2_val) in enumerate(model_l2, 1):
            color_rgb = identity_colors[identity][:3]
            has_freq = identity in freq_coverage
            freq_note = "HAS FREQ DATA" if has_freq else "NO FREQ DATA"
            color_str = f"RGB({color_rgb[0]:.3f}, {color_rgb[1]:.3f}, {color_rgb[2]:.3f})"
            print(f"    Rank {rank:2d}: {identity[0]}L-{identity[1]:>3s} (L2={l2_val:.6f}) | Color: {color_str} | {freq_note}")
    
    # Create figure: frequency plots on top, two tables on bottom
    # Extra width for legend on right side
    fig = plt.figure(figsize=(max(5 * n_pdes, 16) + 2, 14))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1.2], hspace=0.25)
    # Leave space on right for shared legend
    fig.subplots_adjust(right=0.88)
    
    # Top: Frequency coverage subplots
    gs_top = gs[0].subgridspec(1, n_pdes, wspace=0.15)
    
    print("\n=== PLOTTING FREQUENCY COVERAGE ===")
    for col_idx, pde_name in enumerate(pde_names):
        ax = fig.add_subplot(gs_top[0, col_idx])
        
        freq_coverage = pde_freq_coverage.get(pde_name, {})
        print(f"\nPlotting {pde_name}:")
        
        for identity in sorted_identities:
            if identity not in freq_coverage:
                continue
            
            k_bins, final_error = freq_coverage[identity]
            color = identity_colors[identity]
            label = identity_labels[identity]
            
            # Debug: show what's being plotted
            color_rgb = color[:3]
            error_stats = f"min={final_error.min():.2e}, max={final_error.max():.2e}, mean={final_error.mean():.2e}"
            l2_rank = pde_ranks_l2[pde_name].get(identity, 'N/A')
            print(f"  {label}: Color=RGB({color_rgb[0]:.3f}, {color_rgb[1]:.3f}, {color_rgb[2]:.3f}), L2_Rank={l2_rank}, Freq_errors={error_stats}")
            
            ax.plot(k_bins, final_error, color=color, linewidth=2, alpha=0.8, label=label)
        
        ax.set_xlabel('Radial Frequency |k| (Hz)', fontsize=10)
        if col_idx == 0:
            ax.set_ylabel('Relative Error [log]', fontsize=10)
        ax.set_title(f'{pde_name.capitalize()}', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    # Create a shared legend for ALL models (not just those in last plot)
    # Place it to the right of the frequency plots
    legend_handles = []
    legend_labels = []
    for identity in sorted_identities:
        color = identity_colors[identity]
        label = identity_labels[identity]
        handle = plt.Line2D([0], [0], color=color, linewidth=2, alpha=0.8)
        legend_handles.append(handle)
        legend_labels.append(label)
    
    # Add legend to the right of the last subplot
    fig.legend(legend_handles, legend_labels, 
               loc='center right', 
               bbox_to_anchor=(0.99, 0.72),
               fontsize=8, 
               ncol=1,
               title='Models',
               title_fontsize=9)
    
    # Bottom: Two rank tables side by side (sorted by layers, sorted by weights)
    gs_bottom = gs[1].subgridspec(1, 2, wspace=0.3)
    
    # Build table data: rows = model identities
    # Columns: all L2 ranks first, then all L-inf ranks
    col_headers = []
    for pde_name in pde_names:
        col_headers.append(f'{pde_name.capitalize()}\nL2')
    for pde_name in pde_names:
        col_headers.append(f'{pde_name.capitalize()}\nL-inf')
    
    # Helper function to build table data for a given ordering
    def build_table_data(ordered_identities):
        table_data = []
        row_labels = []
        cell_colors = []
        
        for identity in ordered_identities:
            row_labels.append(identity_labels[identity])
            row = []
            row_colors = []
            
            # First all L2 columns
            for pde_name in pde_names:
                rank_l2 = pde_ranks_l2[pde_name].get(identity, '-')
                if isinstance(rank_l2, int):
                    row.append(str(rank_l2))
                    # Color: green (rank 1) to red (rank 12)
                    norm_rank = (rank_l2 - 1) / max(n_models - 1, 1)  # 0 = best, 1 = worst
                    # Green to red gradient
                    r = min(1, 2 * norm_rank)
                    g = min(1, 2 * (1 - norm_rank))
                    row_colors.append((r, g, 0.3, 0.7))
                else:
                    row.append('-')
                    row_colors.append((0.9, 0.9, 0.9, 0.5))
            
            # Then all L-inf columns
            for pde_name in pde_names:
                rank_linf = pde_ranks_linf[pde_name].get(identity, '-')
                if isinstance(rank_linf, int):
                    row.append(str(rank_linf))
                    norm_rank = (rank_linf - 1) / max(n_models - 1, 1)
                    r = min(1, 2 * norm_rank)
                    g = min(1, 2 * (1 - norm_rank))
                    row_colors.append((r, g, 0.3, 0.7))
                else:
                    row.append('-')
                    row_colors.append((0.9, 0.9, 0.9, 0.5))
            
            table_data.append(row)
            cell_colors.append(row_colors)
        
        return table_data, row_labels, cell_colors
    
    # Left table: sorted by layers, then weights
    ax_table_layers = fig.add_subplot(gs_bottom[0, 0])
    ax_table_layers.axis('off')
    ax_table_layers.set_title('Sorted by Layers → Weights', fontsize=10, fontweight='bold', pad=10)
    
    table_data, row_labels, cell_colors = build_table_data(sorted_identities)
    
    if table_data:
        table1 = ax_table_layers.table(
            cellText=table_data,
            rowLabels=row_labels,
            colLabels=col_headers,
            cellColours=cell_colors,
            rowColours=[identity_colors[id] for id in sorted_identities],
            loc='center',
            cellLoc='center'
        )
        table1.auto_set_font_size(False)
        table1.set_fontsize(8)
        table1.scale(0.65, 1.4)
        table1.auto_set_column_width(col=list(range(len(col_headers))))
        
        for (row, col), cell in table1.get_celld().items():
            if row == 0:
                cell.set_text_props(fontweight='bold')
                cell.set_facecolor('#e0e0e0')
    
    # Right table: sorted by weights, then layers
    ax_table_weights = fig.add_subplot(gs_bottom[0, 1])
    ax_table_weights.axis('off')
    ax_table_weights.set_title('Sorted by Weights → Layers', fontsize=10, fontweight='bold', pad=10)
    
    table_data_w, row_labels_w, cell_colors_w = build_table_data(sorted_identities_by_weight)
    
    if table_data_w:
        table2 = ax_table_weights.table(
            cellText=table_data_w,
            rowLabels=row_labels_w,
            colLabels=col_headers,
            cellColours=cell_colors_w,
            rowColours=[identity_colors[id] for id in sorted_identities_by_weight],
            loc='center',
            cellLoc='center'
        )
        table2.auto_set_font_size(False)
        table2.set_fontsize(8)
        table2.scale(0.65, 1.4)
        table2.auto_set_column_width(col=list(range(len(col_headers))))
        
        for (row, col), cell in table2.get_celld().items():
            if row == 0:
                cell.set_text_props(fontweight='bold')
                cell.set_facecolor('#e0e0e0')
    
    fig.suptitle('Frequency Coverage Comparison & Model Rankings Across PDEs', 
                 fontsize=14, fontweight='bold')
    
    save_path = freq_dir / "cross_pde_frequency_coverage.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Cross-PDE frequency coverage comparison saved")


def generate_complexity_frequency_table(pde_data: Dict[str, Dict], output_dir: Path):
    """Generate PDE complexity vs frequency metrics table."""
    freq_dir = output_dir / "frequency_analysis"
    freq_dir.mkdir(parents=True, exist_ok=True)
    
    pde_names = sorted(pde_data.keys())
    rows = []
    
    for pde_name in pde_names:
        # Get PDE stats
        stats = extract_pde_stats(pde_name, pde_data[pde_name]['config'])
        
        # Get frequency data
        freq_data = get_frequency_data_for_pde(pde_data[pde_name])
        
        # Calculate GT high-freq percentage (power in upper 50% of frequencies)
        gt_high_freq_pct = 0
        if freq_data['gt_radial_power'] is not None and len(freq_data['gt_radial_power']) > 0:
            gt_power = freq_data['gt_radial_power']
            mid_idx = len(gt_power) // 2
            high_freq_power = gt_power[mid_idx:].sum()
            total_power = gt_power.sum()
            if total_power > 0:
                gt_high_freq_pct = high_freq_power / total_power * 100
        
        # Calculate mean error at high frequencies
        mean_high_freq_error = 0
        best_leftover = 1.0
        if freq_data['model_errors']:
            all_errors = list(freq_data['model_errors'].values())
            mean_error = np.mean(all_errors, axis=0)
            mid_idx = len(mean_error) // 2
            mean_high_freq_error = np.mean(mean_error[mid_idx:])
            
            # Get best model's leftover ratio
            models = pde_data[pde_name]['models']
            for model_name, model_data in models.items():
                freq = model_data.get('frequency_metrics')
                if freq:
                    leftover = freq.get('final_layer_leftover_ratio', 1.0)
                    best_leftover = min(best_leftover, leftover)
        
        rows.append({
            'pde': pde_name.capitalize(),
            'deriv_order': stats['deriv_order'],
            'nonlin_rank': stats['nonlin_rank'],
            'gt_high_freq': f"{gt_high_freq_pct:.1f}%",
            'mean_high_freq_err': f"{mean_high_freq_error:.4f}",
            'best_leftover': f"{best_leftover:.4f}"
        })
    
    df = pd.DataFrame(rows)
    
    # Save CSV
    df.to_csv(freq_dir / "complexity_frequency_table.csv", index=False)
    
    # Create table image
    columns = ['pde', 'deriv_order', 'nonlin_rank', 'gt_high_freq', 'mean_high_freq_err', 'best_leftover']
    col_labels = ['PDE', 'Deriv Order', 'Nonlin Rank', 'GT High-Freq %', 'Mean High-Freq Error', 'Best Leftover']
    
    # Simple table image
    fig, ax = plt.subplots(figsize=(14, 3 + len(rows) * 0.5))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = df[columns].values.tolist()
    
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    for i in range(len(col_labels)):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')
    
    plt.title('PDE Complexity vs Frequency Metrics', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(freq_dir / "complexity_frequency_table.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Complexity-frequency table saved")


def get_best_model_for_pde(pde_data: Dict, metric: str = 'eval_rel_l2') -> Optional[Dict]:
    """Get the best model for a PDE by a given metric.
    
    Args:
        pde_data: Dict with 'models' key containing model data
        metric: 'eval_rel_l2' or 'eval_linf'
    
    Returns:
        Best model data dict, or None if no valid model found
    """
    models = pde_data['models']
    best_model = None
    best_val = float('inf')
    
    for model_name, model_data in models.items():
        val = model_data.get(metric)
        if val is not None and val < best_val:
            best_val = val
            best_model = model_data
    
    return best_model


def generate_best_models_frequency_comparison(pde_data: Dict[str, Dict], output_dir: Path):
    """Generate frequency comparison figure for best models across PDEs.
    
    Creates a figure with:
    - Top left: GT frequency power spectra for all PDEs (overlaid, different colors)
    - Top right: Same GT spectra (duplicated for side-by-side layout)
    - Middle left: Error spectrum of best model by Rel-L2 for each PDE
    - Middle right: Error spectrum of best model by L-inf for each PDE
    - Bottom: Stats table with model info (layers, weights, PDE derivative/linearity order)
    """
    freq_dir = output_dir / "frequency_analysis"
    freq_dir.mkdir(parents=True, exist_ok=True)
    
    pde_names = sorted(pde_data.keys())
    
    # Collect data for each PDE
    pde_freq_data = {}
    best_models_l2 = {}
    best_models_linf = {}
    
    for pde_name in pde_names:
        # Get frequency data
        freq_data = get_frequency_data_for_pde(pde_data[pde_name])
        if freq_data['gt_radial_power'] is None or len(freq_data['gt_radial_power']) == 0:
            continue
        
        pde_freq_data[pde_name] = freq_data
        
        # Get best models
        best_l2 = get_best_model_for_pde(pde_data[pde_name], 'eval_rel_l2')
        best_linf = get_best_model_for_pde(pde_data[pde_name], 'eval_linf')
        
        if best_l2:
            best_models_l2[pde_name] = best_l2
        if best_linf:
            best_models_linf[pde_name] = best_linf
    
    if not pde_freq_data:
        print("  No frequency data found for best models comparison")
        return
    
    # Create figure with 3 rows: GT spectra, Error spectra (L2 vs Linf), Stats table
    fig = plt.figure(figsize=(16, 14))
    
    # Use GridSpec for flexible layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.2, 0.6], hspace=0.3, wspace=0.15)
    
    # Top row: GT frequency power spectra (same plot on both sides for symmetry)
    ax_gt_l2 = fig.add_subplot(gs[0, 0])
    ax_gt_linf = fig.add_subplot(gs[0, 1])
    
    # Middle row: Error spectra for best models
    ax_err_l2 = fig.add_subplot(gs[1, 0])
    ax_err_linf = fig.add_subplot(gs[1, 1])
    
    # Bottom row: Stats table (spans both columns)
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')
    
    # ----- Plot GT spectra (top row) -----
    for pde_name in pde_names:
        if pde_name not in pde_freq_data:
            continue
        
        freq_data = pde_freq_data[pde_name]
        k_bins = freq_data['k_radial_bins']
        gt_power = freq_data['gt_radial_power']
        color = PDE_COLORS.get(pde_name, '#333333')
        
        # Normalize for comparison
        gt_normalized = gt_power / (gt_power.sum() + 1e-10)
        
        # Plot on both GT axes
        ax_gt_l2.plot(k_bins, gt_normalized, color=color, linewidth=2, 
                     label=pde_name.capitalize(), alpha=0.8)
        ax_gt_linf.plot(k_bins, gt_normalized, color=color, linewidth=2, 
                       label=pde_name.capitalize(), alpha=0.8)
    
    # Configure GT axes
    for ax, title in [(ax_gt_l2, 'Ground Truth Frequency Content\n(Best by Rel-L2)'),
                      (ax_gt_linf, 'Ground Truth Frequency Content\n(Best by L-inf)')]:
        ax.set_xlabel('Radial Frequency |k| (Hz)', fontsize=10)
        ax.set_ylabel('Normalized Power', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    # ----- Plot error spectra for best models (middle row) -----
    for pde_name in pde_names:
        if pde_name not in pde_freq_data:
            continue
        
        freq_data = pde_freq_data[pde_name]
        k_bins = freq_data['k_radial_bins']
        color = PDE_COLORS.get(pde_name, '#333333')
        
        # Best by Rel-L2
        if pde_name in best_models_l2:
            best_model = best_models_l2[pde_name]
            model_name = best_model['model_name']
            if model_name in freq_data['model_errors']:
                error = freq_data['model_errors'][model_name]
                ax_err_l2.plot(k_bins, error, color=color, linewidth=2,
                              label=f"{pde_name.capitalize()}", alpha=0.8)
        
        # Best by L-inf
        if pde_name in best_models_linf:
            best_model = best_models_linf[pde_name]
            model_name = best_model['model_name']
            if model_name in freq_data['model_errors']:
                error = freq_data['model_errors'][model_name]
                ax_err_linf.plot(k_bins, error, color=color, linewidth=2,
                                label=f"{pde_name.capitalize()}", alpha=0.8)
    
    # Configure error axes
    for ax, title in [(ax_err_l2, 'Error Spectrum of Best Model (by Rel-L2)'),
                      (ax_err_linf, 'Error Spectrum of Best Model (by L-inf)')]:
        ax.set_xlabel('Radial Frequency |k| (Hz)', fontsize=10)
        ax.set_ylabel('Relative Error (Error/GT)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    # ----- Create stats table (bottom) -----
    table_data = []
    
    for pde_name in pde_names:
        if pde_name not in pde_freq_data:
            continue
        
        # Get PDE stats
        stats = extract_pde_stats(pde_name, pde_data[pde_name]['config'])
        
        # Get best model by Rel-L2
        best_l2 = best_models_l2.get(pde_name)
        if best_l2:
            l2_layers = best_l2['num_layers']
            l2_weights = best_l2.get('weight_label') or f"{best_l2['num_parameters']:,}"
            l2_error = f"{best_l2.get('eval_rel_l2', 0):.6f}"
            # Get peak error frequency
            model_name_l2 = best_l2['model_name']
            freq_data = pde_freq_data.get(pde_name, {})
            if model_name_l2 in freq_data.get('model_errors', {}):
                error_spectrum = freq_data['model_errors'][model_name_l2]
                k_bins = freq_data.get('k_radial_bins')
                if k_bins is not None and len(error_spectrum) > 0:
                    peak_idx = np.argmax(error_spectrum)
                    l2_peak_freq = f"{k_bins[peak_idx]:.1f}"
                else:
                    l2_peak_freq = 'N/A'
            else:
                l2_peak_freq = 'N/A'
        else:
            l2_layers = 'N/A'
            l2_weights = 'N/A'
            l2_error = 'N/A'
            l2_peak_freq = 'N/A'
        
        # Get best model by L-inf
        best_linf = best_models_linf.get(pde_name)
        if best_linf:
            linf_layers = best_linf['num_layers']
            linf_weights = best_linf.get('weight_label') or f"{best_linf['num_parameters']:,}"
            linf_error = f"{best_linf.get('eval_linf', 0):.6f}"
            # Get peak error frequency
            model_name_linf = best_linf['model_name']
            freq_data = pde_freq_data.get(pde_name, {})
            if model_name_linf in freq_data.get('model_errors', {}):
                error_spectrum = freq_data['model_errors'][model_name_linf]
                k_bins = freq_data.get('k_radial_bins')
                if k_bins is not None and len(error_spectrum) > 0:
                    peak_idx = np.argmax(error_spectrum)
                    linf_peak_freq = f"{k_bins[peak_idx]:.1f}"
                else:
                    linf_peak_freq = 'N/A'
            else:
                linf_peak_freq = 'N/A'
        else:
            linf_layers = 'N/A'
            linf_weights = 'N/A'
            linf_error = 'N/A'
            linf_peak_freq = 'N/A'

        table_data.append([
            pde_name.capitalize(),
            stats['deriv_order'],
            stats['nonlin_rank'],
            l2_layers, l2_weights, l2_error, l2_peak_freq,
            linf_layers, linf_weights, linf_error, linf_peak_freq
        ])

    col_labels = [
        'PDE', 'Deriv\nOrder', 'Nonlin\nRank',
        'L2 Best\nLayers', 'L2 Best\nWeights', 'Rel-L2\nError', 'L2 Peak\nFreq',
        'Linf Best\nLayers', 'Linf Best\nWeights', 'L-inf\nError', 'Linf Peak\nFreq'
    ]

    if table_data:
        table = ax_table.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc='center',
            loc='center',
            bbox=[0.02, 0.1, 0.96, 0.85]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)
        
        # Style header
        for i in range(len(col_labels)):
            cell = table[(0, i)]
            cell.set_facecolor('#34495e')
            cell.set_text_props(weight='bold', color='white')
        
        # Color PDE cells by their color
        for row_idx, pde_name in enumerate(pde_names):
            if pde_name in pde_freq_data:
                color = PDE_COLORS.get(pde_name, '#333333')
                cell = table[(row_idx + 1, 0)]
                cell.set_facecolor(color)
                cell.set_text_props(color='white', weight='bold')
    
    plt.suptitle('Best Models Frequency Analysis Across PDEs', fontsize=14, fontweight='bold', y=0.98)
    
    save_path = freq_dir / "best_models_frequency_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Best models frequency comparison saved to {save_path}")


def generate_best_model_learned_frequencies(pde_data: Dict[str, Dict], output_dir: Path):
    """Generate learned frequencies plot for best models across PDEs.
    
    Creates a 3-row figure (similar to training's learned_frequencies.png):
    - Row 1: GT frequency power spectrum (one column per PDE)
    - Row 2: Error at each layer (one column per PDE, log scale)
    - Row 3: Error ratio Eᵢ / Eᵢ₋₁ (one column per PDE, layer 2+ only, y=1 reference line)
    """
    freq_dir = output_dir / "frequency_analysis"
    freq_dir.mkdir(parents=True, exist_ok=True)
    
    pde_names = sorted(pde_data.keys())
    n_pdes = len(pde_names)
    
    if n_pdes == 0:
        return
    
    # Collect best models data
    best_models_data = {}
    
    for pde_name in pde_names:
        # Get best model by Rel-L2
        best_model = get_best_model_for_pde(pde_data[pde_name], 'eval_rel_l2')
        if not best_model:
            continue
        
        model_name = best_model['model_name']
        freq_metrics = best_model.get('frequency_metrics')
        
        if freq_metrics is None:
            continue
        
        spectral = freq_metrics.get('spectral_efficiency', {})
        k_bins = np.array(spectral.get('k_radial_bins', []))
        gt_power = np.array(spectral.get('gt_radial_power', []))
        error_matrix = np.array(spectral.get('error_matrix', []))
        
        if len(k_bins) == 0 or len(error_matrix) == 0:
            continue
        
        best_models_data[pde_name] = {
            'model_name': model_name,
            'k_bins': k_bins,
            'gt_power': gt_power,
            'error_matrix': error_matrix,
            'n_layers': len(error_matrix)
        }
    
    if not best_models_data:
        print("  No frequency data found for learned frequencies plot")
        return
    
    valid_pdes = [p for p in pde_names if p in best_models_data]
    n_valid = len(valid_pdes)
    
    # Create figure: 3 rows x n_pdes columns
    fig, axes = plt.subplots(3, n_valid, figsize=(5 * n_valid, 12), squeeze=False)
    
    # Color palette for layers
    layer_cmap = plt.cm.viridis
    
    for col_idx, pde_name in enumerate(valid_pdes):
        data = best_models_data[pde_name]
        k_bins = data['k_bins']
        gt_power = data['gt_power']
        error_matrix = data['error_matrix']
        n_layers = data['n_layers']
        pde_color = PDE_COLORS.get(pde_name, '#333333')
        
        # Layer colors
        layer_colors = [layer_cmap(i / max(n_layers - 1, 1)) for i in range(n_layers)]
        
        # ----- Row 1: GT Power Spectrum -----
        ax_gt = axes[0, col_idx]
        ax_gt.fill_between(k_bins, 0, gt_power, alpha=0.4, color=pde_color)
        ax_gt.plot(k_bins, gt_power, color=pde_color, linewidth=2, label='GT Power')
        ax_gt.set_xlabel('|k| (Hz)', fontsize=10)
        if col_idx == 0:
            ax_gt.set_ylabel('Power |FFT|²', fontsize=10)
        if np.all(gt_power[gt_power > 0] > 0):
            ax_gt.set_yscale('log')
        ax_gt.set_title(f'{pde_name.capitalize()}\nGT Frequency Content', 
                       fontsize=11, fontweight='bold')
        ax_gt.grid(True, alpha=0.3)
        ax_gt.legend(loc='upper right', fontsize=8)
        
        # ----- Row 2: Error at each layer -----
        ax_err = axes[1, col_idx]
        for layer_idx in range(n_layers):
            layer_error = error_matrix[layer_idx]
            ax_err.plot(k_bins, layer_error, color=layer_colors[layer_idx],
                       linewidth=2, alpha=0.8, label=f'Layer {layer_idx + 1}')
        
        ax_err.set_xlabel('|k| (Hz)', fontsize=10)
        if col_idx == 0:
            ax_err.set_ylabel('Relative Error', fontsize=10)
        ax_err.set_yscale('log')
        ax_err.set_title('Error at Layer i\n(Lower = Better) [log]', 
                        fontsize=10, fontweight='bold')
        ax_err.grid(True, alpha=0.3)
        ax_err.legend(loc='upper right', fontsize=7, ncol=1)
        
        # ----- Row 3: Error ratio Eᵢ / Eᵢ₋₁ (skip layer 1) -----
        ax_ratio = axes[2, col_idx]
        
        # Reference line at y=1
        ax_ratio.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, 
                        alpha=0.7, label='No change (ratio = 1)')
        
        all_ratios = []
        eps = 1e-10
        
        # Start from layer 2 (index 1) - skip layer 1 since nothing to compare
        for layer_idx in range(1, n_layers):
            prev_error = error_matrix[layer_idx - 1]
            curr_error = error_matrix[layer_idx]
            
            # Error ratio: < 1 = improvement, > 1 = degradation
            error_ratio = curr_error / (prev_error + eps)
            all_ratios.append(error_ratio)
            
            ax_ratio.plot(k_bins, error_ratio, color=layer_colors[layer_idx],
                         linewidth=2, alpha=0.8, label=f'Layer {layer_idx + 1}')
        
        ax_ratio.set_xlabel('|k| (Hz)', fontsize=10)
        if col_idx == 0:
            ax_ratio.set_ylabel('Error Ratio (Eᵢ / Eᵢ₋₁)', fontsize=10)
        
        # Set y-limits symmetric around 1
        if all_ratios:
            all_flat = np.concatenate([r.flatten() for r in all_ratios])
            valid_values = all_flat[np.isfinite(all_flat) & (all_flat > 0)]
            if len(valid_values) > 0:
                min_val = max(valid_values.min(), 0.01)
                max_val = min(valid_values.max(), 100)
                margin = max(1.0 - min_val, max_val - 1.0) * 1.2
                ax_ratio.set_ylim(max(0.01, 1.0 - margin), 1.0 + margin)
        
        # Add shading for improvement (< 1) and degradation (> 1) regions
        ylim = ax_ratio.get_ylim()
        ax_ratio.axhspan(ylim[0], 1.0, alpha=0.1, color='green')
        ax_ratio.axhspan(1.0, ylim[1], alpha=0.1, color='red')
        
        ax_ratio.set_title('Error Ratio: Eᵢ / Eᵢ₋₁\n(<1: Improved, >1: Degraded)', 
                          fontsize=10, fontweight='bold')
        ax_ratio.grid(True, alpha=0.3)
        ax_ratio.legend(loc='upper right', fontsize=7, ncol=1)
    
    fig.suptitle('Frequency Learning Analysis - Best Models (by Rel-L2)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    save_path = freq_dir / "best_model_learned_frequencies.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Best model learned frequencies saved to {save_path}")


def generate_cross_pde_frequency_analysis(pde_data: Dict[str, Dict], output_dir: Path):
    """Generate all cross-PDE frequency analysis plots."""
    print("\nSection 5: Generating frequency domain analysis...")
    
    freq_dir = output_dir / "frequency_analysis"
    freq_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 0: Best models frequency comparison
    generate_best_models_frequency_comparison(pde_data, output_dir)
    
    # Plot 1: Best model learned frequencies (GT, Error per layer, Error ratio)
    generate_best_model_learned_frequencies(pde_data, output_dir)
    
    # Plot 2: Learning progression comparison
    generate_learning_progression_comparison(pde_data, output_dir)
    
    # Plot 3: Per-metric aggregated frequency reduction across PDEs
    generate_per_metric_aggregated_reduction(pde_data, output_dir)
    
    # Plot 4: Cross-PDE frequency coverage with rank table
    generate_cross_pde_frequency_coverage(pde_data, output_dir)
    
    print(f"  Frequency analysis saved to {freq_dir}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def get_next_analysis_index(analysis_base_dir: Path) -> str:
    """Get the next running index for analysis output."""
    if not analysis_base_dir.exists():
        return "analysis_1"
    
    existing_indices = []
    for item in analysis_base_dir.iterdir():
        if item.is_dir() and item.name.startswith("analysis_"):
            try:
                idx = int(item.name.split("_")[1])
                existing_indices.append(idx)
            except (ValueError, IndexError):
                pass
    
    if not existing_indices:
        return "analysis_1"
    
    return f"analysis_{max(existing_indices) + 1}"


def main(experiment_paths: List[str]):
    """Main analysis function."""
    
    print("=" * 70)
    print("Cross-PDE Capacity Experiment Analysis")
    print("=" * 70)
    print(f"Analyzing {len(experiment_paths)} experiment folders")
    print()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    # Get output directory
    analysis_base_dir = Path(__file__).parent.parent / "analysis" / "cross_pde_comparison"
    index = get_next_analysis_index(analysis_base_dir)
    output_dir = analysis_base_dir / index
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print()
    
    # Load data from all experiments
    pde_data = {}
    
    for path in experiment_paths:
        path = Path(path)
        if not path.exists():
            print(f"  Warning: Path does not exist: {path}")
            continue
        
        pde_name = extract_pde_from_path(path)
        print(f"\n{'='*60}")
        print(f"Loading {pde_name} from {path.name}...")
        print(f"{'='*60}")
        
        config = load_experiment_config(path)
        exp_plan = load_experiment_plan(path)
        models = load_all_model_metrics(path, exp_plan, device)
        violations = detect_non_monotonic_metrics(models)
        
        pde_data[pde_name] = {
            'models': models,
            'violations': violations,
            'config': config,
            'path': path
        }
        
        print(f"\n  Loaded {len(models)} models, {sum(len(v) for v in violations.values())} violations")
    
    print()
    
    if not pde_data:
        print("No valid experiments found!")
        return
    
    # Section 1: PDE Statistics Table
    print("Section 1: Generating PDE statistics table...")
    generate_pde_stats_table(pde_data, output_dir)
    print()
    
    # Section 2: Best Model Summary
    print("Section 2: Generating best model summary...")
    generate_best_models_summary(pde_data, output_dir)
    print()
    
    # Section 3: Rank Comparisons
    print("Section 3: Generating rank comparisons...")
    generate_rank_comparisons(pde_data, output_dir)
    print()
    
    # Section 4: Non-Monotonic Analysis
    print("Section 4: Generating non-monotonic analysis plots...")
    generate_cross_pde_violation_plots(pde_data, output_dir)
    print()
    
    # Section 5: Frequency Domain Analysis
    generate_cross_pde_frequency_analysis(pde_data, output_dir)
    print()
    
    print("=" * 70)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_capacity_experiments_across_PDEs.py <exp_path1> <exp_path2> ...")
        print("Example: python analyze_capacity_experiments_across_PDEs.py outputs/experiments/Schrodinger_Capacity_* outputs/experiments/Wave1D_Capacity_*")
        sys.exit(1)
    
    experiment_paths = sys.argv[1:]
    main(experiment_paths)

