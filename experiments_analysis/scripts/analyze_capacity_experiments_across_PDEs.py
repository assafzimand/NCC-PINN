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
    experiment_plan: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, Any]]:
    """Load all metrics from all models in experiment."""
    models_data = {}
    experiment_path = Path(experiment_path)
    
    arch_to_weight_label = {}
    if experiment_plan and 'arch_to_weight_label' in experiment_plan:
        arch_to_weight_label = experiment_plan['arch_to_weight_label']
    
    for model_dir in experiment_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        if model_name.startswith('.') or model_name in ['comparison_summary.csv', 'experiments_plan.yaml']:
            continue
        
        run_dirs = [d for d in model_dir.iterdir() if d.is_dir() and re.match(r'\d{8}_\d{6}', d.name)]
        if not run_dirs:
            continue
        
        run_dir = max(run_dirs, key=lambda x: x.stat().st_mtime)
        
        metrics_file = run_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
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
        
        architecture = parse_model_name(model_name)
        num_layers = len(architecture) - 2
        num_parameters = calculate_num_parameters(architecture)
        weight_label = arch_to_weight_label.get(tuple(architecture))
        problem_name = get_problem_from_model_name(model_name)
        
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
        print(f"Loading {pde_name} from {path.name}...")
        
        config = load_experiment_config(path)
        exp_plan = load_experiment_plan(path)
        models = load_all_model_metrics(path, exp_plan)
        violations = detect_non_monotonic_metrics(models)
        
        pde_data[pde_name] = {
            'models': models,
            'violations': violations,
            'config': config,
            'path': path
        }
        
        print(f"  Loaded {len(models)} models, {sum(len(v) for v in violations.values())} violations")
    
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

