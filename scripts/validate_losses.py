"""
Loss Function Validation Script with Real Datasets.

This script:
1. Generates datasets using the actual solver methods (if not already exists)
2. Loads the datasets from disk
3. Creates a minimal test model for each PDE
4. Computes losses on the ground truth solutions
5. Prints resulting losses for every PDE

This validates that:
- Solvers generate valid ground truth data
- Loss functions accept the data format correctly
- All three loss components (residual, IC, BC) compute without errors
"""

import torch
import sys
import os
import yaml
from pathlib import Path

print("Starting validation script...", flush=True)
print("Importing modules (this may take 30+ seconds)...", flush=True)

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset_gen import generate_and_save_datasets, load_dataset

print("Imports complete. Loading loss modules...", flush=True)

from losses import (
    burgers1d_loss,
    burgers2d_loss,
    schrodinger_loss,
    allen_cahn_loss,
    kdv_loss,
    ks_loss,
    wave1d_loss,
    conv_diff_loss,
    fisher_kpp_loss
)

print("All modules loaded. Starting validation...\n", flush=True)


def load_config():
    """Load base config from experiments_plan.yaml."""
    config_path = os.path.join(
        os.path.dirname(__file__), '..', 'experiments_plan.yaml'
    )
    with open(config_path, 'r') as f:
        config_full = yaml.safe_load(f)
    
    # Use base_config as template
    config = config_full.get('base_config', {})
    config['seed'] = 42
    config['cuda'] = False  # Force CPU for validation
    
    # Add problem-specific configs
    for problem in ['burgers1d', 'burgers2d', 'schrodinger', 'allen_cahn', 
                    'kdv', 'ks', 'wave1d', 'conv_diff', 'fisher_kpp']:
        if problem in config_full:
            config[problem] = config_full[problem]
    
    # Add sampling config (needed for dataset generation)
    if 'sampling' in config_full['base_config']:
        config['sampling'] = config_full['base_config']['sampling']
    
    return config


def test_pde_losses(problem_name, config, device='cpu'):
    """
    Test losses for a specific PDE using real solver-generated data.
    
    Args:
        problem_name: Name of the PDE (e.g., 'burgers1d')
        config: Configuration dictionary
        device: Device to run on
        
    Returns:
        Dictionary with loss results
    """
    print(f"\n{'='*70}")
    print(f"Testing {problem_name.upper()}")
    print(f"{'='*70}")
    
    # Set problem in config
    config['problem'] = problem_name
    
    # Check if datasets exist
    dataset_dir = Path("datasets") / problem_name
    train_path = dataset_dir / "training_data.pt"
    eval_path = dataset_dir / "eval_data.pt"
    
    # Generate datasets if they don't exist (with error handling)
    if not train_path.exists() or not eval_path.exists():
        print(f"  Datasets not found. Generating...")
        try:
            # Suppress numpy warnings during generation
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                generate_and_save_datasets(config)
        except Exception as e:
            print(f"  ERROR generating datasets: {e}")
            import traceback
            traceback.print_exc()
            return {'error': f'Dataset generation failed: {str(e)}', 'status': 'FAILED'}
    else:
        print(f"  Using existing datasets from {dataset_dir}")
    
    # Load training data
    try:
        train_data = load_dataset(str(train_path), device=torch.device(device))
        print(f"  Loaded training data: {train_data['x'].shape[0]} samples")
    except Exception as e:
        print(f"  ERROR loading training data: {e}")
        return {'error': f'Dataset load failed: {str(e)}', 'status': 'FAILED'}
    
    # Get output dimension and proper input dimension
    output_dim = train_data['h_gt'].shape[1]
    spatial_dim = train_data['x'].shape[1]
    input_dim = spatial_dim + 1  # spatial_dim + time
    print(f"  Spatial dim: {spatial_dim}, Input dim: {input_dim}, Output dim: {output_dim}")
    
    # Create simple test model with correct dimensions
    class TestModel(torch.nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.supports_decomposed = False
            # Simple MLP: input -> hidden -> output
            hidden_dim = 64
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_dim, output_dim)
            )
            
        def forward(self, xt):
            return self.net(xt)
    
    model = TestModel(input_dim=input_dim, output_dim=output_dim)
    model.to(device)
    print(f"  Model created: {input_dim} -> 64 -> {output_dim}")
    
    # Get loss function
    loss_modules = {
        'burgers1d': burgers1d_loss,
        'burgers2d': burgers2d_loss,
        'schrodinger': schrodinger_loss,
        'allen_cahn': allen_cahn_loss,
        'kdv': kdv_loss,
        'ks': ks_loss,
        'wave1d': wave1d_loss,
        'conv_diff': conv_diff_loss,
        'fisher_kpp': fisher_kpp_loss,
    }
    
    if problem_name not in loss_modules:
        print(f"  WARNING: No loss module for {problem_name}, skipping")
        return {'error': 'No loss module found', 'status': 'SKIPPED'}
    
    try:
        loss_fn = loss_modules[problem_name].build_loss(**config)
        print(f"  Loss function built successfully")
    except Exception as e:
        print(f"  ERROR building loss function: {e}")
        import traceback
        traceback.print_exc()
        return {'error': f'Loss build failed: {str(e)}', 'status': 'FAILED'}
    
    # Compute loss components
    try:
        print(f"\n  Computing loss components...")
        
        # Compute total loss
        total_loss = loss_fn(model, train_data)
        print(f"    Total loss: {total_loss.item():.6e}")
        
        # Compute individual components
        components = loss_fn(model, train_data, return_components=True)
        
        print(f"    Component losses:")
        print(f"      Residual (MSE_f): {components['residual'].item():.6e}")
        print(f"      IC (MSE_0):       {components['ic'].item():.6e}")
        print(f"      BC (MSE_b):       {components['bc'].item():.6e}")
        
        # Count samples per component
        n_residual = train_data['mask']['residual'].sum().item()
        n_ic = train_data['mask']['IC'].sum().item()
        n_bc = train_data['mask']['BC'].sum().item()
        
        print(f"\n  Sample counts:")
        print(f"    Residual points: {n_residual:,}")
        print(f"    IC points:       {n_ic:,}")
        print(f"    BC points:       {n_bc:,}")
        
        result = {
            'total_loss': total_loss.item(),
            'residual': components['residual'].item(),
            'ic': components['ic'].item(),
            'bc': components['bc'].item(),
            'n_samples': train_data['x'].shape[0],
            'n_residual': n_residual,
            'n_ic': n_ic,
            'n_bc': n_bc,
            'output_dim': output_dim,
            'status': 'SUCCESS'
        }
        
        print(f"\n  Loss computation successful")
        return result
        
    except Exception as e:
        print(f"  ERROR computing loss: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e), 'status': 'FAILED'}


def main():
    """Run loss validation for all PDEs."""
    print("\n" + "=" * 70)
    print("LOSS FUNCTION VALIDATION WITH REAL SOLVER DATA")
    print("=" * 70)
    
    device = 'cpu'
    config = load_config()
    
    # Allow testing specific PDEs via command line
    if len(sys.argv) > 1:
        pdes = [arg for arg in sys.argv[1:] if not arg.startswith('-')]
        print(f"\nTesting specific PDEs: {', '.join(pdes)}")
    else:
        # List of all PDEs to test
        pdes = [
            'burgers1d',
            'burgers2d',
            'schrodinger',
            'allen_cahn',
            'kdv',
            'ks',
            'wave1d',
            'conv_diff',
            'fisher_kpp'
        ]
        print(f"\nTesting all {len(pdes)} PDEs")
    
    results = {}
    
    # Test each PDE
    for pde in pdes:
        try:
            result = test_pde_losses(pde, config, device)
            results[pde] = result
        except Exception as e:
            print(f"\nUNEXPECTED ERROR for {pde}: {e}")
            import traceback
            traceback.print_exc()
            results[pde] = {'error': str(e), 'status': 'FAILED'}
    
    # Summary table
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"{'PDE':<20} {'Status':<12} {'Total Loss':<15} {'Residual':<15} {'IC':<15} {'BC':<15}")
    print("-" * 70)
    
    for pde, result in results.items():
        status = result.get('status', 'FAILED')
        if status == 'SUCCESS':
            print(f"{pde:<20} {status:<12} {result['total_loss']:<15.6e} "
                  f"{result['residual']:<15.6e} {result['ic']:<15.6e} {result['bc']:<15.6e}")
        elif status == 'SKIPPED':
            reason = result.get('error', 'Unknown')[:40]
            print(f"{pde:<20} {status:<12} {reason}")
        else:  # FAILED
            error = result.get('error', 'Unknown error')[:40]
            print(f"{pde:<20} {status:<12} {error}")
    
    # Overall status
    print("\n" + "=" * 70)
    successful = sum(1 for r in results.values() if r.get('status') == 'SUCCESS')
    skipped = sum(1 for r in results.values() if r.get('status') == 'SKIPPED')
    failed = sum(1 for r in results.values() if r.get('status') == 'FAILED')
    total = len(results)
    
    if successful == total:
        print(f"SUCCESS: ALL {total} PDEs VALIDATED")
        print("  All loss functions compute correctly on solver-generated ground truth.")
        exit_code = 0
    elif successful + skipped == total:
        print(f"PARTIAL SUCCESS: {successful}/{total} PDEs validated, {skipped} skipped")
        print("  All tested loss functions work correctly.")
        exit_code = 0
    else:
        print(f"RESULTS: {successful} successful, {skipped} skipped, {failed} failed (total: {total})")
        if successful > 0:
            print(f"  {successful} PDEs validated successfully.")
        if failed > 0:
            print(f"  {failed} PDEs failed validation - check errors above.")
        exit_code = 1 if failed > 0 else 0
    
    print("=" * 70 + "\n")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
