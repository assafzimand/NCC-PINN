"""Test dataset generation and loading."""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.io import load_config
from utils.dataset_gen import generate_and_save_datasets, load_dataset


def test_dataset_generation_problem1():
    """Test dataset generation for problem1 with small counts."""
    print("Testing dataset generation for problem1...")
    
    # Load config and override with small test values
    config = load_config()
    config['problem'] = 'problem1'
    
    # Use tiny counts for fast testing
    n_residual = 50
    n_ic = 10
    n_bc = 10
    N_total = n_residual + n_ic + n_bc
    
    config['n_residual_train'] = n_residual
    config['n_initial_train'] = n_ic
    config['n_boundary_train'] = n_bc
    config['n_residual_eval'] = 20
    config['n_initial_eval'] = 5
    config['n_boundary_eval'] = 5
    
    device = torch.device('cuda' if config['cuda'] and torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    # Import solver
    from solvers.problem1_solver import generate_dataset
    
    # Generate dataset
    data = generate_dataset(n_residual, n_ic, n_bc, device, config)
    
    # Check keys
    assert 'x' in data, "Missing 'x' key"
    assert 't' in data, "Missing 't' key"
    assert 'u_gt' in data, "Missing 'u_gt' key"
    assert 'mask' in data, "Missing 'mask' key"
    
    # Check shapes
    problem_config = config[config['problem']]
    spatial_dim = problem_config['spatial_dim']
    assert data['x'].shape == (N_total, spatial_dim), \
        f"x shape should be ({N_total}, {spatial_dim}), got {data['x'].shape}"
    assert data['t'].shape == (N_total, 1), \
        f"t shape should be ({N_total}, 1), got {data['t'].shape}"
    assert data['u_gt'].shape[0] == N_total, \
        f"u_gt should have {N_total} samples, got {data['u_gt'].shape[0]}"
    assert data['u_gt'].shape[1] == 2, \
        f"u_gt should have 2 outputs, got {data['u_gt'].shape[1]}"
    
    # Check masks
    assert 'residual' in data['mask'], "Missing 'residual' mask"
    assert 'IC' in data['mask'], "Missing 'IC' mask"
    assert 'BC' in data['mask'], "Missing 'BC' mask"
    
    assert data['mask']['residual'].sum() == n_residual, \
        f"Residual mask should sum to {n_residual}, got {data['mask']['residual'].sum()}"
    assert data['mask']['IC'].sum() == n_ic, \
        f"IC mask should sum to {n_ic}, got {data['mask']['IC'].sum()}"
    assert data['mask']['BC'].sum() == n_bc, \
        f"BC mask should sum to {n_bc}, got {data['mask']['BC'].sum()}"
    
    # Check device
    assert data['x'].device.type == device.type, f"x should be on {device}"
    assert data['t'].device.type == device.type, f"t should be on {device}"
    assert data['u_gt'].device.type == device.type, f"u_gt should be on {device}"
    
    # Check IC constraint (t should be at t_min for IC points)
    t_min = problem_config['temporal_domain'][0]
    ic_mask = data['mask']['IC']
    assert torch.allclose(data['t'][ic_mask], torch.tensor(t_min, device=device)), \
        "IC points should have t = t_min"
    
    # Check BC constraint for 1D (x should be at boundaries)
    if spatial_dim == 1:
        bc_mask = data['mask']['BC']
        x_bc = data['x'][bc_mask, 0]
        x_min, x_max = problem_config['spatial_domain'][0]
        # All BC points should be at either x_min or x_max
        at_boundaries = (torch.abs(x_bc - x_min) < 1e-6) | (torch.abs(x_bc - x_max) < 1e-6)
        assert at_boundaries.all(), "BC points should be at domain boundaries"
    
    print("  ✓ Dataset generation successful")
    print(f"  ✓ Shapes: x{data['x'].shape}, t{data['t'].shape}, u_gt{data['u_gt'].shape}")
    print(f"  ✓ Mask sums: residual={n_residual}, IC={n_ic}, BC={n_bc}")


def test_dataset_save_and_load():
    """Test saving and loading datasets."""
    print("\nTesting dataset save and load...")
    
    # Clean up test datasets if they exist
    test_paths = [
        Path("datasets/problem1/training_data.pt"),
        Path("datasets/problem1/eval_data.pt")
    ]
    for p in test_paths:
        if p.exists():
            p.unlink()
            print(f"  Cleaned up existing {p}")
    
    # Load config with tiny counts
    config = load_config()
    config['problem'] = 'problem1'
    config['n_residual_train'] = 50
    config['n_initial_train'] = 10
    config['n_boundary_train'] = 10
    config['n_residual_eval'] = 20
    config['n_initial_eval'] = 5
    config['n_boundary_eval'] = 5
    
    # Generate and save
    generate_and_save_datasets(config)
    
    # Check files exist
    assert test_paths[0].exists(), "Training data file should exist"
    assert test_paths[1].exists(), "Eval data file should exist"
    
    # Load and verify
    device = torch.device('cuda' if config['cuda'] and torch.cuda.is_available() else 'cpu')
    train_data = load_dataset(str(test_paths[0]), device=device)
    eval_data = load_dataset(str(test_paths[1]), device=device)
    
    # Verify training data
    N_train = config['n_residual_train'] + config['n_initial_train'] + config['n_boundary_train']
    assert train_data['x'].shape[0] == N_train, f"Train data should have {N_train} points"
    
    # Verify eval data
    N_eval = config['n_residual_eval'] + config['n_initial_eval'] + config['n_boundary_eval']
    assert eval_data['x'].shape[0] == N_eval, f"Eval data should have {N_eval} points"
    
    print("  ✓ Datasets saved and loaded successfully")
    print(f"  ✓ Training size: {N_train}, Eval size: {N_eval}")


def test_problem2_solver():
    """Test that problem2 solver works similarly."""
    print("\nTesting problem2 solver...")
    
    config = load_config()
    config['problem'] = 'problem2'
    
    n_residual, n_ic, n_bc = 30, 10, 10
    device = torch.device('cuda' if config['cuda'] and torch.cuda.is_available() else 'cpu')
    
    from solvers.problem2_solver import generate_dataset
    
    data = generate_dataset(n_residual, n_ic, n_bc, device, config)
    
    # Basic checks
    N_total = n_residual + n_ic + n_bc
    assert data['x'].shape[0] == N_total, "Correct total number of points"
    assert data['mask']['residual'].sum() == n_residual, "Correct residual count"
    assert data['mask']['IC'].sum() == n_ic, "Correct IC count"
    assert data['mask']['BC'].sum() == n_bc, "Correct BC count"
    
    print("  ✓ Problem2 solver works correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 2 — Placeholder solvers + dataset generation — Tests")
    print("=" * 60)
    
    try:
        test_dataset_generation_problem1()
        test_dataset_save_and_load()
        test_problem2_solver()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

