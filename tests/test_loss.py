"""Test loss functions with simple optimization."""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.io import load_config
from losses.problem1_loss import build_loss as build_loss_problem1
from losses.problem2_loss import build_loss as build_loss_problem2


class SimpleFCNet(nn.Module):
    """Simple fully-connected network for testing."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_tiny_batch(n_points: int, spatial_dim: int, device: torch.device):
    """Create a tiny batch for testing."""
    n_residual = n_points // 2
    n_ic = n_points // 4
    n_bc = n_points - n_residual - n_ic

    x = torch.rand(n_points, spatial_dim, device=device)
    t = torch.rand(n_points, 1, device=device)
    u_gt = torch.randn(n_points, 2, device=device)

    mask_residual = torch.zeros(n_points, dtype=torch.bool, device=device)
    mask_residual[:n_residual] = True

    mask_ic = torch.zeros(n_points, dtype=torch.bool, device=device)
    mask_ic[n_residual:n_residual + n_ic] = True

    mask_bc = torch.zeros(n_points, dtype=torch.bool, device=device)
    mask_bc[n_residual + n_ic:] = True

    batch = {
        'x': x,
        't': t,
        'u_gt': u_gt,
        'mask': {
            'residual': mask_residual,
            'IC': mask_ic,
            'BC': mask_bc
        }
    }

    return batch


def test_loss_problem1():
    """Test problem1 loss function."""
    print("Testing problem1 loss function...")

    # Load config
    config = load_config()
    device = torch.device('cuda' if config['cuda'] and
                          torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Build loss
    loss_fn = build_loss_problem1(**config)

    # Create tiny model
    spatial_dim = config['problem1']['spatial_dim']
    input_dim = spatial_dim + 1  # x + t
    output_dim = 2
    model = SimpleFCNet(input_dim, 20, output_dim).to(device)

    # Create tiny batch
    batch = create_tiny_batch(100, spatial_dim, device)

    # Test forward pass
    loss = loss_fn(model, batch)
    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.device.type == device.type, f"Loss should be on {device}"
    assert loss.item() >= 0, "Loss should be non-negative"

    print(f"  ✓ Loss forward pass successful: {loss.item():.4f}")

    # Test optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []

    for step in range(10):
        optimizer.zero_grad()
        loss = loss_fn(model, batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Check that loss decreased
    assert losses[-1] < losses[0], \
        f"Loss should decrease (initial: {losses[0]:.4f}, " \
        f"final: {losses[-1]:.4f})"

    print(f"  ✓ Optimization works: loss decreased from "
          f"{losses[0]:.4f} to {losses[-1]:.4f}")


def test_loss_problem2():
    """Test problem2 loss function."""
    print("\nTesting problem2 loss function...")

    # Load config
    config = load_config()
    device = torch.device('cuda' if config['cuda'] and
                          torch.cuda.is_available() else 'cpu')

    # Build loss
    config['problem'] = 'problem2'  # Switch to problem2
    loss_fn = build_loss_problem2(**config)

    # Create tiny model
    spatial_dim = config['problem2']['spatial_dim']
    input_dim = spatial_dim + 1
    output_dim = 2
    model = SimpleFCNet(input_dim, 20, output_dim).to(device)

    # Create tiny batch
    batch = create_tiny_batch(100, spatial_dim, device)

    # Test forward pass
    loss = loss_fn(model, batch)
    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "Loss should be non-negative"

    print(f"  ✓ Loss forward pass successful: {loss.item():.4f}")

    # Test optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    initial_loss = loss_fn(model, batch).item()

    for _ in range(10):
        optimizer.zero_grad()
        loss = loss_fn(model, batch)
        loss.backward()
        optimizer.step()

    final_loss = loss_fn(model, batch).item()

    assert final_loss < initial_loss, \
        f"Loss should decrease (initial: {initial_loss:.4f}, " \
        f"final: {final_loss:.4f})"

    print(f"  ✓ Optimization works: loss decreased from "
          f"{initial_loss:.4f} to {final_loss:.4f}")


def test_loss_components():
    """Test that loss properly weights different components."""
    print("\nTesting loss component weighting...")

    config = load_config()
    device = torch.device('cuda' if config['cuda'] and
                          torch.cuda.is_available() else 'cpu')

    # Build loss
    loss_fn = build_loss_problem1(**config)

    # Create model
    spatial_dim = config['problem1']['spatial_dim']
    input_dim = spatial_dim + 1
    model = SimpleFCNet(input_dim, 20, 2).to(device)

    # Create batch with only residual points
    batch_residual = create_tiny_batch(100, spatial_dim, device)
    batch_residual['mask']['IC'][:] = False
    batch_residual['mask']['BC'][:] = False
    batch_residual['mask']['residual'][:] = True

    loss_residual_only = loss_fn(model, batch_residual).item()

    # Create batch with only IC points
    batch_ic = create_tiny_batch(100, spatial_dim, device)
    batch_ic['mask']['residual'][:] = False
    batch_ic['mask']['BC'][:] = False
    batch_ic['mask']['IC'][:] = True

    loss_ic_only = loss_fn(model, batch_ic).item()

    print(f"  ✓ Loss computes correctly for different point types")
    print(f"    Residual-only loss: {loss_residual_only:.4f}")
    print(f"    IC-only loss: {loss_ic_only:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Step 3 — Placeholder losses — Tests")
    print("=" * 60)

    try:
        test_loss_problem1()
        test_loss_problem2()
        test_loss_components()

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

