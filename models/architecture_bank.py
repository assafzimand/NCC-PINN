"""Architecture bank for AToE adaptive expert sizing.

Maps approximate parameter capacity to hidden layer configurations.
Entries span 1k-20k parameters (reference: input_dim=2, output_dim=1)
with sub-monotonic growth in both width and depth (2-7 hidden layers).
Width may dip at depth transitions but is non-decreasing within each
depth group.
"""

from collections import OrderedDict
from typing import List


ARCHITECTURE_BANK = OrderedDict([
    # 2 hidden layers
    (1000,  [30, 30]),
    (1500,  [37, 37]),
    # 3 hidden layers
    (2000,  [30, 30, 30]),
    (2500,  [34, 34, 34]),
    (3000,  [37, 37, 37]),
    (3500,  [40, 40, 40]),
    # 4 hidden layers
    (4000,  [35, 35, 35, 35]),
    (4500,  [38, 38, 38, 38]),
    (5000,  [40, 40, 40, 40]),
    (5500,  [42, 42, 42, 42]),
    # 5 hidden layers
    (6000,  [38, 38, 38, 38, 38]),
    (6500,  [40, 40, 40, 40, 40]),
    (7000,  [41, 41, 41, 41, 41]),
    (7500,  [43, 43, 43, 43, 43]),
    (8000,  [44, 44, 44, 44, 44]),
    # 6 hidden layers
    (8500,  [40, 40, 40, 40, 40, 40]),
    (9000,  [42, 42, 42, 42, 42, 42]),
    (9500,  [43, 43, 43, 43, 43, 43]),
    (10000, [44, 44, 44, 44, 44, 44]),
    (10500, [45, 45, 45, 45, 45, 45]),
    (11000, [46, 46, 46, 46, 46, 46]),
    (11500, [47, 47, 47, 47, 47, 47]),
    (12000, [48, 48, 48, 48, 48, 48]),
    (12500, [49, 49, 49, 49, 49, 49]),
    (13000, [50, 50, 50, 50, 50, 50]),
    (13500, [51, 51, 51, 51, 51, 51]),
    (14000, [52, 52, 52, 52, 52, 52]),
    (14500, [53, 53, 53, 53, 53, 53]),
    (15000, [54, 54, 54, 54, 54, 54]),
    (15500, [55, 55, 55, 55, 55, 55]),
    (16000, [56, 56, 56, 56, 56, 56]),
    (16500, [57, 57, 57, 57, 57, 57]),
    (17000, [58, 58, 58, 58, 58, 58]),
    (17500, [59, 59, 59, 59, 59, 59]),
    # 7 hidden layers
    (18000, [54, 54, 54, 54, 54, 54, 54]),
    (18500, [55, 55, 55, 55, 55, 55, 55]),
    (19000, [56, 56, 56, 56, 56, 56, 56]),
    (19500, [57, 57, 57, 57, 57, 57, 57]),
    (20000, [57, 57, 57, 57, 57, 57, 57]),
])


def compute_param_count(
    hidden_layers: List[int], input_dim: int, output_dim: int
) -> int:
    """Compute total parameter count for a fully-connected network.

    Args:
        hidden_layers: List of hidden layer widths (no input/output dims).
        input_dim: Input dimension of the network.
        output_dim: Output dimension of the network.

    Returns:
        Total number of trainable parameters (weights + biases).
    """
    layers = [input_dim] + hidden_layers + [output_dim]
    total = 0
    for i in range(len(layers) - 1):
        total += layers[i] * layers[i + 1] + layers[i + 1]
    return total


def get_architecture_for_capacity(
    target_capacity: float, input_dim: int, output_dim: int
) -> List[int]:
    """Look up the bank entry whose actual param count is closest to target.

    Computes exact parameter counts using the provided input/output dims,
    then returns the full architecture [input_dim, *hidden, output_dim]
    for the best-matching entry. Capacities above the bank maximum return
    the largest entry.

    Args:
        target_capacity: Desired number of parameters.
        input_dim: Problem input dimension (spatial_dim + 1).
        output_dim: Problem output dimension.

    Returns:
        Full architecture list [input_dim, h1, h2, ..., output_dim].
    """
    best_hidden = None
    best_diff = float('inf')

    for _label, hidden_layers in ARCHITECTURE_BANK.items():
        actual = compute_param_count(hidden_layers, input_dim, output_dim)
        diff = abs(actual - target_capacity)
        if diff < best_diff:
            best_diff = diff
            best_hidden = hidden_layers

    return [input_dim] + list(best_hidden) + [output_dim]
