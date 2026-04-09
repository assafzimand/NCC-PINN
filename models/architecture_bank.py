"""Architecture bank for AToE adaptive expert sizing.

Maps approximate parameter capacity to hidden layer configurations.
Entries span 1k-20k parameters (reference: input_dim=2, output_dim=1)
with sub-monotonic growth in both width and depth (2-7 hidden layers).
Width may dip at depth transitions but is non-decreasing within each
depth group.

ResNet actual param counts for each bank entry (input_dim=2, output_dim=1).
Use this when selecting a fixed architecture for resnet experts:

  MLP label  ->  ResNet params    Architecture (hidden layers)
  ──────────────────────────────────────────────────────────────
#    1000  ->    1857    [29, 29]
#    1500  ->    2809    [36, 36]
#    2000  ->    2911    [30, 30, 30]
#    2500  ->    3707    [34, 34, 34]
#    3000  ->    4367    [37, 37, 37]
#    3500  ->    5081    [40, 40, 40]
#    4000  ->    5181    [35, 35, 35, 35]
#    4500  ->    6081    [38, 38, 38, 38]
#    5000  ->    6721    [40, 40, 40, 40]      ← ~6.7k resnet
#    5500  ->    7393    [42, 42, 42, 42]
#    6000  ->    7563    [38, 38, 38, 38, 38]
#    6500  ->    7957    [39, 39, 39, 39, 39]
#    7000  ->    8775    [41, 41, 41, 41, 41]
#    7500  ->    9199    [42, 42, 42, 42, 42]
#    8000  ->   10077    [44, 44, 44, 44, 44]
#    8500  ->   10001    [40, 40, 40, 40, 40, 40]
#    9000  ->   11005    [42, 42, 42, 42, 42, 42]
#    9500  ->   11525    [43, 43, 43, 43, 43, 43]
#   10000  ->   12057    [44, 44, 44, 44, 44, 44]
#   10500  ->   12601    [45, 45, 45, 45, 45, 45]
#   11000  ->   13157    [46, 46, 46, 46, 46, 46]
#   11500  ->   13725    [47, 47, 47, 47, 47, 47]
#   12000  ->   14305    [48, 48, 48, 48, 48, 48]
#   12500  ->   14897    [49, 49, 49, 49, 49, 49]
#   13000  ->   15501    [50, 50, 50, 50, 50, 50]
#   13500  ->   16117    [51, 51, 51, 51, 51, 51]
#   14000  ->   16745    [52, 52, 52, 52, 52, 52]
#   14500  ->   17385    [53, 53, 53, 53, 53, 53]
#   15000  ->   18037    [54, 54, 54, 54, 54, 54]
#   15500  ->   18701    [55, 55, 55, 55, 55, 55]
#   16000  ->   19377    [56, 56, 56, 56, 56, 56]
#   16500  ->   20065    [57, 57, 57, 57, 57, 57]
#   17000  ->   20765    [58, 58, 58, 58, 58, 58]
#   17500  ->   21477    [59, 59, 59, 59, 59, 59]
#   18000  ->   21007    [54, 54, 54, 54, 54, 54, 54]
#   18500  ->   21781    [55, 55, 55, 55, 55, 55, 55]
#   19000  ->   22569    [56, 56, 56, 56, 56, 56, 56]
#   19500  ->   22257    [52, 52, 52, 52, 52, 52, 52, 52]
#   20000  ->   23371    [57, 57, 57, 57, 57, 57, 57]
"""

from collections import OrderedDict
from typing import List


ARCHITECTURE_BANK = OrderedDict([
    # 2 hidden layers
    (1000,  [29, 29]),
    (1500,  [36, 36]),
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
    (6500,  [39, 39, 39, 39, 39]),
    (7000,  [41, 41, 41, 41, 41]),
    (7500,  [42, 42, 42, 42, 42]),
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
    (19500, [52, 52, 52, 52, 52, 52, 52, 52]),
    (20000, [57, 57, 57, 57, 57, 57, 57]),
])


def compute_param_count(
    hidden_layers: List[int], input_dim: int, output_dim: int,
    expert_type: str = 'mlp',
) -> int:
    """Compute total parameter count for a network.

    For 'mlp': standard fully-connected count (weights + biases per layer).
    For 'resnet': mirrors ResNetModel — input_proj + ResBlocks (pairs of
    hidden layers) + optional leftover plain layer + output_proj.
    All hidden widths must be identical for resnet.

    Args:
        hidden_layers: List of hidden layer widths (no input/output dims).
        input_dim: Input dimension of the network.
        output_dim: Output dimension of the network.
        expert_type: 'mlp' or 'resnet'.

    Returns:
        Total number of trainable parameters (weights + biases).
    """
    if expert_type == 'resnet':
        h = hidden_layers[0]
        n_hidden = len(hidden_layers)
        # input_proj: Linear(input_dim, h)
        total = input_dim * h + h
        # ResBlocks: each block has 2 × Linear(h, h)
        n_blocks = n_hidden // 2
        total += n_blocks * 2 * (h * h + h)
        # leftover plain layer: Linear(h, h) if odd number of hidden layers
        if n_hidden % 2 == 1:
            total += h * h + h
        # output_proj: Linear(h, output_dim)
        total += h * output_dim + output_dim
        return total

    # MLP
    layers = [input_dim] + list(hidden_layers) + [output_dim]
    total = 0
    for i in range(len(layers) - 1):
        total += layers[i] * layers[i + 1] + layers[i + 1]
    return total


def get_architecture_for_capacity(
    target_capacity: float, input_dim: int, output_dim: int,
    expert_type: str = 'mlp',
) -> List[int]:
    """Look up the bank entry whose actual param count is closest to target.

    Computes exact parameter counts (respecting expert_type) using the
    provided input/output dims, then returns the full architecture
    [input_dim, *hidden, output_dim] for the best-matching entry.
    Capacities above the bank maximum return the largest entry.

    Args:
        target_capacity: Desired number of parameters.
        input_dim: Problem input dimension (spatial_dim + 1).
        output_dim: Problem output dimension.
        expert_type: 'mlp' or 'resnet' — affects how params are counted.

    Returns:
        Full architecture list [input_dim, h1, h2, ..., output_dim].
    """
    best_hidden = None
    best_diff = float('inf')

    for _label, hidden_layers in ARCHITECTURE_BANK.items():
        actual = compute_param_count(
            hidden_layers, input_dim, output_dim, expert_type)
        diff = abs(actual - target_capacity)
        if diff < best_diff:
            best_diff = diff
            best_hidden = hidden_layers

    return [input_dim] + list(best_hidden) + [output_dim]
