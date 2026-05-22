"""Smart initialization for PINN models.

Three strategies (all configurable, architecture-agnostic):
  hidden: glorot  — Glorot uniform + zero bias for all hidden linear layers.
                    Also zero-inits fc2 in ResNet blocks for identity-like start.
  output: zero    — Zero-initialize the output linear layer.
  output: ls      — Least-squares fit of output layer to IC data (base model only).

Reference: Wang et al. (2024) "PirateNets: Physics-informed Deep Learning with
Residual Adaptive Networks." JMLR.
"""

import torch
import torch.nn as nn
from typing import Dict


def _get_output_layer(model: nn.Module) -> nn.Linear:
    """Return the final output nn.Linear for any supported architecture."""
    if hasattr(model, 'output_proj'):
        return model.output_proj          # ResNetModel or PirateNet
    names = model.get_layer_names()
    return model.network[names[-1]]       # FCNet


def apply_hidden_init(model: nn.Module, cfg: dict) -> None:
    """Apply Glorot uniform + zero bias to all hidden linear layers.

    When init.hidden == 'glorot':
    - All nn.Linear / RWFLinear except the output layer get xavier_uniform_ with
      the gain appropriate for the configured activation function.
    - Biases are set to zero.
    - For ResNet: fc2 in every ResBlock is additionally zero-initialized so each
      block starts as an identity map (x + F(x) ≈ x), matching PirateNet's alpha=0.
    """
    init_cfg = cfg.get('init', {})
    if init_cfg.get('hidden', 'default') != 'glorot':
        return

    try:
        from models.rwf_layer import RWFLinear
        linear_types = (nn.Linear, RWFLinear)
    except ImportError:
        linear_types = (nn.Linear,)

    activation = cfg.get('activation', 'tanh')
    gain = nn.init.calculate_gain(activation)
    out_layer = _get_output_layer(model)

    n_hidden = 0
    for module in model.modules():
        if isinstance(module, linear_types) and module is not out_layer:
            nn.init.xavier_uniform_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            n_hidden += 1

    # ResNet identity-like start: zero fc2 in every ResBlock after Glorot
    try:
        from models.resnet_model import ResBlock
        n_resblocks = 0
        for module in model.modules():
            if isinstance(module, ResBlock):
                nn.init.zeros_(module.fc2.weight)
                n_resblocks += 1
        if n_resblocks:
            print(f"  [Init] ResNet: zeroed fc2 in {n_resblocks} residual blocks (identity start)")
    except ImportError:
        pass

    print(f"  [Init] Glorot uniform (gain={gain:.4f}) applied to {n_hidden} hidden layers")


def apply_output_init(
    model: nn.Module,
    train_data: Dict[str, torch.Tensor],
    cfg: dict,
    device: torch.device,
) -> None:
    """Initialize the output layer of the BASE MODEL (only).

    Modes (init.output):
      'zero'    — zero weight and bias
      'ls'      — least-squares fit to IC data
      'default' — no-op (keep PyTorch default)

    Raises ValueError if there are too few IC points for a well-determined LS system.
    The ValueError is caught by the atexit emergency handler and saved to metrics.
    """
    init_cfg = cfg.get('init', {})
    output_mode = init_cfg.get('output', 'default')
    out_layer = _get_output_layer(model)

    if output_mode == 'zero':
        with torch.no_grad():
            nn.init.zeros_(out_layer.weight)
            if out_layer.bias is not None:
                nn.init.zeros_(out_layer.bias)
        print("  [Init] Output layer: zero-initialized")

    elif output_mode == 'ls':
        use_bias = init_cfg.get('ls_use_bias', True)

        mask_ic = train_data['mask']['IC']
        n_ic = int(mask_ic.sum().item())
        hidden_dim = out_layer.weight.shape[1]
        required = hidden_dim + (1 if use_bias else 0)

        if n_ic < required:
            raise ValueError(
                f"[Init] LS-init requires ≥{required} IC points "
                f"(hidden_dim={hidden_dim}, use_bias={use_bias}), got {n_ic}. "
                f"Increase sampling.initial_train_ratio or disable ls_init."
            )

        x_ic = train_data['x'][mask_ic].to(device)
        t_ic = train_data['t'][mask_ic].to(device)
        h_gt_ic = train_data['h_gt'][mask_ic].to(device)
        inputs = torch.cat([x_ic, t_ic], dim=1)

        model.eval()
        with torch.no_grad():
            _, features = model(inputs, return_activation=True)  # (N, hidden_dim)

        if use_bias:
            H = torch.cat(
                [features, torch.ones(n_ic, 1, device=device)], dim=1
            ).float()  # (N, hidden_dim+1)
        else:
            H = features.float()  # (N, hidden_dim)

        solution = torch.linalg.lstsq(H, h_gt_ic.float()).solution
        # solution shape: (hidden_dim[+1], output_dim)

        with torch.no_grad():
            if use_bias:
                out_layer.weight.copy_(solution[:-1].T)   # (output_dim, hidden_dim)
                if out_layer.bias is not None:
                    out_layer.bias.copy_(solution[-1])     # (output_dim,)
            else:
                out_layer.weight.copy_(solution.T)

        model.train()
        output_dim = out_layer.weight.shape[0]
        print(
            f"  [Init] Output layer: LS-init from {n_ic} IC points "
            f"(hidden_dim={hidden_dim}, output_dim={output_dim}, use_bias={use_bias})"
        )

    # output_mode == 'default' → no-op


def apply_expert_init(expert: nn.Module, cfg: dict) -> None:
    """Initialize a newly spawned expert network.

    Always applies:
    - Glorot hidden init (if init.hidden == 'glorot')
    - Zero output layer (residual learning: expert starts at u=0 contribution)

    LS-init is NEVER applied to experts (only the base model gets it).
    """
    apply_hidden_init(expert, cfg)

    out_layer = _get_output_layer(expert)
    with torch.no_grad():
        nn.init.zeros_(out_layer.weight)
        if out_layer.bias is not None:
            nn.init.zeros_(out_layer.bias)


def apply_parent_copy_init(expert: nn.Module, parent_model: nn.Module) -> None:
    """Copy hidden layer weights from parent_model into a newly spawned expert.

    The parent's hidden layers are in a trained, stable regime — their tanh activations
    are well-behaved and higher-order spatial derivatives are bounded. Copying them
    prevents the 4th-order derivative overflow (NaN in h_xxxx) that occurs when
    freshly Glorot-initialized random weights are differentiated 4 times via autograd.

    Output layer is NOT copied — it must be zeroed after this call so the expert
    contributes u_k=0 at spawn time (residual learning principle).

    Only layers with matching weight shapes are copied; mismatched layers are skipped.
    """
    try:
        from models.rwf_layer import RWFLinear
        linear_types = (nn.Linear, RWFLinear)
    except ImportError:
        linear_types = (nn.Linear,)

    out_layer_new = _get_output_layer(expert)
    n_copied = 0
    for mod_new, mod_par in zip(expert.modules(), parent_model.modules()):
        if (isinstance(mod_new, linear_types)
                and mod_new is not out_layer_new
                and mod_new.weight.shape == mod_par.weight.shape):
            mod_new.weight.data.copy_(mod_par.weight.data)
            if mod_new.bias is not None and mod_par.bias is not None:
                mod_new.bias.data.copy_(mod_par.bias.data)
            n_copied += 1

    out_layer_new = _get_output_layer(expert)
    with torch.no_grad():
        nn.init.zeros_(out_layer_new.weight)
        if out_layer_new.bias is not None:
            nn.init.zeros_(out_layer_new.bias)

    print(f"  [Init] Copied {n_copied} hidden layers from parent; output zeroed")
