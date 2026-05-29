# PAFA: Physics-informed Adaptive Framework with Adaptive domain decomposition

## Project Overview

PAFA is a Physics-Informed Neural Network (PINN) framework implementing adaptive domain decomposition via a tree-based mixture-of-experts architecture. The method allocates computational resources according to local function complexity by recursively partitioning the domain and spawning regional expert networks.

### Composition Modes

Three global composition formulations are supported via the `model` config key:

| Model | Forward Formula | Expert Participation |
|-------|-----------------|---------------------|
| **AToE** | `u = Σ ψ̃_k · u_k` (all experts) | Base + ALL experts contribute |
| **AToELeaves** | `u = Σ_{leaves} ψ̃_j · u_j` | Only leaf experts (parents retired at spawn) |
| **ANT** | Leaf outputs with parent activation propagation | Leaves only, hierarchical hidden state |

All use soft indicators (`BatchedIndicators`) with sigmoid-based partition of unity normalization.

### Supported PDEs

The framework includes benchmark implementations for:
- **schrodinger** - Nonlinear Schrödinger equation (complex-valued, output_dim=2)
- **burgers1d** - 1D Burgers equation with viscosity
- **allen_cahn** - Allen-Cahn reaction-diffusion
- **kdv** - Korteweg-de Vries equation
- **ks** - Kuramoto-Sivashinsky (4th-order, chaotic)

---

## Orchestration Flow

### Entry Points

```
run_experiments.py  →  Batch runner: loads experiments_plan.yaml, runs multiple experiments
run_ncc.py          →  Single run: trains model + optional NCC analysis
trainer/trainer.py  →  Core training loop
```

### Config Loading and Merging

1. `run_experiments.py` loads `experiments_plan.yaml`
2. For each experiment entry:
   - Deep-merges `base_config` + experiment overrides via `_deep_merge()`
   - Nested dicts are recursively merged (not replaced)
   - Writes merged config to `config/config.yaml`
   - Spawns `run_ncc.py` as subprocess
3. `run_ncc.py` calls `load_config()` from `utils/io.py`
4. Creates model via `create_network()` from `models/network_factory.py`
5. Builds loss via dynamic import: `losses.{problem}_loss.build_loss()`
6. Calls `trainer.train()` with model, loss_fn, data paths, config, run_dir

### Config Resolution

```python
problem = cfg['problem']           # e.g. 'ks'
problem_cfg = cfg[problem]         # nested dict with PDE params, loss_weights, etc.
```

Per-problem config blocks contain: domain bounds, PDE parameters, loss weights, and per-problem feature overrides (causal_training, fourier_features.scale, adaptive_sampling.phi).

---

## Model Architecture

### expert_type Options

Controlled by `adaptive_pinn.expert_type`:

| Value | Class | Description |
|-------|-------|-------------|
| `'mlp'` | `FCNet` | Standard fully-connected network |
| `'resnet'` | `ResNetModel` | ResNet with residual blocks |
| `'piratenet'` | `PirateNet` | Physics-informed residual adaptive network |

### base_architecture

Format: `[input_dim, hidden_1, ..., hidden_n, output_dim]`

Example: `[2, 128, 128, 128, 1]` = 2 inputs (x,t), 3 hidden layers of 128, 1 output

### Expert Architecture Sizing

When `AToE_threshold_capacity` is set:
1. Get metric value from region based on `variable_for_expert_size` (norm/new_norm/smoothness)
2. Compute ratio: `metric_value / threshold`
3. Scale capacity: `target_capacity = AToE_threshold_capacity × max(ratio, 1)`
4. Look up architecture in `models/architecture_bank.py`

For ANT: Uses `ANT_threshold_architecture` or `ANT_default_hidden_layers` with same scaling.

---

## Initialization

### Overview

Initialization is split between **base model** (once, before training) and **new experts** (at spawn time). Controlled by `init.*` config keys.

### init.hidden

| Value | Base Model Behavior | Expert Behavior |
|-------|---------------------|-----------------|
| `'default'` | PyTorch default (Kaiming uniform) | PyTorch default |
| `'glorot'` | Xavier uniform + zero bias; ResNet fc2 zeroed for identity start | Same via `apply_expert_init()` |
| `'parent_weights'` | Uses glorot (parent copy is expert-only) | Copies hidden layers from parent expert (or base if parent_idx=-1) |

**Code path** (`trainer/init.py`):
- `apply_hidden_init()`: Glorot on all hidden layers
- `apply_expert_init()`: Glorot hidden + zero output
- `apply_parent_copy_init()`: Copies matching layers from parent model

### init.output

| Value | Base Model Behavior | Expert Behavior |
|-------|---------------------|-----------------|
| `'default'` | PyTorch default | N/A (experts always zero output) |
| `'zero'` | Zero weight and bias | Same |
| `'ls'` | Least-squares fit to IC data | N/A (base only) |

**LS-init details** (`apply_output_init()`):
- Requires `init.ls_use_bias` (whether to include bias column)
- Forward pass on IC points to get hidden activations
- Solve `H @ W = h_gt` via `torch.linalg.lstsq()`
- Requires `n_ic >= hidden_dim + (1 if ls_use_bias else 0)`
- Raises `ValueError` if insufficient IC points

### init.spectral_norm

When `true`:
- Wraps ALL linear layers (hidden + output) with `nn.utils.parametrizations.spectral_norm()`
- Bounds Lipschitz constant to ≤1 per layer
- Output layer uses tiny random init (std=1e-6) instead of strict zero (avoids sigma=0 → NaN)
- Applied to base at init AND to each new expert at spawn

### Expert Init at Spawn

After `model.spawn_expert()`, trainer applies (`trainer.py` ~lines 1953-1975):

```python
if init.hidden == 'parent_weights':
    # Resolve parent model (base if parent_idx=-1, else parent expert)
    apply_parent_copy_init(new_expert, parent_model, cfg,
                           copy_output=isinstance(model, (AToELeaves, ANT)))
else:
    apply_expert_init(new_expert, cfg)
apply_spectral_norm(new_expert, cfg)
```

For **AToE**: `copy_output=False` — output zeroed so expert contributes u=0 at spawn
For **AToELeaves/ANT**: `copy_output=True` — parent is retired, children must start from parent's solution

---

## Training Phases

### Spawning Method Determines Phase Structure

| spawning_method | Phases | Description |
|-----------------|--------|-------------|
| `'accept_split_by_norm'` | Single-phase | Spawn every N epochs, incremental |
| `'by_mean_residual'` | Single-phase | Spawn one leaf with highest mean residual |
| `'M_term_tree_by_norm'` | 3-phase | Phase 1 → Phase 2 (select top M) → Phase 3 |
| `'use_perfect_trees'` | 2-phase (skip Phase 1) | Load pre-computed tree → Phase 3 |

### 3-Phase Training (M_term_tree_by_norm)

**Phase 1**: Train base model alone
- Duration: `adaptive_pinn.initial_train.epochs`
- Uses Phase 1 config overrides if specified

**Phase 2**: Fit decision tree + select top M experts
- Calls `region_detector.fit_full_tree_and_prune(M=adaptive_pinn.M_experts_num)`
- Selects top M nodes by `variable_for_node_accept` metric (norm/new_norm/smoothness)
- Ensures valid binary tree structure (every node has 0 or 2 children)
- AToELeaves spawns only leaf nodes from the final tree
- AToE/ANT spawns all accepted nodes (top M + closure nodes)

**Phase 3**: Train full model (base + all experts)
- Duration: `epochs` from top-level config
- Fresh optimizer created
- LR scheduler reset

### use_perfect_trees

- Loads tree from JSON at `perfect_trees_path`
- Skips Phase 1 entirely
- Spawns experts from loaded tree nodes
- Then runs Phase 3

---

## Optimizer & Scheduler Flow

### Optimizer Chain

```yaml
optimizer_1: soap      # Primary optimizer
optimizer_2: lbfgs     # Secondary (or null)
optimizer_switch_epoch: 5000  # When to switch
```

Supported optimizers:
- `'adam'` — Standard Adam with configurable betas/eps
- `'soap'` — Shampoo-preconditioned Adam (quasi-second-order)
- `'lbfgs'` — Full-batch L-BFGS (no LR scheduler)
- `'ssbroyden'` — Self-Scaled Broyden via torchmin

### LR Scheduler

Created by `_create_lr_scheduler()`:

```
[LinearLR warmup: 0.01× → 1.0× over lr_warmup_steps]
        ↓
[StepLR (exponential) OR CosineAnnealingLR]
```

| Config Key | Effect |
|------------|--------|
| `lr_schedule: 'none'` | No decay phase |
| `lr_schedule: 'exponential'` | StepLR: lr *= lr_decay_rate every lr_decay_steps |
| `lr_schedule: 'cosine'` | CosineAnnealingLR to 0 over remaining steps |
| `lr_warmup_steps` | Linear warmup from 1% to full LR |

### Grouped Optimizer at Spawn/Unfreeze

When experts are spawned with freeze mechanism:
1. Ancestors frozen (their params excluded from optimizer)
2. Optimizer rebuilt with two param groups:
   - **Untouched leaves**: Same LR, preserved Adam/SOAP moments
   - **New experts**: LR = `current_lr × new_expert_lr_decay`
3. LR scheduler set to `None` during freeze period
4. At unfreeze: ancestors restored as new param group at pre-freeze LR with saved state

---

## Spawning & Plateau Gating

### Spawn Trigger Logic

```python
# Base eligibility check
spawn_eligible = is_adaptive and not spawning_complete and num_experts < max_experts

# Trigger at interval OR retry
at_spawn_interval = (epoch % spawn_every_epochs == 0)
at_retry = (spawn_retry_after and epoch == last_fail_epoch + spawn_retry_after)
at_interval = at_spawn_interval or at_retry

# Plateau gating (if enabled)
if spawn_require_plateau and at_interval:
    lookback = metrics['train_loss'][-spawn_plateau_epochs:]
    rel_drop = (lookback[0] - lookback[-1]) / abs(lookback[0])
    plateau_met = (rel_drop <= spawn_plateau_delta)
    spawn_triggered = plateau_met
else:
    spawn_triggered = spawn_eligible and at_interval
```

### Spawn Config Keys

| Key | Description |
|-----|-------------|
| `spawn_every_epochs` | Minimum interval between spawn checks |
| `spawn_require_plateau` | Enable plateau gating |
| `spawn_plateau_epochs` | Look-back window for loss comparison |
| `spawn_plateau_delta` | Max relative improvement to trigger spawn (e.g., 0.05 = 5%) |
| `spawn_retries_before_stop` | Max retries after failed spawn; `false` = never stop |
| `spawn_retry_after` | Epochs to wait before retry after failed spawn |

### Node Acceptance Metrics

Controlled by `variable_for_node_accept`:

| Value | M_term_tree_by_norm (ranking) | Threshold-based methods | Threshold Key |
|-------|------------------------------|------------------------|---------------|
| `'norm'` | Select top M by wavelet_norm_squared (highest) | wavelet_norm_squared ≥ threshold | `wavelet_threshold` |
| `'new_norm'` | Select top M by new_wavelet_norm_squared (highest) | new_wavelet_norm_squared ≥ threshold | `new_norm_threshold` |
| `'smoothness'` | Select top M by smoothness_alpha (lowest/roughest, R²≥0.5) | smoothness_alpha < threshold AND R² ≥ 0.5 | `tree_smoothness_threshold` |

**Note**: `M_term_tree_by_norm` uses `M_experts_num` for top-M selection, not thresholds. Thresholds are still used by `accept_split_by_norm` method.

---

## Freeze Mechanism

### freeze_mode

| Value | Behavior |
|-------|----------|
| `'none'` | All models trainable; disables freeze_epochs_after_spawn |
| `'previous'` | Freeze base + all but newest expert |
| `'base_only'` | Freeze base, train all experts |

**AToELeaves override**: Only leaf experts (+ base if still a leaf) get `requires_grad=True`

### freeze_epochs_after_spawn

When > 0 and `freeze_mode != 'none'`:

1. **At spawn**: Identify ancestors via `get_ancestor_indices()` (walk parent_idx chains)
2. **Freeze ancestors only** — sibling leaves on other branches keep training
3. **Rebuild optimizer**:
   - Untouched leaves: same LR + preserved moments
   - New experts: LR = `current_lr × new_expert_lr_decay`
4. **Scheduler disabled** during freeze
5. **At epoch + freeze_epochs_after_spawn**:
   - Unfreeze all via `model.freeze_models(mode='none')`
   - Add ancestor params back as new param group at pre-freeze LR
   - Restore saved optimizer state for ancestors

---

## Causal Training

### Overview

Implements Wang et al. 2022 "Respecting Causality":
- Sort residual points by time
- Split into `num_chunks` temporal bins
- Weight each chunk: `w_i = exp(-ε × cumsum(L_j for j<i))`
- When `min(w_i) > min_weight_threshold`, advance ε to next in schedule

### Config Keys

Under `problem.causal_training`:

| Key | Description |
|-----|-------------|
| `enabled` | Enable causal weighting |
| `num_chunks` | Number of temporal bins (default 16) |
| `tol_schedule` | List of ε values, e.g., `[0.01, 0.1, 1.0, 10.0]` |
| `min_weight_threshold` | Convergence threshold (default 0.99) |
| `per_leaf_causal` | Each leaf gets independent causal state |

### Per-Leaf Causal (per_leaf_causal: true)

- Used for KS where different regions converge at different rates
- Each leaf expert has its own `causal_state` in `loss_fn._leaf_state`
- Residual computed per leaf region, weighted by sample count
- After spawn: new leaf states created via `create_causal_state(problem_cfg)`

### Code Flow

```python
# In loss function:
causal_state = loss_fn.causal_state  # or leaf_states for per_leaf_causal
r2_weighted = compute_causal_residual(residual_squared, t_residual, causal_state)

# Each epoch in trainer:
if advance_causal_schedule(causal_state):
    print(f"Epsilon advanced to {causal_state['tol']}")
causal_state['min_weight'] = 1.0  # reset for next epoch
```

---

## Adaptive Sampling

### Overview

Mixes uniform + residual-adaptive collocation points. At resample epochs, points with higher residuals are more likely to be resampled.

### Config Keys

Under `sampling.adaptive_sampling`:

| Key | Description |
|-----|-------------|
| `enabled` | Enable adaptive sampling |
| `adaptive_ratio` | Fraction of residual points that are adaptive (rest uniform) |
| `per_leaf_sampling` | Divide adaptive budget across leaves |

Under `sampling`:

| Key | Description |
|-----|-------------|
| `resample_every_epochs` | How often to regenerate training data |
| `min_points_per_leaf` | Minimum adaptive points per leaf (per_leaf_sampling only) |

### Residual Caching

On resample epochs:
1. Set `model._residual_cache_enabled = True`
2. Loss functions append `(x, t, r²)` per batch to `model._residual_cache`
3. At next epoch's start, call `regenerate_training_data()` with cached residuals

### Per-Problem Phi Config

Under `problem.adaptive_sampling`:

| Key | Options |
|-----|---------|
| `phi` | `'quadratic'` \| `'exponential'` \| `'power'` |
| `phi_epsilon` | Scale parameter for exponential phi |

The phi function transforms residuals into sampling weights.

### Per-Leaf Sampling (per_leaf_sampling: true)

- Divide adaptive budget across leaves by region volume
- Filter cached residuals per region
- Fallback to uniform-in-region if no cache for that leaf
- Respects `min_points_per_leaf`

---

## Other Features

### RWF (Random Weight Factorization)

Config: `rwf: true`

Applies `W_eff = diag(exp(s)) @ W` to all hidden layers, improving trainability.

### Fourier Features

Config under `fourier_features`:

| Key | Description |
|-----|-------------|
| `enabled` | Enable input mapping |
| `dim` | Number of Fourier features (output = 2×dim) |
| `scale` | Std of random projection matrix B |
| `periodic` | Use periodic features |

Maps input `z → [cos(Bz), sin(Bz)]` before network.

### LRA (Loss Rate Annealing)

Config under `lra`:

| Key | Description |
|-----|-------------|
| `enabled` | Enable adaptive loss weighting |
| `update_every` | Epochs between weight updates |
| `alpha` | EMA smoothing factor |
| `scheme` | `'grad_norm'` (symmetric) or `'lra'` (residual anchored) |

Two schemes:
- **grad_norm**: All weights adapt so each term's gradient has same L2 norm
- **lra**: Residual weight fixed; IC/BC boosted to match residual gradient scale

### Gradient Clipping

| Config Key | Description |
|------------|-------------|
| `grad_clip_norm` | Clip for base model params |
| `expert_grad_clip_norm` | Clip for ALL expert params (when experts exist) |

When experts exist:
- Expert params clipped at `expert_grad_clip_norm`
- Base params clipped at `grad_clip_norm`

When no experts: `grad_clip_norm` applies to all.

---

## Per-Problem Config Keys

Each problem section (`schrodinger`, `burgers1d`, etc.) contains:

### Domain & Dimensions
- `spatial_dim`: Number of spatial dimensions
- `output_dim`: Output dimension (2 for complex-valued)
- `spatial_domain`: `[[x_min, x_max], ...]` per spatial axis
- `temporal_domain`: `[t_min, t_max]`

### PDE Parameters
- Problem-specific: `nu` (Burgers), `D` (Allen-Cahn), `mu` (KdV), `alpha/beta/gamma` (KS), etc.

### Tree Thresholds
- `wavelet_threshold`: For norm-based acceptance (used by `accept_split_by_norm`)
- `new_norm_threshold`: For new_norm-based acceptance (used by `accept_split_by_norm`)
- `tree_smoothness_threshold`: For smoothness-based acceptance (used by `accept_split_by_norm`)

### M-term Tree Selection
- `M_experts_num`: Number of top-ranked nodes to select (used by `M_term_tree_by_norm`)
  - After selection, closure is built to ensure valid binary tree structure
  - Final expert count = M + closure nodes (siblings + ancestors)

### Loss Weights
```yaml
loss_weights:
  residual: 1.0
  ic: 1.0      # May be higher (e.g., 100) for stiff problems
  bc: 1.0
```

### Per-Problem Feature Overrides
- `causal_training`: Full causal config (enabled, num_chunks, tol_schedule, etc.)
- `fourier_features.scale`: Override global scale
- `adaptive_sampling.phi`: Phi function type for this problem

---

## File Reference

| File | Purpose |
|------|---------|
| `experiments_plan.yaml` | Main experiment configuration |
| `run_experiments.py` | Batch experiment runner |
| `run_ncc.py` | Single run orchestrator |
| `trainer/trainer.py` | Core training loop |
| `trainer/init.py` | Initialization functions |
| `models/network_factory.py` | Model creation dispatch |
| `models/atoe.py` | AToE model class |
| `models/atoe_leaves.py` | AToELeaves model class |
| `models/ant.py` | ANT model class |
| `losses/causal_weighting.py` | Causal training implementation |
| `losses/lra.py` | Adaptive loss weighting |
| `utils/dataset_gen.py` | Dataset generation + adaptive sampling |
| `adaptive/region_detector.py` | Tree fitting and pruning |
| `adaptive/indicators.py` | Soft indicator functions |
