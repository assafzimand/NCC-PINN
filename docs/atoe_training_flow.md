# AToE Variants - Training Flow Summary

This document describes the training flow for the Adaptive Tree of Experts (AToE) variants in the NCC-PINN codebase.

---

## 1. Model Variants Overview

| Variant | Class | Description |
|---------|-------|-------------|
| **AToE** | `AToE` | Additive Tree of Experts - base stays active, experts add corrections |
| **AToELeaves** | `AToELeaves` | Only leaf experts participate in composition (configurable additive mode) |
| **ANT** | `ANT` | Adaptive Network Tree - staged training |

---

## 2. Training Phases by Variant

### Non-Adaptive (vanilla PINN)
```
main segment → done
```

### AToE (Staged, always additive)
```
Phase 1 (root) → per-level spawn+train → fine_tune → done
```

### AToELeaves
```
Phase 1 (root) → spawn ALL leaves → Phase 3 → [fine_tune if additive] → done
```

---

## 3. AToELeaves Deep Dive

### 3.1 Phase 1: Root Training
- **Trainable**: Base model only (`_set_trainable(model, 'base')`)
- **Config**: Uses `adaptive_pinn.initial_train` settings
- **Optimizer**: `optimizer_1: adam` → `optimizer_2: lbfgs/ssbroyden` at `optimizer_switch_epoch`
- **Output**: Trained root model

### 3.2 Spawning (between Phase 1 and Phase 3)

**Tree Building:**
- Build M-term tree from residual analysis
- Select **only leaf nodes** (internal nodes skipped)
- `retain_siblings=True` for AToELeaves

**Expert Creation:**
- All experts copy weights from **base model** (parent_idx = -1 for all)
- Hidden layers: Always copied from base
- Output layer: **Depends on `additive` setting**

| `additive` | `copy_output` | Output Init |
|------------|---------------|-------------|
| `false` | `true` | Copied from base |
| `true` | `false` | **Zeroed** (residual learning) |

**Leaf Index Management:**
```python
self.leaf_indices.add(expert_idx)
self.leaf_indices.discard(region.parent_idx)  # removes -1 (base)
```
After spawning, `leaf_indices = {0, 1, 2, ...}` (no -1).
- **Non-additive mode**: Base is retired from composition (only leaves are used).
- **Additive mode**: Base remains in composition via `forward()` returning `base_model(x) + leaf_output`.

### 3.3 Phase 3: Joint Leaf Training

**Trainable**: All leaf experts, base frozen (`_set_trainable(model, 'leaves')`)

**Loss Function**: Depends on `split_icbc.enabled`

---

## 4. Split ICBC Mode (`split_icbc.enabled: true`)

### 4.1 What It Does
Instead of computing loss on the composed output, it trains each expert **independently** on its subdomain:

```python
loss_j = w_res * residual_j + w_ic * ic_j + w_bc * bc_j + w_cont * continuity
```

### 4.2 Data Generation (`build_subdomain_data`)

For each expert, generates:
- **Residual points**: Within expert's region bounds
- **IC points**: On expert's t=0 face (true IC) or interior t-boundary (interface)
- **BC points**: On expert's spatial boundaries (true BC) or interior x-faces (interface)
- **Continuity points**: On shared faces between neighboring experts

### 4.3 Target Values by Mode

| Point Type | `additive=false` | `additive=true` |
|------------|------------------|-----------------|
| IC (true, t=0) | Analytic IC | **0** (leaf correction) |
| BC (true, global) | Analytic BC (Dirichlet) | **0** (leaf correction) |
| BC (periodic, e.g., Allen-Cahn) | **Cross-expert pairing**: penalize `(u_left - u_right)²` at matching t-values | **Leaves → 0**: penalize `u_j² + (∂u_j/∂x)²` (root satisfies periodic BC) |
| Interface IC | Minted from frozen model | **0** |
| Interface BC | Minted from frozen model | **0** |

### 4.4 Residual Computation

```python
if additive:
    u_field = model.base_model(xt) + model.forward_single_expert(expert_idx, xt)
else:
    u_field = model.forward_single_expert(expert_idx, xt)
```

### 4.5 Continuity Loss
On shared faces between neighbors:
- Value match: `|u_a - u_b|²`
- Derivative match (if `pde_order >= 1`):
  - Spatial face: `|∂u_a/∂x - ∂u_b/∂x|²` (normal to face)
  - Temporal face: `|∂u_a/∂t - ∂u_b/∂t|²`
- Second derivative match (if `pde_order >= 2`): Similar for second derivatives

---

## 5. Composition (Forward Pass)

### 5.1 Blending Modes

**Soft (`blending_mode: soft`):**
```python
u(x,t) = Σ_{j ∈ leaves} (ψ_j / Σ_k ψ_k) · u_j
```
Uses smooth indicators (sigmoid/smoothstep).
- **Requirements**: Use with `split_icbc.enabled: false` (global loss)

**Hard (`blending_mode: hard`):**
```python
u(x,t) = Σ_{j ∈ leaves} (hard_j / Z) · u_j
```
Step functions; on shared faces, Z=2 → each expert gets weight 0.5.
- **Requirements**: Must use with `split_icbc.enabled: true` (per-expert loss with continuity)

### 5.2 Additive Mode

When `additive=true`:
```python
return self.base_model(inputs) + leaf_output
```

When `additive=false`:
```python
return leaf_output  # base retired from composition
```

---

## 6. Fine-Tune Phase (AToELeaves + Additive Only)

If `additive=true` AND `adaptive_pinn.fine_tune` config exists:

```python
if additive and fine_tune_cfg:
    _set_trainable(model, 'all')  # unfreeze base + leaves
    _train_segment(ctx, 'fine_tune', ...)
```

This allows joint fine-tuning of base and leaves after Phase 3.

---

## 7. Config Combinations Summary

| `additive` | `split_icbc` | `blending_mode` | Expert Init | Phase 3 Loss | Fine-tune |
|------------|--------------|-----------------|-------------|--------------|-----------|
| `false` | `false` | `soft` | Full copy | Composed loss | No |
| `false` | `true` | `hard` | Full copy | Per-expert (targets from model) | No |
| `true` | `false` | `soft` | **Output zeroed** | Composed loss | **Yes** |
| `true` | `true` | `hard` | **Output zeroed** | Per-expert (targets=0) | **Yes** |

---

## 8. Key Config Parameters

### `experiments_plan.yaml` structure:

```yaml
base_config:
  precision: float32/float64
  optimizer_1: adam
  optimizer_2: lbfgs/ssbroyden
  optimizer_switch_epoch: 5000
  
  adaptive_pinn:
    enabled: true
    M_experts_num: 10
    expert_type: mlp/resnet
    blending_mode: soft/hard
    additive: true/false           # AToELeaves composition mode
    split_icbc: { enabled: true }  # Per-expert loss vs composed loss
    
    initial_train:                 # Phase 1 config
      epochs: 35000
      optimizer_1: adam
      optimizer_2: lbfgs
      
    fine_tune:                     # Optional Phase 4 (additive only)
      epochs: 30000
      optimizer_1: soap

experiments:
  - name: experiment_name
    model: AToELeaves
    base_architecture: [2, 30, 30, 30, 1]
    adaptive_pinn:
      additive: true
      blending_mode: soft
      split_icbc: { enabled: false }
```

---

## 9. Key Code Locations

| Component | File | Key Function/Class |
|-----------|------|-------------------|
| Orchestrator | `trainer/trainer.py` | `train_orchestrator()` (line ~2679) |
| Spawning | `trainer/trainer.py` | `_spawn_nodes()` (line ~2437) |
| Expert Init | `trainer/init.py` | `apply_parent_copy_init()` (line ~186) |
| Model | `models/atoe_leaves.py` | `AToELeaves`, `forward()` |
| Split Data | `adaptive/subdomain_data.py` | `build_subdomain_data()` |
| Split Loss | `losses/split_loss.py` | `build_split_loss()` |
| Split Segment | `trainer/trainer.py` | `_run_split_segment()` (line ~2874) |

---

## 10. Important Implementation Details

### Expert Initialization
- All AToELeaves experts copy from **base model** (not tree parent)
- The `parent_idx` stored in regions refers to tree structure, but weight copy always uses base
- `copy_output` is `False` when `additive=True` → output layer zeroed

### Leaf Index Tracking
- `-1` in `leaf_indices` means base is a leaf (before any spawning)
- After spawning: `-1` is removed, experts `{0, 1, 2, ...}` become leaves
- Base is "retired" from composition (unless `additive=True`)

### Split Loss Training
- Creates frozen model snapshot for interface target minting
- Each expert trained on its own subdomain data
- Continuity loss enforces agreement on shared faces
- Periodic BC handled via paired points with shared t-values

### SSBroyden/LBFGS Precision
- SSBroyden **requires float64** (quasi-Newton numerical stability)
- LBFGS works with float32 or float64
- Adam/SOAP work with either precision
