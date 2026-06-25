# PAFA Training-Flow Spec — Per-Variant Flows with Compact Windows

**Status:** contract for the additive coarse-to-fine implementation.
**Scope:** the training flow for the three model variants **AToE**, **ANT**, and **AToE-Leaves**, using the
new compact smoothstep windows and variant-specific composition, spawn timing, and training schedules.

The variants share the **window function** and **tree construction**, but differ in **tree closure**
(siblings), **composition/normalization**, **spawn timing**, **initialization**, and **training flow**.
The `model` config key selects all variant-specific behavior.

---

## 0. Shared — Compact Smoothstep Windows

All variants use the same compactly-supported flat-top window, replacing the previous product-of-sigmoids.

### 0.1 Symbols

- `d = n + 1` — total input dimension (`n` spatial coords + time). A point is `X = (X_1, …, X_d)`.
- `Ω_i = [a_{i,1}, b_{i,1}] × … × [a_{i,d}, b_{i,d}]` — axis-aligned box of region `i`.
- `W_{i,j} = b_{i,j} − a_{i,j}` — region width along dim `j`.
- `α` — collar fraction (config key `sigma_fraction`, default 0.2).
- `δ_{i,j} = α · W_{i,j}` — collar (transition) half-width along dim `j`.
- `N` — window smoothness order (config key `window_smoothness_order`). Rule: `N ≥` PDE spatial order.

### 0.2 Smoothstep Polynomial `S_N` (C^N)

```
S_N(t) for t in [0,1]:
  S_1(t) = 3t² − 2t³                                    (C¹)
  S_2(t) = 6t⁵ − 15t⁴ + 10t³                            (C²)
  S_3(t) = 35t⁴ − 84t⁵ + 70t⁶ − 20t⁷                    (C³)
  S_4(t) = 126t⁵ − 420t⁶ + 540t⁷ − 315t⁸ + 70t⁹         (C⁴)

Properties: S_N(0)=0, S_N(1)=1, S_N^(k)(0)=S_N^(k)(1)=0 for k=1..N.
```

PDE order → minimum `N`: Burgers/Allen–Cahn/Schrödinger (order 2) → `N≥2`; KdV (order 3) → `N≥3`;
KS (order 4) → `N≥4`.

### 0.3 One-Sided Ramp and 1D Window

```
One-sided compact ramp:
  ρ_N(s) = 0           for s ≤ 0
         = S_N(s)      for 0 < s < 1
         = 1           for s ≥ 1

1D window for region i, dim j (δ = α · (b_{ij} − a_{ij})):
  s_lo = (X_j − (a_{ij} − δ)) / δ       # 0 at a−δ, 1 at a
  s_hi = ((b_{ij} + δ) − X_j) / δ       # 1 at b, 0 at b+δ
  ω_{ij}(X_j) = ρ_N(s_lo) · ρ_N(s_hi)
```

This is a **flat-top** window: `=1` on `[a, b]`, smooth `C^N` ramps in the collars `[a−δ, a]` and
`[b, b+δ]`, **exactly 0** outside `[a−δ, b+δ]`.

### 0.4 Region Indicator (Tensor Product)

```
Ψ_i(X) = ∏_{j=1..d} ω_{ij}(X_j)         # d = n+1, includes time
Root/base: Ψ_0(X) = 1 everywhere (constant)
```

### 0.5 Derivative Computation

All derivatives are computed via **autograd on the composed forward output**. The decomposed/analytical
indicator-derivative paths are **legacy** and disabled. The smoothstep polynomial is bounded in the
collar and has exact-zero derivative outside.

---

## 1. Config Surface

### 1.1 `adaptive_pinn` Block

- `enabled`, `spawning_method` (**must** be `M_term_tree_by_norm`; validated at startup).
- `M_experts_num`, `spawn_every_epochs`, `max_experts`, `tree_max_depth`, `tree_min_samples_leaf`.
- `variable_for_node_accept`, `variable_for_expert_size`, `blending_mode`.
- `reinitialize_base_after_spawn` (only for AToE-Leaves; ignored for AToE/ANT staged flow).
- `spawn_require_plateau`, `spawn_plateau_epochs`, `spawn_plateau_delta`.
- `spawn_retries_before_stop`, `spawn_retry_after`.
- `initial_train` (root/base training block; required unless a checkpoint is given).
- `fine_tune` (**new**; final joint training block; see below).

**The three training-phase configs (each supports two optimizers):**
The staged flow has three distinct training phases, each governed by its own optimizer/scheduler config.
Every one of them exposes the full `optimizer_1`, `optimizer_2`, `optimizer_switch_epoch`, `lr`,
`lr_schedule`, decay/warmup keys, so any phase can switch Adam/SOAP → L-BFGS/SSBroyden mid-phase.

| Phase | Config source | Trains | Used by |
|-------|---------------|--------|---------|
| Root | `adaptive_pinn.initial_train` | base/root only | all variants |
| Per-level | the effective top-level config (today's "Phase 3"/`base_config` after merge) | one level at a time | AToE / ANT staged |
| Fine-tune | `adaptive_pinn.fine_tune` (**new**) | all params together | AToE / ANT (and = AToE-Leaves Phase 3) |

**New keys:**
- `fine_tune` — full optimizer/scheduler block (epochs, `optimizer_1`/`optimizer_2`/`optimizer_switch_epoch`,
  `lr`, `lr_schedule`, decay, warmup) for the final all-together phase. Mirrors `initial_train`.
- `new_expert_lr_decay` — per-level LR decay factor (default `1.0`), applied to **every spawned expert
  (all non-root levels)**. Levels are indexed by `RegionDescriptor.depth`: root/base = level 0, spawned
  experts have level ≥ 1. The level optimizer's starting LR is
  `lr_level(ℓ) = lr_level(ℓ-1) × new_expert_lr_decay = base_lr × new_expert_lr_decay^ℓ`, with
  `lr_level(0) = base_lr` (the per-level config's `lr`). So level 1 is already decayed once relative to the
  root and finer levels compound; `1.0` is a no-op. (Precedent: AB-PINN's per-subdomain decaying LR
  groups, arXiv:2510.08924.)
- `min_epochs_per_level` — minimum epochs to train each level before stopping (AToE/ANT staged).
- `max_epochs_per_level` — maximum epochs per level (AToE/ANT staged).

**Removed (legacy):**
- `perfect_trees_path`, `freeze_mode`, `freeze_epochs_after_spawn`.

### 1.2 Per-Problem Block (`cfg[problem]`)

- `model` — **`AToE`**, **`ANT`**, or **`AToE-Leaves`**. Selects tree closure, composition, spawn
  timing, init, and training flow.
- `init.hidden` (`glorot` | `parent_weights`), `init.output`, `init.spectral_norm`.
- `wavelet_threshold`, `new_norm_threshold`, `tree_smoothness_threshold`.
- `patience_epochs`, `min_epochs`, `patience_rel_delta`, `grad_clip_norm`, `expert_grad_clip_norm`.
- `pretrained_base_checkpoint` — `null` or path.
- `base_weight` — **must be `1`** for AToE additive composition (root coefficient = 1).
- `sigma_fraction` — collar fraction α (default 0.2).

**New keys:**
- `window_smoothness_order` — smoothstep order N (default = PDE spatial order).

### 1.3 Startup Validation

1. `spawning_method == 'M_term_tree_by_norm'` else `ValueError`.
2. Exactly one Phase-1 source: if `pretrained_base_checkpoint` is `null`, `initial_train` is required.
3. `base_weight == 1` for AToE (warning if not).
4. `window_smoothness_order >= PDE_order` (warning if not).

---

## 2. Variant Matrix (Summary)

| Aspect | AToE | ANT | AToE-Leaves |
|--------|------|-----|-------------|
| **Tree closure** | ancestors only (drop siblings) | ancestors + siblings | ancestors + siblings |
| **Composition** | additive, per-level background | leaf-frontier PoU, staged | leaves-only PoU, all leaves |
| **Spawn timing** | incremental (level by level) | incremental (level by level) | all leaves at once |
| **Training flow** | staged + final fine-tune | staged + final fine-tune | joint (current 3-phase) |
| **Init** | copy parent hidden + zero output | current (routing input) | copy parent hidden + output |
| **Expert forward input** | raw `(x,t)` | parent activation `A_parent` | raw `(x,t)` |

---

## 3. AToE — Additive Coarse-to-Fine

### 3.1 Tree Closure

Keep **ancestors only**, drop siblings. Unselected regions get no expert and are carried by the
additive root (the background term in the normalization handles partial coverage).

```
retain_siblings = False
accepted = top_M_nodes + their ancestors (for valid parent_idx linkage)
```

### 3.2 Composition (Additive, Per-Level Background Normalization)

```
u(X) = u_0(X) + ∑_{ℓ=1..L} ∑_{i: level(i)=ℓ} w_i(X) · u_i(X)

w_i(X) = Ψ_i(X) / (1 + ∑_{k: level(k)=ℓ(i)} Ψ_k(X))
```

- Root/base contributes with coefficient **exactly 1** (not normalized).
- Each level has its own denominator `Z_ℓ = 1 + ∑_{level ℓ} Ψ`.
- The constant `1` is the background slot — keeps a lone child regional (`w = Ψ/(1+Ψ) < 1`), avoids `0/0`.
- The sum grows incrementally as levels are spawned (no wasted forward on unspawned levels).

### 3.3 Spawn Timing (Incremental)

Experts are spawned **level by level** (on the go), not all at once. After root training, the tree
structure (all region bounds + levels) is computed once; NN experts are created and added to the
composition one level at a time. Each child copies its **already-trained parent** (not the untrained root).

### 3.4 Initialization

```
init.hidden == 'parent_weights':
  apply_parent_copy_init(expert, parent, copy_output=False)
  → copy trained-parent hidden layers, ZERO output layer
  → expert contributes u_i ≡ 0 at spawn (additive residual)

init.hidden == 'glorot':
  apply_expert_init(expert)
  → Glorot uniform hidden, zeroed output
```

AToE experts use base architecture, so parent-hidden copy is shape-compatible.

### 3.5 Training Flow (Staged Coarse-to-Fine)

```
Phase Root: Train root/base on full PDE loss (residual + BC + IC) using `initial_train`.

Tree Build: Compute M-term tree ONCE from root eval.
            retain_siblings=False; group nodes by depth → L levels (level 0 = root).

For ℓ = 1..L (coarse → fine):     [per-level config = effective top-level config]
  1. Spawn level-ℓ NN experts; init from trained parent (copy_output=False).
  2. Add their terms to the additive composition.
  3. Freeze base + all lower levels (requires_grad=False).
  4. Build a FRESH optimizer + scheduler over level-ℓ params only:
     - starting LR = lr_level(ℓ) = lr_level(ℓ-1) × new_expert_lr_decay = base_lr × decay^ℓ
       (lr_level(0) = per-level base lr = root anchor; decay applies to every level ℓ ≥ 1).
     - step_count = 0; short per-level warmup (LinearLR) + decay (StepLR/cosine) sized to max_epochs_per_level.
     - per-level optimizer switch (optimizer_1 → optimizer_2 at the per-level optimizer_switch_epoch) allowed.
  5. Train on full PDE loss until per-level stopping:
     - relative-improvement test (patience_rel_delta)
     - min_epochs_per_level / max_epochs_per_level
  6. Hard-freeze level ℓ.

Final Joint Fine-Tune:             [config = `fine_tune` block]
  1. Unfreeze ALL params.
  2. Build a fresh optimizer + scheduler over all params from the `fine_tune` block
     (its own lr/schedule/warmup; optimizer_1 → optimizer_2 switch allowed).
  3. Train full composition on full PDE loss to convergence (patience from fine_tune/config).
```

**Why the per-level optimizer reset is safe here** (cf. PIKAN optimizer-reset spike, arXiv:2407.17611):
prior levels are frozen / out of the optimizer (their Adam moments are never reset), and new experts are
zero-output initialized so the composition is continuous across the level boundary. The only reset that
re-touches trained params is the **final fine-tune** — hence it gets its own `fine_tune` block where a
lower starting LR + warmup absorb the moment reset.

### 3.6 Console Markers (AToE)

- `[3-Phase] Phase 1 …`
- `[Tree] Computing M-term tree (retain_siblings=False) …`
- `[Staged] Training level ℓ (K experts) …`
- `[ParentInit] Expert k: hidden copied from <parent>, output zeroed`
- `[Freeze] Level ℓ frozen`
- `[FinalTune] Unfreezing all params for final joint fine-tune`
- `[EarlyStop] …` (per-level or final)

---

## 4. ANT — Staged with Leaf-Frontier Normalization

### 4.1 Tree Closure

Keep **ancestors + siblings** (complete binary routing tree required for parent→child activation flow).

```
retain_siblings = True
accepted = top_M_nodes + ancestors + siblings (current behavior)
```

### 4.2 Composition (Leaf-Frontier PoU, Staged)

At stage ℓ, the leaves are the freshly added level-ℓ nodes plus any earlier nodes still without
children. Normalize only among the **current leaf frontier**:

```
At stage ℓ, let F_ℓ = current leaf frontier:
  u(X) = ∑_{j ∈ F_ℓ} Ψ̃_j(X) · u*_j(X)

  Ψ̃_j = Ψ_j / ∑_{k ∈ F_ℓ} Ψ_k
```

- Parents **retire from the blend** when they get children, but **still route activations** to children.
- The leaf frontier and normalization are re-evaluated at each stage.

### 4.3 Expert Forward (Routing)

ANT experts take their **parent's last-hidden activation** as input, not raw `(x,t)`:

```
u_i(A_parent) = (u*_i, A_i)    # output prediction + activation for children
```

Frozen ancestors are still evaluated during forward to supply `A_parent` to active leaves.

### 4.4 Spawn Timing (Incremental)

Same as AToE: spawn **level by level** (on the go). Each child is initialized after its parent has trained.

### 4.5 Initialization

Keep current ANT init behavior. Because the routing input dimension differs from the parent's hidden
dimension, a parent-hidden copy is **not** shape-compatible. Use `copy_output=True` (existing).
Only the **timing** becomes incremental.

### 4.6 Training Flow (Staged with Routing)

```
Phase Root: Train root/base on full PDE loss using `initial_train`.

Tree Build: Compute M-term tree ONCE from root eval.
            retain_siblings=True; group nodes by depth → L levels (level 0 = root).

For ℓ = 1..L (coarse → fine):     [per-level config = effective top-level config]
  1. Spawn level-ℓ NN experts; init from parent (copy_output=True).
  2. Update leaf frontier (level-ℓ nodes become leaves; their parents retire from blend).
  3. Freeze previous levels (requires_grad=False); they still run for routing.
  4. Build a FRESH optimizer + scheduler over level-ℓ params only:
     - starting LR = lr_level(ℓ) = lr_level(ℓ-1) × new_expert_lr_decay = base_lr × decay^ℓ
       (lr_level(0) = per-level base lr = root anchor; decay applies to every level ℓ ≥ 1).
     - step_count = 0; per-level warmup + decay sized to max_epochs_per_level.
     - per-level optimizer_1 → optimizer_2 switch allowed.
  5. Train on full PDE loss (current leaf-frontier composition) until per-level stopping
     (patience_rel_delta; min_epochs_per_level / max_epochs_per_level).
  6. Hard-freeze level ℓ.

Final Joint Fine-Tune:             [config = `fine_tune` block]
  1. Unfreeze ALL params.
  2. Build a fresh optimizer + scheduler over all params from the `fine_tune` block
     (own lr/schedule/warmup; optimizer_1 → optimizer_2 switch allowed).
  3. Train full composition on full PDE loss to convergence.
```

Same safety note as AToE §3.5 applies: frozen prior levels keep their moments; the final fine-tune's
`fine_tune` block (lower LR + warmup) absorbs the all-params optimizer reset.

### 4.7 Console Markers (ANT)

- `[3-Phase] Phase 1 …`
- `[Tree] Computing M-term tree (retain_siblings=True) …`
- `[Staged] Training level ℓ (K experts), leaf frontier size = F`
- `[LeafUpdate] Level ℓ: parents retire, new leaves active`
- `[Freeze] Level ℓ frozen (still routing)`
- `[FinalTune] Unfreezing all params for final joint fine-tune`
- `[EarlyStop] …`

---

## 5. AToE-Leaves — Straight to Leaves (Joint)

### 5.1 Tree Closure

Keep **ancestors + siblings** (leaves must tile the domain).

```
retain_siblings = True
accepted = top_M_nodes + ancestors + siblings
nodes_to_spawn = leaves of the pruned tree only
```

### 5.2 Composition (Leaves-Only PoU)

```
u(X) = ∑_{j ∈ L} Ψ̃_j(X) · u_j(X)

Ψ̃_j = Ψ_j / ∑_{k ∈ L} Ψ_k
```

- Computed once over **all leaves** (the complete leaf set, not a frontier).
- Base is excluded from the blend once it has children.

### 5.3 Spawn Timing (All at Once)

All leaves spawn in a **single epoch** (current behavior). No staged build.

### 5.4 Initialization

```
init.hidden == 'parent_weights':
  apply_parent_copy_init(expert, parent, copy_output=True)
  → copy parent hidden + output (child starts from parent's full solution)

init.hidden == 'glorot':
  apply_expert_init(expert)
  → Glorot uniform hidden, zeroed output
```

### 5.5 Training Flow (Joint, Current 3-Phase)

```
Phase 1: Train root/base on full PDE loss.

Tree Build + Spawn: Compute M-term tree; spawn ALL leaves in one epoch.

Phase 3 (Leaf Training):
  1. Recreate optimizer/scheduler over leaf params only (base retired from composition).
  2. Train leaves on full PDE loss to convergence (early stopping via patience).
```

No staged build, no per-level freezing. Base is retired once leaves spawn (discarded from `leaf_indices`);
the forward pass computes a partition-of-unity sum over leaf experts only.

### 5.6 Console Markers (AToE-Leaves)

- `[Orchestrator] [3-Phase] Phase 1: training root/base for N epochs`
- `[Tree] Computing M-term tree (retain_siblings=True) …`
- `[FullTree] Spawning complete. K leaves spawned.`
- `[Phase 3] Training K leaf experts (base retired from composition)`
- `[Segment:phase3] start | epochs …`
- `[EarlyStop] …`

---

## 6. Optimizer and Scheduler Lifecycle

Each training phase builds its optimizer/scheduler from its own config block, over only the params that
are trainable at that moment (factories filter `requires_grad=True`). Every phase may run an internal
`optimizer_1 → optimizer_2` switch at its own `optimizer_switch_epoch`.

### 6.1 AToE / ANT (Staged)

| Moment | Config | Action |
|--------|--------|--------|
| Root start | `initial_train` | build optimizer over base params; build scheduler; `step_count = 0` |
| Each level start | effective top-level cfg | fresh optimizer over level-ℓ params only; LR = `base_lr × new_expert_lr_decay^ℓ` (decay applied to every level ℓ ≥ 1); fresh scheduler (warmup + decay, sized to `max_epochs_per_level`); `step_count = 0`; reset patience |
| Fine-tune start | `fine_tune` | fresh optimizer over ALL params; own lr/schedule/warmup; `step_count = 0`; reset patience |
| `optimizer_switch_epoch` (any phase) | that phase's `optimizer_2` | build `optimizer_2`; `lr_scheduler = None` (line-search); reset patience |

`lr_level(ℓ) = lr_level(ℓ-1) × new_expert_lr_decay = base_lr × new_expert_lr_decay^ℓ`, with
`lr_level(0) = base_lr` (the per-level config's `lr`). Levels are indexed by `RegionDescriptor.depth`:
level 0 = root (its own training uses `initial_train`); the decay applies to every spawned level ℓ ≥ 1.

### 6.2 AToE-Leaves (Joint)

| Moment | Config | Action |
|--------|--------|--------|
| Root start | `initial_train` | build optimizer over base params; build scheduler; `step_count = 0` |
| Phase 3 transition | effective top-level cfg | recreate optimizer over leaf params only (base retired); recreate scheduler; reset patience |
| `optimizer_switch_epoch` | `optimizer_2` | build `optimizer_2`; `lr_scheduler = None`; reset patience |

Optimizer factories: `_create_primary_optimizer`, `_create_optimizer_by_name`, `_create_lr_scheduler`.
Factories filter to `requires_grad=True` params automatically.

---

## 7. Early Stopping / Patience

Training phases terminate early when the loss plateaus. Patience is active for **both** optimizers
(not only after an optimizer switch), enabling faster iteration when either optimizer stalls.

### 7.1 General Mechanism

1. **Plateau detection** uses a relative improvement test: the loss must drop by at least
   `patience_rel_delta` (e.g., 0.1%) relative to the best seen so far to count as improvement.
   Formally: improvement iff `loss < best_loss × (1 − patience_rel_delta)`.

2. **Grace period:** no stopping is allowed until `min_epochs` have passed since the phase (or
   optimizer window) started. This gives the optimizer time to warm up.

3. **Patience counter:** after the grace period, if the loss fails to improve for
   `patience_epochs` consecutive epochs, a plateau is declared.

4. **Plateau action — depends on optimizer:**
   - **`optimizer_1` plateau with `optimizer_2` configured:** do not stop; instead, fast-forward
     to `optimizer_switch_epoch` (skip remaining optimizer_1 epochs), reset `best_loss` and
     patience counter, and continue with `optimizer_2`. This preserves optimizer_2's configured
     epoch budget while avoiding wasted computation on a stalled optimizer_1.
   - **`optimizer_1` plateau without `optimizer_2`:** stop the current training phase.
   - **`optimizer_2` plateau:** stop the current training phase.

5. **Reset at transitions:** `best_loss` and the patience counter reset at each phase start
   (root → level-1, level-ℓ → level-ℓ+1, levels → fine-tune) and at the optimizer switch.

### 7.2 Per-Level Stopping (AToE / ANT Staged Training)

For staged training, each level uses:

- **Relative-improvement test:** as above.
- **Minimum epochs:** `min_epochs_per_level` must pass before stopping.
- **Maximum epochs:** hard cap at `max_epochs_per_level`.
- **Reset:** `best_loss` / patience counter reset at each level start.

Final joint fine-tune uses `patience_epochs` / `min_epochs` from the `fine_tune` block (falling
back to the top-level config).

---

## 8. Legacy / Removed

The following are **disabled** or **removed**:

- **Decomposed/analytical indicator derivatives** — all loss files use autograd on composed output only.
  The `compute_analytical_indicator_derivatives` functions and `use_decomposed`/`use_analytical` branches
  are marked legacy.
- **`freeze_mode`, `freeze_epochs_after_spawn`** — config keys removed. Staged freezing is now clean
  `requires_grad=False` per level, not a global mode.
- **`new_expert_lr_decay`** — *re-introduced* with new meaning: a per-level starting-LR decay factor for
  the staged flow (§1.1), **not** the old grouped-optimizer freeze-LR multiplier.
- **`freeze_events` metric slot** — dead, removed.
- **Product-of-sigmoids indicators** — replaced by compact smoothstep windows.

---

## 9. Dispatch Logic (Trainer)

The `model` config key drives all variant-specific behavior:

```python
model_type = cfg[problem]['model']  # 'AToE', 'ANT', or 'AToE-Leaves'

# Tree closure
retain_siblings = (model_type != 'AToE')

# Init
if model_type == 'AToE':
    copy_output = False  # zero output for additive residual
else:
    copy_output = True   # copy output for PoU continuity

# Training flow
if model_type in ('AToE', 'ANT'):
    run_staged_schedule()  # level-by-level + final fine-tune
else:
    run_joint_schedule()   # spawn-all + joint Phase 3
```

No `isinstance` plumbing; all dispatch is config-driven.
