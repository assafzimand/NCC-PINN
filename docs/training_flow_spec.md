# PAFA Training-Flow Spec — `M_term_tree_by_norm` (the only spawning method)

**Status:** contract for the cleanup on branch `M_term_AToE-AToE-leaves-ANT`.
**Scope:** the desired final training flow for the single spawning method `M_term_tree_by_norm`, across
the three model variants **AToE**, **AToELeaves**, **ANT**. Every config-driven branch is enumerated so
the implementation can be verified against this document.

After this cleanup the trainer supports exactly one spawning method and **no freezing mechanism**. The
flow is a 3-phase pipeline: **Phase 1** (base only) → **one-shot tree build + spawn** → **Phase 3**
(base + experts together).

---

## 0. Config surface (what the flow reads)

`adaptive_pinn` block:
- `enabled`, `spawning_method` (**must** be `M_term_tree_by_norm`; validated at startup),
  `M_experts_num`, `spawn_every_epochs`, `max_experts`, `tree_max_depth`, `tree_min_samples_leaf`,
  `variable_for_node_accept`, `variable_for_expert_size`, `blending_mode`, `reinitialize_base_after_spawn`,
  `spawn_require_plateau`, `spawn_plateau_epochs`, `spawn_plateau_delta`, `spawn_retries_before_stop`,
  `spawn_retry_after`, `initial_train` (Phase-1 optimizer/scheduler block; required unless a checkpoint is
  given).
- **Removed:** `perfect_trees_path`, `freeze_mode`, `freeze_epochs_after_spawn`, `new_expert_lr_decay`.

Per-problem block (`cfg[problem]`):
- `init.hidden` (`glorot` | `parent_weights`), `init.output`, `init.spectral_norm`, …
- `wavelet_threshold`, `new_norm_threshold`, `tree_smoothness_threshold` (consumed by tree fit).
- `patience_epochs`, `min_epochs`, **`patience_rel_delta`** (new), `grad_clip_norm`,
  `expert_grad_clip_norm`.
- **`pretrained_base_checkpoint`** (new): `null` or a path to a base checkpoint.

Top-level / phase config (and the `initial_train` sub-block): `epochs`, `optimizer_1`, `optimizer_2`,
`optimizer_switch_epoch`, `lr`, `soap_betas`/`adam_betas`, `lr_schedule`, `lr_decay_rate`,
`lr_decay_steps`, `lr_warmup_steps`, `lr_warmup_start_factor`, `batch_size`.

**Startup validation (fail fast):**
1. `spawning_method == 'M_term_tree_by_norm'` else `ValueError`.
2. Exactly one Phase-1 source: if `pretrained_base_checkpoint` is `null`, `initial_train` is required; if
   it is a path, `initial_train` is ignored and `reinitialize_base_after_spawn` **must** be `false`.

---

## 1. Phase 1 — base only

Phase 1 produces a trained base network whose residuals drive the tree. Two mutually exclusive sources:

### 1a. `pretrained_base_checkpoint == null` — train the base
- `active_cfg = cfg overridden by adaptive_pinn.initial_train` (epochs, optimizer_1/2, betas, lr,
  schedule, decay, warmup all come from `initial_train`).
- Build the primary optimizer (or the full-batch path for `lbfgs`/`ssbroyden`) over **base params only**,
  build the LR scheduler, set `step_count = 0`.
- Train for `initial_train.epochs`. `current_phase = 1`. **No early stopping in Phase 1.**

### 1b. `pretrained_base_checkpoint == <path>` — load the base
- `_load_pretrained_base(model, path, cfg)` loads the checkpoint with `weights_only=False` (trusted local
  file) and copies **only the base** into `model.base_model`:
  - adaptive/MoE checkpoint → uses `adaptive_state['base_model']` (its experts are ignored);
  - plain base checkpoint → uses `model_state_dict`.
- **Architecture adoption:** if the checkpoint's base architecture differs from the run's
  `base_architecture`, the base is rebuilt to the checkpoint's architecture and that architecture is
  written back into `cfg['base_architecture']` and `model.config_base_architecture`. This is required so
  that experts spawned later — especially `init.hidden == 'parent_weights'`, which copies the parent's
  layers — are shape-compatible with the loaded base.
- **No Phase-1 training.** Force the spawn check to fire on the first loop epoch so the tree is built from
  the loaded base, then transition immediately to Phase 3.
- `reinitialize_base_after_spawn` must be `false` (loading then reinitializing would discard the
  checkpoint).

---

## 2. Tree build + one-shot spawn (Phase 1 only)

When the spawn check triggers (`epoch % spawn_every_epochs == 0`, optionally plateau-gated by
`spawn_require_plateau`; or forced on the first epoch in case 1b) **and** `current_phase == 1` **and**
`not spawning_complete`:

1. Evaluate the model on the eval grid → `(X_eval, y_eval)`.
2. `region_detector.fit_full_tree_and_prune(X_eval, y_eval, M=M_experts_num,
   variable_for_node_accept=…)` fits a single decision tree and prunes to the top-`M` accepted nodes by
   the chosen norm variable.
3. Choose `nodes_to_spawn`:
   - **AToELeaves:** leaves of the pruned tree only.
   - **AToE, ANT:** all accepted nodes.
4. Spawn each node's expert (`model.spawn_expert(region, …)`), recording parent linkage. All `M` experts
   spawn in this single epoch.
5. `spawning_complete = True`. Spawning never happens again (it is gated to Phase 1).

**Expert initialization at spawn** (`init.hidden`, applied right after spawn):

| `init.hidden` | Hidden layers | Output layer (ALL variants) | Parent resolution |
|---|---|---|---|
| `glorot` | Glorot uniform + zero bias (`apply_expert_init`) | zeroed (tiny-random if `spectral_norm`) → expert starts at identity (residual) | n/a |
| `parent_weights` | copied from parent (`apply_parent_copy_init`); prints `[ParentInit]` | **copied** from parent (`copy_output=True`) | base if `parent_idx == -1`, else `experts[parent_idx]` |

- `apply_spectral_norm` is applied to every newly spawned expert afterward.
- **Why `copy_output=True` is correct for AToE:** the soft indicators `ψ̃` are normalized (partition of
  unity). When a child spawns inside a parent's region, the parent's normalized weight there is reduced
  and the child's weight takes its place; a child copying the parent output therefore makes the total
  blended output continuous across the spawn (no jump). AToELeaves/ANT retire the parent leaf, so the
  child inheriting the parent's full output is likewise the continuous choice.

**Optional base reinitialization** (case 1a only): if `reinitialize_base_after_spawn`, call
`model.reinitialize_base()` (prints `[Reinit]`). **AToE additionally re-syncs `BatchedModels`** via
`sync_from_models`; AToELeaves/ANT have no batched container.

---

## 3. Phase 3 — base + experts together

The spawn epoch is also the Phase-1 → Phase-3 transition (`current_phase 1 → 3`). At the transition:

1. (case 1a) optional `reinitialize_base()` as above.
2. `active_cfg = cfg` (top-level config now governs).
3. Extend the loop: `total_epochs = epoch + phase3_epochs`.
4. **Recreate the optimizer + LR scheduler** from `active_cfg` over **all** params (base + every expert);
   reset `step_count = 0`; reset patience/best-loss. Prints `[3-Phase FIX] …` (optimizer name, betas,
   decay steps, warmup, total param count). This recreation is **unconditional** (no freeze branch).
5. Recompute the optimizer-switch schedule (see §4) and the patience window (see §5).

Because Phase 3 rebuilds the optimizer from `model.parameters()`, **every parameter is in the optimizer
exactly once** — there are no add-param-group steps and no freeze groups.

---

## 4. Optimizer lifecycle

| Moment | Action |
|---|---|
| Phase 1 start (case 1a) | build primary optimizer (or full-batch `lbfgs`/`ssbroyden`) over base params; build scheduler; `step_count = 0` |
| Phase 1 (case 1b) | no optimizer training; go straight to spawn → Phase 3 |
| Phase 3 transition | **recreate** optimizer + scheduler from top-level cfg over all params; `step_count = 0`; reset patience/best (`[3-Phase FIX]`) |
| `optimizer_switch_epoch` (Phase 3, if `optimizer_2` set) | build `optimizer_2` over all params; `lr_scheduler = None` (line-search optimizers); reset patience/best |

Optimizer factories: `_create_primary_optimizer` (Adam/SOAP), `_create_optimizer_by_name` (incl.
full-batch `lbfgs`/`ssbroyden`). Full-batch optimizers run without an LR scheduler.

---

## 5. LR scheduler lifecycle

`_create_lr_scheduler(optimizer, cfg, total_steps)`:

| `lr_schedule` | Decay scheduler | Warmup | Notes |
|---|---|---|---|
| `exponential` | `StepLR(step_size=lr_decay_steps, gamma=lr_decay_rate)` | optional `LinearLR(start_factor=lr_warmup_start_factor, total_iters=lr_warmup_steps)` composed via `SequentialLR` | most common |
| `cosine` | `CosineAnnealingLR(T_max=total_steps − lr_warmup_steps)` | optional, as above | |
| `none` | — | optional warmup only | returns `None` if no warmup and no decay |

- Scheduler is stepped once per mini-batch **except** for full-batch optimizers (skipped when
  `current_optimizer_name == 'LBFGS'`).
- `step_count` is reset to 0 at Phase-1 start and at the Phase-3 transition, so any configured Phase-3
  warmup re-applies from the start of Phase 3.

---

## 6. Patience / early stopping (only early-stop in the flow)

**Intent:** stop wasted training when Phase-3 loss plateaus — and only then.

- **Active window:** only when `current_phase == 3` **and** `patience_epochs > 0` **and** `epoch >=
  patience_start_epoch`.
  - If Phase 3 configures `optimizer_2`: `patience_start_epoch = switch_epoch` → patience is active **only
    after the optimizer switch** (the second optimizer).
  - Else: `patience_start_epoch =` the Phase-3 start epoch → active for all of Phase 3.
  - **Phase 1 never early-stops.**
- **Plateau test (min-delta):** an epoch counts as an improvement **only if**
  `train_loss < best_train_loss * (1 − patience_rel_delta)`. Otherwise `epochs_without_improvement += 1`.
  Stop (print `[EarlyStop]`) when `epochs_without_improvement >= patience_epochs` (after the `min_epochs`
  grace period within the active window). This replaces the old strict-`<` test, under which a loss
  creeping down by a negligible amount each epoch reset the counter forever and never stopped.
- **`min_epochs`:** a grace period measured **from `patience_start_epoch`** (within the active window),
  not a global epoch count.
- **Resets:** `best_train_loss`/`epochs_without_improvement` reset at the Phase-3 transition and at the
  optimizer switch, so each active window starts fresh.

---

## 7. Per-variant differences (summary)

| Aspect | AToE | AToELeaves | ANT |
|---|---|---|---|
| Nodes spawned | all accepted | leaves only | all accepted |
| Parent at spawn | stays active (additive, normalized) | retired from leaves | retired (leaf_status=False) |
| Experts in forward | all (via `BatchedModels`) | leaves (+ base) | leaves only |
| `reinitialize_base()` | resets base **+ re-syncs `BatchedModels`** | resets base | resets base |
| Expert arch source | base arch | base arch | parent hidden dim → output dim |
| `parent_weights` output | copied (continuous via normalization) | copied | copied |

---

## 8. Expected console markers (for verification)

- `[3-Phase] Phase 1 …` / `[3-Phase] Transitioning to Phase 3 …`
- `[3-Phase FIX] Phase 3 optimizer recreated: …` (+ betas/decay/warmup + total params)
- `[ParentInit] Expert k: hidden layers copied from <base|expert i>, output copied` (when
  `init.hidden == parent_weights`)
- `[Reinit] Base model reinitialized (… params)` (when `reinitialize_base_after_spawn` and case 1a)
- `[FullTree] Spawning complete. No further spawning steps.`
- `OPTIMIZER SWITCH: <opt1> -> <OPT2> at epoch …` (when `optimizer_2` configured)
- `[EarlyStop] …` (only in Phase 3, only after the switch if `optimizer_2` is set)
