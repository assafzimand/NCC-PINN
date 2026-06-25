# Coarse-to-Fine Additive Training Plan for the Tree-Structured MoE (AToE)

## Motivation

The collapse-to-vanilla behavior on easier PDEs is structural, not a tuning
issue. The original AToE normalizes **all** tree nodes (root + internal +
leaves) into a single partition of unity. Because the root window is
≈ 1 over the whole domain, that single normalization makes the experts
*compete* to represent the same quantity: either the root dominates (the model
degenerates to a single vanilla network) or experts at different levels average
their predictions of the same field, with no incentive to specialize.

The remedy is to make experts **add up** to the solution rather than compete for
it. The root expert learns the global, low-frequency solution; each finer tree
level learns a *correction* on top of the coarser composition, blended only
locally within that level. This is the multilevel domain-decomposition /
multiresolution principle: coarse levels carry global structure, fine levels
carry localized high-frequency detail, and the levels are summed additively.
This mirrors multilevel FBPINNs (Dolean, Heinlein, Mishra, Moseley, 2024),
Progressive Domain Decomposition (Luo et al., 2025), classical multigrid, and
the wavelet/multiresolution structure already implicit in the geometric-wavelet
M-term tree. With an additive correction hierarchy, specialization is the
default: once the coarse levels have absorbed the smooth part of the solution,
the only thing a leaf expert *can* do is represent the local detail that remains.

---

## Part 0 — Indicators and Windows (choose before implementing)

This part fixes two coupled issues with the per-level blending and presents two
options. It does **not** prescribe a single choice; it lays out the math of both
and gives a recommendation.

### The two issues per-level blending must handle

1. **Smooth transitions between sibling experts** within a level (the overlap
   collars), smooth to at least the PDE order so high-order derivatives of the
   composed solution stay bounded.
2. **The lone-child / partial-coverage problem.** With siblings removed (see
   Part C), a level may contain a single child over part of the parent region.
   Naive per-level normalization `Ψ_i / Σ_{j∈ℓ} Ψ_j` then gives
   `w_i = Ψ_i/Ψ_i = 1` everywhere the window is nonzero — the child becomes
   global and stops being a regional expert, and uncovered sub-regions yield
   `0/0`.

Both issues are solved by **(i)** a sufficiently smooth window and **(ii)** a
per-level normalization with a constant background term in the denominator.

### Per-level normalization with background term (used by both options)

Replacing the original all-nodes / leaves-only normalization, the per-level
weight for region `i` at level `ℓ` is

```
              Ψ_i(X)
w_i(X) = ─────────────────────
         1 + Σ_{j ∈ level ℓ} Ψ_j(X)
```

The constant `1` is a background slot (equivalently, the coarser-level / root
contribution). For a lone child this gives `w_i = Ψ_i/(1+Ψ_i) < 1`, which decays
as `Ψ_i` decays — so the child stays regional (near-full weight in its core,
fading to 0 outside) and the missing weight falls back to the coarser levels.
No phantom sibling and no `0/0`. This is the FBPINN partition-of-unity idea —
the denominator sums over the windows actually present, not a forced complete
set *(Moseley, Markham, Nissen-Meyer, 2021/2023; Dolean et al., 2024)*.

### Option 0a — Current sigmoid indicators + background normalization

Keep the existing product-of-sigmoids indicator,

```
Ψ_i(X) = Π_{j=1..d}  σ((x_j − a_{i,j}) / σ_{i,j}) · σ((b_{i,j} − x_j) / σ_{i,j}),
         σ_{i,j} = α (b_{i,j} − a_{i,j}),
```

and apply the background normalization above. This is the **minimal change**:
only the normalization is modified.

Caveats: the sigmoid window is C^∞ but **not compactly supported** — it is never
exactly zero, only small. So (a) the background `1+` term is *required* (there is
no exact-zero region to fall back on cleanly), and (b) its high-order
derivatives scale like `σ_{i,j}^{−k} ∝ (α·width)^{−k}`, which is the suspected
source of the explosion on high-order-derivative PDEs (KdV ∂³, etc.). There is
no value of α that is simultaneously sharp enough to specialize and smooth enough
for high-order autodiff.

### Option 0b — Compactly-supported windows (FBPINN-style)

Replace the sigmoid factor with a **compactly-supported** 1D window
`φ` (tensor-product across dimensions), exactly zero outside the region+collar:

```
Ψ_i(X) = Π_{j=1..d}  φ( (x_j − c_{i,j}) / h_{i,j} ),     φ(r) = 0 for |r| ≥ 1,
```

where `c_{i,j}`, `h_{i,j}` are the region center and half-width-plus-collar.
Choose `φ` with continuity matched to the PDE order:

- **Polynomial smoothstep** `S_N` on the ramp (C^N): `S_1 = 3r²−2r³` (C¹),
  `S_2 = 6r⁵−15r⁴+10r³` (C²), `S_3 = 35r⁴−84r⁵+70r⁶−20r⁷` (C³). Pick
  `N ≥` PDE order (use N = order or order+1 for margin).
- **Wendland RBF** `φ_{3,k}` (C^{2k}): `φ_{3,1}=(1−r)₊⁴(4r+1)` (C²),
  `φ_{3,2}=(1−r)₊⁶(35r²+18r+3)` (C⁴).

Apply the same background normalization. Because the window is exactly 0 outside
its support, this option natively handles smooth transitions (C^k decay at
support edges) **and** the inactive-outside-region behavior — uncovered
sub-regions are simply carried by coarser levels, no special casing. The `1+`
background term is still needed for the lone-child case (inside a lone window,
`Ψ/Ψ = 1` without it). Derives from FBPINN / multilevel FBPINN
*(Moseley et al., 2021/2023; Dolean et al., 2024)*; Wendland (1995);
Babuška–Melenk partition-of-unity FEM.

### Recommendation

Start with **0a** for the minimum implementation (Part A): smallest change, keeps
the current indicators, only adds the background term. Treat **0b** as the
principled upgrade — adopt it especially for the high-order-derivative PDEs,
where the sigmoid window's `(α·width)^{−k}` derivative growth is the likely cause
of divergence. 0b is the version that makes sibling-free partial refinement fully
native and citable to multilevel FBPINN.

---

## Composition

Let the tree be organized into levels `ℓ = 0, 1, …, L`, where level 0 is the
root. Using the per-level background normalization `w_i` from Part 0, the global
solution is the additive sum of the per-level blended fields:

```
u_θ(X) = u_root(X)
       + Σ_{i ∈ level 1} w_i(X) · u_i(X)
       + Σ_{i ∈ level 2} w_i(X) · u_i(X)
       + …
```

Each level's contribution is a piecewise field that equals one expert inside its
core region and transitions smoothly to its neighbors only in the collars — it
is **not** a global average. Because the levels are summed (not jointly
normalized), each level can only contribute the residual detail the coarser
levels left behind.

**Note on training the residual.** "Each level learns the solution minus the
previous levels" does not mean regressing against a subtracted target — in a
PINN there is no `u_true`, only the PDE residual. It means: extend the additive
sum with the new (trainable) level on top of the frozen composition, and
minimize the **same** PDE residual loss on the *full current composition*. The
new level's experts are the only trainable terms, so the physics loss drives them
to supply exactly the missing correction.

---

## Part A — Minimum Implementation

The smallest version that captures the structural fix. Sigmoid indicators with
background normalization (Option 0a), hard freeze after each level, no H
envelope, all loss terms (residual + BC + IC) active throughout on the full
composition.

**Step 1 — Train the root.** Train the single root expert to convergence on the
full PDE loss (residual + BC + IC). This establishes a stable global,
low-frequency solution and satisfies the boundary/initial conditions. *(Standard
PINN training, Raissi, Perdikaris, Karniadakis, 2019.)*

**Step 2 — Spawn the M-term tree.** Construct the geometric-wavelet tree via the
existing M-term-by-norm procedure. For AToE, **drop the sibling-retention rule**
(see Part C); keep only the nodes selected by wavelet norm plus their ancestors.
Organize the retained nodes into levels.

**Step 3 — Train level by level, coarse to fine.** For each level
`ℓ = 1, …, L`, top-down:
- Add level-ℓ correction experts to the composition. **Initialize each by
  copying its parent's weights and zeroing its output layer** (see init note
  below), so the child contributes exactly 0 at init and the composed solution
  equals the previous frozen composition.
- Apply the per-level background normalization `w_i` from Part 0.
- Minimize the **same** full PDE loss (residual + BC + IC) on the **full current
  composition** (root + all frozen levels + the new level). Only the level-ℓ
  experts are trainable.
- Train until a relative-improvement stopping criterion is met (allow early exit
  — on easy PDEs the residual may already be tiny after the root, in which case
  fine levels correctly have little to do; do not force a fixed epoch budget that
  makes idle experts fit noise).

This coarse-to-fine schedule is the multilevel-FBPINN / Progressive Domain
Decomposition philosophy *(Dolean et al., 2024; Luo et al., 2025)* and is
analogous to multigrid and curriculum learning. Introducing high-frequency /
high-derivative content only after the global field is stable also reduces the
risk of divergence.

**Expert initialization (Part A default).** When spawning a child expert, **copy
the parent's weights and zero the output layer** (or zero an output gate).
Copying the parent puts the child in the right function regime (it "speaks the
same language" as the region it corrects); zeroing the output makes its
contribution exactly zero at init, so adding a level never perturbs the
converged composition — this is what makes "freeze, then add a level" safe,
particularly for the high-order-derivative PDEs. This is the zero-initialized
residual idea of PirateNet *(Wang, Sankaran, Wang, Perdikaris, 2024)*; the
parent-warm-start half echoes coarse-to-fine / progressive growing. (Lighter
variant if copying is awkward: random body + zero output gate. Avoid: fresh
random init with nonzero output — a random child perturbs the converged
composition and can destabilize high-order PDEs.)

**Step 4 — Hard-freeze each completed level.** After level ℓ converges, freeze
all its parameters and proceed to level ℓ+1. Freezing during the staged build
keeps training stable: each new level optimizes against a fixed background.

**Step 5 — Final joint fine-tuning.** After all levels are built, unfreeze the
entire stack and train all experts together on the full PDE loss for a final
pass. This is part of the minimum implementation, not an add-on: staged freezing
leaves a calibration gap because each coarse level converged against the true
solution rather than against "true minus the corrections added later." The joint
pass closes that gap. This two-phase pattern — staged construction followed by
global refinement — is standard in multilevel/multigrid training and in
progressive/coarse-to-fine PINN schemes *(Dolean et al., 2024; Luo et al.,
2025)*.

### Optional stabilizers (independent of the staged structure)

Orthogonal to the level-by-level mechanism; enable as needed for the high-order
PDEs without changing the flow:
- **Double (float64) precision** for stable high-order autodiff.
- **Learning-rate annealing / gradient balancing** across loss terms
  *(Wang, Teng, Perdikaris, 2021)*.
- **Causal training** along the time axis for time-dependent problems
  *(Wang, Sankaran, Perdikaris, 2022)*.
- **Second-stage optimizer** (e.g. Adam → L-BFGS) per level and in the final
  joint pass *(Kiyani et al., 2025)*.

---

## Part B — Add-ons

Layered on top of Part A once it is working. Each is independently optional.

**B1 — Low learning rate instead of hard freeze.** Replace the hard freeze in
Step 4 with a small (nonzero) learning rate on completed levels, so coarse levels
can make minor adjustments as finer corrections come online instead of being
fully locked until the final joint pass. Closer to how multilevel methods couple
levels *(Dolean et al., 2024)*. Trade-off: more coupling can reduce stability, so
keep the rate well below the active level's.

**B2 — Boundary/initial-condition envelope H.** Multiply each level's blended
contribution (the root **excluded**) by a single shared smooth envelope `H(X)`
that is 1 over the interior and decays smoothly to 0 near the domain boundary and
at `t = 0`:

```
u_θ(X) = u_root(X) + H(X) · Σ_ℓ ( Σ_{i ∈ ℓ} w_i(X) · u_i(X) )
```

Because `H` multiplies each level term **after** normalization (outside the
ratio), it genuinely drives the corrections' contribution to zero on the
boundary — unlike shrinking the raw indicators, which normalization would cancel
(`ε_i / Σ_j ε_j` stays O(1)). All per-level PU properties remain intact
internally. Effect: BC/IC become the root's responsibility by construction, so
the fine levels are not dragged into low-frequency BC cleanup (a subtle
re-introduction of the collapse problem, and especially wasteful on easy PDEs).
Requirements: `H` must be smooth to at least the PDE order (use the same
smoothstep / Wendland family as the windows — a C¹ taper reintroduces seam
blow-up at the boundary), and `H` must **not** multiply the root. This is a soft
architectural bias toward correct BC/IC, not a hard constraint, consistent with
how domain-decomposition PINNs handle BCs via the loss on the full composition
*(Moseley et al., 2021/2023; Dolean et al., 2024)*.

**B3 — Trainable correction gate with a turn-on schedule.** Part A already
zero-initializes the child output. As an optional refinement, make the output
gate a *trainable* scalar (still initialized to 0) and/or anneal a turn-on
schedule, so the rate at which each correction activates is learned/controlled
rather than fixed. Same zero-init residual-gate lineage as PirateNet
*(Wang, Sankaran, Wang, Perdikaris, 2024)*.

---

## Part C — Variant-specific notes (sibling retention)

Sibling retention is **variant-dependent**, not a global choice. The M-term
construction currently keeps the siblings of selected nodes to form a complete
binary tree; whether that is needed depends on the composition variant.

**AToE.** **Remove siblings.** In the additive coarse-to-fine scheme a complete
tree is not required: regions the M-term criterion did not select (low local
complexity) simply receive no correction and are carried by coarser levels — this
is strictly more faithful to the M-term selection. The lone-child / partial-
coverage issue this creates is handled by the **background-term normalization**
of Part 0 (`w_i = Ψ_i/(1 + Σ_ℓ Ψ_j)`), which keeps a lone child regional instead
of `w = 1` everywhere. With compactly-supported windows (Option 0b) this is the
native multilevel-FBPINN partial-refinement setup.

**AToE-Leaves.** **Keep siblings.** Leaves-only blending requires the leaves to
*tile* the domain (a genuine cover), so completeness at the **leaf frontier** is
needed — every point must lie in some leaf core, or there are holes. (Note: this
is leaf-frontier completeness, not internal-sibling completeness.) Because the
leaves already form a cover, AToE-Leaves can **train the leaf layer directly**,
without the coarse-to-fine staged build — the staged additive schedule is mainly
an AToE construct.

**ANT.** **Keep siblings.** ANT passes each parent's last-hidden activation into
its children, so the tree must be a proper complete binary routing structure
where every node either predicts (leaf) or feeds its children's inner layers. A
node with one child but no sibling breaks the "predict-or-route" invariant.
Consistent with the routing-tree structure of Adaptive Neural Trees
*(Tanno et al., 2019)*.

---

## Summary of the flow

```
Part 0 (choose indicators):
  0a. Current sigmoids + background normalization  w = Ψ / (1 + Σ_ℓ Ψ).   [minimal]
  0b. Compactly-supported windows (smoothstep/Wendland), same background.  [upgrade]
      Recommendation: 0a to start; 0b for high-order PDEs.

Part A (minimum):
  1. Train root to convergence (full PDE loss).
  2. Build M-term tree; AToE = no siblings; organize into levels.
  3. For ℓ = 1..L (coarse → fine):
        add level-ℓ experts (copy parent weights, zero output layer),
        per-level background normalization,
        minimize full PDE loss on full composition (only level ℓ trainable),
        stop on relative-improvement criterion.
  4. Hard-freeze level ℓ, continue.
  5. Final joint fine-tune: unfreeze all, train together.
  (+ optional stabilizers: float64, LRA, causal, 2nd-stage optimizer)

Part B (add-ons):
  B1. Low LR on completed levels instead of hard freeze.
  B2. Shared smooth envelope H on the correction levels (not the root).
  B3. Trainable zero-init correction gate with turn-on schedule.

Part C (sibling retention, per variant):
  AToE        -> remove siblings (background normalization handles lone child).
  AToE-Leaves -> keep siblings (leaves must tile domain); train leaf layer directly.
  ANT         -> keep siblings (complete binary routing tree required).
```

## References

- Raissi, Perdikaris, Karniadakis (2019) — Physics-Informed Neural Networks.
- Wang, Teng, Perdikaris (2021) — gradient pathologies / learning-rate annealing.
- Wang, Sankaran, Perdikaris (2022) — causal training for time-dependent PINNs.
- Moseley, Markham, Nissen-Meyer (2021/2023) — Finite Basis PINNs (FBPINN);
  overlapping-subdomain partition-of-unity windows (compact support).
- Dolean, Heinlein, Mishra, Moseley (2024) — multilevel FBPINNs; additive
  per-level partition of unity, coarse-to-fine composition, partial refinement.
- Luo et al. (2025) — Progressive Domain Decomposition (PDD).
- Wang, Sankaran, Wang, Perdikaris (2024) — PirateNet; zero-initialized adaptive
  residual gates.
- Kiyani et al. (2025) — optimizer study for PINNs (second-stage / quasi-Newton).
- Wendland (1995) — compactly-supported positive-definite RBFs (C^{2k}).
- Babuška, Melenk (1997) — partition-of-unity finite element method.
- Tanno, Arulkumaran, Alexander, Criminisi, Nori (2019) — Adaptive Neural Trees
  (the gating/routing mechanism the original AToE indicators follow).

*Reference details (years, venues, exact mechanisms) should be confirmed against
the primary sources before citing in the paper.*
