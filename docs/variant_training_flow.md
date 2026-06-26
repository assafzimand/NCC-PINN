# Per-Variant Training Flow and Loss

This note specifies, for each variant, **what is trained when, what is frozen, and
what loss is minimized at each stage**. It assumes the indicators $\Psi_{\Omega_i}$,
the per-variant normalization $w_i$ / $\tilde\Psi_i$, and the composition formulas
are already defined. Throughout, the standard PINN loss on a field $u$ over a point
set $\mathcal{S}$ is

$$
\mathcal{L}[u;\mathcal{S}]
= \mathcal{L}_F[u;\mathcal{S}_F]
+ \mathcal{L}_{IC}[u;\mathcal{S}_{IC}]
+ \mathcal{L}_{BC}[u;\mathcal{S}_{BC}],
$$

with PDE-residual, initial-condition, and boundary-condition terms

$$
\mathcal{L}_F[u;\mathcal{S}_F]=\frac{1}{|\mathcal{S}_F|}\sum_{x\in\mathcal{S}_F}\big|F(u)(x)\big|^2,\quad
\mathcal{L}_{IC}[u;\mathcal{S}_{IC}]=\frac{1}{|\mathcal{S}_{IC}|}\sum_{x\in\mathcal{S}_{IC}}\big|u(x)-u_0(x)\big|^2,\quad
\mathcal{L}_{BC}[u;\mathcal{S}_{BC}]=\frac{1}{|\mathcal{S}_{BC}|}\sum_{x\in\mathcal{S}_{BC}}\big|u(x)-g(x)\big|^2 .
$$

The variants differ in **which field** enters the loss (the full composition vs. a
local expert), **which points** $\mathcal{S}$ are used (global vs. a subdomain), and
**where the IC/BC targets come from** (the true problem data vs. a frozen earlier
network's prediction).

---

## AToE — Additive, staged, no boundary handoff

AToE trains coarse-to-fine with the previous state kept **live in the composition**,
so coupling is implicit and there is no explicit boundary handoff. **Root phase:**
train the base expert $u_0$ on the full global PINN loss
$\mathcal{L}[u_0;\mathcal{S}]$ (true IC/BC targets) until convergence; then build the
$M$-term tree and group experts by level $\ell=1,\dots,L$. **Per-level phase
($\ell=1,\dots,L$, coarse to fine):** spawn level-$\ell$ experts (parent hidden
weights copied, output zeroed, so each contributes $0$ at spawn), and **freeze the
base and all levels $\ell'<\ell$**; only level-$\ell$ parameters
$\theta_\ell=\{\theta_i: \text{level}(i)=\ell\}$ are trainable. The loss is the
**same full global PINN loss evaluated on the entire current composition**,

$$
u^{(\ell)}_\theta(x)= u_0(x)+\sum_{\ell'=1}^{\ell}\;\sum_{i:\text{level}(i)=\ell'} w_i(x)\,u_i(x),
\qquad
\min_{\theta_\ell}\;\mathcal{L}\big[u^{(\ell)}_\theta;\,\mathcal{S}\big],
$$

with frozen levels contributing but not receiving gradients. Because the residual
$F(u^{(\ell)}_\theta)$ is already small wherever the frozen levels succeeded,
minimizing it w.r.t. $\theta_\ell$ forces the new experts to supply exactly the
remaining local error — the previous state acts as a fixed additive background, **not**
as a boundary target. **Final phase:** unfreeze all parameters and minimize
$\mathcal{L}[u^{(L)}_\theta;\mathcal{S}]$ jointly at a lower learning rate to remove the
staged-freezing calibration gap. No IC/BC handoff is used anywhere; smoothness at
region edges is provided geometrically by the compact windows.

---

## AToE-Leaves — parallel leaves, root supplies each leaf's IC/BC

AToE-Leaves trains the leaf experts **independently and in parallel**, each confined to
its own subdomain, using the **frozen root's prediction as the IC/BC target** on that
subdomain's boundary. **Root phase:** identical to AToE — train $u_0$ on
$\mathcal{L}[u_0;\mathcal{S}]$ to convergence; build the $M$-term tree with siblings
retained so the leaves $\mathcal{L}$ tile the domain. **Leaf phase:** freeze $u_0$;
spawn all leaves (parent hidden **and** output weights copied, since the root retires
from the blend). Each leaf $u_j$, $j\in\mathcal{L}$, is trained on its **own subdomain
points** $\mathcal{S}\cap\Omega_j^{+}$ (the region plus its collar), with the PDE
residual inside the region and the IC/BC taken from the **frozen root** on the part of
$\partial\Omega_j$ interior to the global domain:

$$
\min_{\theta_j}\;
\underbrace{\mathcal{L}_F\big[u_j;\,\mathcal{S}_F\cap\Omega_j^{+}\big]}_{\text{local PDE residual}}
\;+\;
\underbrace{\lambda\,\frac{1}{|\partial\Omega_j|}\!\!\sum_{x\in\partial\Omega_j\setminus\partial\Omega}\!\!\big|u_j(x)-u_0(x)\big|^2}_{\text{root-supplied subdomain BC}}
\;+\;
\underbrace{\mathcal{L}_{IC}\!+\!\mathcal{L}_{BC}\ \text{(true, where $\Omega_j$ meets $\partial\Omega$ or $t=0$)}}_{\text{true problem data on real boundaries}} .
$$

Leaves share no parameters and can be optimized fully in parallel. The trained leaves
are then assembled through the leaf-only partition of unity. Because the leaves were
fit independently, neighboring leaves agree only approximately in the collars; an
**optional joint fine-tuning stage** — unfreeze all leaves and minimize the global
$\mathcal{L}[u_\theta;\mathcal{S}]$ on the blended composition — removes any residual
seam left by the partition-of-unity blend.

---

## ANT — parallel leaf frontier, parent's prediction supplies each child's IC/BC

ANT trains the **current leaf frontier** $\mathcal{F}_\ell$ in parallel, with each child
using its **direct frozen parent's saved prediction** as the IC/BC target (the
PDD-style "previous network as boundary," sourced from the parent rather than the
root). Unlike AToE-Leaves, experts are not independent: each child consumes its
parent's last-hidden activation $A_{\text{parent}(i)}$ as input, so **frozen ancestors
are still evaluated in the forward pass** to route activations to active descendants.
**Root phase:** train $u_0$ on $\mathcal{L}[u_0;\mathcal{S}]$; build the $M$-term tree
with siblings retained (complete tree, so no hidden layer is simultaneously a router
and an output). **Per-stage phase ($\ell=1,\dots,L$):** the frontier $\mathcal{F}_\ell$
advances one level; the children at the new frontier are trainable, all ancestors are
**frozen** (their parameters fixed, hence their predictions and routed activations are
fixed). Each child $u_j\in\mathcal{F}_\ell$ has its output layer copied from its parent
at spawn (so it starts from the parent's prediction), and is trained on its subdomain
with the PDE residual inside the region and the **frozen parent's prediction**
$u^{*}_{\text{parent}(j)}$ as the IC/BC target on the child's interior boundary:

$$
\min_{\theta_j}\;
\mathcal{L}_F\big[u^{*}_j;\,\mathcal{S}_F\cap\Omega_j^{+}\big]
\;+\;
\lambda\,\frac{1}{|\partial\Omega_j|}\!\!\sum_{x\in\partial\Omega_j\setminus\partial\Omega}\!\!\big|u^{*}_j(x)-u^{*}_{\text{parent}(j)}(x)\big|^2
\;+\;
\big(\mathcal{L}_{IC}+\mathcal{L}_{BC}\ \text{true, where $\Omega_j$ meets $\partial\Omega$ or $t=0$}\big),
$$

where $u^{*}_j$ is the child's prediction obtained from the forward pass that feeds
$A_{\text{parent}(j)}$ into $u_j$. Children sharing a parent can be trained in parallel
(parent fixed); across the frontier, the active experts are blended through the
leaf-frontier partition of unity $\tilde\Psi_j$ over $\mathcal{F}_\ell$. When the
frontier advances, the just-trained children freeze and become the parents/routers for
the next stage. As in AToE-Leaves, an optional final joint pass over the active
frontier can absorb partition-of-unity seams.

---

## Summary

| Variant | Trained unit | Frozen | Loss field | Points | IC/BC target source |
|---|---|---|---|---|---|
| **AToE** | one level at a time | base + lower levels (live in composition) | full composition $u^{(\ell)}_\theta$ | global $\mathcal{S}$ | none (implicit, additive background) |
| **AToE-Leaves** | all leaves, parallel | root (retired) | local expert $u_j$ | subdomain $\Omega_j^{+}$ | frozen **root** prediction $u_0$ |
| **ANT** | leaf frontier, parallel | ancestors (route activations) | local expert $u^{*}_j$ | subdomain $\Omega_j^{+}$ | frozen **parent** prediction $u^{*}_{\text{parent}(j)}$ |

The common thread for the split variants (Leaves, ANT) is that a **frozen earlier
network's prediction provides the subdomain IC/BC target** — the root for Leaves, the
direct parent for ANT — whereas AToE couples implicitly by training each level on the
full additive composition with no boundary handoff.
