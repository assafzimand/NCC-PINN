"""Per-expert split loss for PDD-style subdomain training.

Each expert is trained on its OWN output (no PoU), with:
  - PDE residual inside its subdomain
  - Dirichlet matching to frozen composed model on interior
    faces (interface)
  - True IC/BC on faces coinciding with global domain bounds

For Allen-Cahn with periodic BC, global boundary points are
paired across experts (left/right at same t), penalizing both
value and spatial derivative mismatches.

Total loss = SUM over experts of:
    w_res*L_res + w_ic*L_ic + w_bc*L_bc
where interface faces inherit the IC or BC weight by face type.
"""

import torch
import importlib
from typing import Dict, Callable
from adaptive.subdomain_data import (
    KIND_RESIDUAL, KIND_IC_TRUE, KIND_INTERFACE, KIND_INTERFACE_BC, KIND_BC_TRUE,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)


def build_split_loss(
    model,
    cfg: Dict,
    *,
    variant: str,
    orig_loss_fn: Callable = None,
) -> Callable:
    """Build a split loss for per-expert subdomain training.

    Returns a callable ``loss_fn(model, batch)`` compatible
    with ``_train_segment``.
    
    If ``orig_loss_fn`` is provided, batches missing split-specific
    keys (expert_id, kind) will fall back to the original loss
    (used for eval batches).
    """
    problem = cfg['problem']
    pc = cfg[problem]
    loss_weights = pc['loss_weights']
    w_res = loss_weights['residual']
    w_ic = loss_weights['ic']
    w_bc = loss_weights['bc']

    # Allen-Cahn uses periodic BC pairing, so skip Dirichlet-to-zero
    is_allen_cahn = (problem == 'allen_cahn')

    pde_res_fn, deriv_fn = _import_pde_helpers(problem)
    pde_params = _get_pde_params(problem, pc)

    per_expert_history: Dict[int, Dict[str, list]] = {}

    def split_loss_fn(
        model, batch, return_components=False, **kw
    ):
        # Fix 1: Fall back to original loss if batch lacks split keys (eval batches)
        if 'expert_id' not in batch or 'kind' not in batch:
            if orig_loss_fn is not None:
                return orig_loss_fn(model, batch, return_components=return_components, **kw)
            else:
                # No fallback available, compute simple MSE
                x = batch['x']
                t = batch['t']
                h_gt = batch['h_gt']
                device = x.device
                xt = torch.cat([x, t], dim=1)
                h_pred = model(xt)
                return torch.mean((h_pred - h_gt) ** 2)
        
        x = batch['x']
        t = batch['t']
        h_gt = batch['h_gt']
        expert_ids = batch['expert_id']
        kinds = batch['kind']
        bc_face_ids = batch.get('bc_face_id', None)
        device = x.device

        unique_experts = expert_ids.unique().tolist()
        total_loss = torch.tensor(0.0, device=device)
        all_comps = {}

        for eidx in unique_experts:
            emask = (expert_ids == eidx)
            comps = _compute_expert_loss(
                model, eidx,
                x[emask], t[emask], h_gt[emask],
                kinds[emask],
                pde_res_fn, deriv_fn, pde_params,
                w_res, w_ic, w_bc, is_allen_cahn,
                variant, device,
            )
            total_loss = total_loss + comps['total']
            _record(per_expert_history, eidx, comps)
            if return_components:
                all_comps[eidx] = comps

        # ── Periodic BC (Allen-Cahn): cross-expert pairing ──
        if is_allen_cahn and bc_face_ids is not None:
            bc_loss_contrib = _compute_periodic_bc_loss(
                model, x, t, expert_ids, kinds,
                bc_face_ids, deriv_fn, device,
            )
            if bc_loss_contrib.item() > 0:
                logger.debug(
                    f"[SplitLoss] Periodic BC contrib: "
                    f"{bc_loss_contrib.item():.6e}"
                )
            total_loss = total_loss + w_bc * bc_loss_contrib

        if return_components:
            return all_comps
        return total_loss

    split_loss_fn._per_expert_history = per_expert_history
    split_loss_fn._variant = variant
    return split_loss_fn


def _compute_expert_loss(
    model, expert_idx, x, t, h_gt, kinds,
    pde_res_fn, deriv_fn, pde_params,
    w_res, w_ic, w_bc, is_allen_cahn, variant, device,
):
    """Per-expert local loss (no PoU)."""
    z = torch.tensor(0.0, device=device)
    comps = {
        'residual': z.clone(),
        'ic': z.clone(),
        'interface_ic': z.clone(),
        'interface_bc': z.clone(),
        'bc': z.clone(),
    }

    # ── Residual ──
    rmask = (kinds == KIND_RESIDUAL)
    if rmask.sum() > 0:
        xf = x[rmask].clone().detach().requires_grad_(True)
        tf = t[rmask].clone().detach().requires_grad_(True)
        xt = torch.cat([xf, tf], dim=1)
        u_j = model.forward_single_expert(expert_idx, xt)
        hf = u_j[:, 0]
        ht, hx, hxx = deriv_fn(hf, xf, tf)
        res = pde_res_fn(hf, ht, hx, hxx, **pde_params)
        comps['residual'] = torch.mean(res ** 2)

    # ── IC true (real t=0) ──
    ic_mask = (kinds == KIND_IC_TRUE)
    if ic_mask.sum() > 0:
        xt_ic = torch.cat(
            [x[ic_mask], t[ic_mask]], dim=1
        )
        u_ic = model.forward_single_expert(
            expert_idx, xt_ic
        )
        comps['ic'] = torch.mean(
            (u_ic - h_gt[ic_mask]) ** 2
        )

    # ── Interface IC (t-face interior boundary → w_ic) ──
    ifm_ic = (kinds == KIND_INTERFACE)
    if ifm_ic.sum() > 0:
        xt_if = torch.cat(
            [x[ifm_ic], t[ifm_ic]], dim=1
        )
        u_if = model.forward_single_expert(
            expert_idx, xt_if
        )
        comps['interface_ic'] = torch.mean(
            (u_if - h_gt[ifm_ic]) ** 2
        )

    # ── Interface BC (x-face interior boundary → w_bc) ──
    ifm_bc = (kinds == KIND_INTERFACE_BC)
    if ifm_bc.sum() > 0:
        xt_if_bc = torch.cat(
            [x[ifm_bc], t[ifm_bc]], dim=1
        )
        u_if_bc = model.forward_single_expert(
            expert_idx, xt_if_bc
        )
        comps['interface_bc'] = torch.mean(
            (u_if_bc - h_gt[ifm_bc]) ** 2
        )

    # ── BC true: Dirichlet (for non-periodic problems only) ──
    # Fix 2: Allen-Cahn uses periodic BC pairing at batch level, skip here
    bc_mask = (kinds == KIND_BC_TRUE)
    if (not is_allen_cahn) and bc_mask.sum() > 0:
        xt_bc = torch.cat(
            [x[bc_mask], t[bc_mask]], dim=1
        )
        u_bc = model.forward_single_expert(
            expert_idx, xt_bc
        )
        comps['bc'] = torch.mean(
            (u_bc - h_gt[bc_mask]) ** 2
        )

    comps['total'] = (
        w_res * comps['residual']
        + w_ic * (comps['ic'] + comps['interface_ic'])
        + w_bc * (comps['interface_bc'] + comps['bc'])
    )
    return comps


def _compute_periodic_bc_loss(
    model, x, t, expert_ids, kinds, bc_face_ids, deriv_fn, device,
):
    """Compute periodic BC loss for Allen-Cahn (cross-expert pairing).
    
    Pairs left/right boundary points by sorting on t-value,
    penalizes (u_left - u_right)² + (∂u/∂x_left - ∂u/∂x_right)².
    
    Since left and right sides share the same t-samples (per dimension),
    sorting by t ensures we pair points at matching times.
    
    Vectorized by grouping points by (expert_left, expert_right) pairs
    to minimize forward passes and autograd calls.
    """
    bc_mask = (kinds == KIND_BC_TRUE)
    if bc_mask.sum() == 0:
        return torch.tensor(0.0, device=device)
    
    x_bc = x[bc_mask]
    t_bc = t[bc_mask]
    eid_bc = expert_ids[bc_mask]
    fid_bc = bc_face_ids[bc_mask]
    
    # Group by dimension (face_id // 2)
    dims = fid_bc // 2
    sides = fid_bc % 2
    
    unique_dims = dims.unique().tolist()
    total_bc_loss = torch.tensor(0.0, device=device)
    n_pairs = 0
    
    for d in unique_dims:
        d_mask = (dims == d)
        x_d = x_bc[d_mask]
        t_d = t_bc[d_mask]
        eid_d = eid_bc[d_mask]
        side_d = sides[d_mask]
        
        # Separate left (side=0) and right (side=1)
        left_mask = (side_d == 0)
        right_mask = (side_d == 1)
        
        n_left = left_mask.sum().item()
        n_right = right_mask.sum().item()
        
        if n_left == 0 or n_right == 0:
            continue
        
        # Extract left and right data
        x_left_all = x_d[left_mask]
        t_left_all = t_d[left_mask]
        eid_left_all = eid_d[left_mask]
        
        x_right_all = x_d[right_mask]
        t_right_all = t_d[right_mask]
        eid_right_all = eid_d[right_mask]
        
        # Sort both sides by t-value for matching
        t_left_vals = t_left_all[:, 0]
        t_right_vals = t_right_all[:, 0]
        sort_left = torch.argsort(t_left_vals)
        sort_right = torch.argsort(t_right_vals)
        
        n_match = min(n_left, n_right)
        
        x_left = x_left_all[sort_left[:n_match]]
        t_left = t_left_all[sort_left[:n_match]]
        eid_left = eid_left_all[sort_left[:n_match]]
        
        x_right = x_right_all[sort_right[:n_match]]
        t_right = t_right_all[sort_right[:n_match]]
        eid_right = eid_right_all[sort_right[:n_match]]
        
        # Vectorized evaluation: group by (expert_left, expert_right) pairs
        # Create pair keys for grouping
        pair_keys = eid_left * 10000 + eid_right  # assumes < 10000 experts
        unique_pairs = pair_keys.unique().tolist()
        
        for pair_key in unique_pairs:
            pair_mask = (pair_keys == pair_key)
            eid_l = pair_key // 10000
            eid_r = pair_key % 10000
            
            # Batch all points with this expert pair
            x_l_batch = x_left[pair_mask].clone().detach()
            x_l_batch.requires_grad_(True)
            t_l_batch = t_left[pair_mask].clone().detach()
            x_r_batch = x_right[pair_mask].clone().detach()
            x_r_batch.requires_grad_(True)
            t_r_batch = t_right[pair_mask].clone().detach()
            
            xt_l = torch.cat([x_l_batch, t_l_batch], dim=1)
            xt_r = torch.cat([x_r_batch, t_r_batch], dim=1)
            
            # Single batched forward pass per expert
            u_l = model.forward_single_expert(eid_l, xt_l)[:, 0]
            u_r = model.forward_single_expert(eid_r, xt_r)[:, 0]
            
            # Batched spatial derivative computation
            ux_l = torch.autograd.grad(
                u_l, x_l_batch,
                grad_outputs=torch.ones_like(u_l),
                create_graph=True, retain_graph=True,
            )[0][:, d]
            
            ux_r = torch.autograd.grad(
                u_r, x_r_batch,
                grad_outputs=torch.ones_like(u_r),
                create_graph=True, retain_graph=True,
            )[0][:, d]
            
            # Periodic penalty (vectorized sum)
            total_bc_loss = (
                total_bc_loss
                + torch.sum((u_l - u_r) ** 2)
                + torch.sum((ux_l - ux_r) ** 2)
            )
            n_pairs += pair_mask.sum().item()
    
    if n_pairs > 0:
        return total_bc_loss / n_pairs
    return torch.tensor(0.0, device=device)


def _record(history, expert_idx, comps):
    if expert_idx not in history:
        history[expert_idx] = {
            k: [] for k in [
                'residual', 'ic', 'interface_ic',
                'interface_bc', 'bc', 'total',
            ]
        }
    for k in history[expert_idx]:
        val = comps[k]
        history[expert_idx][k].append(
            val.item() if torch.is_tensor(val) else val
        )


def _import_pde_helpers(problem: str):
    """Import problem-specific PDE residual + derivatives."""
    mod = importlib.import_module(f'losses.{problem}_loss')
    return mod.pde_residual, mod.compute_derivatives


def _get_pde_params(problem: str, pc: Dict) -> Dict:
    """PDE-specific kwargs for the residual function."""
    if problem == 'allen_cahn':
        return {'D': pc['D']}
    if problem == 'burgers1d':
        return {'nu': pc['nu']}
    if problem == 'kdv':
        return {'mu': pc['mu']}
    if problem == 'ks':
        return {
            'alpha': pc.get('alpha', 1.0),
            'beta': pc.get('beta', 1.0),
            'gamma': pc.get('gamma', 1.0),
        }
    return {}
