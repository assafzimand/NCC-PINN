"""Per-expert subdomain data builder for split IC/BC training.

Builds a combined training dataset where each point is tagged with the
owning expert, its kind (residual / ic_true / interface / bc_true),
and bc_face_id for periodic BC pairing.

For bc_true points on global spatial boundaries, bc_face_id encodes
dim*2 + side (side=0 lower, side=1 upper) to enable cross-expert
pairing in periodic BC loss (Allen-Cahn).

Used by the split-loss training path for AToE-Leaves and ANT.
"""

import torch
from typing import Dict, List
from adaptive.indicators import RegionDescriptor  # noqa: F401
from utils.dataset_gen import _analytic_ic
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Integer codes stored in the ``kind`` tensor
KIND_RESIDUAL = 0
KIND_IC_TRUE = 1
KIND_INTERFACE = 2      # t-face interface (weighted by w_ic)
KIND_INTERFACE_BC = 3   # x-face interface (weighted by w_bc)
KIND_BC_TRUE = 4

KIND_NAMES = {
    KIND_RESIDUAL: 'residual',
    KIND_IC_TRUE: 'ic_true',
    KIND_INTERFACE: 'interface_ic',
    KIND_INTERFACE_BC: 'interface_bc',
    KIND_BC_TRUE: 'bc_true',
}


def build_subdomain_data(
    model: torch.nn.Module,
    new_expert_indices: List[int],
    regions,
    cfg: Dict,
    device: torch.device,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Build per-expert dataset for split-loss training.

    Returns dict with keys ``x``, ``t``, ``h_gt``,
    ``expert_id``, ``kind``, ``bc_face_id``.
    
    ``bc_face_id`` encodes which spatial boundary face
    for periodic pairing: ``dim * 2 + side`` where
    side=0 for lower, side=1 for upper.
    
    For periodic BC, left (side=0) and right (side=1) points
    on the same dimension share identical t-values to enable
    cross-expert pairing.
    """
    torch.manual_seed(seed)

    problem = cfg['problem']
    pc = cfg[problem]
    spatial_dim = pc['spatial_dim']
    spatial_domain = pc['spatial_domain']
    temporal_domain = pc['temporal_domain']
    t_min_global = temporal_domain[0]
    t_max_global = temporal_domain[1]
    output_dim = pc['output_dim']

    sampling = cfg.get('sampling', {})
    n_res_total = sampling.get('n_residual_train', 4096)
    ic_ratio = sampling.get('initial_train_ratio', 0.026)
    bc_ratio = sampling.get('boundary_train_ratio', 0.026)
    n_ic_per_face = max(1, int(round(n_res_total * ic_ratio)))
    n_bc_per_face = max(1, int(round(n_res_total * bc_ratio)))

    num_experts = len(new_expert_indices)
    if num_experts == 0:
        return _empty(spatial_dim, output_dim, device)

    # Fix 3+4: Generate shared t-values per dimension (full global range)
    # Each expert will filter to its own temporal range
    bc_t_global = {}
    for d in range(spatial_dim):
        bc_t_global[d] = (
            torch.rand(n_bc_per_face, 1, device=device)
            * (t_max_global - t_min_global) + t_min_global
        )

    # ── Residual: global uniform, filter into regions ──
    x_g = torch.zeros(n_res_total, spatial_dim, device=device)
    t_g = torch.zeros(n_res_total, 1, device=device)
    for d in range(spatial_dim):
        lo, hi = spatial_domain[d]
        x_g[:, d] = (torch.rand(n_res_total, device=device)
                      * (hi - lo) + lo)
    t_g[:, 0] = (torch.rand(n_res_total, device=device)
                  * (t_max_global - t_min_global) + t_min_global)

    xs, ts, gs, eids, ks, bc_fids = [], [], [], [], [], []

    for eidx in new_expert_indices:
        region = regions[eidx]
        bl, bu = region.bounds_lower, region.bounds_upper

        mask = torch.ones(
            n_res_total, dtype=torch.bool, device=device
        )
        for d in range(spatial_dim):
            mask &= ((x_g[:, d] >= bl[d])
                      & (x_g[:, d] <= bu[d]))
        mask &= ((t_g[:, 0] >= bl[spatial_dim])
                  & (t_g[:, 0] <= bu[spatial_dim]))

        n = mask.sum().item()
        if n > 0:
            xs.append(x_g[mask])
            ts.append(t_g[mask])
            gs.append(torch.zeros(n, output_dim, device=device))
            eids.append(torch.full(
                (n,), eidx, dtype=torch.long, device=device
            ))
            ks.append(torch.full(
                (n,), KIND_RESIDUAL, dtype=torch.long,
                device=device
            ))
            bc_fids.append(torch.full(
                (n,), -1, dtype=torch.long, device=device
            ))

    # ── IC / BC faces per expert ──
    for eidx in new_expert_indices:
        region = regions[eidx]
        _add_ic_face(
            eidx, region, spatial_dim, spatial_domain,
            t_min_global, n_ic_per_face, output_dim,
            problem, pc, device, xs, ts, gs, eids, ks,
            bc_fids,
        )
        _add_bc_faces_periodic(
            eidx, region, spatial_dim, spatial_domain,
            n_bc_per_face, output_dim, device,
            xs, ts, gs, eids, ks, bc_fids,
            bc_t_global,
        )

    # ── Mint interface targets from frozen composed model ──
    x_cat = torch.cat(xs, dim=0)
    t_cat = torch.cat(ts, dim=0)
    h_gt_cat = torch.cat(gs, dim=0)
    eid_cat = torch.cat(eids, dim=0)
    kind_cat = torch.cat(ks, dim=0)
    bc_fid_cat = torch.cat(bc_fids, dim=0)

    # Mint interface targets from frozen composed model
    # t-face interfaces (KIND_INTERFACE, weighted by w_ic)
    iface_mask = (kind_cat == KIND_INTERFACE)
    if iface_mask.sum() > 0:
        with torch.no_grad():
            xt_if = torch.cat(
                [x_cat[iface_mask], t_cat[iface_mask]], dim=1
            )
            h_gt_cat[iface_mask] = model(xt_if)

    # x-face interfaces (KIND_INTERFACE_BC, weighted by w_bc)
    iface_bc_mask = (kind_cat == KIND_INTERFACE_BC)
    if iface_bc_mask.sum() > 0:
        with torch.no_grad():
            xt_if_bc = torch.cat(
                [x_cat[iface_bc_mask], t_cat[iface_bc_mask]], dim=1
            )
            h_gt_cat[iface_bc_mask] = model(xt_if_bc)

    # Log BC statistics for periodic pairing
    bc_true_mask = (kind_cat == KIND_BC_TRUE)
    n_bc_true = bc_true_mask.sum().item()
    if n_bc_true > 0:
        unique_fids = bc_fid_cat[bc_true_mask].unique().tolist()
        logger.info(
            f"[SplitData] bc_true points: {n_bc_true}, "
            f"unique face_ids: {unique_fids}"
        )

    return {
        'x': x_cat,
        't': t_cat,
        'h_gt': h_gt_cat,
        'expert_id': eid_cat,
        'kind': kind_cat,
        'bc_face_id': bc_fid_cat,
    }


# ── Helpers ─────────────────────────────────────────────


def _empty(spatial_dim, output_dim, device):
    return {
        'x': torch.zeros(0, spatial_dim, device=device),
        't': torch.zeros(0, 1, device=device),
        'h_gt': torch.zeros(0, output_dim, device=device),
        'expert_id': torch.zeros(
            0, dtype=torch.long, device=device
        ),
        'kind': torch.zeros(
            0, dtype=torch.long, device=device
        ),
        'bc_face_id': torch.zeros(
            0, dtype=torch.long, device=device
        ),
    }


def _is_global_boundary(val, global_lo, global_hi, tol=1e-8):
    return (abs(val - global_lo) < tol
            or abs(val - global_hi) < tol)


def _add_ic_face(
    eidx, region, spatial_dim, spatial_domain,
    t_min_global, n_pts, output_dim, problem, pc,
    device, xs, ts, gs, eids, ks, bc_fids,
):
    """Add IC face points (t = region lower-t boundary)."""
    bl, bu = region.bounds_lower, region.bounds_upper
    t_face = bl[spatial_dim]
    is_true = abs(t_face - t_min_global) < 1e-8

    x_ic = torch.zeros(n_pts, spatial_dim, device=device)
    for d in range(spatial_dim):
        lo, hi = bl[d], bu[d]
        x_ic[:, d] = (torch.rand(n_pts, device=device)
                       * (hi - lo) + lo)
    t_ic = torch.full((n_pts, 1), t_face, device=device)

    if is_true:
        h_gt = _analytic_ic(problem, x_ic, pc)
        kind_val = KIND_IC_TRUE
    else:
        h_gt = torch.zeros(n_pts, output_dim, device=device)
        kind_val = KIND_INTERFACE

    xs.append(x_ic)
    ts.append(t_ic)
    gs.append(h_gt)
    eids.append(torch.full(
        (n_pts,), eidx, dtype=torch.long, device=device
    ))
    ks.append(torch.full(
        (n_pts,), kind_val, dtype=torch.long, device=device
    ))
    bc_fids.append(torch.full(
        (n_pts,), -1, dtype=torch.long, device=device
    ))


def _add_bc_faces_periodic(
    eidx, region, spatial_dim, spatial_domain,
    n_pts, output_dim, device,
    xs, ts, gs, eids, ks, bc_fids,
    bc_t_global,
):
    """Add BC face points with periodic pairing support.
    
    For bc_true faces on global boundaries:
    - Uses shared t-values per dimension (both left and right sides)
    - Filters to t-values within this expert's temporal range
    - Assigns bc_face_id = dim*2 + side (side=0 lower, 1 upper)
    
    For interior x-face interfaces (non-global boundaries):
    - Uses KIND_INTERFACE_BC (weighted by w_bc)
    """
    bl, bu = region.bounds_lower, region.bounds_upper
    t_lo = bl[spatial_dim]
    t_hi = bu[spatial_dim]

    for d in range(spatial_dim):
        g_lo, g_hi = spatial_domain[d]
        for side_idx, face_val in enumerate([bl[d], bu[d]]):
            is_true = _is_global_boundary(
                face_val, g_lo, g_hi
            )

            if is_true:
                # Fix 3+4: Use shared t per dimension, filter to expert's t-range
                t_global = bc_t_global[d]
                # Filter t-values that fall within this expert's temporal range
                t_mask = (t_global[:, 0] >= t_lo) & (t_global[:, 0] <= t_hi)
                t_bc = t_global[t_mask]
                n_actual = t_bc.shape[0]
                
                if n_actual == 0:
                    continue
                
                kind_val = KIND_BC_TRUE
                face_id = d * 2 + side_idx
            else:
                # Fix 6: x-face interface uses KIND_INTERFACE_BC (weighted by w_bc)
                t_bc = (
                    torch.rand(n_pts, 1, device=device)
                    * (t_hi - t_lo) + t_lo
                )
                n_actual = n_pts
                kind_val = KIND_INTERFACE_BC
                face_id = -1

            x_bc = torch.zeros(
                n_actual, spatial_dim, device=device
            )
            x_bc[:, d] = face_val
            for d2 in range(spatial_dim):
                if d2 != d:
                    lo2, hi2 = bl[d2], bu[d2]
                    x_bc[:, d2] = (
                        torch.rand(n_actual, device=device)
                        * (hi2 - lo2) + lo2
                    )

            h_gt = torch.zeros(
                n_actual, output_dim, device=device
            )

            xs.append(x_bc)
            ts.append(t_bc)
            gs.append(h_gt)
            eids.append(torch.full(
                (n_actual,), eidx,
                dtype=torch.long, device=device
            ))
            ks.append(torch.full(
                (n_actual,), kind_val,
                dtype=torch.long, device=device
            ))
            bc_fids.append(torch.full(
                (n_actual,), face_id,
                dtype=torch.long, device=device
            ))
