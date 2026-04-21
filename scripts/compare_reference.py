"""
Compare our solvers against Raissi's reference .mat datasets.

For each available reference dataset, evaluates our solver on the exact
same (x, t) grid used by Raissi and computes the relative L2 error.
Produces a summary figure saved to datasets/reference_comparison.png.

Prerequisites:
  Run `python scripts/download_raissi_reference.py` first to get .mat files.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

REF_DIR = Path("datasets") / "reference"


def compare_burgers1d():
    import scipy.io as sio
    from solvers.burgers1d_solver import cole_hopf_exact

    mat_path = REF_DIR / 'burgers1d' / 'burgers_shock.mat'
    if not mat_path.exists():
        return None
    mat = sio.loadmat(str(mat_path))
    x_ref = mat['x'].flatten()       # (256,)
    t_ref = mat['t'].flatten()       # (100,)
    u_ref = np.real(mat['usol'])     # (256, 100)

    nu = 0.003141592653589793
    u_ours = np.zeros_like(u_ref)
    for j, tv in enumerate(t_ref):
        u_ours[:, j] = cole_hopf_exact(x_ref, tv, nu)

    diff = u_ours - u_ref
    norm_ref = np.linalg.norm(u_ref)
    rel_l2 = np.linalg.norm(diff) / norm_ref
    max_abs = np.abs(diff).max()

    return {
        'name': 'Burgers 1D',
        'rel_l2': rel_l2,
        'max_abs': max_abs,
        'x': x_ref, 't': t_ref,
        'u_ref': u_ref, 'u_ours': u_ours, 'diff': diff,
    }


def compare_schrodinger():
    import scipy.io as sio
    from solvers.schrodinger_solver import solve_nlse_splitstep

    mat_path = REF_DIR / 'schrodinger' / 'NLS.mat'
    if not mat_path.exists():
        return None
    mat = sio.loadmat(str(mat_path))
    x_ref = mat['x'].flatten()       # (256,)
    t_ref = mat['tt'].flatten()      # (201,)
    uu_ref = mat['uu']               # complex (256, 201)

    x_min, x_max = -5.0, 5.0
    t_min, t_max = 0.0, np.pi / 2
    x_grid, t_grid, h_sol = solve_nlse_splitstep(
        x_min, x_max, t_min, t_max, nx=2048, nt=1000)

    from scipy.interpolate import RegularGridInterpolator
    dx = x_grid[1] - x_grid[0]
    x_closed = np.append(x_grid, x_grid[0] + len(x_grid) * dx)
    h_closed = np.concatenate([h_sol, h_sol[:, :1]], axis=1)

    interp_re = RegularGridInterpolator(
        (t_grid, x_closed), h_closed.real,
        method='cubic', bounds_error=False, fill_value=None)
    interp_im = RegularGridInterpolator(
        (t_grid, x_closed), h_closed.imag,
        method='cubic', bounds_error=False, fill_value=None)

    domain_len = x_closed[-1] - x_closed[0]
    X, T = np.meshgrid(x_ref, t_ref, indexing='ij')
    x_q = x_closed[0] + np.mod(
        X.ravel() - x_closed[0], domain_len)
    pts = np.column_stack([T.ravel(), x_q])
    u_re = interp_re(pts).reshape(uu_ref.shape)
    u_im = interp_im(pts).reshape(uu_ref.shape)
    h_ours = u_re + 1j * u_im

    diff = h_ours - uu_ref
    norm_ref = np.linalg.norm(uu_ref)
    rel_l2 = np.linalg.norm(diff) / norm_ref
    max_abs = np.abs(diff).max()

    return {
        'name': 'Schrodinger (NLS)',
        'rel_l2': rel_l2,
        'max_abs': max_abs,
        'x': x_ref, 't': t_ref,
        'u_ref': np.abs(uu_ref),
        'u_ours': np.abs(h_ours),
        'diff': np.abs(diff),
    }


def compare_kdv():
    import scipy.io as sio
    from solvers.kdv_solver import solve_kdv

    mat_path = REF_DIR / 'kdv' / 'KdV.mat'
    if not mat_path.exists():
        return None
    mat = sio.loadmat(str(mat_path))
    x_ref = mat['x'].flatten()       # (512,)
    t_ref = mat['tt'].flatten()      # (201,)
    u_ref = np.real(mat['uu'])       # (512, 201)

    mu = 0.000484
    x_grid, t_grid, h_sol = solve_kdv(
        x_min=-1.0, x_max=1.0, t_min=0.0, t_max=1.0,
        nx=512, nt=500, mu=mu)

    from scipy.interpolate import RegularGridInterpolator
    dx = x_grid[1] - x_grid[0]
    x_closed = np.append(x_grid, x_grid[0] + len(x_grid) * dx)
    h_closed = np.concatenate([h_sol, h_sol[:, :1]], axis=1)
    domain_len = x_closed[-1] - x_closed[0]

    interp = RegularGridInterpolator(
        (t_grid, x_closed), h_closed,
        method='cubic', bounds_error=False, fill_value=None)

    X, T = np.meshgrid(x_ref, t_ref, indexing='ij')
    x_q = x_closed[0] + np.mod(
        X.ravel() - x_closed[0], domain_len)
    pts = np.column_stack([T.ravel(), x_q])
    u_ours = interp(pts).reshape(u_ref.shape)

    diff = u_ours - u_ref
    norm_ref = np.linalg.norm(u_ref)
    rel_l2 = np.linalg.norm(diff) / norm_ref
    max_abs = np.abs(diff).max()

    return {
        'name': 'KdV',
        'rel_l2': rel_l2,
        'max_abs': max_abs,
        'x': x_ref, 't': t_ref,
        'u_ref': u_ref, 'u_ours': u_ours, 'diff': diff,
    }


def compare_allen_cahn():
    import scipy.io as sio
    from solvers.allen_cahn_solver import solve_allen_cahn

    mat_path = REF_DIR / 'allen_cahn' / 'AC.mat'
    if not mat_path.exists():
        return None
    mat = sio.loadmat(str(mat_path))
    x_ref = mat['x'].flatten()       # (512,)
    t_ref = mat['tt'].flatten()      # (201,)
    u_ref = np.real(mat['uu'])       # (512, 201)

    D = 0.0001
    x_grid, t_grid, h_sol = solve_allen_cahn(
        x_min=-1.0, x_max=1.0, t_min=0.0, t_max=1.0,
        nx=512, nt=500, D=D)

    from scipy.interpolate import RegularGridInterpolator
    dx = x_grid[1] - x_grid[0]
    x_closed = np.append(x_grid, x_grid[0] + len(x_grid) * dx)
    h_closed = np.concatenate([h_sol, h_sol[:, :1]], axis=1)
    domain_len = x_closed[-1] - x_closed[0]

    interp = RegularGridInterpolator(
        (t_grid, x_closed), h_closed,
        method='cubic', bounds_error=False, fill_value=None)

    X, T = np.meshgrid(x_ref, t_ref, indexing='ij')
    x_q = x_closed[0] + np.mod(
        X.ravel() - x_closed[0], domain_len)
    pts = np.column_stack([T.ravel(), x_q])
    u_ours = interp(pts).reshape(u_ref.shape)

    diff = u_ours - u_ref
    norm_ref = np.linalg.norm(u_ref)
    rel_l2 = np.linalg.norm(diff) / norm_ref
    max_abs = np.abs(diff).max()

    return {
        'name': 'Allen-Cahn',
        'rel_l2': rel_l2,
        'max_abs': max_abs,
        'x': x_ref, 't': t_ref,
        'u_ref': u_ref, 'u_ours': u_ours, 'diff': diff,
    }


COMPARATORS = [
    compare_burgers1d,
    compare_schrodinger,
    compare_kdv,
    compare_allen_cahn,
]


def main():
    print("=" * 60)
    print("REFERENCE COMPARISON: Our solvers vs Raissi .mat")
    print("=" * 60)

    results = []
    for cmp_fn in COMPARATORS:
        try:
            r = cmp_fn()
            if r is None:
                print(f"  [{cmp_fn.__name__}] .mat not found, skipping.")
                continue
            print(f"  {r['name']:20s}  rel-L2 = {r['rel_l2']:.6e}  "
                  f"max|diff| = {r['max_abs']:.6e}")
            results.append(r)
        except Exception as e:
            print(f"  [{cmp_fn.__name__}] FAILED: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("\nNo comparisons completed. "
              "Run download_raissi_reference.py first.")
        return 1

    n = len(results)
    fig, axes = plt.subplots(n, 4, figsize=(20, 4.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, r in enumerate(results):
        x, t = r['x'], r['t']
        X, T = np.meshgrid(x, t, indexing='ij')

        ax = axes[row, 0]
        im = ax.pcolormesh(T, X, r['u_ref'], shading='auto',
                           cmap='RdBu_r')
        ax.set_title(f"{r['name']} -- Raissi reference")
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        plt.colorbar(im, ax=ax)

        ax = axes[row, 1]
        im = ax.pcolormesh(T, X, r['u_ours'], shading='auto',
                           cmap='RdBu_r')
        ax.set_title(f"{r['name']} -- Our solver")
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        plt.colorbar(im, ax=ax)

        ax = axes[row, 2]
        im = ax.pcolormesh(T, X, np.abs(r['diff']),
                           shading='auto', cmap='hot_r')
        ax.set_title(f"|diff|  (max={r['max_abs']:.2e})")
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        plt.colorbar(im, ax=ax)

        ax = axes[row, 3]
        ax.axis('off')
        txt = (f"{r['name']}\n"
               f"{'=' * 30}\n"
               f"Rel-L2: {r['rel_l2']:.6e}\n"
               f"Max |diff|: {r['max_abs']:.6e}\n"
               f"Ref grid: {r['u_ref'].shape}\n"
               f"x: [{x.min():.2f}, {x.max():.2f}]\n"
               f"t: [{t.min():.4f}, {t.max():.4f}]")
        ax.text(0.1, 0.5, txt, fontsize=12, family='monospace',
                verticalalignment='center',
                transform=ax.transAxes)

    plt.tight_layout()
    out_path = Path("datasets") / "reference_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved to {out_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'PDE':<22s} {'Rel-L2':<14s} {'Max |diff|':<14s}")
    print("-" * 50)
    for r in results:
        print(f"{r['name']:<22s} {r['rel_l2']:<14.6e} "
              f"{r['max_abs']:<14.6e}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
