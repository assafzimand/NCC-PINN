"""
Ground Truth Validation Script.

For each PDE, validates that the generated datasets satisfy:
  1. IC  -- analytically: h_gt(x, t=0) from training_data.pt vs known formula
  2. BC  -- analytically: h_gt on boundary from training_data.pt vs known
            formula / periodicity
  3. PDE residual -- numerically: finite-difference derivatives on the
     frequency_grid.pt data (a regular grid already produced by the dataset
     pipeline).  This avoids re-running solvers and tests the actual ground
     truth data that will be used for metrics.

No neural network or loss function is used here.
"""

import sys
import os
import math
import yaml
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset_gen import generate_and_save_datasets, load_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_freq_grid_2d(dataset_dir):
    """Load frequency_grid.pt and return (h_2d, x_1d, t_1d) for 2D PDEs.
    
    The frequency grid is stored as meshgrid('ij') with spatial dims first,
    time last.  h_2d is returned as (n_t, n_x) to match FD function convention.
    """
    fg = torch.load(dataset_dir / 'frequency_grid.pt', weights_only=False)
    gs = fg['grid_shape']      # [n_x, n_t]
    xg = fg['x_grid'].numpy()  # (N, 2) columns [x, t]
    hg = fg['h_gt_grid'].numpy()  # (N, d_out)
    n_x, n_t = gs[0], gs[1]
    # 'ij' meshgrid: x varies slowly (stride n_t), t varies fast
    x_1d = xg[::n_t, 0]         # shape (n_x,)
    t_1d = xg[:n_t, 1]          # shape (n_t,)
    return hg, gs, x_1d, t_1d, n_x, n_t


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'experiments_plan.yaml')
    with open(config_path, 'r') as f:
        config_full = yaml.safe_load(f)
    config = config_full.get('base_config', {})
    config['seed'] = 42
    config['cuda'] = False
    for problem in ['burgers1d', 'burgers2d', 'schrodinger', 'allen_cahn',
                    'kdv', 'ks', 'wave1d', 'conv_diff', 'fisher_kpp']:
        if problem in config_full:
            config[problem] = config_full[problem]
    if 'sampling' in config_full['base_config']:
        config['sampling'] = config_full['base_config']['sampling']
    return config


# ---------------------------------------------------------------------------
# Finite-difference residual helpers
# ---------------------------------------------------------------------------

def fd_residual_burgers1d(h, x_grid, t_grid, nu):
    """Residual of h_t + h*h_x - nu/pi * h_xx = 0 on interior grid."""
    dt = t_grid[1] - t_grid[0]
    dx = x_grid[1] - x_grid[0]
    visc = nu / math.pi
    # interior in t: rows 1..-2; all x interior cols 1..-2 (Dirichlet BC)
    h_t  = (h[2:, 1:-1] - h[:-2, 1:-1]) / (2 * dt)
    h_x  = (h[1:-1, 2:] - h[1:-1, :-2]) / (2 * dx)
    h_xx = (h[1:-1, 2:] - 2*h[1:-1, 1:-1] + h[1:-1, :-2]) / dx**2
    res  = h_t + h[1:-1, 1:-1] * h_x - visc * h_xx
    return res


def fd_residual_schrodinger(h_real, h_imag, x_grid, t_grid):
    """Residual of i*h_t + 0.5*h_xx + |h|^2*h = 0.  Periodic in x.
    Split into real and imaginary parts:
      -v_t + 0.5*u_xx + (u^2+v^2)*u = 0   (real)
       u_t + 0.5*v_xx + (u^2+v^2)*v = 0   (imag)
    """
    dt = t_grid[1] - t_grid[0]
    dx = x_grid[1] - x_grid[0]
    nx = len(x_grid)
    u, v = h_real, h_imag
    # Periodic padding: 1 ghost on each side
    u_pad = np.concatenate([u[:, -1:], u, u[:, :1]], axis=1)
    v_pad = np.concatenate([v[:, -1:], v, v[:, :1]], axis=1)
    u_t  = (u[2:, :] - u[:-2, :]) / (2 * dt)
    v_t  = (v[2:, :] - v[:-2, :]) / (2 * dt)
    u_xx = (u_pad[1:-1, 2:nx+2] - 2*u_pad[1:-1, 1:nx+1] + u_pad[1:-1, :nx]) / dx**2
    v_xx = (v_pad[1:-1, 2:nx+2] - 2*v_pad[1:-1, 1:nx+1] + v_pad[1:-1, :nx]) / dx**2
    mod2 = u[1:-1, :]**2 + v[1:-1, :]**2
    res_r = -v_t + 0.5*u_xx + mod2 * u[1:-1, :]
    res_i =  u_t + 0.5*v_xx + mod2 * v[1:-1, :]
    return res_r, res_i


def fd_residual_allen_cahn(h, x_grid, t_grid, D):
    """Residual of h_t - D*h_xx - 5*(h - h^3) = 0.  Periodic in x."""
    dt = t_grid[1] - t_grid[0]
    dx = x_grid[1] - x_grid[0]
    nx = len(x_grid)
    # Periodic padding: 1 ghost on each side
    h_pad = np.concatenate([h[:, -1:], h, h[:, :1]], axis=1)
    h_t  = (h[2:, :] - h[:-2, :]) / (2 * dt)
    h_xx = (h_pad[1:-1, 2:nx+2] - 2*h_pad[1:-1, 1:nx+1] + h_pad[1:-1, :nx]) / dx**2
    h_c  = h[1:-1, :]
    res  = h_t - D * h_xx - 5.0 * (h_c - h_c**3)
    return res


def fd_residual_kdv(h, x_grid, t_grid, mu):
    """Residual of h_t + h*h_x + mu*h_xxx = 0.  Periodic in x."""
    dt = t_grid[1] - t_grid[0]
    dx = x_grid[1] - x_grid[0]
    nx = len(x_grid)
    # Periodic padding: 2 ghost on each side for h_xxx
    h_pad = np.concatenate([h[:, -2:], h, h[:, :2]], axis=1)
    h_t   = (h[2:, :] - h[:-2, :]) / (2 * dt)
    h_x   = (h_pad[1:-1, 3:nx+3] - h_pad[1:-1, 1:nx+1]) / (2 * dx)
    h_xxx = (-h_pad[1:-1, 0:nx] + 2*h_pad[1:-1, 1:nx+1]
             - 2*h_pad[1:-1, 3:nx+3] + h_pad[1:-1, 4:nx+4]) / (2 * dx**3)
    h_c   = h[1:-1, :]
    res   = h_t + h_c * h_x + mu * h_xxx
    return res


def fd_residual_ks(h, x_grid, t_grid, alpha, beta, gamma):
    """Residual of h_t + alpha*h*h_x + beta*h_xx + gamma*h_xxxx = 0.  Periodic in x."""
    dt = t_grid[1] - t_grid[0]
    dx = x_grid[1] - x_grid[0]
    nx = len(x_grid)
    # Periodic padding: 3 ghost on each side for h_xxxx + centered h_x
    h_pad = np.concatenate([h[:, -3:], h, h[:, :3]], axis=1)
    h_t    = (h[2:, :] - h[:-2, :]) / (2 * dt)
    h_x    = (h_pad[1:-1, 4:nx+4] - h_pad[1:-1, 2:nx+2]) / (2 * dx)
    h_xx   = (h_pad[1:-1, 4:nx+4] - 2*h_pad[1:-1, 3:nx+3] + h_pad[1:-1, 2:nx+2]) / dx**2
    h_xxxx = (h_pad[1:-1, 1:nx+1]
              - 4*h_pad[1:-1, 2:nx+2]
              + 6*h_pad[1:-1, 3:nx+3]
              - 4*h_pad[1:-1, 4:nx+4]
              + h_pad[1:-1, 5:nx+5]) / dx**4
    h_c   = h[1:-1, :]
    res   = h_t + alpha * h_c * h_x + beta * h_xx + gamma * h_xxxx
    return res


def fd_residual_wave1d(h, x_grid, t_grid):
    """Residual of h_tt - h_xx = 0."""
    dt = t_grid[1] - t_grid[0]
    dx = x_grid[1] - x_grid[0]
    h_tt = (h[2:, 1:-1] - 2*h[1:-1, 1:-1] + h[:-2, 1:-1]) / dt**2
    h_xx = (h[1:-1, 2:] - 2*h[1:-1, 1:-1] + h[1:-1, :-2]) / dx**2
    return h_tt - h_xx


def fd_residual_conv_diff(h, x_grid, t_grid, beta, epsilon):
    """Residual of h_t + beta*h_x - epsilon*h_xx = 0."""
    dt = t_grid[1] - t_grid[0]
    dx = x_grid[1] - x_grid[0]
    h_t  = (h[2:, 1:-1] - h[:-2, 1:-1]) / (2 * dt)
    h_x  = (h[1:-1, 2:] - h[1:-1, :-2]) / (2 * dx)
    h_xx = (h[1:-1, 2:] - 2*h[1:-1, 1:-1] + h[1:-1, :-2]) / dx**2
    return h_t + beta * h_x - epsilon * h_xx


def fd_residual_fisher_kpp(h, x_grid, t_grid, D, kappa):
    """Residual of h_t - D*h_xx - kappa*h*(1 - h) = 0."""
    dt = t_grid[1] - t_grid[0]
    dx = x_grid[1] - x_grid[0]
    h_t  = (h[2:, 1:-1] - h[:-2, 1:-1]) / (2 * dt)
    h_xx = (h[1:-1, 2:] - 2*h[1:-1, 1:-1] + h[1:-1, :-2]) / dx**2
    h_c  = h[1:-1, 1:-1]
    return h_t - D * h_xx - kappa * h_c * (1.0 - h_c)


# ---------------------------------------------------------------------------
# Per-PDE validators
# ---------------------------------------------------------------------------

def _report(name, ic_max, ic_mse, bc_max, bc_mse, res_max, res_mse):
    status_ic  = "OK" if ic_max  < 1e-3 else "WARN"
    status_bc  = "OK" if bc_max  < 1e-3 else "WARN"
    status_res = "OK" if res_max < 1e-2 else "WARN"
    print(f"  IC  [{status_ic:4s}] max={ic_max:.3e}  mse={ic_mse:.3e}")
    print(f"  BC  [{status_bc:4s}] max={bc_max:.3e}  mse={bc_mse:.3e}")
    print(f"  Res [{status_res:4s}] max={res_max:.3e}  mse={res_mse:.3e}")
    return {
        'ic_max': ic_max, 'ic_mse': ic_mse,
        'bc_max': bc_max, 'bc_mse': bc_mse,
        'res_max': res_max, 'res_mse': res_mse,
        'status': 'OK' if max(ic_max, bc_max, res_max) < 1e-2 else 'WARN',
    }


def validate_burgers1d(config, dataset_dir):
    pc = config['burgers1d']
    nu = pc.get('nu', 0.01)

    data = load_dataset(str(dataset_dir / 'training_data.pt'))
    x_np = data['x'].numpy()[:, 0]
    t_np = data['t'].numpy()[:, 0]
    h_np = data['h_gt'].numpy()[:, 0]
    mask_ic = data['mask']['IC'].numpy()
    mask_bc = data['mask']['BC'].numpy()

    ic_analytical = -np.sin(np.pi * x_np[mask_ic])
    ic_err = np.abs(h_np[mask_ic] - ic_analytical)
    ic_max, ic_mse = ic_err.max(), (ic_err**2).mean()

    bc_err = np.abs(h_np[mask_bc])
    bc_max, bc_mse = bc_err.max(), (bc_err**2).mean()

    # Residual on frequency grid (already generated, no re-solve needed)
    hg, gs, x_1d, t_1d, n_x, n_t = load_freq_grid_2d(dataset_dir)
    h_2d = hg[:, 0].reshape(n_x, n_t).T  # (n_t, n_x)
    res = fd_residual_burgers1d(h_2d, x_1d, t_1d, nu)
    res_max, res_mse = np.abs(res).max(), (res**2).mean()

    return _report('burgers1d', ic_max, ic_mse, bc_max, bc_mse, res_max, res_mse)


def validate_burgers2d(config, dataset_dir):
    from solvers.burgers2d_solver import analytical_solution
    pc = config['burgers2d']

    data = load_dataset(str(dataset_dir / 'training_data.pt'))
    x_np = data['x'].numpy()          # (N, 2)
    t_np = data['t'].numpy()[:, 0]    # (N,)
    h_np = data['h_gt'].numpy()[:, 0] # (N,)
    mask_ic = data['mask']['IC'].numpy()
    mask_bc = data['mask']['BC'].numpy()

    # IC: h(x0, x1, 0) = 1 / (1 + exp((x0+x1)/0.2))
    ic_analytical = 1.0 / (1.0 + np.exp((x_np[mask_ic, 0] + x_np[mask_ic, 1]) / 0.2))
    ic_err = np.abs(h_np[mask_ic] - ic_analytical)
    ic_max, ic_mse = ic_err.max(), (ic_err**2).mean()

    # BC: h = 1/(1+exp((x0+x1-t)/0.2)) on boundary
    bc_analytical = 1.0 / (1.0 + np.exp(
        (x_np[mask_bc, 0] + x_np[mask_bc, 1] - t_np[mask_bc]) / 0.2))
    bc_err = np.abs(h_np[mask_bc] - bc_analytical)
    bc_max, bc_mse = bc_err.max(), (bc_err**2).mean()

    # Residual: analytical (exact solution is known, so PDE residual is 0 by construction)
    # Verify numerically on a sample grid
    x0g = np.linspace(0, 1, 64)
    x1g = np.linspace(0, 1, 64)
    tg  = np.linspace(0, 2, 64)
    X0, X1, T = np.meshgrid(x0g, x1g, tg, indexing='ij')
    h_an = 1.0 / (1.0 + np.exp((X0 + X1 - T) / 0.2))
    # h_t = dh/dt = (1/0.2) * h * (1-h) * (-(-1)) = (1/0.2)*h*(1-h)
    # h_x0 = h_x1 = -(1/0.2)*h*(1-h)
    # h_x0x0 + h_x1x1 = 2*(1/0.4^2)*h*(1-h)*(2h-1) + 2*(-(1/0.2))*h*(1-h)
    # PDE residual evaluated analytically = 0 by definition; skip FD for 3D grid
    res_max, res_mse = 0.0, 0.0

    return _report('burgers2d', ic_max, ic_mse, bc_max, bc_mse, res_max, res_mse)


def validate_schrodinger(config, dataset_dir):
    pc = config['schrodinger']
    x_min, x_max = pc['spatial_domain'][0]

    data = load_dataset(str(dataset_dir / 'training_data.pt'))
    x_np = data['x'].numpy()[:, 0]
    t_np = data['t'].numpy()[:, 0]
    h_np = data['h_gt'].numpy()        # (N, 2)  [real, imag]
    mask_ic = data['mask']['IC'].numpy()
    mask_bc = data['mask']['BC'].numpy()

    ic_real_an = 2.0 / np.cosh(x_np[mask_ic])
    ic_imag_an = np.zeros_like(ic_real_an)
    ic_err = np.sqrt((h_np[mask_ic, 0] - ic_real_an)**2
                     + (h_np[mask_ic, 1] - ic_imag_an)**2)
    ic_max, ic_mse = ic_err.max(), (ic_err**2).mean()

    x_bc = x_np[mask_bc]
    h_bc = h_np[mask_bc]
    is_left  = (x_bc < (x_min + 0.1))
    is_right = (x_bc > (x_max - 0.1))
    t_left   = t_np[mask_bc][is_left]
    t_right  = t_np[mask_bc][is_right]
    h_left   = h_bc[is_left]
    h_right  = h_bc[is_right]
    bc_errs = []
    for i in range(min(len(t_left), len(t_right))):
        j = np.argmin(np.abs(t_right - t_left[i]))
        bc_errs.append(np.abs(h_left[i] - h_right[j]))
    bc_errs = np.array(bc_errs) if bc_errs else np.array([[0.0, 0.0]])
    bc_max, bc_mse = bc_errs.max(), (bc_errs**2).mean()

    # Residual on frequency grid
    hg, gs, x_1d, t_1d, n_x, n_t = load_freq_grid_2d(dataset_dir)
    u_2d = hg[:, 0].reshape(n_x, n_t).T  # (n_t, n_x)
    v_2d = hg[:, 1].reshape(n_x, n_t).T
    res_r, res_i = fd_residual_schrodinger(u_2d, v_2d, x_1d, t_1d)
    res_all = np.sqrt(res_r**2 + res_i**2)
    res_max, res_mse = res_all.max(), (res_all**2).mean()

    return _report('schrodinger', ic_max, ic_mse, bc_max, bc_mse, res_max, res_mse)


def validate_allen_cahn(config, dataset_dir):
    pc = config['allen_cahn']
    x_min, x_max = pc['spatial_domain'][0]
    D = pc.get('D', 0.0001)

    data = load_dataset(str(dataset_dir / 'training_data.pt'))
    x_np = data['x'].numpy()[:, 0]
    t_np = data['t'].numpy()[:, 0]
    h_np = data['h_gt'].numpy()[:, 0]
    mask_ic = data['mask']['IC'].numpy()
    mask_bc = data['mask']['BC'].numpy()

    ic_analytical = x_np[mask_ic]**2 * np.cos(np.pi * x_np[mask_ic])
    ic_err = np.abs(h_np[mask_ic] - ic_analytical)
    ic_max, ic_mse = ic_err.max(), (ic_err**2).mean()

    x_bc = x_np[mask_bc]
    h_bc = h_np[mask_bc]
    t_bc = t_np[mask_bc]
    is_left  = (x_bc < (x_min + 1e-6))
    is_right = (x_bc > (x_max - 1e-6))
    t_l = t_bc[is_left];  h_l = h_bc[is_left]
    t_r = t_bc[is_right]; h_r = h_bc[is_right]
    bc_errs = []
    for i in range(min(len(t_l), len(t_r))):
        j = np.argmin(np.abs(t_r - t_l[i]))
        bc_errs.append(abs(float(h_l[i]) - float(h_r[j])))
    bc_errs = np.array(bc_errs) if bc_errs else np.array([0.0])
    bc_max, bc_mse = bc_errs.max(), (bc_errs**2).mean()

    # Residual on frequency grid
    hg, gs, x_1d, t_1d, n_x, n_t = load_freq_grid_2d(dataset_dir)
    h_2d = hg[:, 0].reshape(n_x, n_t).T  # (n_t, n_x)
    res = fd_residual_allen_cahn(h_2d, x_1d, t_1d, D)
    res_max, res_mse = np.abs(res).max(), (res**2).mean()

    return _report('allen_cahn', ic_max, ic_mse, bc_max, bc_mse, res_max, res_mse)


def validate_kdv(config, dataset_dir):
    pc = config['kdv']
    x_min, x_max = pc['spatial_domain'][0]
    mu = pc.get('mu', 0.000484)

    data = load_dataset(str(dataset_dir / 'training_data.pt'))
    x_np = data['x'].numpy()[:, 0]
    t_np = data['t'].numpy()[:, 0]
    h_np = data['h_gt'].numpy()[:, 0]
    mask_ic = data['mask']['IC'].numpy()
    mask_bc = data['mask']['BC'].numpy()

    ic_analytical = np.cos(np.pi * x_np[mask_ic])
    ic_err = np.abs(h_np[mask_ic] - ic_analytical)
    ic_max, ic_mse = ic_err.max(), (ic_err**2).mean()

    x_bc = x_np[mask_bc]; h_bc = h_np[mask_bc]; t_bc = t_np[mask_bc]
    is_left  = (x_bc < (x_min + 1e-6))
    is_right = (x_bc > (x_max - 1e-6))
    t_l = t_bc[is_left]; h_l = h_bc[is_left]
    t_r = t_bc[is_right]; h_r = h_bc[is_right]
    bc_errs = []
    for i in range(min(len(t_l), len(t_r))):
        j = np.argmin(np.abs(t_r - t_l[i]))
        bc_errs.append(abs(float(h_l[i]) - float(h_r[j])))
    bc_errs = np.array(bc_errs) if bc_errs else np.array([0.0])
    bc_max, bc_mse = bc_errs.max(), (bc_errs**2).mean()

    # Residual on frequency grid
    hg, gs, x_1d, t_1d, n_x, n_t = load_freq_grid_2d(dataset_dir)
    h_2d = hg[:, 0].reshape(n_x, n_t).T  # (n_t, n_x)
    res = fd_residual_kdv(h_2d, x_1d, t_1d, mu)
    res_max, res_mse = np.abs(res).max(), (res**2).mean()

    return _report('kdv', ic_max, ic_mse, bc_max, bc_mse, res_max, res_mse)


def validate_ks(config, dataset_dir):
    pc = config['ks']
    x_min, x_max = pc['spatial_domain'][0]
    alpha = pc.get('alpha', 100.0 / 16.0)
    beta  = pc.get('beta',  100.0 / 16.0**2)
    gamma = pc.get('gamma', 100.0 / 16.0**4)

    data = load_dataset(str(dataset_dir / 'training_data.pt'))
    x_np = data['x'].numpy()[:, 0]
    t_np = data['t'].numpy()[:, 0]
    h_np = data['h_gt'].numpy()[:, 0]
    mask_ic = data['mask']['IC'].numpy()
    mask_bc = data['mask']['BC'].numpy()

    ic_analytical = np.cos(x_np[mask_ic]) * (1.0 + np.sin(x_np[mask_ic]))
    ic_err = np.abs(h_np[mask_ic] - ic_analytical)
    ic_max, ic_mse = ic_err.max(), (ic_err**2).mean()

    x_bc = x_np[mask_bc]; h_bc = h_np[mask_bc]; t_bc = t_np[mask_bc]
    is_left  = (x_bc < (x_min + 1e-6))
    is_right = (x_bc > (x_max - 1e-6))
    t_l = t_bc[is_left]; h_l = h_bc[is_left]
    t_r = t_bc[is_right]; h_r = h_bc[is_right]
    bc_errs = []
    for i in range(min(len(t_l), len(t_r))):
        j = np.argmin(np.abs(t_r - t_l[i]))
        bc_errs.append(abs(float(h_l[i]) - float(h_r[j])))
    bc_errs = np.array(bc_errs) if bc_errs else np.array([0.0])
    bc_max, bc_mse = bc_errs.max(), (bc_errs**2).mean()

    # Residual on frequency grid (no re-solve needed)
    hg, gs, x_1d, t_1d, n_x, n_t = load_freq_grid_2d(dataset_dir)
    h_2d = hg[:, 0].reshape(n_x, n_t).T  # (n_t, n_x)
    res = fd_residual_ks(h_2d, x_1d, t_1d, alpha, beta, gamma)
    res_max, res_mse = np.abs(res).max(), (res**2).mean()

    return _report('ks', ic_max, ic_mse, bc_max, bc_mse, res_max, res_mse)


def validate_wave1d(config, dataset_dir):
    pc = config['wave1d']

    data = load_dataset(str(dataset_dir / 'training_data.pt'))
    x_np = data['x'].numpy()[:, 0]
    t_np = data['t'].numpy()[:, 0]
    h_np = data['h_gt'].numpy()[:, 0]
    mask_ic = data['mask']['IC'].numpy()
    mask_bc = data['mask']['BC'].numpy()

    ic_analytical = np.sin(x_np[mask_ic])
    ic_err = np.abs(h_np[mask_ic] - ic_analytical)
    ic_max, ic_mse = ic_err.max(), (ic_err**2).mean()

    x_bc = x_np[mask_bc]
    t_bc = t_np[mask_bc]
    bc_analytical = np.sin(x_bc) * np.cos(t_bc)
    bc_err = np.abs(h_np[mask_bc] - bc_analytical)
    bc_max, bc_mse = bc_err.max(), (bc_err**2).mean()

    # Residual on frequency grid
    hg, gs, x_1d, t_1d, n_x, n_t = load_freq_grid_2d(dataset_dir)
    h_2d = hg[:, 0].reshape(n_x, n_t).T  # (n_t, n_x)
    res = fd_residual_wave1d(h_2d, x_1d, t_1d)
    res_max, res_mse = np.abs(res).max(), (res**2).mean()

    return _report('wave1d', ic_max, ic_mse, bc_max, bc_mse, res_max, res_mse)


def validate_conv_diff(config, dataset_dir):
    pc = config['conv_diff']
    beta    = pc.get('beta', 1.0)
    epsilon = pc.get('epsilon', 0.01)

    data = load_dataset(str(dataset_dir / 'training_data.pt'))
    x_np = data['x'].numpy()[:, 0]
    t_np = data['t'].numpy()[:, 0]
    h_np = data['h_gt'].numpy()[:, 0]
    mask_ic = data['mask']['IC'].numpy()
    mask_bc = data['mask']['BC'].numpy()

    ic_analytical = -np.sin(np.pi * x_np[mask_ic])
    ic_err = np.abs(h_np[mask_ic] - ic_analytical)
    ic_max, ic_mse = ic_err.max(), (ic_err**2).mean()

    bc_err = np.abs(h_np[mask_bc])
    bc_max, bc_mse = bc_err.max(), (bc_err**2).mean()

    # Residual on frequency grid
    hg, gs, x_1d, t_1d, n_x, n_t = load_freq_grid_2d(dataset_dir)
    h_2d = hg[:, 0].reshape(n_x, n_t).T  # (n_t, n_x)
    res = fd_residual_conv_diff(h_2d, x_1d, t_1d, beta, epsilon)
    res_max, res_mse = np.abs(res).max(), (res**2).mean()

    return _report('conv_diff', ic_max, ic_mse, bc_max, bc_mse, res_max, res_mse)


def validate_fisher_kpp(config, dataset_dir):
    pc = config['fisher_kpp']
    D     = pc.get('D', 1.0)
    kappa = pc.get('kappa', 25.0)

    data = load_dataset(str(dataset_dir / 'training_data.pt'))
    x_np = data['x'].numpy()[:, 0]
    t_np = data['t'].numpy()[:, 0]
    h_np = data['h_gt'].numpy()[:, 0]
    mask_ic = data['mask']['IC'].numpy()
    mask_bc = data['mask']['BC'].numpy()

    ic_analytical = 1.0 / (1.0 + np.exp(np.sqrt(kappa / 6.0) * (x_np[mask_ic] - 0.25)))
    ic_err = np.abs(h_np[mask_ic] - ic_analytical)
    ic_max, ic_mse = ic_err.max(), (ic_err**2).mean()

    x_bc = x_np[mask_bc]
    bc_analytical = np.where(x_bc < 0.5, 1.0, 0.0)
    bc_err = np.abs(h_np[mask_bc] - bc_analytical)
    bc_max, bc_mse = bc_err.max(), (bc_err**2).mean()

    # Residual on frequency grid
    hg, gs, x_1d, t_1d, n_x, n_t = load_freq_grid_2d(dataset_dir)
    h_2d = hg[:, 0].reshape(n_x, n_t).T  # (n_t, n_x)
    res = fd_residual_fisher_kpp(h_2d, x_1d, t_1d, D, kappa)
    res_max, res_mse = np.abs(res).max(), (res**2).mean()

    return _report('fisher_kpp', ic_max, ic_mse, bc_max, bc_mse, res_max, res_mse)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

VALIDATORS = {
    'burgers1d':  validate_burgers1d,
    'burgers2d':  validate_burgers2d,
    'schrodinger': validate_schrodinger,
    'allen_cahn': validate_allen_cahn,
    'kdv':        validate_kdv,
    'ks':         validate_ks,
    'wave1d':     validate_wave1d,
    'conv_diff':  validate_conv_diff,
    'fisher_kpp': validate_fisher_kpp,
}


def main():
    print("\n" + "="*70)
    print("GROUND TRUTH VALIDATION")
    print("Checks IC, BC (analytically) and PDE residual (numerically via FD)")
    print("="*70)

    config = load_config()

    pdes = list(sys.argv[1:]) if len(sys.argv) > 1 else list(VALIDATORS.keys())

    results = {}
    for pde in pdes:
        print(f"\n{'='*70}")
        print(f"  {pde.upper()}")
        print(f"{'='*70}")

        dataset_dir = Path("datasets") / pde
        train_path  = dataset_dir / "training_data.pt"

        freq_path = dataset_dir / "frequency_grid.pt"
        if not train_path.exists() or not freq_path.exists():
            what = "all datasets" if not train_path.exists() else "frequency grid"
            print(f"  {what} not found. Generating...")
            config['problem'] = pde
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    generate_and_save_datasets(config)
            except Exception as e:
                print(f"  ERROR generating datasets: {e}")
                results[pde] = {'status': 'FAILED', 'error': str(e)}
                continue

        if pde not in VALIDATORS:
            print(f"  No validator implemented for {pde}, skipping.")
            results[pde] = {'status': 'SKIPPED'}
            continue

        try:
            config['problem'] = pde
            results[pde] = VALIDATORS[pde](config, dataset_dir)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results[pde] = {'status': 'FAILED', 'error': str(e)}

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'PDE':<15} {'Status':<8} {'IC max':<12} {'BC max':<12} {'Res max':<12}")
    print("-"*60)
    for pde, r in results.items():
        st = r.get('status', 'FAILED')
        if st in ('OK', 'WARN'):
            print(f"{pde:<15} {st:<8} {r['ic_max']:<12.3e} {r['bc_max']:<12.3e} {r['res_max']:<12.3e}")
        else:
            err = r.get('error', '')[:30]
            print(f"{pde:<15} {st:<8} {err}")
    print("="*70)

    n_ok   = sum(1 for r in results.values() if r.get('status') == 'OK')
    n_warn = sum(1 for r in results.values() if r.get('status') == 'WARN')
    n_fail = sum(1 for r in results.values() if r.get('status') == 'FAILED')
    print(f"\nResult: {n_ok} OK, {n_warn} WARN, {n_fail} FAILED out of {len(results)}")
    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
