# PDE Benchmark Summary

All PDEs implemented in this project with their mathematical formulation, domain configuration, and state-of-the-art PINN results from the literature.

---

## Quick Reference

| # | PDE | Equation | Domain | Key Parameters | Output Dim |
|---|-----|----------|--------|----------------|------------|
| 1 | Burgers 1D | $h_t + h h_x - \frac{\nu}{\pi} h_{xx} = 0$ | $x \in [-1,1],\; t \in [0,1]$ | $\nu = \pi/1000$ | 1 |
| 2 | Schrödinger (NLS) | $i h_t + \tfrac{1}{2} h_{xx} + \|h\|^2 h = 0$ | $x \in [-5,5],\; t \in [0,\pi/2]$ | — | 2 |
| 3 | Wave 1D | $h_{tt} - h_{xx} = 0$ | $x \in [-5,5],\; t \in [0,2\pi]$ | — | 1 |
| 4 | Burgers 2D | $h_t + h(h_{x_0} + h_{x_1}) - \nu(h_{x_0 x_0} + h_{x_1 x_1}) = 0$ | $(x_0,x_1) \in [0,1]^2,\; t \in [0,2]$ | $\nu = 0.1$ | 1 |
| 5 | Allen-Cahn | $h_t - D h_{xx} - 5(h - h^3) = 0$ | $x \in [-1,1],\; t \in [0,1]$ | $D = 0.001$ | 1 |
| 6 | KdV | $h_t + h h_x + \mu h_{xxx} = 0$ | $x \in [0,1],\; t \in [0,1]$ | $\mu = 0.0025$ | 1 |
| 7 | Fisher-KPP | $h_t - D h_{xx} - \kappa h(1-h) = 0$ | $x \in [0,1],\; t \in [0,1]$ | $D=1,\; \kappa=25$ | 1 |
| 8 | Convection-Diffusion | $h_t + \beta h_x - \varepsilon h_{xx} = 0$ | $x \in [-1,1],\; t \in [0,1]$ | $\beta=1,\; \varepsilon=0.01$ | 1 |

---

## Detailed PDE Descriptions

### 1. Burgers 1D (Hard Variant)

$$h_t + h\, h_x - \frac{\nu}{\pi}\, h_{xx} = 0$$

| Property | Value |
|----------|-------|
| **Spatial domain** | $x \in [-1, 1]$ |
| **Temporal domain** | $t \in [0, 1]$ |
| **Parameters** | $\nu = \pi/1000 \approx 0.00314$ (effective viscosity $\nu/\pi = 1/1000$) |
| **Initial condition** | $h(x, 0) = -\sin(\pi x)$ |
| **Boundary conditions** | Dirichlet: $h(-1, t) = h(1, t) = 0$ |
| **Character** | Sharp shock formation with very thin viscous layer; standard "hard" PINN benchmark |

**PINN Benchmark Results (Rel. L₂ Error):**

| Method | Year | Optimizer | Rel. L₂ | Reference |
|--------|------|-----------|---------|-----------|
| vRBA ($\Phi = r^2$) + FF | 2025 | SSBroyden | **2.68 × 10⁻⁷** | [Hag et al., 2025 (Nature Comput. Sci.)](https://www.nature.com/articles/s44387-026-00084-4) |
| Vanilla PINN | 2019 | L-BFGS | 6.7 × 10⁻⁴ | [Raissi et al., J. Comput. Phys. 378](https://doi.org/10.1016/j.jcp.2018.10.045) |

> Note: results above are for the $\nu/\pi = 1/1000$ variant matching our config. For the easier $\nu/\pi = 1/100$ variant, SOTA is vRBA: 8.25 × 10⁻⁹ and PirateNet (Adam): 8.20 × 10⁻⁵.

---

### 2. Nonlinear Schrödinger Equation (NLS)

$$i\, h_t + \tfrac{1}{2}\, h_{xx} + |h|^2 h = 0$$

| Property | Value |
|----------|-------|
| **Spatial domain** | $x \in [-5, 5]$ |
| **Temporal domain** | $t \in [0, \pi/2]$ |
| **Parameters** | None (coefficients fixed at 1/2 and 1) |
| **Initial condition** | $h(x, 0) = 2\,\mathrm{sech}(x)$ |
| **Boundary conditions** | Periodic: $h(-5,t) = h(5,t)$, $h_x(-5,t) = h_x(5,t)$ |
| **Output** | Complex-valued: $h = [u, v]$ (real, imaginary parts); output dim = 2 |
| **Character** | Peregrine soliton dynamics; amplitude $|h|$ develops sharp localized peaks |

**PINN Benchmark Results (Rel. L₂ Error):**

| Method | Year | Rel. L₂ | Reference |
|--------|------|---------|-----------|
| Vanilla PINN | 2019 | ~1.97 × 10⁻² | [Raissi et al., J. Comput. Phys. 378](https://doi.org/10.1016/j.jcp.2018.10.045) |
| PirateNet + FF + WF + CS | 2024 | ~10⁻⁴ range | [Wang et al., JMLR 25, 2024](https://jmlr.org/papers/v25/24-0313.html) |
| PINNacle benchmark (multi-method) | 2024 | varies | [Hao et al., NeurIPS 2024](https://arxiv.org/abs/2306.08827) |

---

### 3. Wave 1D

$$h_{tt} - h_{xx} = 0$$

| Property | Value |
|----------|-------|
| **Spatial domain** | $x \in [-5, 5]$ |
| **Temporal domain** | $t \in [0, 2\pi]$ |
| **Parameters** | None |
| **Initial conditions** | $h(x,0) = \sin(x)$, $h_t(x,0) = 0$ |
| **Boundary conditions** | Dirichlet from analytical solution: $h(\pm 5, t) = \sin(\pm 5)\cos(t)$ |
| **Analytical solution** | $h(x,t) = \sin(x)\cos(t)$ |
| **Character** | Linear; smooth standing wave solution. Useful baseline for method validation |

**PINN Benchmark Results (Rel. L₂ Error):**

| Method | Year | Rel. L₂ | Reference |
|--------|------|---------|-----------|
| PINNacle benchmark (multi-method) | 2024 | varies by method | [Hao et al., NeurIPS 2024](https://arxiv.org/abs/2306.08827) |
| Vanilla PINN | 2019 | ~10⁻³ range | [Raissi et al., J. Comput. Phys. 378](https://doi.org/10.1016/j.jcp.2018.10.045) |

> Note: The 1D wave equation with smooth IC is relatively easy for PINNs. It is included primarily as a validation/baseline problem rather than a challenging benchmark.

---

### 4. Burgers 2D

$$h_t + h(h_{x_0} + h_{x_1}) - \nu(h_{x_0 x_0} + h_{x_1 x_1}) = 0$$

| Property | Value |
|----------|-------|
| **Spatial domain** | $(x_0, x_1) \in [0, 1] \times [0, 1]$ |
| **Temporal domain** | $t \in [0, 2]$ |
| **Parameters** | $\nu = 0.1$ |
| **Initial condition** | $h(x_0, x_1, 0) = \frac{1}{1 + \exp\!\left(\frac{x_0 + x_1}{2\nu}\right)}$ |
| **Boundary conditions** | Dirichlet from analytical solution: $h = \frac{1}{1 + \exp\!\left(\frac{x_0 + x_1 - t}{2\nu}\right)}$ on all edges |
| **Analytical solution** | $h(x_0, x_1, t) = \frac{1}{1 + \exp\!\left(\frac{x_0 + x_1 - t}{2\nu}\right)}$ |
| **Character** | Traveling sigmoid front in 2D; tests multi-dimensional spatial derivative handling |

**PINN Benchmark Results:**

| Method | Year | Notes | Reference |
|--------|------|-------|-----------|
| WF-PINNs | 2025 | Weak-form PINNs for Burgers-type models including 2D | [Alghamdi et al., Sci. Reports, 2025](https://www.nature.com/articles/s41598-025-24427-4) |
| PINNacle benchmark | 2024 | Includes 2D Burgers among 20+ PDEs | [Hao et al., NeurIPS 2024](https://arxiv.org/abs/2306.08827) |

> Note: 2D Burgers with $\nu = 0.1$ is moderately diffusive. Few papers report comparable benchmark numbers for this exact configuration.

---

### 5. Allen-Cahn

$$h_t - D\, h_{xx} - 5(h - h^3) = 0$$

| Property | Value |
|----------|-------|
| **Spatial domain** | $x \in [-1, 1]$ |
| **Temporal domain** | $t \in [0, 1]$ |
| **Parameters** | $D = 0.001$ |
| **Initial condition** | $h(x, 0) = x^2 \cos(\pi x)$ |
| **Boundary conditions** | Dirichlet: $h(-1, t) = h(1, t) = -1$ |
| **Character** | Stiff nonlinear reaction-diffusion; sharp moving interface between $h \approx +1$ and $h \approx -1$ regions. One of the hardest standard PINN benchmarks |

**PINN Benchmark Results (Rel. L₂ Error):**

| Method | Year | Optimizer | Rel. L₂ | Reference |
|--------|------|-----------|---------|-----------|
| vRBA ($\Phi = r^2$) + FF | 2025 | SSBroyden | **1.88 × 10⁻⁶** | [Hag et al., 2025 (Nature Comput. Sci.)](https://www.nature.com/articles/s44387-026-00084-4) |
| RAD + FF | 2024 | SSBroyden | 2.20 × 10⁻⁶ | [Wu et al., JMLR 24, 2023](https://jmlr.org/papers/v24/22-1258.html) |
| PirateNet + FF + WF + CS + LRA | 2025 | SOAP | 3.48 × 10⁻⁶ | [Huang et al., 2025](https://arxiv.org/abs/2412.09009) |
| PirateNet + FF + WF + CS + NTK | 2024 | Adam | 2.24 × 10⁻⁵ | [Wang et al., JMLR 25, 2024](https://jmlr.org/papers/v25/24-0313.html) |
| BRDR + FF + mMLP | 2025 | Adam | 1.45 × 10⁻⁵ | [Kim & Perdikaris, 2025](https://www.nature.com/articles/s44387-026-00084-4#ref-CR16) |
| DASA-PINN + FF | 2023 | Adam | 8.57 × 10⁻⁵ | [Anagnostopoulos et al., 2023](https://arxiv.org/abs/2302.08894) |
| Vanilla PINN | 2017 | Adam | 4.98 × 10⁻¹ | [Raissi et al., 2019](https://doi.org/10.1016/j.jcp.2018.10.045) |

---

### 6. Korteweg-de Vries (KdV)

$$h_t + h\, h_x + \mu\, h_{xxx} = 0$$

| Property | Value |
|----------|-------|
| **Spatial domain** | $x \in [0, 1]$ |
| **Temporal domain** | $t \in [0, 1]$ |
| **Parameters** | $\mu = 0.0025$ |
| **Initial condition** | $h(x, 0) = \cos(2\pi x)$ |
| **Boundary conditions** | Periodic: $h(0,t) = h(1,t)$, $h_x(0,t) = h_x(1,t)$ |
| **Character** | Nonlinear dispersive; initial cosine breaks into multi-soliton train with sharp localized peaks on flat background |

**PINN Benchmark Results (Rel. L₂ Error):**

| Method | Year | Optimizer | Rel. L₂ | Reference |
|--------|------|-----------|---------|-----------|
| vRBA ($\Phi = e^r$) + FF | 2025 | SSBroyden | **2.17 × 10⁻⁶** | [Hag et al., 2025 (Nature Comput. Sci.)](https://www.nature.com/articles/s44387-026-00084-4) |
| RAD + FF | 2024 | SSBroyden | 6.00 × 10⁻⁶ | [Wu et al., JMLR 24, 2023](https://jmlr.org/papers/v24/22-1258.html) |
| PirateNet + FF + WF + CS + LRA | 2025 | SOAP | 3.40 × 10⁻⁴ | [Huang et al., 2025](https://arxiv.org/abs/2412.09009) |
| PirateNet + FF + WF + CS + LRA | 2024 | Adam | 4.27 × 10⁻⁴ | [Wang et al., JMLR 25, 2024](https://jmlr.org/papers/v25/24-0313.html) |

---

### 7. Fisher-KPP

$$h_t - D\, h_{xx} - \kappa\, h(1 - h) = 0$$

| Property | Value |
|----------|-------|
| **Spatial domain** | $x \in [0, 1]$ |
| **Temporal domain** | $t \in [0, 1]$ |
| **Parameters** | $D = 1.0$, $\kappa = 25.0$ |
| **Initial condition** | $h(x, 0) = \frac{1}{1 + \exp\!\left(\sqrt{\kappa/6}\,(x - 0.25)\right)}$ |
| **Boundary conditions** | Dirichlet: $h(0, t) = 1$, $h(1, t) = 0$ |
| **Character** | Nonlinear reaction-diffusion; sharp traveling wavefront propagating rightward. High $\kappa$ makes the front steeper |

**PINN Benchmark Results:**

| Method | Year | Notes | Reference |
|--------|------|-------|-----------|
| PINN (various architectures) | 2025 | ~10⁻⁶ errors reported | [Oruç, Accscience, 2025](https://www.accscience.com/journal/NSCE/articles/online_first/6222) |
| Residual-weighted PINN | 2024 | Specialized for steep traveling waves | [Hale & Sheraton, 2024](https://arxiv.org/abs/2402.08313) |
| PIKAN (KAN-based PINN) | 2026 | Includes Fisher-type reaction-diffusion | [Rigas et al., 2026](https://arxiv.org/abs/2602.15068) |

> Note: Fisher-KPP is less standardized as a PINN benchmark. The large $\kappa = 25$ makes the wavefront very steep, providing a good test case for adaptive methods.

---

### 8. Convection-Diffusion

$$h_t + \beta\, h_x - \varepsilon\, h_{xx} = 0$$

| Property | Value |
|----------|-------|
| **Spatial domain** | $x \in [-1, 1]$ |
| **Temporal domain** | $t \in [0, 1]$ |
| **Parameters** | $\beta = 1.0$ (convection), $\varepsilon = 0.01$ (diffusion) |
| **Initial condition** | $h(x, 0) = -\sin(\pi x)$ |
| **Boundary conditions** | Dirichlet: $h(-1, t) = h(1, t) = 0$ |
| **Péclet number** | $Pe = \beta L / \varepsilon = 100$ (convection-dominated) |
| **Character** | Linear but numerically challenging; develops sharp boundary layers where convection pushes solution against boundaries. High Péclet number tests adaptive resolution |

**PINN Benchmark Results:**

| Method | Year | Notes | Reference |
|--------|------|-------|-----------|
| PINNacle benchmark (multi-method) | 2024 | Includes convection-diffusion problems | [Hao et al., NeurIPS 2024](https://arxiv.org/abs/2306.08827) |
| Specialized loss functionals | 2024 | 17% L² error reduction vs vanilla for convection-dominated problems | [Brüning et al., J. Numer. Math., 2024](https://link.springer.com/article/10.1007/s42967-024-00433-7) |

> Note: Convection-diffusion with $Pe = 100$ is convection-dominated, producing sharp layers. This makes it a natural test case for adaptive expert methods even though it is linear.

---

## Key References

| Abbrev. | Full Reference |
|---------|---------------|
| **Raissi et al., 2019** | M. Raissi, P. Perdikaris, G.E. Karniadakis. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *J. Comput. Phys.* 378, 2019. |
| **PirateNet (Wang et al., 2024)** | S. Wang, B. Li, Y. Chen, P. Perdikaris. "PirateNets: Physics-informed Deep Learning with Residual Adaptive Networks." *JMLR* 25, 2024. |
| **RAD (Wu et al., 2023)** | C. Wu, M. Zhu, Q. Tan, Y. Kartha, L. Lu. "A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks." *JMLR* 24, 2023. |
| **vRBA (Hag et al., 2025)** | J. Hag et al. "Variance-reduced residual-based adaptive sampling for physics-informed neural networks." *Nature Comput. Sci.*, 2025. |
| **PINNacle (Hao et al., 2024)** | Z. Hao et al. "PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs." *NeurIPS Datasets & Benchmarks*, 2024. |
| **SOAP (Huang et al., 2025)** | Z. Huang, T. Zhang. "SOAP optimizer for PINNs." arXiv:2412.09009, 2025. |

---

*Last updated: 2026-02-19*
