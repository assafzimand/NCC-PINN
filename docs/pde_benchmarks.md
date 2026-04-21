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
| 6 | KdV | $h_t + h h_x + \mu^2 h_{xxx} = 0$ | $x \in [-1,1],\; t \in [0,1]$ | $\mu = 0.022$ ($\mu^2 = 4.84 \times 10^{-4}$) | 1 |
| 7 | Fisher-KPP | $h_t - D h_{xx} - \kappa h(1-h) = 0$ | $x \in [0,1],\; t \in [0,1]$ | $D=1,\; \kappa=25$ | 1 |
| 8 | Convection-Diffusion | $h_t + \beta h_x - \varepsilon h_{xx} = 0$ | $x \in [-1,1],\; t \in [0,1]$ | $\beta=1,\; \varepsilon=0.01$ | 1 |
| 9 | Kuramoto-Sivashinsky | $h_t + \alpha h h_x + \beta h_{xx} + \gamma h_{xxxx} = 0$ | $x \in [0,2\pi],\; t \in [0,1]$ | $\alpha=100/16,\; \beta=100/16^2,\; \gamma=100/16^4$ | 1 |

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

| Method | Year | Optimizer | Rel. L₂ | Model Capacity | Reference |
|--------|------|-----------|---------|----------------|-----------|
| vRBA ($\Phi = r^2$) + FF | 2025 | SSBroyden | **2.68 × 10⁻⁷** | **2,011 params** (from paper Table 2) | [Hag et al., 2025 (npj AI)](https://www.nature.com/articles/s44387-026-00084-4) |
| Vanilla PINN | 2019 | L-BFGS | 6.7 × 10⁻⁴ | **3,021 params** — MLP `[2,20×8,1]` (from [code](https://github.com/maziarraissi/PINNs)) | [Raissi et al., J. Comput. Phys. 378](https://doi.org/10.1016/j.jcp.2018.10.045) |

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

| Method | Year | Rel. L₂ | Model Capacity | Reference |
|--------|------|---------|----------------|-----------|
| Vanilla PINN | 2019 | ~1.97 × 10⁻² | **30,802 params** — MLP `[2,100×4,2]` (from [code](https://github.com/maziarraissi/PINNs)) | [Raissi et al., J. Comput. Phys. 378](https://doi.org/10.1016/j.jcp.2018.10.045) |
| PirateNet + FF + WF + CS | 2024 | ~10⁻⁴ range | *~500K+ params (est.)* — ModifiedMlp with 256 neurons/layer + FF + adaptive residual connections. Exact param count not stated; jaxpi default is 4×256. | [Wang et al., JMLR 25, 2024](https://jmlr.org/papers/v25/24-0313.html) |
| PINNacle benchmark (multi-method) | 2024 | varies | Varies by method; standardized per-problem configs | [Hao et al., NeurIPS 2024](https://arxiv.org/abs/2306.08827) |

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

| Method | Year | Rel. L₂ | Model Capacity | Reference |
|--------|------|---------|----------------|-----------|
| PINNacle benchmark (multi-method) | 2024 | varies by method | Varies by method | [Hao et al., NeurIPS 2024](https://arxiv.org/abs/2306.08827) |
| Vanilla PINN | 2019 | ~10⁻³ range | *~3K params (est.)* — same MLP style as Burgers (`[2,20×8,1]`). Raissi uses this architecture across 1D problems. | [Raissi et al., J. Comput. Phys. 378](https://doi.org/10.1016/j.jcp.2018.10.045) |

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

| Method | Year | Notes | Model Capacity | Reference |
|--------|------|-------|----------------|-----------|
| WF-PINNs | 2025 | Weak-form PINNs for Burgers-type models including 2D | *Not reported.* Paper states "simple standard neural network architecture"; dual-network for inverse problems. | [Alghamdi et al., Sci. Reports, 2025](https://www.nature.com/articles/s41598-025-24427-4) |
| PINNacle benchmark | 2024 | Includes 2D Burgers among 20+ PDEs | Varies by method | [Hao et al., NeurIPS 2024](https://arxiv.org/abs/2306.08827) |

> Note: 2D Burgers with $\nu = 0.1$ is moderately diffusive. Few papers report comparable benchmark numbers for this exact configuration.

---

### 5. Allen-Cahn

$$h_t - D\, h_{xx} - 5(h - h^3) = 0$$

| Property | Value |
|----------|-------|
| **Spatial domain** | $x \in [-1, 1]$ |
| **Temporal domain** | $t \in [0, 1]$ |
| **Parameters** | $D = 0.0001$ (standard SOTA benchmark) |
| **Initial condition** | $h(x, 0) = x^2 \cos(\pi x)$ |
| **Boundary conditions** | Periodic: $h(-1, t) = h(1, t)$, $h_x(-1, t) = h_x(1, t)$ |
| **Character** | Stiff nonlinear reaction-diffusion; sharp moving interface between $h \approx +1$ and $h \approx -1$ regions. One of the hardest standard PINN benchmarks |

**PINN Benchmark Results (Rel. L₂ Error):**

| Method | Year | Optimizer | Rel. L₂ | Model Capacity | Reference |
|--------|------|-----------|---------|----------------|-----------|
| vRBA ($\Phi = r^2$) + FF | 2025 | SSBroyden | **1.88 × 10⁻⁶** | **2,011 params** (from paper Table 2) | [Hag et al., 2025 (npj AI)](https://www.nature.com/articles/s44387-026-00084-4) |
| RAD + FF | 2024 | SSBroyden | 2.20 × 10⁻⁶ | *~2K params (est.)* — vRBA Table 4 compares RAD under same SSBroyden setup, implying same network size | [Wu et al., JMLR 24, 2023](https://jmlr.org/papers/v24/22-1258.html) |
| PirateNet + FF + WF + CS + LRA | 2025 | SOAP | 3.48 × 10⁻⁶ | *~500K+ params (est.)* — same PirateNet architecture (same research group) | [Huang et al., 2025](https://arxiv.org/abs/2412.09009) |
| PirateNet + FF + WF + CS + NTK | 2024 | Adam | 2.24 × 10⁻⁵ | *~500K+ params (est.)* — paper: "256 neurons in each hidden layer", ModifiedMlp + FF | [Wang et al., JMLR 25, 2024](https://jmlr.org/papers/v25/24-0313.html) |
| BRDR + FF + mMLP | 2025 | Adam | 1.45 × 10⁻⁵ | *~20K–50K params (est.)* — uses mMLP (modified MLP) + FF; architecture details not explicitly reported | [Kim & Perdikaris, 2025](https://www.nature.com/articles/s44387-026-00084-4#ref-CR16) |
| DASA-PINN + FF | 2023 | Adam | 8.57 × 10⁻⁵ | *~200K-270K params (est.)* — uses standard MLP + FF + attention weighting; code at [github](https://github.com/soanagno/rba-pinns), exact arch not reported in paper | [Anagnostopoulos et al., 2023](https://arxiv.org/abs/2307.00379) |
| Vanilla PINN | 2017 | Adam | 4.98 × 10⁻¹ | *~3K params (est.)* — Raissi-style MLP | [Raissi et al., 2019](https://doi.org/10.1016/j.jcp.2018.10.045) |

---

### 6. Korteweg-de Vries (KdV)

$$h_t + \eta\, h\, h_x + \mu^2\, h_{xxx} = 0$$

| Property | Value |
|----------|-------|
| **Spatial domain** | $x \in [-1, 1]$ |
| **Temporal domain** | $t \in [0, 1]$ |
| **Parameters** | $\eta = 1$, $\mu = 0.022$ (so $\mu^2 = 4.84 \times 10^{-4}$); classical Zabusky & Kruskal (1965) values |
| **Initial condition** | $h(x, 0) = \cos(\pi x)$ |
| **Boundary conditions** | Periodic: $h(-1,t) = h(1,t)$ |
| **Character** | Nonlinear dispersive; initial cosine breaks into multi-soliton train with sharp localized peaks on flat background |

**PINN Benchmark Results (Rel. L₂ Error):**

| Method | Year | Optimizer | Rel. L₂ | Model Capacity | Reference |
|--------|------|-----------|---------|----------------|-----------|
| vRBA ($\Phi = e^r$) + FF | 2025 | SSBroyden | **2.17 × 10⁻⁶** | **2,011 params** (from paper Table 2) | [Hag et al., 2025 (npj AI)](https://www.nature.com/articles/s44387-026-00084-4) |
| RAD + FF | 2024 | SSBroyden | 6.00 × 10⁻⁶ | *~2K params (est.)* — same SSBroyden setup as vRBA comparison | [Wu et al., JMLR 24, 2023](https://jmlr.org/papers/v24/22-1258.html) |
| PirateNet + FF + WF + CS + LRA | 2025 | SOAP | 3.40 × 10⁻⁴ | *~500K+ params (est.)* — PirateNet ModifiedMlp + FF | [Huang et al., 2025](https://arxiv.org/abs/2412.09009) |
| PirateNet + FF + WF + CS + LRA | 2024 | Adam | 4.27 × 10⁻⁴ | *~500K+ params (est.)* — PirateNet ModifiedMlp + FF | [Wang et al., JMLR 25, 2024](https://jmlr.org/papers/v25/24-0313.html) |

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

| Method | Year | Notes | Model Capacity | Reference |
|--------|------|-------|----------------|-----------|
| PINN (various architectures) | 2025 | ~10⁻⁶ errors reported | *Not reported.* Paper tests multiple architectures without specifying param counts. | [Oruç, Accscience, 2025](https://www.accscience.com/journal/NSCE/articles/online_first/6222) |
| Residual-weighted PINN | 2024 | Specialized for steep traveling waves | *Not reported.* | [Hale & Sheraton, 2024](https://arxiv.org/abs/2402.08313) |
| PIKAN (KAN-based PINN) | 2026 | Includes Fisher-type reaction-diffusion | *KAN-based* — uses Kolmogorov-Arnold Networks instead of MLP; param count depends on B-spline grid size. Rigas et al. report KANs can match MLPs with 8.5× fewer params. | [Rigas et al., 2026](https://arxiv.org/abs/2602.15068) |

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

| Method | Year | Notes | Model Capacity | Reference |
|--------|------|-------|----------------|-----------|
| PINNacle benchmark (multi-method) | 2024 | Includes convection-diffusion problems | Varies by method; standard baseline: 4 hidden layers × 128-256 neurons | [Hao et al., NeurIPS 2024](https://arxiv.org/abs/2306.08827) |
| Specialized loss functionals | 2024 | 17% L² error reduction vs vanilla for convection-dominated problems | *Not reported.* Paper focuses on loss functional design, not architecture. | [Brüning et al., J. Numer. Math., 2024](https://link.springer.com/article/10.1007/s42967-024-00433-7) |

> Note: Convection-diffusion with $Pe = 100$ is convection-dominated, producing sharp layers. This makes it a natural test case for adaptive expert methods even though it is linear.

---

### 9. Kuramoto-Sivashinsky (KS)

$$h_t + \alpha\, h\, h_x + \beta\, h_{xx} + \gamma\, h_{xxxx} = 0$$

| Property | Value |
|----------|-------|
| **Spatial domain** | $x \in [0, 2\pi]$ |
| **Temporal domain** | $t \in [0, 1]$ |
| **Parameters** | $\alpha = 100/16 = 6.25$, $\beta = 100/16^2 = 0.390625$, $\gamma = 100/16^4 \approx 1.526 \times 10^{-3}$ |
| **Initial condition** | $h(x, 0) = \cos(x)(1 + \sin(x))$ |
| **Boundary conditions** | Periodic: $h(0,t) = h(2\pi,t)$, $h_x(0,t) = h_x(2\pi,t)$ |
| **Character** | Nonlinear, chaotic; anti-diffusion ($\beta h_{xx}$) drives instability, hyper-diffusion ($\gamma h_{xxxx}$) stabilizes short wavelengths, nonlinear convection ($\alpha h h_x$) transfers energy. Exhibits complex spatiotemporal patterns with sharp features |

**PINN Benchmark Results (Rel. L₂ Error):**

| Method | Year | Optimizer | Rel. L₂ | Model Capacity | Reference |
|--------|------|-----------|---------|----------------|-----------|
| PirateNet + FF + WF + CS + NTK | 2024 | Adam | **1.42 × 10⁻⁴** | *~500K+ params (est.)* — PirateNet ModifiedMlp + FF | [Wang et al., JMLR 25, 2024](https://jmlr.org/papers/v25/24-0313.html) |
| PirateNet + FF + WF + CS + LRA | 2025 | SOAP | ~10⁻⁴ range | *~500K+ params (est.)* — same PirateNet architecture | [Huang et al., 2025](https://arxiv.org/abs/2412.09009) |
| BRDR + FF + mMLP | 2025 | Adam | tested | *~20K-50K params (est.)* — mMLP with FF, smaller than PirateNet; exact size not reported | [Kim & Perdikaris, 2025](https://www.nature.com/articles/s44387-026-00084-4#ref-CR16) |

> Note: KS is one of the most challenging 1D PINN benchmarks due to its chaotic dynamics and 4th-order spatial derivative. The PirateNet benchmark above uses the same domain and parameter configuration as our implementation.

---

## Model Capacity Summary

A cross-method summary of network sizes. **Bold** = exact numbers from paper/code. *Italic* = assessment/estimate.

| Method | Architecture | Params | Source |
|--------|-------------|--------|--------|
| **Raissi (Burgers 1D)** | MLP `[2, 20×8, 1]`, Tanh | **3,021** | Code: [github](https://github.com/maziarraissi/PINNs) |
| **Raissi (Schrödinger)** | MLP `[2, 100×4, 2]`, Tanh | **30,802** | Code: [github](https://github.com/maziarraissi/PINNs) |
| **vRBA (Adam)** | MLP + FF | **21,318** | Paper Table 2 |
| **vRBA (SSBroyden)** | MLP + FF | **2,011** | Paper Table 2 |
| *PirateNet* | ModifiedMlp × 256 + FF + adaptive residual connections | *~500K+* | Paper: "256 neurons in each hidden layer"; ModifiedMlp adds U/V encoding branches. Exact count not stated; jaxpi default is 4×256. |
| *RAD (SSBroyden)* | MLP + FF (same setup as vRBA) | *~2K* | Compared in vRBA Table 4 under identical SSBroyden config |
| *SOAP* | PirateNet (same group) | *~500K+* | Same architecture as PirateNet paper |
| *BRDR (mMLP)* | Modified MLP + FF | *~20K–50K* | Uses mMLP variant; no explicit count in paper |
| *DASA-PINN / RBA* | Standard MLP + FF + attention weights | *~200K–270K* | Estimated from standard 4–5 hidden layers × 256; code at [github](https://github.com/soanagno/rba-pinns) |
| *PINNacle* | Varies per method/PDE | *Varies* | Benchmark tool; default baseline ~4 layers × 128–256 |

> **How to read**: "MLP + FF" = multi-layer perceptron with random Fourier feature embedding. PirateNet adds adaptive residual connections on top of the modified MLP (mMLP) architecture. SSBroyden is a quasi-Newton optimizer that converges with far fewer parameters than Adam.

---

## Key References

| Abbrev. | Full Reference |
|---------|---------------|
| **Raissi et al., 2019** | M. Raissi, P. Perdikaris, G.E. Karniadakis. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *J. Comput. Phys.* 378, 2019. |
| **PirateNet (Wang et al., 2024)** | S. Wang, B. Li, Y. Chen, P. Perdikaris. "PirateNets: Physics-informed Deep Learning with Residual Adaptive Networks." *JMLR* 25, 2024. |
| **RAD (Wu et al., 2023)** | C. Wu, M. Zhu, Q. Tan, Y. Kartha, L. Lu. "A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks." *CMAME* 403, 2023. |
| **vRBA (Hag et al., 2025)** | J. Hag et al. "A variational framework for residual-based adaptivity in neural PDE solvers and operator learning." *npj Artificial Intelligence*, 2025. |
| **PINNacle (Hao et al., 2024)** | Z. Hao et al. "PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs." *NeurIPS Datasets & Benchmarks*, 2024. |
| **SOAP (Huang et al., 2025)** | Z. Huang, T. Zhang. "SOAP optimizer for PINNs." arXiv:2412.09009, 2025. |
| **DASA-PINN / RBA (Anagnostopoulos et al., 2024)** | S. Anagnostopoulos, J.D. Toscano, N. Stergiopulos, G.E. Karniadakis. "Residual-based attention in physics-informed neural networks." *CMAME* 421, 2024. |

---

*Last updated: 2026-02-19*
