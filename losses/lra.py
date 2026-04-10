"""Loss Rate Annealing (LRA) for adaptive loss component weighting.

Adaptively weights residual/IC/BC loss components based on gradient norms so
that all components contribute equally to training. Every `update_every` epochs
the weights are updated using an EMA-smoothed inverse-proportion rule.

Algorithm (Wang et al. 2022 — "When and why PINNs fail to train"):
    1. Compute gradient norm for each loss component:
           g_i = ||∇_θ L_i||₂
    2. Compute target weight (inverse-proportion, normalized to mean=1):
           λ_i* = (Σ_j g_j) / g_i
           λ_i* → λ_i* * n / Σ_j λ_j*     (normalize so mean=1)
    3. Apply EMA:
           λ_i ← (1 - α) * λ_i + α * λ_i*

The loss function must support `return_components=True` which returns a dict
    {'residual': scalar, 'ic': scalar, 'bc': scalar}
of unweighted scalar MSE values.
"""

import torch
from typing import Dict, Callable, Optional


class LRAWeights:
    """Adaptive loss component weights via Loss Rate Annealing.

    Args:
        alpha: EMA smoothing factor. 0 = no update, 1 = instant update.
               Paper recommends 0.1 (slow, stable adaptation).
        update_every: How often (in epochs) to recompute weights.
                      Each update requires 3 separate backward passes —
                      keep this at least 50-100 for efficiency.
    """

    def __init__(self, alpha: float = 0.1, update_every: int = 100):
        self.alpha = alpha
        self.update_every = update_every
        self.weights: Dict[str, float] = {
            'residual': 1.0,
            'ic': 1.0,
            'bc': 1.0,
        }

    def update(
        self,
        model: torch.nn.Module,
        loss_fn: Callable,
        batch: Dict[str, torch.Tensor],
    ) -> None:
        """Recompute LRA weights from current gradient norms.

        Runs 3 separate backward passes (one per loss component) with
        retain_graph=True, then zeros gradients. Caller must ensure the
        optimizer is zero-grad'd before calling this.

        Args:
            model: The PINN model.
            loss_fn: Loss function supporting `return_components=True`.
            batch: Training batch dict (x, t, h_gt, mask).
        """
        components = loss_fn(model, batch, return_components=True)

        grad_norms: Dict[str, float] = {}
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        for key, loss_val in components.items():
            if not isinstance(loss_val, torch.Tensor) or not loss_val.requires_grad:
                grad_norms[key] = 1e-8
                continue
            grads = torch.autograd.grad(
                loss_val, trainable_params,
                retain_graph=True,
                allow_unused=True,
            )
            norm_sq = sum(
                g.norm() ** 2 for g in grads if g is not None
            )
            grad_norms[key] = float(norm_sq.sqrt().clamp(min=1e-8))

        # Zero gradients accumulated during grad norm computation
        model.zero_grad()

        # Compute inverse-proportion target weights, normalized to mean=1
        total = sum(grad_norms.values())
        n = len(grad_norms)
        raw_targets = {k: total / v for k, v in grad_norms.items()}
        raw_sum = sum(raw_targets.values())
        targets = {k: v * n / raw_sum for k, v in raw_targets.items()}

        # Apply EMA
        for key in self.weights:
            self.weights[key] = (
                (1.0 - self.alpha) * self.weights[key]
                + self.alpha * targets[key]
            )

    def get(self, key: str) -> float:
        """Return current weight for a loss component."""
        return self.weights.get(key, 1.0)

    def __repr__(self) -> str:
        w = self.weights
        return (
            f"LRAWeights(residual={w['residual']:.4f}, "
            f"ic={w['ic']:.4f}, bc={w['bc']:.4f}, "
            f"alpha={self.alpha}, update_every={self.update_every})"
        )
