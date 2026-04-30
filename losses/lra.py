"""Loss Rate Annealing (LRA) for adaptive loss component weighting.

Implements the original Wang et al. (2021) algorithm from
"Understanding and mitigating gradient flow pathologies in PINNs":

    1. Residual weight is FIXED (anchored) — never changes.
    2. For each non-residual term i (IC, BC), compute:
           λ_i = max_θ |∂L_res/∂θ| / mean_θ |∂L_i/∂θ|
       This BOOSTS IC/BC to match the dominant residual gradient.
    3. Apply EMA:
           λ_i ← (1 - α) * λ_i + α * λ_i_new

The loss function must support `return_components=True` which returns a dict
    {'residual': scalar, 'ic': scalar, 'bc': scalar}
of unweighted scalar MSE values.
"""

import torch
from typing import Dict, Callable


class LRAWeights:
    """Adaptive loss component weights via Loss Rate Annealing.

    The residual weight is always fixed at its initial value.
    Only IC and BC weights are adapted to match the residual
    gradient magnitude, following Wang et al. (2021).

    Args:
        alpha: EMA smoothing factor. 0 = no update, 1 = instant update.
               Original paper uses beta=0.9 (equivalent to alpha=0.1).
        update_every: How often (in epochs) to recompute weights.
                      Each update requires backward passes —
                      keep this at least 50-100 for efficiency.
        initial_weights: Optional dict of initial weights
                        {'residual': float, 'ic': float, 'bc': float}.
                        If None, defaults to all 1.0.
                        The residual weight is frozen at its initial value.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        update_every: int = 100,
        initial_weights: Dict[str, float] = None,
    ):
        self.alpha = alpha
        self.update_every = update_every
        if initial_weights is not None:
            self.weights: Dict[str, float] = {
                'residual': initial_weights.get('residual', 1.0),
                'ic': initial_weights.get('ic', 1.0),
                'bc': initial_weights.get('bc', 1.0),
            }
        else:
            self.weights: Dict[str, float] = {
                'residual': 1.0,
                'ic': 1.0,
                'bc': 1.0,
            }
        self.fixed_residual_weight = self.weights['residual']
        self.last_grad_norms: Dict[str, float] = {
            'residual': 0.0,
            'ic': 0.0,
            'bc': 0.0,
        }

    def update(
        self,
        model: torch.nn.Module,
        loss_fn: Callable,
        batch: Dict[str, torch.Tensor],
    ) -> None:
        """Recompute LRA weights from current gradient norms.

        Wang et al. (2021) algorithm:
          - Compute max(|grad_res|) across all parameters
          - For each of IC, BC: compute mean(|grad_i|) across all parameters
          - Set lambda_i = max_grad_res / mean_grad_i
          - EMA smooth the lambda values
          - Residual weight stays fixed

        Args:
            model: The PINN model.
            loss_fn: Loss function supporting `return_components=True`.
            batch: Training batch dict (x, t, h_gt, mask).
        """
        components = loss_fn(model, batch, return_components=True)
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        grad_stats: Dict[str, float] = {}

        for key, loss_val in components.items():
            if not isinstance(loss_val, torch.Tensor) or not loss_val.requires_grad:
                grad_stats[key] = 1e-8
                continue
            grads = torch.autograd.grad(
                loss_val, trainable_params,
                retain_graph=True,
                allow_unused=True,
            )
            if key == 'residual':
                max_abs = max(
                    g.abs().max().item() for g in grads if g is not None
                )
                grad_stats[key] = max(max_abs, 1e-8)
            else:
                mean_abs = torch.mean(torch.stack([
                    g.abs().mean() for g in grads if g is not None
                ])).item()
                grad_stats[key] = max(mean_abs, 1e-8)

        self.last_grad_norms = grad_stats.copy()
        model.zero_grad()

        max_grad_res = grad_stats['residual']

        for key in ['ic', 'bc']:
            target = max_grad_res / grad_stats[key]
            self.weights[key] = (
                (1.0 - self.alpha) * self.weights[key]
                + self.alpha * target
            )

        self.weights['residual'] = self.fixed_residual_weight

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
