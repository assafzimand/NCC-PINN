"""Derivatives tracker module for layer-wise physics residual analysis."""

from .derivatives_core import (
    compute_layer_derivatives_via_probe,
    compute_residual_terms,
    track_all_layers
)

from .derivatives_plotting import (
    plot_residual_evolution,
    plot_term_magnitudes,
    plot_derivative_heatmaps,
    plot_residual_heatmaps
)

from .derivatives_runner import run_derivatives_tracker

__all__ = [
    'compute_layer_derivatives_via_probe',
    'compute_residual_terms',
    'track_all_layers',
    'plot_residual_evolution',
    'plot_term_magnitudes',
    'plot_derivative_heatmaps',
    'plot_residual_heatmaps',
    'run_derivatives_tracker'
]

