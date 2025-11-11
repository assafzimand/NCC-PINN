"""
NCC (Nearest Class Centroid) core computations.
All operations are CUDA-compatible and vectorized.
"""

import torch
from typing import Dict, Tuple, List
import numpy as np


def create_class_labels_from_regression(
    outputs: torch.Tensor,
    bins: int,
    device: torch.device
) -> Tuple[torch.Tensor, List[Tuple], Dict]:
    """
    Convert continuous regression outputs to discrete class labels by binning.

    Each output dimension is binned separately using its min/max range.

    Args:
        outputs: Regression outputs (N, output_dim)
        bins: Number of bins per dimension
        device: Device for computation

    Returns:
        class_labels: Integer class labels (N,)
        class_map: List mapping class_id -> (bin_d0, bin_d1, ...)
        bin_info: Dict with binning information
    """
    N, output_dim = outputs.shape

    # Compute min/max per dimension from data
    bin_edges_list = []
    for d in range(output_dim):
        dim_min = outputs[:, d].min().item()
        dim_max = outputs[:, d].max().item()
        # Create bins with small epsilon to handle edge cases
        edges = torch.linspace(dim_min, dim_max + 1e-6, bins + 1, device=device)
        bin_edges_list.append(edges)

    # Bin each dimension
    bin_indices = torch.zeros(N, output_dim, dtype=torch.long, device=device)
    for d in range(output_dim):
        # Use digitize-like operation
        bin_indices[:, d] = torch.searchsorted(
            bin_edges_list[d][:-1],
            outputs[:, d],
            right=False
        )
        # Clamp to valid range [0, bins-1]
        bin_indices[:, d] = torch.clamp(bin_indices[:, d], 0, bins - 1)

    # Create class labels from multi-dimensional bin indices
    # Pre-create ALL possible bin combinations: bins^output_dim classes
    import itertools
    class_map = list(itertools.product(range(bins), repeat=output_dim))
    bin_tuples_to_class = {bin_tuple: i for i, bin_tuple in enumerate(class_map)}
    
    num_classes = bins ** output_dim
    print(f"  Created {num_classes} classes (bins^output_dim = {bins}^{output_dim})")

    # Convert to class labels
    class_labels = torch.zeros(N, dtype=torch.long, device=device)
    for i in range(N):
        bin_tuple = tuple(bin_indices[i].cpu().tolist())
        class_labels[i] = bin_tuples_to_class[bin_tuple]

    bin_info = {
        'bin_edges': bin_edges_list,
        'bins': bins,
        'output_dim': output_dim,
        'num_classes': len(class_map)
    }

    return class_labels, class_map, bin_info


def compute_class_centers(
    embeddings: torch.Tensor,
    class_labels: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    Compute class centers (centroids) for NCC.

    Args:
        embeddings: Layer activations (N, embedding_dim)
        class_labels: Class labels (N,)
        num_classes: Number of classes

    Returns:
        centers: Class centers (num_classes, embedding_dim)
    """
    N, embedding_dim = embeddings.shape
    device = embeddings.device

    centers = torch.zeros(num_classes, embedding_dim, device=device)

    for c in range(num_classes):
        mask = (class_labels == c)
        if mask.sum() > 0:
            centers[c] = embeddings[mask].mean(dim=0)

    return centers


def compute_ncc_predictions(
    embeddings: torch.Tensor,
    centers: torch.Tensor
) -> torch.Tensor:
    """
    Predict classes using nearest class centroid.

    Args:
        embeddings: Layer activations (N, embedding_dim)
        centers: Class centers (num_classes, embedding_dim)

    Returns:
        predictions: Predicted class labels (N,)
    """
    # Compute distances to all centers (vectorized)
    # distances[i, c] = ||embedding[i] - center[c]||_2
    distances = torch.cdist(embeddings, centers, p=2)  # (N, num_classes)

    # Nearest center
    predictions = distances.argmin(dim=1)  # (N,)

    return predictions


def compute_ncc_accuracy(
    predictions: torch.Tensor,
    true_labels: torch.Tensor
) -> float:
    """
    Compute NCC classification accuracy.

    Args:
        predictions: Predicted labels (N,)
        true_labels: True labels (N,)

    Returns:
        accuracy: Fraction of correct predictions
    """
    correct = (predictions == true_labels).sum().item()
    total = len(true_labels)
    return correct / total if total > 0 else 0.0


def compute_compactness_metrics(
    embeddings: torch.Tensor,
    class_labels: torch.Tensor,
    centers: torch.Tensor,
    num_classes: int
) -> Dict[str, float]:
    """
    Compute compactness metrics.

    Args:
        embeddings: Layer activations (N, embedding_dim)
        class_labels: Class labels (N,)
        centers: Class centers (num_classes, embedding_dim)
        num_classes: Number of classes

    Returns:
        Dictionary with intra_class_dist, inter_class_mean, inter_class_std
    """
    device = embeddings.device

    # Intra-class distances (average distance within each class to center)
    intra_dists = []
    for c in range(num_classes):
        mask = (class_labels == c)
        if mask.sum() > 0:
            class_embeddings = embeddings[mask]
            center = centers[c]
            dists = torch.norm(class_embeddings - center, p=2, dim=1)
            intra_dists.append(dists.mean().item())

    intra_class_dist = np.mean(intra_dists) if intra_dists else 0.0

    # Inter-class distances (pairwise distances between centers)
    if num_classes > 1:
        center_dists = torch.cdist(centers, centers, p=2)  # (C, C)
        # Get upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones_like(center_dists), diagonal=1).bool()
        pairwise_dists = center_dists[mask]

        inter_class_mean = pairwise_dists.mean().item()
        inter_class_std = pairwise_dists.std().item()
    else:
        inter_class_mean = 0.0
        inter_class_std = 0.0

    return {
        'intra_class_dist': intra_class_dist,
        'inter_class_mean': inter_class_mean,
        'inter_class_std': inter_class_std
    }


def compute_center_geometry_metrics(
    centers: torch.Tensor
) -> Dict[str, any]:
    """
    Compute geometric properties of class centers.

    Args:
        centers: Class centers (num_classes, embedding_dim)

    Returns:
        Dictionary with center norms and pairwise distances
    """
    # Center norms
    center_norms = torch.norm(centers, p=2, dim=1)  # (num_classes,)

    # Pairwise distances between centers
    pairwise_dists = torch.cdist(centers, centers, p=2)  # (C, C)

    return {
        'center_norms': center_norms.cpu().numpy(),
        'pairwise_distances': pairwise_dists.cpu().numpy(),
        'mean_center_norm': center_norms.mean().item(),
        'std_center_norm': center_norms.std().item()
    }


def compute_margin_metrics(
    embeddings: torch.Tensor,
    class_labels: torch.Tensor,
    centers: torch.Tensor,
    num_classes: int
) -> Dict[str, any]:
    """
    Compute margin metrics.

    Margin = (distance to nearest wrong-class center) - (distance to correct center)

    Args:
        embeddings: Layer activations (N, embedding_dim)
        class_labels: True class labels (N,)
        centers: Class centers (num_classes, embedding_dim)
        num_classes: Number of classes

    Returns:
        Dictionary with margins and fraction_positive
    """
    N = embeddings.shape[0]
    device = embeddings.device

    # Compute all distances
    distances = torch.cdist(embeddings, centers, p=2)  # (N, num_classes)

    margins = torch.zeros(N, device=device)

    for i in range(N):
        true_class = class_labels[i].item()
        dist_to_correct = distances[i, true_class]

        # Find nearest wrong-class center
        # Mask out the correct class
        wrong_class_dists = distances[i].clone()
        wrong_class_dists[true_class] = float('inf')
        dist_to_nearest_wrong = wrong_class_dists.min()

        margins[i] = dist_to_nearest_wrong - dist_to_correct

    fraction_positive = (margins > 0).sum().item() / N

    return {
        'margins': margins.cpu().numpy(),
        'mean_margin': margins.mean().item(),
        'std_margin': margins.std().item(),
        'fraction_positive': fraction_positive
    }


def compute_confusion_matrix(
    predictions: torch.Tensor,
    true_labels: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    Compute confusion matrix.

    Args:
        predictions: Predicted labels (N,)
        true_labels: True labels (N,)
        num_classes: Number of classes

    Returns:
        confusion_matrix: (num_classes, num_classes) normalized by row
    """
    device = predictions.device

    # Initialize confusion matrix
    confusion = torch.zeros(num_classes, num_classes, device=device)

    # Fill confusion matrix
    for i in range(len(predictions)):
        true_c = true_labels[i].item()
        pred_c = predictions[i].item()
        confusion[true_c, pred_c] += 1

    # Normalize by row (true class)
    row_sums = confusion.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    confusion_normalized = confusion / row_sums

    return confusion_normalized


def compute_all_ncc_metrics(
    embeddings_dict: Dict[str, torch.Tensor],
    ground_truth_outputs: torch.Tensor,
    bins: int,
    device: torch.device
) -> Dict:
    """
    Compute all NCC metrics for all layers.

    Args:
        embeddings_dict: Dict mapping layer_name -> embeddings (N, embed_dim)
        ground_truth_outputs: Ground truth outputs (N, output_dim)
        bins: Number of bins per dimension
        device: Device for computation

    Returns:
        Dictionary with all metrics per layer
    """
    # Create class labels from ground truth
    class_labels, class_map, bin_info = create_class_labels_from_regression(
        ground_truth_outputs, bins, device
    )
    num_classes = bin_info['num_classes']

    results = {
        'class_labels': class_labels,
        'class_map': class_map,
        'bin_info': bin_info,
        'layer_metrics': {}
    }

    # Compute metrics for each layer
    for layer_name, embeddings in embeddings_dict.items():
        # Compute centers
        centers = compute_class_centers(embeddings, class_labels, num_classes)

        # Compute predictions
        predictions = compute_ncc_predictions(embeddings, centers)

        # Compute accuracy
        accuracy = compute_ncc_accuracy(predictions, class_labels)

        # Compute compactness
        compactness = compute_compactness_metrics(
            embeddings, class_labels, centers, num_classes
        )

        # Compute center geometry
        center_geometry = compute_center_geometry_metrics(centers)

        # Compute margins
        margins = compute_margin_metrics(
            embeddings, class_labels, centers, num_classes
        )

        # Compute confusion matrix
        confusion = compute_confusion_matrix(
            predictions, class_labels, num_classes
        )

        results['layer_metrics'][layer_name] = {
            'accuracy': accuracy,
            'compactness': compactness,
            'center_geometry': center_geometry,
            'margins': margins,
            'confusion_matrix': confusion.cpu().numpy(),
            'predictions': predictions.cpu().numpy(),
            'centers': centers.cpu().numpy()
        }

    return results

