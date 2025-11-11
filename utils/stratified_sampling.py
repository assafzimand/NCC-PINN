"""Stratified sampling utilities for NCC dataset generation."""

import torch
from typing import Dict


def stratify_by_bins(
    large_data: Dict,
    bins: int,
    output_dim: int,
    target_size: int,
    min_samples_per_class: int,
    device: torch.device
) -> Dict:
    """
    Stratified sampling ensuring all bins^output_dim classes are represented.
    
    Algorithm:
    1. Bin all samples by ground truth (u, v, ...) -> assign to bins^output_dim classes
    2. Sample proportionally based on class frequency
    3. For rare classes with < min_samples_per_class: add more to reach minimum
    4. Return dataset with samples from all classes
    
    Args:
        large_data: Large dataset dict with 'x', 't', 'u_gt', 'mask'
        bins: Number of bins per output dimension
        output_dim: Output dimensionality (e.g., 2 for complex field u+iv)
        target_size: Target total number of samples in stratified dataset
        min_samples_per_class: Minimum samples per class/bin
        device: Device for computation
        
    Returns:
        Stratified dataset dict with same structure as input
    """
    # Import binning function from NCC
    from ncc.ncc_core import create_class_labels_from_regression
    
    print(f"  Stratifying {len(large_data['x'])} samples...")
    
    # Bin all samples
    class_labels, class_map, bin_info = create_class_labels_from_regression(
        large_data['u_gt'], bins, device
    )
    num_classes = bins ** output_dim  # Generic: bins^output_dim
    
    print(f"  Target classes: {num_classes} (bins={bins}, output_dim={output_dim})")
    
    # Count samples per class
    class_counts = torch.bincount(class_labels, minlength=num_classes)
    classes_with_samples = (class_counts > 0).sum().item()
    print(f"  Classes with samples in large dataset: {classes_with_samples}/{num_classes}")
    
    # Step 1: Calculate proportional sampling
    total_large = len(class_labels)
    samples_per_class = (class_counts.float() / total_large * target_size).int()
    
    # Step 2: Ensure minimums (add samples to rare bins)
    samples_per_class = torch.maximum(
        samples_per_class, 
        torch.full((num_classes,), min_samples_per_class, dtype=torch.int32, device=device)
    )
    
    # Step 3: Sample indices for each class
    selected_indices = []
    classes_sampled = 0
    
    for c in range(num_classes):
        class_mask = (class_labels == c)
        class_indices = torch.where(class_mask)[0]
        
        if len(class_indices) == 0:
            # No samples for this class in large dataset
            print(f"  Warning: Class {c} (bin {class_map[c]}) has no samples in large dataset")
            continue
        
        # Sample with replacement if needed
        n_to_sample = samples_per_class[c].item()
        if len(class_indices) >= n_to_sample:
            # Sample without replacement
            sampled = class_indices[torch.randperm(len(class_indices), device=device)[:n_to_sample]]
        else:
            # Sample with replacement to reach minimum
            sampled = class_indices[torch.randint(0, len(class_indices), (n_to_sample,), device=device)]
        
        selected_indices.append(sampled)
        classes_sampled += 1
    
    # Concatenate all selected indices
    selected_indices = torch.cat(selected_indices)
    
    print(f"  ✓ Stratified sampling complete:")
    print(f"    - Classes represented: {classes_sampled}/{num_classes}")
    print(f"    - Total samples: {len(selected_indices)}")
    print(f"    - Target size: {target_size}")
    print(f"    - Min samples per class: {min_samples_per_class}")
    
    # Extract stratified subset
    return {
        'x': large_data['x'][selected_indices],
        't': large_data['t'][selected_indices],
        'u_gt': large_data['u_gt'][selected_indices],
        'mask': {
            'residual': large_data['mask']['residual'][selected_indices],
            'IC': large_data['mask']['IC'][selected_indices],
            'BC': large_data['mask']['BC'][selected_indices]
        }
    }

