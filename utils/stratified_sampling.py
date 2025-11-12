"""Stratified sampling utilities for NCC dataset generation."""

import torch
from typing import Dict


def stratify_by_bins(
    large_data: Dict,
    bins: int,
    output_dim: int,
    target_size: int,
    device: torch.device
) -> Dict:
    """
    Uniform sampling - equal samples from each non-empty class.
    
    Algorithm:
    1. Bin all samples by ground truth (u, v, ...) -> assign to bins^output_dim classes
    2. Identify non-empty classes
    3. Sample equal number of samples from each non-empty class
    4. Return dataset with uniform class distribution
    
    Args:
        large_data: Large dataset dict with 'x', 't', 'u_gt', 'mask'
        bins: Number of bins per output dimension
        output_dim: Output dimensionality (e.g., 2 for complex field u+iv)
        target_size: Target total number of samples
        device: Device for computation
        
    Returns:
        Uniformly sampled dataset dict with same structure as input
    """
    # Import binning function from NCC
    from ncc.ncc_core import create_class_labels_from_regression
    
    print(f"  Stratifying {len(large_data['x'])} samples...")
    
    # Bin all samples (empty classes already filtered out)
    class_labels, class_map, bin_info = create_class_labels_from_regression(
        large_data['u_gt'], bins, device
    )
    num_classes = bin_info['num_classes']  # Number of non-empty classes after filtering
    
    print(f"  Using {num_classes} non-empty classes (out of {bin_info['original_num_classes']} total)")
    
    # Uniform sampling: equal samples per class (all classes have samples after filtering)
    samples_per_class_value = target_size // num_classes
    
    # Sample indices for each class (all classes have samples after filtering)
    selected_indices = []
    
    for c in range(num_classes):
        class_mask = (class_labels == c)
        class_indices = torch.where(class_mask)[0]
        
        # All classes should have samples after filtering
        n_to_sample = samples_per_class_value
        
        if len(class_indices) >= n_to_sample:
            # Sample without replacement
            sampled = class_indices[torch.randperm(len(class_indices), device=device)[:n_to_sample]]
        else:
            # Sample with replacement if needed
            sampled = class_indices[torch.randint(0, len(class_indices), (n_to_sample,), device=device)]
        
        selected_indices.append(sampled)
    
    # Concatenate all selected indices
    selected_indices = torch.cat(selected_indices)
    
    print(f"  ✓ Uniform sampling complete:")
    print(f"    - Classes sampled: {num_classes} (all non-empty classes)")
    print(f"    - Total samples: {len(selected_indices)}")
    print(f"    - Target size: {target_size}")
    print(f"    - Samples per class: {samples_per_class_value}")
    
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

