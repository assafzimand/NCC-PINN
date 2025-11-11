<!-- b8798a4a-1193-4d41-924c-ad831260f63b f1d4f081-50aa-4f49-a1c8-8dce907e579e -->
# NCC Stratified Dataset + Checkpoint Directory Fix

## Overview

Create a third dataset (ncc_data.pt) with stratified sampling to ensure all bins**output_dim classes have samples. Also fix checkpoint directory structure to mirror outputs structure.

## Changes

### 1. Update Configuration (config/config.yaml)

Add new parameters for NCC dataset generation:

```yaml
# dataset sizes
n_residual_train: 20000
n_initial_train: 50
n_boundary_train: 50
n_residual_eval: 1000
n_initial_eval: 20
n_boundary_eval: 20

# NCC dataset (stratified sampling to ensure all bins are represented)
n_samples_ncc: 2000  # target total NCC dataset size
rarest_bin_samples_num: 5  # minimum samples per class/bin
```

### 2. Create Stratified Sampling Utility (utils/stratified_sampling.py)

New file with generic stratified sampling function.

Key algorithm:

```python
def stratify_by_bins(large_data, bins, output_dim, target_size, min_samples_per_class, device):
    """
    Stratified sampling ensuring all bins^output_dim classes are represented.
    
 1. Generate large dataset (much larger than target_size)
 2. Bin by ground truth (u, v, ...) -> assign to bins^output_dim classes
 3. Sample proportionally based on class frequency
 4. For rare classes with < min_samples_per_class: add more to reach minimum
 5. Return dataset with exactly target_size samples (or slightly more if needed for minimums)
    """
    # Import binning function from NCC
    from ncc.ncc_core import create_class_labels_from_regression
    
    # Bin all samples
    class_labels, class_map, bin_info = create_class_labels_from_regression(
        large_data['u_gt'], bins, device
    )
    num_classes = bins ** output_dim  # Generic: bins^output_dim
    
    # Count samples per class
    class_counts = torch.bincount(class_labels, minlength=num_classes)
    
    # Step 1: Calculate proportional sampling
    total_large = len(class_labels)
    samples_per_class = (class_counts / total_large * target_size).int()
    
    # Step 2: Ensure minimums (add samples to rare bins)
    samples_per_class = torch.maximum(samples_per_class, 
                                     torch.tensor(min_samples_per_class))
    
    # Step 3: Sample indices for each class
    selected_indices = []
    for c in range(num_classes):
        class_mask = (class_labels == c)
        class_indices = torch.where(class_mask)[0]
        
        if len(class_indices) == 0:
            # Generate synthetic sample or skip (should not happen with large enough dataset)
            print(f"Warning: Class {c} has no samples in large dataset")
            continue
            
        # Sample with replacement if needed
        n_to_sample = samples_per_class[c].item()
        if len(class_indices) >= n_to_sample:
            sampled = class_indices[torch.randperm(len(class_indices))[:n_to_sample]]
        else:
            # Sample with replacement to reach minimum
            sampled = class_indices[torch.randint(0, len(class_indices), (n_to_sample,))]
        
        selected_indices.append(sampled)
    
    selected_indices = torch.cat(selected_indices)
    
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
```

### 3. Update Dataset Generation (utils/dataset_gen.py)

Add NCC dataset generation after eval dataset (around line 95):

```python
# Generate NCC data if missing (stratified)
ncc_path = dataset_dir / "ncc_data.pt"
if not ncc_path.exists():
    print(f"Generating stratified NCC data for {problem}...")
    
    # Generate large dataset for stratification (5-10x target size)
    n_large = config['n_samples_ncc'] * 10
    print(f"  Generating large dataset ({n_large} samples) for stratification...")
    large_data = solver_module.generate_dataset(
        n_residual=n_large,
        n_ic=0,  # NCC only needs residual points
        n_bc=0,
        device=device,
        config=config
    )
    
    # Determine output dimension (problem-specific)
    # For regression: output_dim = model output dimension
    # Generic: check u_gt shape
    output_dim = large_data['u_gt'].shape[1]
    
    # Apply stratified sampling
    print(f"  Applying stratified sampling (target: {config['n_samples_ncc']} samples)...")
    from utils.stratified_sampling import stratify_by_bins
    ncc_data = stratify_by_bins(
        large_data, 
        bins=config['bins'],
        output_dim=output_dim,
        target_size=config['n_samples_ncc'],
        min_samples_per_class=config['rarest_bin_samples_num'],
        device=device
    )
    
    torch.save(ncc_data, ncc_path)
    print(f"  ✓ Saved {len(ncc_data['x'])} samples to {ncc_path}")
    print(f"  ✓ All {config['bins']**output_dim} classes should be represented")
    
    # Create visualizations
    plot_path = dataset_dir / "ncc_data_visualization.png"
    title = f"{problem} - NCC Data (Stratified)"
    plot_dataset(ncc_data, str(plot_path), title=title)
    
    stats_path = dataset_dir / "ncc_data_statistics.png"
    plot_dataset_statistics(ncc_data, str(stats_path))
    
    # Problem-specific visualization
    try:
        from utils.problem_specific import get_visualization_module
        visualize_dataset, _ = get_visualization_module(problem)
        visualize_dataset(ncc_data, dataset_dir, config, 'ncc')
    except ValueError:
        pass
else:
    print(f"NCC data already exists: {ncc_path}")
```

### 4. Fix Checkpoint Directory Structure (trainer/trainer.py)

Update line 95 to include architecture subdirectory:

```python
# OLD: checkpoint_dir = Path("checkpoints") / cfg['problem']
# NEW:
architecture_str = "-".join(map(str, cfg['architecture']))
checkpoint_dir = Path("checkpoints") / cfg['problem'] / f"layers-{architecture_str}_act-{cfg['activation']}"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
```

### 5. Update NCC Runner (run_ncc.py and ncc/ncc_runner.py)

Change to use ncc_data.pt instead of eval_data.pt:

**run_ncc.py** (around line 20-30):

```python
# OLD: eval_data_path = f"datasets/{problem}/eval_data.pt"
# NEW:
ncc_data_path = f"datasets/{problem}/ncc_data.pt"

# Call NCC runner with ncc_data_path
ncc_summary = ncc_runner.run_ncc(
    model=model,
    eval_data_path=ncc_data_path,  # Using stratified NCC dataset
    cfg=config,
    run_dir=run_dir
)
```

**ncc/ncc_runner.py** (docstrings and prints):

Update references from "eval_data" to "ncc_data" for clarity.

### 6. Fix Confusion Matrix NaN Display (ncc/ncc_core.py)

Update confusion matrix computation (line 306-309) to mark empty classes as NaN:

```python
# Normalize by row (true class)
row_sums = confusion.sum(dim=1, keepdim=True)
confusion_normalized = confusion / row_sums
# Mark empty classes (no samples) as NaN for proper visualization
confusion_normalized[row_sums.squeeze() == 0] = float('nan')
```

Note: With stratified sampling, we shouldn't have empty classes, but this handles edge cases.

### 7. Update Confusion Matrix Plotting (ncc/ncc_plotting.py)

Handle NaN values in heatmap (line 251):

```python
# Plot heatmap with NaN masking
sns.heatmap(confusion_subset, annot=False, fmt='.2f', cmap='Blues',
            cbar=True, ax=ax, vmin=0, vmax=1,
            mask=torch.isnan(torch.tensor(confusion_subset)) if isinstance(confusion_subset, np.ndarray) 
                 else torch.isnan(confusion_subset))
```

### 8. Update Summary Prints

In `ncc/ncc_core.py` line 61, update print to be generic:

```python
# OLD: print(f"  Created 36 classes (bins^output_dim = {bins}^{output_dim})")
# Keep as is - already uses bins^output_dim calculation
```

## Testing

After implementation:

1. Delete existing `datasets/schrodinger/` to force regeneration
2. Run `python run_ncc.py`
3. Verify:

                                                                                                                                                                                                - `ncc_data.pt` created with ~2000 samples
                                                                                                                                                                                                - All bins**2 = 36 classes have at least 5 samples
                                                                                                                                                                                                - Confusion matrices show clear differences between layers
                                                                                                                                                                                                - Checkpoints saved to `checkpoints/schrodinger/layers-2-100-100-100-2_act-tanh/`
                                                                                                                                                                                                - No "showing X/36 classes" in confusion matrix titles (all 36 shown)

## Files to Modify

1. config/config.yaml (add n_samples_ncc, rarest_bin_samples_num)
2. utils/stratified_sampling.py (NEW FILE)
3. utils/dataset_gen.py (add NCC dataset generation)
4. trainer/trainer.py (fix checkpoint directory structure)
5. run_ncc.py (use ncc_data_path instead of eval_data_path)
6. ncc/ncc_runner.py (update prints/docs for clarity)
7. ncc/ncc_core.py (NaN for empty classes in confusion matrix)
8. ncc/ncc_plotting.py (handle NaN in heatmap)

### To-dos

- [ ] Add n_ncc_large and rarest_bin_samples_num to config.yaml
- [ ] Create utils/stratified_sampling.py with generic stratify_by_bins function
- [ ] Add NCC dataset generation to utils/dataset_gen.py
- [ ] Update checkpoint directory structure in trainer/trainer.py
- [ ] Update run_ncc.py and ncc_runner.py to use ncc_data.pt
- [ ] Fix empty class centers (inf) and confusion matrix (NaN) in ncc_core.py
- [ ] Handle NaN values in confusion matrix plotting
- [ ] Test complete pipeline with stratified NCC dataset