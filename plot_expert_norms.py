"""Quick visualization of expert region norm distributions."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the expert regions data
json_path = Path("outputs/experiments/AToE-New/schrodinger_tests_20260210_055733-non-pretrained-10k-epochs/schrodinger-2-30-30-30-30-30-2-tanh/20260210_055737/adaptive_plots/expert_regions.json")

with open(json_path) as f:
    data = json.load(f)

regions = data['regions']

# Extract data
norms = [r['wavelet_norm'] for r in regions]
depths = [r['depth'] for r in regions]
spawn_epochs = [r['spawn_epoch'] for r in regions]

# Group by depth and spawn_epoch
depth_norms = {}
epoch_norms = {}

for r in regions:
    depth = r['depth']
    epoch = r['spawn_epoch']
    norm = r['wavelet_norm']

    if depth not in depth_norms:
        depth_norms[depth] = []
    depth_norms[depth].append(norm)

    if epoch not in epoch_norms:
        epoch_norms[epoch] = []
    epoch_norms[epoch].append(norm)

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Overall distribution
ax = axes[0]
ax.hist(norms, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
ax.set_xlabel('Wavelet Norm', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title(f'Overall Norm Distribution\n({len(norms)} experts)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axvline(np.median(norms), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(norms):.4f}')
ax.legend()

# 2. Distribution by depth (violin plot)
ax = axes[1]
sorted_depths = sorted(depth_norms.keys())
data_by_depth = [depth_norms[d] for d in sorted_depths]

# Create violin plot
parts = ax.violinplot(data_by_depth, positions=sorted_depths, widths=0.6,
                       showmeans=True, showmedians=True)

# Color the violins
for pc in parts['bodies']:
    pc.set_facecolor('steelblue')
    pc.set_alpha(0.7)
    pc.set_edgecolor('black')
    pc.set_linewidth(1.5)

# Add scatter points
for depth, norms_at_depth in depth_norms.items():
    x = [depth] * len(norms_at_depth)
    ax.scatter(x, norms_at_depth, alpha=0.4, s=30, color='darkblue', zorder=3)

ax.set_xlabel('Depth', fontsize=12, fontweight='bold')
ax.set_ylabel('Wavelet Norm', fontsize=12, fontweight='bold')
ax.set_title('Norm Distribution by Depth', fontsize=13, fontweight='bold')
ax.set_xticks(sorted_depths)
ax.grid(True, alpha=0.3, axis='y')

# Add count annotations
for depth in sorted_depths:
    count = len(depth_norms[depth])
    ax.text(depth, ax.get_ylim()[1] * 0.95, f'n={count}',
            ha='center', va='top', fontsize=9, fontweight='bold')

# 3. Distribution by spawn epoch (violin plot)
ax = axes[2]
sorted_epochs = sorted(epoch_norms.keys())
data_by_epoch = [epoch_norms[e] for e in sorted_epochs]

# Create violin plot
parts = ax.violinplot(data_by_epoch, positions=range(len(sorted_epochs)), widths=0.6,
                       showmeans=True, showmedians=True)

# Color the violins
for pc in parts['bodies']:
    pc.set_facecolor('coral')
    pc.set_alpha(0.7)
    pc.set_edgecolor('black')
    pc.set_linewidth(1.5)

# Add scatter points
for i, epoch in enumerate(sorted_epochs):
    norms_at_epoch = epoch_norms[epoch]
    x = [i] * len(norms_at_epoch)
    ax.scatter(x, norms_at_epoch, alpha=0.4, s=30, color='darkred', zorder=3)

ax.set_xlabel('Spawn Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Wavelet Norm', fontsize=12, fontweight='bold')
ax.set_title('Norm Distribution by Spawn Iteration', fontsize=13, fontweight='bold')
ax.set_xticks(range(len(sorted_epochs)))
ax.set_xticklabels([str(e) for e in sorted_epochs], rotation=45)
ax.grid(True, alpha=0.3, axis='y')

# Add count annotations
for i, epoch in enumerate(sorted_epochs):
    count = len(epoch_norms[epoch])
    ax.text(i, ax.get_ylim()[1] * 0.95, f'n={count}',
            ha='center', va='top', fontsize=9, fontweight='bold')

plt.tight_layout()

# Save plot
output_path = json_path.parent / "expert_norm_distributions.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

# Print statistics
print(f"\n{'='*60}")
print("Expert Norm Statistics")
print(f"{'='*60}")
print(f"Total experts: {len(norms)}")
print(f"Overall: mean={np.mean(norms):.4f}, median={np.median(norms):.4f}, std={np.std(norms):.4f}")
print(f"Min: {np.min(norms):.4f}, Max: {np.max(norms):.4f}")
print(f"\nBy Depth:")
for depth in sorted_depths:
    norms_d = depth_norms[depth]
    print(f"  Depth {depth}: n={len(norms_d)}, mean={np.mean(norms_d):.4f}, median={np.median(norms_d):.4f}")
print(f"\nBy Spawn Epoch:")
for epoch in sorted_epochs:
    norms_e = epoch_norms[epoch]
    print(f"  Epoch {epoch}: n={len(norms_e)}, mean={np.mean(norms_e):.4f}, median={np.median(norms_e):.4f}")

# plt.show()  # Commented out to avoid blocking
