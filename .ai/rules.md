---

### RULES.md
```markdown
# NCC-PINN — Working Rules

## Coding
- Always use CUDA if available:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  x = x.to(device)
Prefer vectorized, batched CUDA ops; avoid Python loops.

No .cpu() mid-pipeline unless necessary.

Functions have type hints + docstrings.

Keep modules small and single-responsibility.

Placeholders (v1)
solvers/* and losses/* are runnable TODOs:

Accept correct inputs.

Return correctly shaped CUDA tensors.

Use deterministic random math so pipeline works end-to-end.

Testing (lightweight)
Each step adds a fast smoke test under tests/:

CUDA available, shapes correct, files created, quick forward/backward OK.

Keep tests under a few seconds.

Outputs (per run)
Folder name: <problem>_layers-<...>_act-<activation>

Contains training_plots/, ncc_plots/, metrics.json, config_used.yaml, summary.txt.

Efficiency Expectations
Batch everything; leverage broadcasting.

Use DataLoaders or batched tensors for residual/IC/BC.

NCC computations parallelized (no per-class loops if avoidable).