
---

### STRUCTURE.md
```markdown
# NCC-PINN — Repository Structure

ncc_pinn/
│
├─ solvers/                        # problem-specific (placeholder now)
│   ├─ problem1_solver.py
│   └─ problem2_solver.py
│
├─ datasets/                       # persisted reusable data
│   ├─ problem1/
│   │   ├─ training_data.pt
│   │   └─ eval_data.pt
│   └─ problem2/
│       ├─ training_data.pt
│       └─ eval_data.pt
│
├─ losses/                         # problem-specific (placeholder now)
│   ├─ problem1_loss.py
│   └─ problem2_loss.py
│
├─ models/
│   └─ fc_model.py                 # FCNet + named layers + NCC hooks
│
├─ trainer/
│   ├─ trainer.py                  # train() → logging, ckpts, plots
│   ├─ plotting.py
│   └─ utils.py
│
├─ ncc/
│   ├─ ncc_core.py                 # centers, assignments, metrics
│   ├─ ncc_plotting.py             # compact plot set (+ confusion)
│   └─ ncc_runner.py               # run_ncc()
│
├─ config/
│   └─ config.yaml
│
├─ checkpoints/                    # saved model weights
│
├─ outputs/                        # per-run bundles (training + NCC)
│   └─ <problem>_layers-<...>_act-<activation>/
│       ├─ training_plots/
│       ├─ ncc_plots/
│       ├─ metrics.json
│       ├─ config_used.yaml
│       └─ summary.txt
│
├─ tests/                          # fast smoke tests
│   ├─ test_config.py
│   ├─ test_datasets.py
│   ├─ test_model.py
│   ├─ test_loss.py
│   ├─ test_trainer.py
│   ├─ test_ncc_core.py
│   └─ test_ncc_runner.py
│
├─ run_training.py                 # orchestrates training (or eval_only)
└─ run_ncc.py                      # orchestrates NCC on a checkpoint

Flow:
config.yaml → run_training.py → (generate datasets if missing) → model + loss → trainer → checkpoint
→ run_ncc.py → NCC analysis → outputs/<problem>_layers-..._act-.../
