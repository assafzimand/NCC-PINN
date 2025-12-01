## AWS Scripts Usage

This folder contains helper scripts to make it easier to run NCC-PINN on an AWS EC2 GPU instance and download results back to your PC.

---

### 1. `prepare_AWS_run.sh` – Setup on EC2

**Purpose**: Prepare an EC2 GPU machine for NCC-PINN (Python, venv, repo clone/pull, dependencies).

#### When to use
- After you **SSH into the EC2 instance**:

  ```powershell
  # From your local Windows machine (PowerShell)
  # Navigate to the Master folder first:
  cd C:\Users\assaf\Desktop\Coding\Msc\Master
  ssh -i .\NCC-PINN-ASSAF.pem ubuntu@13.60.229.209
  ```

- On a **fresh instance** or whenever you want to make sure the environment + repo are ready.

#### One-time setup (already done if this file exists on EC2)
On EC2, make sure the script is executable:

```bash
chmod +x ~/NCC-PINN/AWS_scripts/prepare_AWS_run.sh
```

#### Morning workflow
1. **From your local machine**, SSH into EC2:

```powershell
# Make sure you're in the Master directory where the .pem file is
cd C:\Users\assaf\Desktop\Coding\Msc\Master
ssh -i .\NCC-PINN-ASSAF.pem ubuntu@13.60.229.209
```

2. **On the EC2 shell**, run:

```bash
cd ~/NCC-PINN/AWS_scripts
bash prepare_AWS_run.sh
```

This will:
- `sudo apt update` and install `python3`, `python3-venv`, `git`, `screen`
- Create a virtualenv at `~/.venv_ncc_pinn` (if missing)
- Activate that venv
- Clone or `git pull` the `~/NCC-PINN` repo
- Install `requirements.txt`

3. **Run experiments in a screen session** (recommended - allows you to disconnect safely):

```bash
# Start a new screen session
screen -S ncc_experiment

# Inside the screen session, activate venv and navigate to repo:
source ~/.venv_ncc_pinn/bin/activate
cd ~/NCC-PINN
python run_experiments.py              # or: python run_ncc.py

# To detach (leave it running): Press Ctrl+A, then D
# Now you can safely disconnect from EC2!
```

4. **Later: Check on your running experiment**:

```bash
# SSH back into EC2 (from Master directory on your PC)
ssh -i .\NCC-PINN-ASSAF.pem ubuntu@13.60.229.209

# Reattach to your screen session to see live progress
screen -r ncc_experiment

# When done viewing: Press Ctrl+A, then D to detach again
```

**Useful screen commands:**
- `screen -ls` - List all screen sessions
- `screen -r ncc_experiment` - Reattach to a session
- `Ctrl+A, then D` - Detach from a session (keeps it running)
- `Ctrl+C` - Stop the running program (while attached)
- `exit` - Close the screen session (while attached)

You can safely re-run `prepare_AWS_run.sh` on the same instance; it will just reuse the venv and update the repo.

---

### 2. `download_AWS_results.ps1` – Download outputs to your PC

**Purpose**: Copy a folder from `outputs/` on EC2 to your local machine using `scp`.

#### Requirements
- Run this **on your Windows PC** in the repo root.
- OpenSSH client installed (on recent Windows 10/11 it usually is).
- Your `.pem` key accessible (default in script: `NCC-PINN-ASSAF.pem` in repo root).

#### Usage
1. Open **PowerShell** and navigate to your repo folder:

```powershell
cd C:\Users\assaf\Desktop\Coding\Msc\Master\NCC-PINN
.\AWS_scripts\download_AWS_results.ps1
```

2. The script will:
   - Check for active **screen sessions** on EC2 (to warn you if experiments are still running)
   - Ask for **EC2 Public IP** (press Enter to keep the default in the script, or type a new one)
   - Automatically detect the **latest experiment** under `outputs/experiments/`

3. It will:
   - Build the remote path to the latest experiment
   - Copy that directory to a local folder named `AWS_scripts\aws_outputs` in the repo root

After it finishes, you'll find your experiment results at:

```text
NCC-PINN\AWS_scripts\aws_outputs\<experiment_folder>\...
```

You can then view plots and metrics locally as usual.

**Note:** If experiments are still running in a screen session, you can still download partial results. The script will warn you if it detects an active screen session.

---

## Quick Reference: Screen Commands

| Task | Command |
|------|---------|
| Start a new screen session | `screen -S ncc_experiment` |
| Detach from session (keeps running) | Press `Ctrl+A`, then `D` |
| List all screen sessions | `screen -ls` |
| Reattach to a session | `screen -r ncc_experiment` |
| Stop the running program | `Ctrl+C` (while attached) |
| Exit/close the screen session | `exit` (while attached) |

**Why use screen?**
- Your experiment continues running even if your PC goes to sleep or loses connection
- You can check progress anytime by reattaching
- Perfect for long-running neural network training jobs


