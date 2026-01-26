## AWS Scripts Usage

This folder contains helper scripts to make it easier to run NCC-PINN on an AWS EC2 GPU instance and download results back to your PC.

---

## Quick Start

1. **First time?** Create an EC2 instance (see [Creating a New EC2 Instance](#creating-a-new-ec2-instance) below)
2. **Set up S3 auto-shutdown** (recommended): Follow [AWS_S3_SETUP.md](AWS_S3_SETUP.md)
3. **SSH into your instance** and run experiments

---

## Scripts Overview

| Script | Purpose | Run On |
|--------|---------|--------|
| `prepare_AWS_run.sh` | Setup EC2 environment | EC2 |
| `run_and_terminate.sh` | Run experiments + upload to S3 + auto-shutdown | EC2 |
| `download_AWS_results.ps1` | Download from EC2 via SSH (legacy) | Windows |
| `sync_from_s3.ps1` | Download from S3 (recommended) | Windows |

---

## Creating a New EC2 Instance

If your instance was terminated or you need a new one, follow these steps to recreate an identical setup:

### Step 1: Launch Instance

1. Go to [EC2 Console](https://console.aws.amazon.com/ec2/) → **"Launch instance"**

2. **Name**: `NCC-PINN-GPU` (or any name you prefer)

3. **Application and OS Images (AMI)**:
   - Search for: `Deep Learning AMI GPU PyTorch` 
   - Select: **Deep Learning AMI GPU PyTorch 2.x (Ubuntu 20.04)** or similar
   - This comes with CUDA and PyTorch pre-installed

4. **Instance type**:
   - For GPU training: `g4dn.xlarge` (cheapest GPU option, ~$0.526/hour)
   - For testing without GPU: `t3.medium` (~$0.042/hour)

5. **Key pair**:
   - Select your existing key: `NCC-PINN-ASSAF`
   - Or create a new one and save the `.pem` file

6. **Network settings**:
   - Allow SSH traffic from: **My IP** (more secure) or **Anywhere** (if IP changes often)

7. **Configure storage**:
   - Root volume: **50 GB** gp3 (enough for datasets and checkpoints)

8. Click **"Launch instance"**

### Step 2: Note the Public IP

1. Select your instance in EC2 Console
2. Copy the **Public IPv4 address** (e.g., `13.60.229.209`)
3. Update your scripts if the IP changed

### Step 3: Connect and Setup

```powershell
# From your Windows PC (in the Master directory)
cd C:\Users\assaf\Desktop\Coding\Msc\Master
ssh -i .\NCC-PINN-ASSAF.pem ubuntu@<NEW-IP-ADDRESS>

# On EC2: Run setup script
bash ~/NCC-PINN/AWS_scripts/prepare_AWS_run.sh

# Configure AWS CLI for S3 uploads (enter your AWS credentials)
aws configure
```

### Quick Reference: Instance Types

| Type | GPU | vCPU | RAM | Cost/hour | Best For |
|------|-----|------|-----|-----------|----------|
| `t3.medium` | None | 2 | 4GB | ~$0.04 | Testing, debugging |
| `g4dn.xlarge` | T4 (16GB) | 4 | 16GB | ~$0.53 | Training (recommended) |
| `g4dn.2xlarge` | T4 (16GB) | 8 | 32GB | ~$0.75 | Larger batch sizes |
| `p3.2xlarge` | V100 (16GB) | 8 | 61GB | ~$3.06 | Faster training |

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
- `sudo apt update` and install `python3`, `python3-venv`, `python3-dev`, `git`, `screen`
- Create a virtualenv at `~/.venv_ncc_pinn` (if missing)
- Activate that venv
- Clone or **force update** the `~/NCC-PINN` repo to match GitHub exactly
  - Uses `git reset --hard origin/main` to ensure EC2 matches your latest push
  - Clears Python cache to prevent using outdated `.pyc` files
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

---

### 3. `run_and_terminate.sh` – Auto-Shutdown After Experiments (Recommended!)

**Purpose**: Run experiments, upload results to S3, and automatically shut down the instance to save money.

#### Prerequisites

Before using this script, you must:
1. Create an S3 bucket (see [AWS_S3_SETUP.md](AWS_S3_SETUP.md))
2. Configure AWS CLI on EC2 with `aws configure`

#### Usage

```bash
# SSH into EC2
ssh -i .\NCC-PINN-ASSAF.pem ubuntu@<EC2-IP>

# (Optional) Run prepare script if needed
bash ~/NCC-PINN/AWS_scripts/prepare_AWS_run.sh

# Start experiments with auto-terminate
screen -S ncc_experiment
bash ~/NCC-PINN/AWS_scripts/run_and_terminate.sh

# Detach: Ctrl+A, then D
# You can now safely disconnect!
```

The script will:
1. Run `python run_experiments.py`
2. Upload `outputs/` and `checkpoints/` to S3
3. Wait 60 seconds (giving you time to cancel with Ctrl+C if needed)
4. Stop the EC2 instance

#### Configuration

Edit `run_and_terminate.sh` line 24 to set your S3 bucket:
```bash
S3_BUCKET="ncc-pinn-results"
```

---

### 4. `sync_from_s3.ps1` – Download Results from S3

**Purpose**: Download experiment results from S3 to your Windows PC.

#### Prerequisites

- AWS CLI installed on Windows
- AWS credentials configured (`aws configure`)
- See [AWS_S3_SETUP.md](AWS_S3_SETUP.md) for setup instructions

#### Usage

```powershell
cd C:\Users\assaf\Desktop\Coding\Msc\Master\NCC-PINN
.\AWS_scripts\sync_from_s3.ps1
```

The script will:
1. List all available experiments in S3
2. Show which ones are complete vs in-progress
3. Let you choose which experiment to download
4. Download to `AWS_scripts\aws_outputs\`

#### Configuration

Edit `sync_from_s3.ps1` lines 2-6 to set your S3 bucket and region:
```powershell
[string]$S3Bucket = "ncc-pinn-results",
[string]$Region = "eu-north-1"
```

---

## Recommended Workflow (Auto-Shutdown)

This workflow ensures you **never forget to stop your instance**:

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. SSH into EC2                                                  │
│    ssh -i .\NCC-PINN-ASSAF.pem ubuntu@<IP>                      │
├─────────────────────────────────────────────────────────────────┤
│ 2. (If needed) Run setup                                         │
│    bash ~/NCC-PINN/AWS_scripts/prepare_AWS_run.sh               │
├─────────────────────────────────────────────────────────────────┤
│ 3. Start experiments with auto-shutdown                          │
│    screen -S ncc_experiment                                      │
│    bash ~/NCC-PINN/AWS_scripts/run_and_terminate.sh             │
│    (Ctrl+A, D to detach)                                        │
├─────────────────────────────────────────────────────────────────┤
│ 4. Disconnect and go do other things!                           │
│    Instance will auto-stop when experiments complete            │
├─────────────────────────────────────────────────────────────────┤
│ 5. Later: Download results from S3                              │
│    .\AWS_scripts\sync_from_s3.ps1                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Legacy Workflow (Manual Download via SSH)

If you prefer not to use S3, you can still use the original `download_AWS_results.ps1`:

```powershell
cd C:\Users\assaf\Desktop\Coding\Msc\Master\NCC-PINN
.\AWS_scripts\download_AWS_results.ps1
```

**Note**: This requires the EC2 instance to still be running. You must manually stop it afterward!


