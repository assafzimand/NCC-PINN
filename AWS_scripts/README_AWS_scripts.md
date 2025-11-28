## AWS Scripts Usage

This folder contains helper scripts to make it easier to run NCC-PINN on an AWS EC2 GPU instance and download results back to your PC.

---

### 1. `prepare_AWS_run.sh` – Setup on EC2

**Purpose**: Prepare an EC2 GPU machine for NCC-PINN (Python, venv, repo clone/pull, dependencies).

#### When to use
- After you **SSH into the EC2 instance**:

  ```powershell
  # From your local Windows machine (PowerShell)
  cd C:\Users\assaf\Desktop\Coding\Msc\Master\NCC-PINN
  ssh -i .\NCC-PINN-ASSAF.pem ubuntu@<EC2_PUBLIC_IP>
  # for now - ssh -i .\NCC-PINN-ASSAF.pem ubuntu@13.60.229.209
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
ssh -i .\NCC-PINN-ASSAF.pem ubuntu@<EC2_PUBLIC_IP>
```

2. **On the EC2 shell**, run:

```bash
cd ~/NCC-PINN/AWS_scripts
bash prepare_AWS_run.sh
```

This will:
- `sudo apt update` and install `python3`, `python3-venv`, `git`
- Create a virtualenv at `~/.venv_ncc_pinn` (if missing)
- Activate that venv
- Clone or `git pull` the `~/NCC-PINN` repo
- Install `requirements.txt`

3. **Run experiments**:

```bash
source ~/.venv_ncc_pinn/bin/activate   # only if not already active
cd ~/NCC-PINN
python run_experiments.py              # or: python run_ncc.py
```

You can safely re-run `prepare_AWS_run.sh` on the same instance; it will just reuse the venv and update the repo.

---

### 2. `download_AWS_results.ps1` – Download outputs to your PC

**Purpose**: Copy a folder from `outputs/` on EC2 to your local machine using `scp`.

#### Requirements
- Run this **on your Windows PC** in the repo root.
- OpenSSH client installed (on recent Windows 10/11 it usually is).
- Your `.pem` key accessible (default in script: `NCC-PINN-ASSAF.pem` in repo root).

#### Usage
1. Open **PowerShell** in your local repo folder:

```powershell
cd C:\Users\assaf\Desktop\Coding\Msc\Master\NCC-PINN
.\AWS_scripts\download_AWS_results.ps1
```

2. The script will:
   - Ask for **EC2 Public IP** (press Enter to keep the default in the script, or type a new one).
   - Ask for the **path under `outputs/`** you want to download, for example:
     - `experiments/testing_new_plots_lr0.0001_ep100_bs512_bins5_20251123_205514`
     - `schrodinger_layers-2-50-50-50-2_act-tanh/20251123_213624`

3. It will:
   - Build the remote path: `/home/ubuntu/NCC-PINN/outputs/<your_subpath>`
   - Copy that directory to a local folder named `aws_outputs` in the repo root.

After it finishes, you’ll find your experiment results at:

```text
NCC-PINN\aws_outputs\<your_subpath>\...
```

You can then view plots and metrics locally as usual.


