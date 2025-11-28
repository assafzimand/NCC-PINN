#!/usr/bin/env bash
set -e

# Helper script to prepare an EC2 GPU instance for running NCC-PINN.
# Run this *on the EC2 machine* after you SSH in.
#
# Usage:
#   bash ~/NCC-PINN/AWS_scripts/prepare_AWS_run.sh
#   # or if copied elsewhere:
#   bash prepare_AWS_run.sh

# === Configuration ===
REPO_URL="https://github.com/assafzimand/NCC-PINN.git"
REPO_DIR="$HOME/NCC-PINN"
VENV_DIR="$HOME/.venv_ncc_pinn"

echo "=== Updating apt and installing dependencies (python3, venv, git) ==="
sudo apt update
sudo apt install -y python3 python3-venv git

echo
echo "=== Creating Python virtual environment (if missing) ==="
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

echo "=== Activating virtual environment ==="
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo
echo "=== Cloning or updating NCC-PINN repo ==="
if [ ! -d "$REPO_DIR" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
else
  cd "$REPO_DIR"
  git pull
fi

cd "$REPO_DIR"

echo
echo "=== Installing Python dependencies ==="
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  echo "WARNING: requirements.txt not found in $REPO_DIR"
fi

echo
echo "=== Environment ready ==="
echo "To start working on this instance next time, run:"
echo "  source $VENV_DIR/bin/activate"
echo "  cd $REPO_DIR"
echo
echo "To launch experiments:"
echo "  python run_experiments.py"
echo "or a single run:"
echo "  python run_ncc.py"


