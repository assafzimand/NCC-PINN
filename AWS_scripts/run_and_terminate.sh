#!/usr/bin/env bash
set -e

# =============================================================================
# run_and_terminate.sh - Run experiments and auto-shutdown EC2 instance
# =============================================================================
# This script:
#   1. Runs python run_experiments.py
#   2. Uploads outputs/ to S3
#   3. Shuts down the EC2 instance (stops billing!)
#
# Usage:
#   screen -S ncc_experiment
#   bash ~/NCC-PINN/AWS_scripts/run_and_terminate.sh
#   # Detach with Ctrl+A, D - safe to disconnect!
#
# Prerequisites:
#   - S3 bucket created (see AWS_S3_SETUP.md)
#   - AWS CLI configured on EC2: run 'aws configure' with your credentials
# =============================================================================

# === Configuration ===
# CHANGE THIS TO MATCH YOUR S3 BUCKET:
S3_BUCKET="ncc-pinn-results"
REPO_DIR="$HOME/NCC-PINN"
VENV_DIR="$HOME/.venv_ncc_pinn"

# === Colors for output ===
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# === Helper functions ===
log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# === Main script ===
echo "============================================================================="
echo "  NCC-PINN: Run Experiments and Auto-Shutdown"
echo "============================================================================="
echo ""

log_info "S3 Bucket: $S3_BUCKET"
echo ""

# Check if AWS CLI is configured
log_info "Checking AWS CLI configuration..."
if ! aws sts get-caller-identity &>/dev/null; then
    log_error "AWS CLI is not configured!"
    log_error "Please run 'aws configure' and enter your AWS credentials."
    log_error ""
    log_error "You need:"
    log_error "  - AWS Access Key ID"
    log_error "  - AWS Secret Access Key"
    log_error "  - Default region (e.g., eu-north-1)"
    exit 1
fi

AWS_USER=$(aws sts get-caller-identity --query 'Arn' --output text 2>/dev/null || echo "unknown")
log_success "AWS CLI configured as: $AWS_USER"
echo ""

# Check S3 bucket access
log_info "Checking S3 bucket access..."
if ! aws s3 ls "s3://$S3_BUCKET" &>/dev/null; then
    log_error "Cannot access S3 bucket: $S3_BUCKET"
    log_error "Make sure:"
    log_error "  1. The bucket exists"
    log_error "  2. Your AWS user has S3 permissions"
    log_error "  3. The bucket name is correct in this script (line 24)"
    exit 1
fi
log_success "S3 bucket accessible"
echo ""

# Activate virtual environment
log_info "Activating virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    log_error "Virtual environment not found at $VENV_DIR"
    log_error "Please run prepare_AWS_run.sh first!"
    exit 1
fi
source "$VENV_DIR/bin/activate"

# Change to repo directory
cd "$REPO_DIR"
log_info "Working directory: $(pwd)"
echo ""

# Create timestamp for this experiment run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
S3_PATH="s3://$S3_BUCKET/experiments_$TIMESTAMP"

log_info "Results will be uploaded to: $S3_PATH"
echo ""

# === Run experiments ===
echo "============================================================================="
log_info "Starting experiments..."
echo "============================================================================="
echo ""

EXPERIMENT_SUCCESS=true
python run_experiments.py || EXPERIMENT_SUCCESS=false

echo ""
if [ "$EXPERIMENT_SUCCESS" = true ]; then
    log_success "Experiments completed successfully!"
else
    log_warn "Experiments completed with errors (some may have failed)"
fi
echo ""

# === Upload to S3 ===
echo "============================================================================="
log_info "Uploading results to S3..."
echo "============================================================================="
echo ""

# Upload outputs directory
log_info "Uploading outputs/ directory..."
if aws s3 sync "$REPO_DIR/outputs/" "$S3_PATH/outputs/" --quiet; then
    log_success "Outputs uploaded."
else
    log_warn "Some outputs may have failed to upload"
fi

# Upload checkpoints directory
log_info "Uploading checkpoints/ directory..."
if aws s3 sync "$REPO_DIR/checkpoints/" "$S3_PATH/checkpoints/" --quiet; then
    log_success "Checkpoints uploaded."
else
    log_warn "Some checkpoints may have failed to upload"
fi

# Upload experiment plan
if [ -f "$REPO_DIR/experiments_plan.yaml" ]; then
    log_info "Uploading experiments_plan.yaml..."
    aws s3 cp "$REPO_DIR/experiments_plan.yaml" "$S3_PATH/experiments_plan.yaml" --quiet
fi

echo ""
log_success "All results uploaded to: $S3_PATH"
echo ""

# === Create completion marker ===
echo "Experiment completed at $(date)" > /tmp/experiment_complete.txt
echo "S3 Path: $S3_PATH" >> /tmp/experiment_complete.txt
aws s3 cp /tmp/experiment_complete.txt "$S3_PATH/_EXPERIMENT_COMPLETE.txt" --quiet
log_success "Completion marker uploaded"
echo ""

# === Shutdown ===
echo "============================================================================="
log_warn "Instance will SHUT DOWN in 60 seconds to save costs!"
log_warn "Press Ctrl+C to cancel shutdown"
echo "============================================================================="
echo ""
log_info "After shutdown:"
log_info "  - Instance will be STOPPED (not terminated)"
log_info "  - No more compute charges"
log_info "  - You can restart the instance later if needed"
log_info "  - Results are safe in S3: $S3_PATH"
echo ""

# Give user a chance to cancel
for i in {60..1}; do
    echo -ne "\r  Shutting down in $i seconds... (Ctrl+C to cancel)  "
    sleep 1
done
echo ""
echo ""

log_info "Initiating shutdown..."
sudo shutdown -h now
