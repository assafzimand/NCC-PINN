# AWS S3 Setup for Auto-Shutdown Feature

This guide walks you through setting up S3 storage so that your EC2 experiments can automatically upload results and shut down the instance to save money.

---

## Overview

The auto-shutdown feature requires:
1. **S3 Bucket** - Storage for experiment results
2. **AWS CLI configured on EC2** - To upload results
3. **AWS CLI configured on your PC** - To download results

No special IAM roles or policies needed - just your existing AWS credentials!

---

## Step 1: Create S3 Bucket

1. Go to [AWS S3 Console](https://console.aws.amazon.com/s3/)

2. Click **"Create bucket"**

3. Configure the bucket:
   - **Bucket name**: `ncc-pinn-results` (or choose your own - must be globally unique)
   - **AWS Region**: `eu-north-1` (same as your EC2 instance for free data transfer)
   - **Block Public Access**: Keep all boxes checked (default - secure)
   - Leave other settings as default

4. Click **"Create bucket"**

5. **Important**: Note down your bucket name - you'll need to update it in the scripts:
   - `run_and_terminate.sh` line 24: `S3_BUCKET="ncc-pinn-results"`
   - `sync_from_s3.ps1` line 3: `[string]$S3Bucket = "ncc-pinn-results"`

---

## Step 2: Configure AWS CLI on EC2

SSH into your EC2 instance and run:

```bash
aws configure
```

Enter the following when prompted:
- **AWS Access Key ID**: Your access key (same as on your PC)
- **AWS Secret Access Key**: Your secret key (same as on your PC)
- **Default region name**: `eu-north-1` (or your region)
- **Default output format**: `json`

### Getting Your Access Keys

If you don't have access keys:

1. Go to [IAM Console → Users](https://console.aws.amazon.com/iam/home#/users)
2. Click on your username
3. Go to **"Security credentials"** tab
4. Click **"Create access key"**
5. Select **"Command Line Interface (CLI)"**
6. **Save both keys** (you won't see the secret again!)

### Test the Configuration

```bash
# Test S3 access
aws s3 ls s3://ncc-pinn-results/
```

If successful, you should see the bucket contents (or empty if new).

---

## Step 3: Configure AWS CLI on Your Windows PC

### Install AWS CLI

Download and install from: https://aws.amazon.com/cli/

Or use winget:
```powershell
winget install Amazon.AWSCLI
```

### Configure Credentials

Open PowerShell and run:

```powershell
aws configure
```

Enter the same credentials you used on EC2.

### Test the Configuration

```powershell
aws s3 ls s3://ncc-pinn-results/
```

---

## Step 4: Update Script Configuration

### On EC2 (run_and_terminate.sh)

Edit line 24 if your bucket name is different:
```bash
S3_BUCKET="your-bucket-name"
```

### On Windows (sync_from_s3.ps1)

Edit lines 3-6 if your bucket name or region is different:
```powershell
[string]$S3Bucket = "your-bucket-name",
[string]$Region = "eu-north-1"
```

---

## Usage

### Running Experiments (on EC2)

```bash
# SSH into EC2
ssh -i .\NCC-PINN-ASSAF.pem ubuntu@<EC2-IP>

# Prepare environment (first time or after changes)
bash ~/NCC-PINN/AWS_scripts/prepare_AWS_run.sh

# Run experiments with auto-shutdown
screen -S ncc_experiment
bash ~/NCC-PINN/AWS_scripts/run_and_terminate.sh

# Detach: Ctrl+A, then D
# Now safe to disconnect!
```

### Downloading Results (on Windows)

```powershell
cd C:\Users\assaf\Desktop\Coding\Msc\Master\NCC-PINN
.\AWS_scripts\sync_from_s3.ps1
```

---

## How It Saves Money

| What Happens | Cost Impact |
|--------------|-------------|
| Instance runs experiments | Normal EC2 charges |
| Results upload to S3 | Free (same region) |
| Instance shuts down | **Compute charges STOP** |
| Instance stays stopped | Only tiny EBS storage cost (~$4/month for 50GB) |

The big cost is **compute time** (~$0.53/hour for g4dn.xlarge). By auto-shutting down, you never forget to stop the instance!

---

## Cost Estimates

| Resource | Cost |
|----------|------|
| S3 Storage | ~$0.023/GB per month |
| S3 Upload (from EC2 same region) | **Free** |
| S3 Download (to your PC) | ~$0.09/GB |
| S3 Requests | ~$0.0004 per 1000 requests |

**Example**: 10GB of results = ~$0.23/month storage + ~$0.90 one-time download

---

## Troubleshooting

### "Unable to locate credentials" on EC2
- Run `aws configure` and enter your credentials
- Check that `~/.aws/credentials` file exists

### "Access Denied" when uploading to S3
- Verify your AWS user has S3 permissions
- Check the bucket name is correct
- Try: `aws s3 ls s3://your-bucket-name/`

### Can't list S3 bucket from Windows
- Run `aws configure` and enter your credentials
- Check the bucket name and region are correct

### Instance didn't shut down
- Check if the script completed (look at screen session output)
- You can manually stop from [EC2 Console](https://console.aws.amazon.com/ec2/)

### How to restart a stopped instance
1. Go to [EC2 Console](https://console.aws.amazon.com/ec2/)
2. Select your stopped instance
3. Click **Actions → Instance state → Start instance**
4. Note: The public IP will change! Check the new IP in the console.
