param(
    # S3 bucket name - CHANGE THIS to match your bucket
    [string]$S3Bucket = "ncc-pinn-results",
    # Local folder where results will be downloaded
    [string]$LocalTarget = "$PSScriptRoot\aws_outputs",
    # AWS region (should match your bucket region)
    [string]$Region = "eu-north-1"
)

Write-Host "=== Download NCC-PINN Results from S3 ===" -ForegroundColor Cyan
Write-Host ""

# Check if AWS CLI is installed
try {
    $awsVersion = (aws --version 2>&1)
    Write-Host "AWS CLI: $awsVersion" -ForegroundColor Gray
} catch {
    Write-Host "ERROR: AWS CLI not found!" -ForegroundColor Red
    Write-Host "Please install AWS CLI: https://aws.amazon.com/cli/" -ForegroundColor Yellow
    Write-Host "Then configure it with: aws configure" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "S3 Bucket: $S3Bucket" -ForegroundColor Cyan
Write-Host "Local Target: $LocalTarget" -ForegroundColor Cyan
Write-Host ""

# List available experiments in S3
Write-Host "Fetching available experiments from S3..." -ForegroundColor Cyan
try {
    $s3Contents = (aws s3 ls "s3://$S3Bucket/" --region $Region 2>&1)
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to list S3 bucket contents" -ForegroundColor Red
        Write-Host $s3Contents -ForegroundColor Red
        Write-Host ""
        Write-Host "Make sure:" -ForegroundColor Yellow
        Write-Host "  1. AWS CLI is configured (run 'aws configure')" -ForegroundColor Yellow
        Write-Host "  2. Your AWS credentials have S3 read permissions" -ForegroundColor Yellow
        Write-Host "  3. The bucket name '$S3Bucket' is correct" -ForegroundColor Yellow
        exit 1
    }
    
    # Parse experiment folders (they start with "experiments_")
    $experiments = @()
    foreach ($line in ($s3Contents -split "`n")) {
        if ($line -match "PRE\s+(experiments_\d+_\d+)/") {
            $experiments += $matches[1]
        }
    }
    
    if ($experiments.Count -eq 0) {
        Write-Host "No experiments found in S3 bucket." -ForegroundColor Yellow
        Write-Host "Make sure experiments have been uploaded from EC2." -ForegroundColor Gray
        exit 0
    }
    
    # Sort by timestamp (newest first)
    $experiments = $experiments | Sort-Object -Descending
    
} catch {
    Write-Host "ERROR: Failed to query S3: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Available experiments:" -ForegroundColor Cyan
for ($i = 0; $i -lt $experiments.Count; $i++) {
    $exp = $experiments[$i]
    # Check if this experiment is complete
    $completeMarker = (aws s3 ls "s3://$S3Bucket/$exp/_EXPERIMENT_COMPLETE.txt" --region $Region 2>&1)
    if ($completeMarker -match "_EXPERIMENT_COMPLETE.txt") {
        $status = "[COMPLETE]"
        $statusColor = "Green"
    } else {
        $status = "[IN PROGRESS]"
        $statusColor = "Yellow"
    }
    
    Write-Host "  $($i + 1)) $exp " -NoNewline
    Write-Host $status -ForegroundColor $statusColor
}

Write-Host ""
Write-Host "Enter experiment number to download (or press Enter for latest): " -NoNewline -ForegroundColor Cyan
$choice = Read-Host

if ([string]::IsNullOrWhiteSpace($choice)) {
    $selectedExp = $experiments[0]
} else {
    $choiceNum = [int]$choice
    if ($choiceNum -lt 1 -or $choiceNum -gt $experiments.Count) {
        Write-Host "Invalid choice. Using latest experiment." -ForegroundColor Yellow
        $selectedExp = $experiments[0]
    } else {
        $selectedExp = $experiments[$choiceNum - 1]
    }
}

Write-Host ""
Write-Host "Selected: $selectedExp" -ForegroundColor Green
Write-Host ""

# Create local target directory
$localExpDir = Join-Path $LocalTarget $selectedExp
New-Item -ItemType Directory -Force -Path $localExpDir | Out-Null

# Download from S3
Write-Host "Downloading from S3..." -ForegroundColor Cyan
Write-Host "  Source: s3://$S3Bucket/$selectedExp/" -ForegroundColor Gray
Write-Host "  Destination: $localExpDir" -ForegroundColor Gray
Write-Host ""

try {
    aws s3 sync "s3://$S3Bucket/$selectedExp/" "$localExpDir/" --region $Region
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Download failed" -ForegroundColor Red
        exit 1
    }
    
    Write-Host ""
    Write-Host "Download complete!" -ForegroundColor Green
    Write-Host "Results saved to: $localExpDir" -ForegroundColor Cyan
    
} catch {
    Write-Host "ERROR: Download failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan

# List what was downloaded
$outputsDir = Join-Path $localExpDir "outputs"
$checkpointsDir = Join-Path $localExpDir "checkpoints"

if (Test-Path $outputsDir) {
    $outputFolders = Get-ChildItem -Path $outputsDir -Directory
    Write-Host "  Outputs: $($outputFolders.Count) folders" -ForegroundColor Gray
}

if (Test-Path $checkpointsDir) {
    $checkpointFolders = Get-ChildItem -Path $checkpointsDir -Directory -Recurse | Where-Object { $_.Name -match "layers-|\.pt$" }
    Write-Host "  Checkpoints: $($checkpointFolders.Count) model(s)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Done!" -ForegroundColor Green
