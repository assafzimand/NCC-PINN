param(
    # Default EC2 public IP - you can override on each run
    [string]$Ec2Ip = "13.60.229.209",
    # Path to your SSH key (relative to repo root by default)
    # Repo structure: Master\NCC-PINN-ASSAF.pem and Master\NCC-PINN\AWS_scripts\this_file
    # So from $PSScriptRoot (NCC-PINN\AWS_scripts) we need to go up two levels.
    [string]$PemPath = "$PSScriptRoot\..\..\NCC-PINN-ASSAF.pem",
    # Remote outputs root on EC2
    [string]$RemoteRoot = "/home/ubuntu/NCC-PINN/outputs",
    # Experiments root (where run_experiments.py writes)
    [string]$ExperimentsRoot = "/home/ubuntu/NCC-PINN/outputs/experiments",
    # Local folder where results will be downloaded
    [string]$LocalTarget = "$PSScriptRoot\aws_outputs"
)

Write-Host "=== Download NCC-PINN outputs from AWS EC2 ===" -ForegroundColor Cyan
Write-Host ""

Write-Host "Current EC2 Public IP: $Ec2Ip"
$ipInput = Read-Host "Enter EC2 Public IP (or press Enter to keep current)"
if ($ipInput) { $Ec2Ip = $ipInput }
Write-Host ""

if (-not (Test-Path $PemPath)) {
    Write-Error "PEM file not found at '$PemPath'. Edit PemPath in this script or pass it as a parameter."
    exit 1
}

# Check if any screen sessions are running (optional info)
Write-Host "Checking for active screen sessions on EC2..." -ForegroundColor Cyan
try {
    $screenSessions = (& ssh -i $PemPath ubuntu@$Ec2Ip "screen -ls 2>&1").Trim()
    if ($screenSessions -match "ncc_experiment") {
        Write-Host "  ⚠ Active screen session detected: experiments may still be running!" -ForegroundColor Yellow
        Write-Host "  Tip: SSH in and run 'screen -r ncc_experiment' to check progress" -ForegroundColor Yellow
    } else {
        Write-Host "  No active screen sessions found" -ForegroundColor Gray
    }
} catch {
    Write-Host "  (Could not check screen sessions)" -ForegroundColor Gray
}
Write-Host ""

# Find the most recently modified experiment under outputs/experiments on EC2
Write-Host "Querying EC2 for latest experiment under outputs/experiments/ ..." -ForegroundColor Cyan
try {
    # This assumes OpenSSH client is installed on Windows
    $lastFolder = (& ssh -i $PemPath ubuntu@$Ec2Ip "cd $ExperimentsRoot && ls -1t | head -1").Trim()
} catch {
    Write-Error "Failed to query EC2 via ssh. Make sure ssh is installed and the IP/key are correct."
    exit 1
}

if (-not $lastFolder) {
    Write-Error "Could not determine latest folder under '$ExperimentsRoot' on EC2."
    exit 1
}

$remotePath = "$ExperimentsRoot/$lastFolder"

Write-Host "Latest folder detected:" -ForegroundColor Cyan
Write-Host "  $lastFolder"
Write-Host ""
Write-Host "Remote path:" -ForegroundColor Cyan
Write-Host "  ubuntu@${Ec2Ip}:${remotePath}"
Write-Host "Local destination:" -ForegroundColor Cyan
Write-Host "  $LocalTarget"
Write-Host ""

# Ensure destination directory exists locally
New-Item -ItemType Directory -Force -Path $LocalTarget | Out-Null

$scpCmd = "scp -i `"$PemPath`" -r ubuntu@${Ec2Ip}:`"$remotePath`" `"$LocalTarget`""

Write-Host "Running:" -ForegroundColor Yellow
Write-Host "  $scpCmd"
Write-Host ""

try {
    Invoke-Expression $scpCmd
    Write-Host ""
    Write-Host "Download complete." -ForegroundColor Green
}
catch {
    Write-Error "scp failed: $_"
}