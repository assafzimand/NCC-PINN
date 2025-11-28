param(
    # Default EC2 public IP - you can override on each run
    [string]$Ec2Ip = "13.60.229.209",
    # Path to your SSH key (relative to repo root by default)
    [string]$PemPath = "$PSScriptRoot\NCC-PINN-ASSAF.pem",
    # Remote outputs root on EC2
    [string]$RemoteRoot = "/home/ubuntu/NCC-PINN/outputs",
    # Local folder where results will be downloaded
    [string]$LocalTarget = "$PSScriptRoot\aws_outputs"
)

Write-Host "=== Download NCC-PINN outputs from AWS EC2 ===" -ForegroundColor Cyan
Write-Host ""

Write-Host "Current EC2 Public IP: $Ec2Ip"
$ipInput = Read-Host "Enter EC2 Public IP (or press Enter to keep current)"
if ($ipInput) { $Ec2Ip = $ipInput }

$subPath = Read-Host "Enter path under outputs/ to download (e.g. 'experiments/...', or 'schrodinger_layers-.../TIMESTAMP')"
if (-not $subPath) {
    Write-Error "No path provided. Exiting."
    exit 1
}

$remotePath = "$RemoteRoot/$subPath"

Write-Host ""
Write-Host "Remote path:" -ForegroundColor Cyan
Write-Host "  ubuntu@$Ec2Ip:$remotePath"
Write-Host "Local destination:" -ForegroundColor Cyan
Write-Host "  $LocalTarget"
Write-Host ""

if (-not (Test-Path $PemPath)) {
    Write-Error "PEM file not found at '$PemPath'. Edit PemPath in this script or pass it as a parameter."
    exit 1
}

# Ensure destination directory exists
New-Item -ItemType Directory -Force -Path $LocalTarget | Out-Null

$scpCmd = "scp -i `"$PemPath`" -r ubuntu@$Ec2Ip:`"$remotePath`" `"$LocalTarget`""

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


