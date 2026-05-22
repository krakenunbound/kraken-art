# Kraken Art — Python sidecar install script
# Re-runnable. Creates venv if missing, installs torch+CUDA, then requirements.txt.

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Py   = Join-Path $Root 'python\venv\Scripts\python.exe'
$Req  = Join-Path $Root 'python\requirements.txt'

if (-not (Test-Path $Py)) {
    Write-Host "[install] creating venv..."
    py -3.11 -m venv (Join-Path $Root 'python\venv')
    & $Py -m pip install --upgrade pip
}

Write-Host "[install] installing torch + torchvision (CUDA 12.4)..."
& $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Write-Host "[install] installing diffusion stack..."
& $Py -m pip install -r $Req

Write-Host "[install] verifying torch CUDA..."
& $Py -c "import torch; print(f'torch {torch.__version__} CUDA={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

Write-Host "[install] done."
