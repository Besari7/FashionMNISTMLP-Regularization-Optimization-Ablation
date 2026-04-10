$ErrorActionPreference = 'Stop'

Set-Location "$PSScriptRoot\.."

$python = "c:/Users/berke/DL_Project/.venv/Scripts/python.exe"

if (-not (Test-Path $python)) {
    throw "Venv python not found at: $python"
}

Write-Host "Using Python: $python"
& $python -c "import sys, torch; print('Executable:', sys.executable); print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_CUDA')"

$env:FORCE_PHASE1_RETRAIN = '1'
Write-Host "FORCE_PHASE1_RETRAIN=$env:FORCE_PHASE1_RETRAIN"

& $python run.py

if ($LASTEXITCODE -ne 0) {
    throw "run.py exited with code $LASTEXITCODE"
}
