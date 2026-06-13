$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

try {
    if (-not (Test-Path ".venv\Scripts\python.exe")) {
        Write-Host "Creating the application environment..."
        python -m venv .venv
    }

    Write-Host "Preparing the application environment..."
    & ".venv\Scripts\python.exe" -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip update failed" }

    $hasNvidiaGpu = $null -ne (Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue)
    if ($hasNvidiaGpu) {
        & ".venv\Scripts\python.exe" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "NVIDIA GPU detected. Installing CUDA-enabled PyTorch..."
            & ".venv\Scripts\python.exe" -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
            if ($LASTEXITCODE -ne 0) { throw "CUDA-enabled PyTorch installation failed" }
        }
    }
    else {
        & ".venv\Scripts\python.exe" -c "import torch, torchvision" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "NVIDIA GPU was not detected. Installing CPU PyTorch..."
            & ".venv\Scripts\python.exe" -m pip install torch torchvision
            if ($LASTEXITCODE -ne 0) { throw "CPU PyTorch installation failed" }
        }
    }

    Write-Host "Installing required components..."
    & ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Application component installation failed" }

    & ".venv\Scripts\python.exe" -c "import torch; print('PyTorch device: ' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))"
    if ($LASTEXITCODE -ne 0) { throw "PyTorch verification failed" }

    Write-Host "Setup completed." -ForegroundColor Green
}
catch {
    Write-Host "Setup failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
