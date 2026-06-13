@echo off
setlocal
cd /d "%~dp0"

echo NVIDIA GPU support setup is starting.
echo The CUDA-enabled PyTorch download may take several minutes.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
if errorlevel 1 (
  echo.
  echo GPU setup failed. Check your internet connection and try again.
  pause
  exit /b 1
)

echo.
".venv\Scripts\python.exe" -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not detected')"
pause
