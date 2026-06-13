@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found.
  echo Install Python 3.10 or later, then run this file again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo First-time setup is starting. This may take several minutes.
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
  if errorlevel 1 (
    echo.
    echo Setup failed. Check your internet connection and try again.
    pause
    exit /b 1
  )
)

".venv\Scripts\python.exe" -c "import PySide6, ultralytics" >nul 2>nul
if errorlevel 1 (
  echo Required components are missing. Setup will run again.
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
  if errorlevel 1 (
    echo.
    echo Setup failed. Check your internet connection and try again.
    pause
    exit /b 1
  )
)

where nvidia-smi >nul 2>nul
if not errorlevel 1 (
  ".venv\Scripts\python.exe" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
  if errorlevel 1 (
    echo NVIDIA GPU was detected, but CUDA PyTorch is not ready.
    echo GPU setup is starting. This may take several minutes.
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
    if errorlevel 1 (
      echo.
      echo GPU setup failed. Check your internet connection and try again.
      pause
      exit /b 1
    )
  )
)

start "NSFW Auto Mosaic Flow" ".venv\Scripts\pythonw.exe" "%~dp0main.py"
exit /b 0
