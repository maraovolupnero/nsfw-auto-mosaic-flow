@echo off
setlocal
cd /d "%~dp0"

if not exist "models" mkdir "models"
if exist "models\nsfw-anime-xl-x1280.pt" (
  echo The recommended model is already installed.
  pause
  exit /b 0
)

echo Downloading the recommended XL segmentation model...
curl.exe -L --fail --progress-bar ^
  "https://huggingface.co/01miku/anime-nsfw-segm-yolo26/resolve/main/nsfw-anime-xl-x1280.pt?download=true" ^
  -o "models\nsfw-anime-xl-x1280.pt"

if errorlevel 1 (
  del /q "models\nsfw-anime-xl-x1280.pt" 2>nul
  echo.
  echo Download failed. Open the model page and download the XL model manually:
  echo https://huggingface.co/01miku/anime-nsfw-segm-yolo26
  pause
  exit /b 1
)

echo.
echo Model installation completed.
pause
