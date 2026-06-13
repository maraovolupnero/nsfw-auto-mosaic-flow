# NSFW Auto Mosaic Flow

**English** | [日本語](README_JA.md)

A Windows desktop application that uses Ultralytics YOLO segmentation models to detect sensitive regions in NSFW images and apply pixelated mosaics to entire folders. It supports NVIDIA GPU acceleration and can also run on CPU-only systems.

Author: [maraovolupnero](https://github.com/maraovolupnero) / License: [MIT](LICENSE)

> [!WARNING]
> Automatic detection is not perfect. Always review every processed image before publishing or distributing it.

## Features

- Batch processing for PNG, JPG, JPEG, and WebP images
- Support for custom YOLO `.pt` models and class selection
- Automatic class list generation from `model.names`
- Segmentation masks with dilation, feathering, and pixelated mosaic compositing
- Automatic fallback to elliptical bounding-box masks for detection-only models
- Adjustable confidence, pixel size, mask expansion, and feathering
- Optional high-recall tiled inference for small or distant targets
- One-click recommended settings with manual slider adjustment
- Original, Detection, and Processed previews
- Detection mask contours displayed in the preview
- Manual mosaic brush, eraser, undo, and save tools
- Preview zoom from 25% to 400%, with drag-to-pan navigation
- Mouse-wheel navigation through the processing queue
- Automatic copying of images with no detections
- CSV logging of all detected classes and the classes actually masked
- Automatic output-folder creation when no output path is specified
- PDF export with one processed image per page
- Automatic NVIDIA GPU use with CPU fallback
- All image processing is performed locally

## Recommended Window Size

The interface is designed for a window size of at least `1280x800`. Its initial size is `1420x940`.

The application can be used in a smaller window, but the settings panels and preview area may become difficult to read. Maximizing the window or using a size close to the default is recommended.

## Processing Behavior

When segmentation masks are available, the application uses `results[0].masks`. Selected masks are expanded using morphological dilation, feathered with Gaussian blur, and combined with the pixelated image.

The Detection preview shows expanded selected-class contours in green and unselected-class contours in gray. Detection-only models fall back to bounding boxes and elliptical masks.

`Expand 15%` expands a segmentation mask by approximately 15%. In bounding-box fallback mode, each side of the box is expanded by 15%. High-recall detection performs both full-image inference and overlapping tiled inference. The inference resolution is fixed at `imgsz=1280`.

After automatic processing, select an image from the queue and open the Processed tab to apply manual corrections. Paint the missing area with the mosaic brush, use the eraser or undo button when necessary, and then save the additional mosaic.

For PDF output, enable automatic PDF creation before processing or use the PDF creation button after manual corrections. Images are ordered by filename, with one image placed on each page.

## Recommended Model

The recommended model is the XL model from [01miku/anime-nsfw-segm-yolo26](https://huggingface.co/01miku/anime-nsfw-segm-yolo26).

- Model file: `nsfw-anime-xl-x1280.pt`
- Destination: `models/nsfw-anime-xl-x1280.pt`
- Model license: MIT. Check the model page for the latest terms.

The model is approximately 135 MB and is not included in this repository. Run `DOWNLOAD_RECOMMENDED_MODEL.bat` or download it manually from Hugging Face.

The class list is generated from `model.names`. By default, only `vagina` and `penis` are selected for masking. Classes such as `nipple`, `anus`, `pubic hair`, `female face`, and `male face` may appear in detection results but are not masked by default.

## CPU and GPU Support

- NVIDIA GPU detected: the application installs or uses CUDA-enabled PyTorch and runs inference on the GPU.
- No NVIDIA GPU detected: the CPU version of PyTorch is used.
- CPU processing works normally but may be significantly slower, especially with high-recall detection enabled.
- If CUDA inference fails, the application automatically retries on CPU.

## Installation

Requires Python 3.10 or later. Run the following commands in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Alternatively, open PowerShell in the project folder and run:

```powershell
./setup.ps1
```

Download the recommended model by double-clicking:

```text
DOWNLOAD_RECOMMENDED_MODEL.bat
```

If an NVIDIA GPU is detected but the installed PyTorch build is CPU-only, the launcher attempts to install the CUDA 12.8 build automatically.

## Running the Application

Double-click `NSFW_Auto_Mosaic_Flow_START.bat`. The launcher prepares the required environment on first use.

You can also start the application directly:

```powershell
python main.py
```

Basic workflow:

1. Select a YOLO `.pt` model.
2. Select an input folder and, optionally, an output folder.
3. Select the classes to mask. The default selection is `vagina,penis`.
4. Adjust the settings or apply the recommended settings.
5. Start batch processing.
6. Review the Detection and Processed previews before publishing the results.

Application settings are saved to `settings.json` when changed and when the application closes.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Building an Executable

```powershell
pyinstaller --noconfirm --clean nsfw_auto_mosaic_flow.spec
```

The executable is generated at:

```text
dist/NSFW Auto Mosaic Flow/NSFW Auto Mosaic Flow.exe
```

Download the model separately and place it in the `models` folder.

## License

The application source code is released under the [MIT License](LICENSE).

The recommended model is a separate project. Review its license and usage terms on the model distribution page.

## Notes and Limitations

- Detection accuracy depends on the selected model and source images.
- Only load `.pt` files from sources you trust. PyTorch model files may contain serialized Python objects.
- This application does not train or fine-tune models.
- Segmentation masks are preferred, but detection-only models can be used through the bounding-box fallback.
- Input images are processed locally and are not uploaded by this application.
- If an image cannot be loaded or processed, the application attempts to copy the original image to the output folder.
