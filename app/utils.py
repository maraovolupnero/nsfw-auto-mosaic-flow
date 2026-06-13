from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np


def list_images(folder: str | Path, extensions: list[str]) -> list[Path]:
    directory = Path(folder)
    if not directory.is_dir():
        return []
    allowed = {ext.lower() for ext in extensions}
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in allowed),
        key=lambda path: path.name.lower(),
    )


def resolve_output_directory(input_dir: str | Path, output_dir: str | Path) -> tuple[Path, bool]:
    requested = str(output_dir).strip()
    if requested:
        destination = Path(requested)
        destination.mkdir(parents=True, exist_ok=True)
        return destination, False

    input_path = Path(input_dir)
    base = input_path / "mosaic_output"
    destination = base
    index = 1
    while destination.exists():
        destination = input_path / f"mosaic_output_{index:03d}"
        index += 1
    destination.mkdir(parents=True, exist_ok=False)
    return destination, True


def read_image(path: str | Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, cv2.error, ValueError):
        return None


def unique_output_path(output_dir: str | Path, filename: str) -> Path:
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    candidate = folder / filename
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    index = 1
    while True:
        candidate = folder / f"{stem}_{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def save_image(path: str | Path, image: np.ndarray) -> tuple[Path, str | None]:
    destination = Path(path)
    extension = destination.suffix.lower()
    fallback_message = None
    try:
        success, encoded = cv2.imencode(extension, image)
        if not success:
            raise cv2.error("encode failed")
        encoded.tofile(str(destination))
        return destination, None
    except (OSError, cv2.error):
        if extension != ".webp":
            raise
        destination = unique_output_path(destination.parent, destination.stem + ".png")
        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise OSError("PNGフォールバック保存に失敗しました。")
        encoded.tofile(str(destination))
        fallback_message = "webp_save_failed_fallback_to_png"
        return destination, fallback_message


def copy_image(source: str | Path, destination: str | Path) -> None:
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
