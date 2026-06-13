from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent
SETTINGS_PATH = ROOT_DIR / "settings.json"
SAMPLE_PATH = ROOT_DIR / "settings.json.sample"


DEFAULT_SETTINGS: dict[str, Any] = {
    "model_path": "models/nsfw-anime-xl-x1280.pt",
    "input_dir": "",
    "output_dir": "",
    "confidence": 0.30,
    "inference_size": 1280,
    "high_recall_detection": True,
    "tile_overlap": 0.20,
    "pixel_size_mode": "auto_1_80",
    "pixel_size": 16,
    "expand_ratio": 0.15,
    "feather_px": 8,
    "target_class_mode": "names",
    "target_class_ids": [],
    "target_class_names": ["vagina", "penis"],
    "preset": "recommended",
    "copy_when_no_detection": True,
    "keep_same_file_count": True,
    "use_gpu_if_available": True,
    "create_pdf_after_processing": False,
    "pdf_filename": "mosaic_images.pdf",
    "preview_mode": "processed",
    "supported_extensions": [".png", ".jpg", ".jpeg", ".webp"],
    "recommended_settings": {
        "confidence": 0.30,
        "pixel_size_mode": "auto_1_80",
        "pixel_size": 20,
        "expand_ratio": 0.15,
        "feather_px": 8,
    },
}


def _merge(defaults: dict[str, Any], loaded: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(defaults)
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_settings(path: Path = SETTINGS_PATH) -> dict[str, Any]:
    source = path if path.exists() else SAMPLE_PATH
    loaded: dict[str, Any] = {}
    if source.exists():
        try:
            loaded = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = {}
    settings = _merge(DEFAULT_SETTINGS, loaded)
    loaded_model_name = Path(str(settings.get("model_path", ""))).name.lower()
    legacy_model_names = {"yolo_nsfw_s.pt", "anime-nsfw-segm-yolo26.pt"}
    if loaded_model_name in legacy_model_names or settings.get("target_class_names") == ["vulva", "penis", "vaginal"]:
        settings["target_class_names"] = ["vagina", "penis"]
        settings["target_class_ids"] = []
        settings["target_class_mode"] = "names"
    if loaded_model_name in legacy_model_names and int(settings.get("inference_size", 1024)) == 1024:
        settings["inference_size"] = 1280
    settings["inference_size"] = 1280
    if settings.get("pixel_size_mode") == "auto_1_100":
        settings["pixel_size_mode"] = "auto_1_80"
    settings["preset"] = "recommended"
    settings.pop("presets", None)
    model_path = Path(settings.get("model_path", ""))
    if model_path and not model_path.is_absolute():
        model_path = ROOT_DIR / model_path
    bundled_model = ROOT_DIR / "models" / "nsfw-anime-xl-x1280.pt"
    if model_path.name.lower() in legacy_model_names or not model_path.is_file():
        model_path = bundled_model
    settings["model_path"] = str(model_path) if model_path else ""
    if not path.exists() or settings != loaded:
        save_settings(settings, path)
    return settings


def save_settings(settings: dict[str, Any], path: Path = SETTINGS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def parse_class_ids(text: str) -> list[int]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("対象クラスIDを入力してください。例: 0,1")
    result = sorted(set(int(value) for value in values))
    if any(value < 0 for value in result):
        raise ValueError("クラスIDは0以上の整数で指定してください。")
    return result


def parse_class_names(text: str) -> list[str]:
    values = [part.strip().lower() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("対象クラス名を入力してください。例: vagina,penis")
    return list(dict.fromkeys(values))


def normalize_model_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def resolve_target_class_ids(
    model_names: dict[int, str], mode: str, class_ids: list[int], class_names: list[str]
) -> list[int]:
    if mode == "ids":
        return sorted(set(int(value) for value in class_ids))
    wanted = {name.strip().lower() for name in class_names}
    return sorted(class_id for class_id, name in model_names.items() if name.strip().lower() in wanted)


def auto_pixel_size(width: int, height: int, mode: str, fixed: int) -> int:
    # Keep the old auto mode compatible while using the slightly coarser
    # recommended calculation for new and migrated settings.
    divisor = {"auto_1_100": 100, "auto_1_80": 80, "auto_1_60": 60}.get(mode)
    if divisor is None:
        return max(2, int(fixed))
    return max(4, round(max(width, height) / divisor))
