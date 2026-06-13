from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil
from threading import Event
from typing import Callable

import numpy as np

from app.detector import Detection, YoloDetector
from app.logger import CsvResultLogger
from app.mosaic import apply_mosaic, draw_detections
from app.pdf_export import create_image_pdf, normalize_pdf_filename
from app.settings import ROOT_DIR
from app.settings import resolve_target_class_ids
from app.utils import copy_image, list_images, read_image, resolve_output_directory, save_image, unique_output_path


@dataclass
class ProcessResult:
    index: int
    total: int
    filename: str
    status: str
    detection_count: int
    masked_count: int
    original_path: str
    detection_preview_path: str
    output_path: str
    error_message: str = ""


ProgressCallback = Callable[[ProcessResult], None]
LogCallback = Callable[[str], None]


class BatchProcessor:
    def __init__(self, settings: dict, on_progress: ProgressCallback | None = None, on_log: LogCallback | None = None) -> None:
        self.settings = settings
        self.on_progress = on_progress or (lambda result: None)
        self.on_log = on_log or (lambda message: None)
        self._stop_event = Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> dict[str, int]:
        images = list_images(self.settings["input_dir"], self.settings["supported_extensions"])
        if not images:
            raise ValueError("入力フォルダに対応画像がありません。")
        output_dir, auto_created = resolve_output_directory(self.settings["input_dir"], self.settings.get("output_dir", ""))
        self.settings["output_dir"] = str(output_dir)
        if auto_created:
            self.on_log(f"出力フォルダが未指定のため作成しました: {output_dir}")
        preview_root = ROOT_DIR / "logs" / "preview_cache"
        shutil.rmtree(preview_root, ignore_errors=True)
        preview_dir = preview_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        preview_dir.mkdir(parents=True, exist_ok=True)
        logger = CsvResultLogger(ROOT_DIR / "logs" / "result.csv")
        self.on_log("YOLOモデルを読み込んでいます...")
        detector = YoloDetector(self.settings["model_path"], self.settings["use_gpu_if_available"])
        self.on_log(f"モデルを読み込みました。使用デバイス: {detector.device_label}")
        target_ids = resolve_target_class_ids(
            detector.names,
            self.settings.get("target_class_mode", "names"),
            self.settings.get("target_class_ids", []),
            self.settings.get("target_class_names", ["vagina", "penis"]),
        )
        if not target_ids:
            raise ValueError("モデル内に選択したモザイク対象クラスがありません。クラス名またはIDを確認してください。")
        selected = ", ".join(f"{class_id}:{detector.names.get(class_id, class_id)}" for class_id in target_ids)
        self.on_log(f"モザイク対象クラス: {selected}")
        self.on_log("処理方式: Segmentation Mask優先 / BBoxフォールバック")
        if self.settings.get("high_recall_detection", True):
            self.on_log("高精度検出: ON（全体推論 + 重なり付き分割推論）")

        counts = {"mosaic": 0, "copy": 0, "error": 0, "stopped": 0}
        output_paths: list[str] = []
        total = len(images)
        for index, source in enumerate(images, start=1):
            if self._stop_event.is_set():
                counts["stopped"] = total - index + 1
                logger.write(
                    filename=source.name, status="stopped", detection_count=0, masked_count=0,
                    detected_classes="", masked_classes="",
                    confidence=self.settings["confidence"], pixel_size="", expand_ratio=self.settings["expand_ratio"],
                    feather_px=self.settings["feather_px"],
                    error_message="user_stopped",
                )
                self.on_log("停止要求を受け付けました。")
                break
            result = self._process_one(detector, target_ids, source, output_dir, preview_dir, index, total, logger)
            counts[result.status] = counts.get(result.status, 0) + 1
            if Path(result.output_path).is_file():
                output_paths.append(result.output_path)
            self.on_progress(result)
        if self.settings.get("create_pdf_after_processing", False) and not counts["stopped"]:
            try:
                pdf_name = normalize_pdf_filename(self.settings.get("pdf_filename", "mosaic_images.pdf"))
                pdf_path, page_count = create_image_pdf(output_paths, output_dir / pdf_name)
                self.on_log(f"PDFを作成しました: {pdf_path}（{page_count}ページ）")
            except Exception as exc:
                self.on_log(f"PDFの作成に失敗しました: {exc}")
        return counts

    def _process_one(self, detector: YoloDetector, target_ids: list[int], source: Path, output_dir: Path, preview_dir: Path, index: int, total: int, logger: CsvResultLogger) -> ProcessResult:
        preview_dir.mkdir(parents=True, exist_ok=True)
        status = "error"
        output_path = unique_output_path(output_dir, source.name)
        error_message = ""
        detection_count = 0
        masked_count = 0
        actual_pixel_size: int | str = ""
        original = read_image(source)
        detections: list[Detection] = []
        processed = None
        detection_preview = None
        detection_preview_path = preview_dir / f"{index:06d}_{source.stem}.jpg"
        try:
            if original is None:
                raise OSError("failed_to_read_image")
            detections = detector.predict(
                original,
                self.settings["confidence"],
                self.settings.get("inference_size", 1280),
                self.settings.get("high_recall_detection", True),
                self.settings.get("tile_overlap", 0.20),
            )
            detection_count = len(detections)
            masked_detections = [detection for detection in detections if detection.class_id in target_ids]
            masked_count = len(masked_detections)
            detection_preview = draw_detections(original, detections, set(target_ids), self.settings["expand_ratio"])
            save_image(detection_preview_path, detection_preview)
            if masked_detections:
                processed, actual_pixel_size, _ = apply_mosaic(
                    original, masked_detections, self.settings["pixel_size_mode"], self.settings["pixel_size"],
                    self.settings["expand_ratio"], self.settings["feather_px"],
                )
                output_path, fallback = save_image(output_path, processed)
                error_message = fallback or ""
                status = "mosaic"
            else:
                copy_image(source, output_path)
                processed = original.copy()
                status = "copy"
            self.on_log(f"[{index}/{total}] {source.name}: {status} / 全検出 {detection_count}件 / マスク {masked_count}件")
        except Exception as exc:
            error_message = str(exc)
            try:
                if not output_path.exists():
                    copy_image(source, output_path)
                self.on_log(f"[{index}/{total}] {source.name}: エラー。元画像をコピーしました ({exc})")
            except Exception as copy_exc:
                error_message = f"{error_message}; copy_failed: {copy_exc}"
                self.on_log(f"[{index}/{total}] {source.name}: エラー ({error_message})")

        logger.write(
            filename=source.name, status=status, detection_count=detection_count, masked_count=masked_count,
            detected_classes=";".join(detection.class_name for detection in detections),
            masked_classes=";".join(detection.class_name for detection in detections if detection.class_id in target_ids),
            confidence=self.settings["confidence"], pixel_size=actual_pixel_size,
            expand_ratio=self.settings["expand_ratio"], feather_px=self.settings["feather_px"],
            error_message=error_message,
        )
        return ProcessResult(
            index, total, source.name, status, detection_count, masked_count,
            str(source), str(detection_preview_path) if detection_preview_path.exists() else "", str(output_path), error_message,
        )
