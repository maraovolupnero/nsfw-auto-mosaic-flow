from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
import cv2

from app.settings import ROOT_DIR, normalize_model_names


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    class_id: int
    confidence: float
    class_name: str
    mask: np.ndarray | None = None


class YoloDetector:
    def __init__(self, model_path: str, use_gpu_if_available: bool = True) -> None:
        path = Path(model_path)
        if not path.is_file() or path.suffix.lower() != ".pt":
            raise ValueError("有効なYOLOモデル(.pt)を選択してください。")
        config_dir = ROOT_DIR / "logs" / "ultralytics"
        (config_dir / "Ultralytics").mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralyticsが未インストールです。requirements.txtをインストールしてください。") from exc

        self._torch: Any = None
        try:
            import torch

            self._torch = torch
        except ImportError:
            pass
        self.device = "cpu"
        self.device_label = "CPU"
        if use_gpu_if_available and self._torch is not None and self._torch.cuda.is_available():
            self.device = 0
            self.device_label = f"GPU: {self._torch.cuda.get_device_name(0)}"
        self.model = YOLO(str(path))
        self.names = normalize_model_names(getattr(self.model, "names", {}))

    def _predict_once(self, image: np.ndarray, confidence: float, inference_size: int) -> list[Detection]:
        try:
            results = self.model.predict(
                source=image,
                conf=float(confidence),
                imgsz=int(inference_size),
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            if self.device != "cpu" and "cuda" in str(exc).lower():
                self.device = "cpu"
                self.device_label = "CPU (CUDAエラーのため切替)"
                results = self.model.predict(
                    source=image, conf=float(confidence), imgsz=int(inference_size), device="cpu", verbose=False
                )
            else:
                raise RuntimeError(f"YOLO推論に失敗しました: {exc}") from exc

        if not results:
            return []
        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        confidences = boxes.conf.detach().cpu().numpy()
        raw_masks: list[np.ndarray | None] = [None] * len(xyxy)
        if result.masks is not None and result.masks.data is not None:
            mask_data = result.masks.data.detach().cpu().numpy()
            height, width = image.shape[:2]
            raw_masks = [
                cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
                for mask in mask_data
            ]
        detections: list[Detection] = []
        for index, box in enumerate(xyxy):
            x1, y1, x2, y2 = (int(round(value)) for value in box)
            class_id = int(classes[index])
            mask = raw_masks[index] if index < len(raw_masks) else None
            detections.append(
                Detection(
                    (x1, y1, x2, y2), class_id, float(confidences[index]),
                    self.names.get(class_id, str(class_id)), mask,
                )
            )
        return detections

    @staticmethod
    def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
        if intersection == 0:
            return 0.0
        first_area = max(1, ax2 - ax1) * max(1, ay2 - ay1)
        second_area = max(1, bx2 - bx1) * max(1, by2 - by1)
        return intersection / max(1, first_area + second_area - intersection)

    @classmethod
    def _deduplicate(cls, detections: list[Detection], threshold: float = 0.50) -> list[Detection]:
        kept: list[Detection] = []
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            duplicate = any(
                detection.class_id == existing.class_id and cls._iou(detection.bbox, existing.bbox) >= threshold
                for existing in kept
            )
            if not duplicate:
                kept.append(detection)
        return kept

    def predict(
        self,
        image: np.ndarray,
        confidence: float,
        inference_size: int = 1280,
        high_recall: bool = False,
        tile_overlap: float = 0.20,
    ) -> list[Detection]:
        detections = self._predict_once(image, confidence, inference_size)
        height, width = image.shape[:2]
        if not high_recall or max(width, height) <= inference_size:
            return detections

        tile_size = min(int(inference_size), max(width, height))
        overlap = max(0.0, min(0.45, float(tile_overlap)))
        stride = max(1, round(tile_size * (1.0 - overlap)))

        def starts(length: int) -> list[int]:
            if length <= tile_size:
                return [0]
            values = list(range(0, max(1, length - tile_size + 1), stride))
            last = length - tile_size
            if values[-1] != last:
                values.append(last)
            return values

        for y1 in starts(height):
            for x1 in starts(width):
                tile = image[y1 : min(height, y1 + tile_size), x1 : min(width, x1 + tile_size)]
                for detection in self._predict_once(tile, confidence, inference_size):
                    bx1, by1, bx2, by2 = detection.bbox
                    full_mask = None
                    if detection.mask is not None:
                        full_mask = np.zeros((height, width), dtype=np.float32)
                        tile_height, tile_width = tile.shape[:2]
                        full_mask[y1 : y1 + tile_height, x1 : x1 + tile_width] = detection.mask[:tile_height, :tile_width]
                    detections.append(
                        Detection(
                            (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1),
                            detection.class_id,
                            detection.confidence,
                            detection.class_name,
                            full_mask,
                        )
                    )
        return self._deduplicate(detections)
