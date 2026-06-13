from __future__ import annotations

import cv2
import numpy as np

from app.detector import Detection
from app.settings import auto_pixel_size


def _expanded_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int], expand_ratio: float) -> tuple[int, int, int, int]:
    height, width = shape
    x1, y1, x2, y2 = bbox
    box_width, box_height = max(1, x2 - x1), max(1, y2 - y1)
    # expand_ratio is applied to every edge. 15% therefore adds 15% of the
    # detected width/height on the left, right, top, and bottom.
    dx, dy = box_width * expand_ratio, box_height * expand_ratio
    return (
        max(0, int(x1 - dx)),
        max(0, int(y1 - dy)),
        min(width, int(x2 + dx)),
        min(height, int(y2 + dy)),
    )


def expand_segmentation_mask(mask: np.ndarray, expand_ratio: float) -> np.ndarray:
    binary = (mask > 0.5).astype(np.uint8)
    points = cv2.findNonZero(binary)
    if points is None or expand_ratio <= 0:
        return binary.astype(np.float32)
    _, _, width, height = cv2.boundingRect(points)
    radius = max(1, round(max(width, height) * expand_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(binary, kernel).astype(np.float32)


def _bbox_ellipse_mask(
    image_shape: tuple[int, int], bbox: tuple[int, int, int, int], expand_ratio: float
) -> np.ndarray:
    height, width = image_shape
    x1, y1, x2, y2 = _expanded_bbox(bbox, image_shape, expand_ratio)
    current = np.zeros((height, width), dtype=np.float32)
    if x2 <= x1 or y2 <= y1:
        return current
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
    cv2.ellipse(current, center, axes, 0, 0, 360, 1.0, -1, lineType=cv2.LINE_AA)
    return current


def build_mask(image_shape: tuple[int, int], detections: list[Detection], expand_ratio: float, feather_px: int) -> np.ndarray:
    height, width = image_shape
    combined = np.zeros((height, width), dtype=np.float32)
    for detection in detections:
        if detection.mask is not None and detection.mask.size:
            current = cv2.resize(detection.mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
            current = expand_segmentation_mask(current, expand_ratio)
        else:
            current = _bbox_ellipse_mask(image_shape, detection.bbox, expand_ratio)
        combined = np.maximum(combined, current)

    if feather_px > 0 and np.any(combined):
        kernel = feather_px * 2 + 1
        combined = cv2.GaussianBlur(combined, (kernel, kernel), 0)
    return np.clip(combined, 0.0, 1.0)


def pixelate(image: np.ndarray, pixel_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    block = max(2, int(pixel_size))
    small_width = max(1, width // block)
    small_height = max(1, height // block)
    small = cv2.resize(image, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def apply_mask_mosaic(image: np.ndarray, mask: np.ndarray, pixel_size: int, feather_px: int = 0) -> np.ndarray:
    if mask.shape != image.shape[:2]:
        raise ValueError("マスクと画像のサイズが一致しません。")
    if not np.any(mask):
        return image.copy()
    mosaic = pixelate(image, pixel_size)
    alpha_mask = mask.astype(np.float32) / 255.0
    if feather_px > 0:
        kernel = feather_px * 2 + 1
        alpha_mask = cv2.GaussianBlur(alpha_mask, (kernel, kernel), 0)
    alpha = alpha_mask[:, :, None]
    output = image.astype(np.float32) * (1.0 - alpha) + mosaic.astype(np.float32) * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def apply_mosaic(
    image: np.ndarray,
    detections: list[Detection],
    pixel_size_mode: str,
    pixel_size: int,
    expand_ratio: float,
    feather_px: int,
) -> tuple[np.ndarray, int, np.ndarray]:
    height, width = image.shape[:2]
    actual_pixel_size = auto_pixel_size(width, height, pixel_size_mode, pixel_size)
    if not detections:
        return image.copy(), actual_pixel_size, np.zeros((height, width), dtype=np.float32)
    mask = build_mask((height, width), detections, expand_ratio, feather_px)
    mosaic = pixelate(image, actual_pixel_size)
    alpha = mask[:, :, None]
    output = image.astype(np.float32) * (1.0 - alpha) + mosaic.astype(np.float32) * alpha
    return np.clip(output, 0, 255).astype(np.uint8), actual_pixel_size, mask


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    masked_class_ids: set[int] | None = None,
    expand_ratio: float = 0.0,
) -> np.ndarray:
    output = image.copy()
    height, width = image.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        is_masked = masked_class_ids is None or detection.class_id in masked_class_ids
        color = (196, 206, 57) if is_masked else (120, 130, 140)
        if detection.mask is not None and detection.mask.size:
            raw_mask = cv2.resize(detection.mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
            display_mask = expand_segmentation_mask(raw_mask, expand_ratio) if is_masked else (raw_mask > 0.5).astype(np.float32)
            contours, _ = cv2.findContours((display_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, color, 3 if is_masked else 2, lineType=cv2.LINE_AA)
        else:
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        if is_masked and detection.mask is None:
            ex1, ey1, ex2, ey2 = _expanded_bbox(detection.bbox, (height, width), expand_ratio)
            center = ((ex1 + ex2) // 2, (ey1 + ey2) // 2)
            axes = (max(1, (ex2 - ex1) // 2), max(1, (ey2 - ey1) // 2))
            cv2.ellipse(output, center, axes, 0, 0, 360, (80, 220, 185), 3, lineType=cv2.LINE_AA)
        label = f"{detection.class_name} (ID:{detection.class_id}) {detection.confidence:.2f}"
        cv2.rectangle(output, (x1, max(0, y1 - 24)), (x1 + max(100, len(label) * 8), y1), color, -1)
        cv2.putText(output, label, (x1 + 4, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (10, 22, 28), 1, cv2.LINE_AA)
    return output
