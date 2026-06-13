from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import QLabel, QSizePolicy

from app.mosaic import apply_mask_mosaic


class ImagePreview(QLabel):
    wheel_navigate = Signal(int)
    mask_changed = Signal()

    def __init__(self) -> None:
        super().__init__("画像を処理すると、ここにプレビューが表示されます")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(300, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setObjectName("imagePreview")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self._image: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._history: list[np.ndarray] = []
        self._editable = False
        self._erase = False
        self._drawing = False
        self._panning = False
        self._last_pan_point: QPoint | None = None
        self._last_image_point: tuple[int, int] | None = None
        self._brush_size = 60
        self._pixel_size = 16
        self._feather_px = 0
        self._display_rect = QRect()
        self._zoom = 1.0
        self._pan = QPoint(0, 0)

    def set_image(
        self,
        image: np.ndarray | None,
        mask: np.ndarray | None = None,
        editable: bool = False,
        pixel_size: int = 16,
        feather_px: int = 0,
    ) -> None:
        self._image = image.copy() if image is not None else None
        self._editable = editable and image is not None
        self._pixel_size = max(2, int(pixel_size))
        self._feather_px = max(0, int(feather_px))
        if image is None:
            self._mask = None
        elif mask is not None and mask.shape == image.shape[:2]:
            self._mask = mask.copy().astype(np.uint8)
        else:
            self._mask = np.zeros(image.shape[:2], dtype=np.uint8)
        self._history = []
        self._pan = QPoint(0, 0)
        self._update_cursor()
        self._refresh()

    def set_zoom_percent(self, percent: int) -> None:
        self._zoom = max(0.25, min(4.0, int(percent) / 100.0))
        self._clamp_pan()
        self._update_cursor()
        self._refresh()

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan = QPoint(0, 0)
        self._update_cursor()
        self._refresh()

    def set_brush_size(self, size: int) -> None:
        self._brush_size = max(2, int(size))

    def set_erase(self, erase: bool) -> None:
        self._erase = erase

    def mask(self) -> np.ndarray | None:
        return self._mask.copy() if self._mask is not None else None

    def has_mask(self) -> bool:
        return self._mask is not None and bool(np.any(self._mask))

    def rendered_image(self) -> np.ndarray | None:
        if self._image is None:
            return None
        if self._mask is None or not np.any(self._mask):
            return self._image.copy()
        return apply_mask_mosaic(self._image, self._mask, self._pixel_size, self._feather_px)

    def undo(self) -> None:
        if self._mask is not None and self._history:
            self._mask = self._history.pop()
            self.mask_changed.emit()
            self._refresh()

    def clear_mask(self) -> None:
        if self._mask is not None and np.any(self._mask):
            self._history.append(self._mask.copy())
            self._mask.fill(0)
            self.mask_changed.emit()
            self._refresh()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self._refresh()

    def wheelEvent(self, event: QWheelEvent) -> None:
        direction = -1 if event.angleDelta().y() > 0 else 1
        self.wheel_navigate.emit(direction)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        should_pan = event.button() == Qt.MouseButton.RightButton or (
            event.button() == Qt.MouseButton.LeftButton and not self._editable and self._zoom > 1.0
        )
        if should_pan and self._image is not None:
            self.setFocus()
            self._panning = True
            self._last_pan_point = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._editable and self._mask is not None:
            point = self._to_image_point(event.position().toPoint())
            if point is not None:
                self.setFocus()
                self._history.append(self._mask.copy())
                self._drawing = True
                self._last_image_point = point
                self._paint_line(point, point)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning and self._last_pan_point is not None:
            point = event.position().toPoint()
            delta = point - self._last_pan_point
            self._pan += delta
            self._last_pan_point = point
            self._clamp_pan()
            self._refresh()
            event.accept()
            return
        if self._drawing and self._last_image_point is not None:
            point = self._to_image_point(event.position().toPoint())
            if point is not None:
                self._paint_line(self._last_image_point, point)
                self._last_image_point = point
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._panning and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self._panning = False
            self._last_pan_point = None
            self._update_cursor()
            event.accept()
            return
        if self._drawing and event.button() == Qt.MouseButton.LeftButton:
            self._drawing = False
            self._last_image_point = None
            self.mask_changed.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _paint_line(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        if self._mask is None:
            return
        color = 0 if self._erase else 255
        cv2.line(self._mask, start, end, color, self._brush_size, lineType=cv2.LINE_AA)
        cv2.circle(self._mask, end, max(1, self._brush_size // 2), color, -1, lineType=cv2.LINE_AA)
        self._refresh()

    def _to_image_point(self, point: QPoint) -> tuple[int, int] | None:
        if self._image is None or not self._display_rect.contains(point):
            return None
        height, width = self._image.shape[:2]
        x = round((point.x() - self._display_rect.x()) * width / max(1, self._display_rect.width()))
        y = round((point.y() - self._display_rect.y()) * height / max(1, self._display_rect.height()))
        return min(width - 1, max(0, x)), min(height - 1, max(0, y))

    def _scaled_size(self) -> tuple[int, int]:
        if self._image is None:
            return 0, 0
        height, width = self._image.shape[:2]
        fit = min(self.width() / max(1, width), self.height() / max(1, height))
        scale = fit * self._zoom
        return max(1, round(width * scale)), max(1, round(height * scale))

    def _clamp_pan(self) -> None:
        scaled_width, scaled_height = self._scaled_size()
        limit_x = max(0, (scaled_width - self.width()) // 2)
        limit_y = max(0, (scaled_height - self.height()) // 2)
        self._pan.setX(max(-limit_x, min(limit_x, self._pan.x())))
        self._pan.setY(max(-limit_y, min(limit_y, self._pan.y())))

    def _update_cursor(self) -> None:
        if self._editable:
            cursor = Qt.CursorShape.CrossCursor
        elif self._image is not None and self._zoom > 1.0:
            cursor = Qt.CursorShape.OpenHandCursor
        else:
            cursor = Qt.CursorShape.ArrowCursor
        self.setCursor(cursor)

    def _refresh(self) -> None:
        if self._image is None:
            self.clear()
            self.setText("画像を処理すると、ここにプレビューが表示されます")
            self._display_rect = QRect()
            return
        display = self.rendered_image()
        if display is None:
            return
        if self._editable and self._mask is not None and np.any(self._mask):
            overlay = display.copy()
            overlay[self._mask > 0] = (60, 200, 170)
            alpha = (self._mask.astype(np.float32) / 255.0 * 0.35)[:, :, None]
            display = np.clip(display.astype(np.float32) * (1.0 - alpha) + overlay.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        scaled_width, scaled_height = self._scaled_size()
        pixmap = QPixmap.fromImage(qimage).scaled(
            scaled_width, scaled_height, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self._clamp_pan()
        self._display_rect = QRect(
            (self.width() - pixmap.width()) // 2 + self._pan.x(),
            (self.height() - pixmap.height()) // 2 + self._pan.y(),
            pixmap.width(),
            pixmap.height(),
        )
        viewport = QPixmap(self.size())
        viewport.fill(Qt.GlobalColor.black)
        painter = QPainter(viewport)
        painter.drawPixmap(self._display_rect.topLeft(), pixmap)
        painter.end()
        self.setPixmap(viewport)
