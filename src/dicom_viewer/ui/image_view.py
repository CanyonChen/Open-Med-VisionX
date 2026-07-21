"""Reusable zoom/pan image view with pixel, ROI, and overlay support."""

from __future__ import annotations

from typing import Literal

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QTransform
from PyQt5.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from ..domain.display import ColorDisplayMapping, GrayscaleDisplayMapping
from .i18n import Language, translate

MeasurementMode = Literal["none", "distance", "area", "roi", "annotation"]


def _screen_pen(color: str, width: float) -> QPen:
    """Return a pen whose width stays stable while the image view zooms."""

    pen = QPen(QColor(color))
    pen.setWidthF(width)
    pen.setCosmetic(True)
    return pen


def array_to_qimage(
    array: np.ndarray,
    *,
    grayscale_mapping: GrayscaleDisplayMapping | None = None,
    color_mapping: ColorDisplayMapping | None = None,
) -> QImage:
    values = np.asarray(array)
    if values.ndim == 2:
        mapping = grayscale_mapping or GrayscaleDisplayMapping.from_percentiles(values)
        display = np.ascontiguousarray(mapping.map(values))
        height, width = display.shape
        return QImage(
            display.data,
            width,
            height,
            display.strides[0],
            QImage.Format_Grayscale8,
        ).copy()
    if values.ndim == 3 and values.shape[2] in {3, 4}:
        mapping = color_mapping or ColorDisplayMapping()
        display = np.ascontiguousarray(mapping.map(values))
        height, width, channels = display.shape
        image_format = QImage.Format_RGB888 if channels == 3 else QImage.Format_RGBA8888
        return QImage(
            display.data,
            width,
            height,
            display.strides[0],
            image_format,
        ).copy()
    raise ValueError(f"Cannot display array with shape {values.shape}.")


class ImageView(QGraphicsView):
    pixelHovered = pyqtSignal(int, int, object)
    measurementCompleted = pyqtSignal(str, object)

    _ACCESSIBLE_DESCRIPTIONS: dict[Language, str] = {
        "en": "Zoomable image view with pan and measurement tools",
        "zh_CN": "支持平移、缩放和测量工具的图像视图",
    }
    _MINIMUM_SCALE = 0.05
    _MAXIMUM_SCALE = 50.0
    _ZOOM_FACTOR_PER_STEP = 1.2
    _EMPTY_SCENE_MARGIN = 12.0

    def __init__(self, title: str = "Image", parent: object | None = None) -> None:
        super().__init__(parent)
        self._language: Language = "en"
        self._title_source = title
        self.setAccessibleName(translate(title, self._language))
        self.setAccessibleDescription(self._ACCESSIBLE_DESCRIPTIONS[self._language])
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor("#07152f"))
        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self._pixmap_item)
        self._title_item = QGraphicsSimpleTextItem(translate(title, self._language))
        self._title_item.setBrush(QColor("#f9fafb"))
        self._title_item.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self._title_item.setZValue(100)
        self._scene.addItem(self._title_item)
        self._array: np.ndarray | None = None
        self._pixel_spacing = (1.0, 1.0)
        self._physical_units = False
        self._measurement_mode: MeasurementMode = "none"
        self._start: QPointF | None = None
        self._preview_item: QGraphicsLineItem | QGraphicsRectItem | None = None
        self._measurement_items: list[object] = []
        self._overlay_items: list[object] = []
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._update_scene_geometry()

    @property
    def array(self) -> np.ndarray | None:
        return self._array

    def clear_image(self) -> None:
        """Clear pixels and all dataset-specific interaction state."""

        self._cancel_pending_measurement()
        self.clear_measurements()
        self.clear_overlays()
        self._array = None
        self._pixmap_item.setPixmap(QPixmap())
        self.resetTransform()
        self._update_scene_geometry()

    def set_title(self, title: str) -> None:
        """Set the canonical title and render it in the active UI language."""

        self._title_source = title
        self._title_item.setText(translate(title, self._language))
        self.setAccessibleName(translate(title, self._language))
        self._update_scene_geometry()

    def set_language(self, language: Language) -> None:
        """Retranslate the scene title without disturbing pixels or overlays."""

        self._language = language
        self._title_item.setText(translate(self._title_source, language))
        self.setAccessibleName(translate(self._title_source, language))
        self.setAccessibleDescription(self._ACCESSIBLE_DESCRIPTIONS[language])
        self._update_scene_geometry()

    def set_array(
        self,
        array: np.ndarray,
        *,
        grayscale_mapping: GrayscaleDisplayMapping | None = None,
        color_mapping: ColorDisplayMapping | None = None,
        pixel_spacing: tuple[float, float] | None = None,
        physical_units: bool = False,
        fit: bool = False,
    ) -> None:
        self._array = np.asarray(array)
        image = array_to_qimage(
            self._array,
            grayscale_mapping=grayscale_mapping,
            color_mapping=color_mapping,
        )
        self._pixmap_item.setPixmap(QPixmap.fromImage(image))
        self._pixel_spacing = pixel_spacing or (1.0, 1.0)
        self._physical_units = physical_units
        self._pixmap_item.setTransform(
            QTransform.fromScale(float(self._pixel_spacing[0]), float(self._pixel_spacing[1]))
        )
        self._update_scene_geometry()
        if fit:
            self.fit_image()

    def fit_image(self) -> None:
        if not self._pixmap_item.pixmap().isNull():
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def set_measurement_mode(self, mode: MeasurementMode) -> None:
        self._cancel_pending_measurement()
        self._measurement_mode = mode
        self.setDragMode(QGraphicsView.ScrollHandDrag if mode == "none" else QGraphicsView.NoDrag)

    def clear_measurements(self) -> None:
        for item in self._measurement_items:
            self._scene.removeItem(item)  # type: ignore[arg-type]
        self._measurement_items.clear()

    def clear_overlays(self) -> None:
        for item in self._overlay_items:
            self._scene.removeItem(item)  # type: ignore[arg-type]
        self._overlay_items.clear()

    def set_mask_overlay(
        self,
        mask: np.ndarray,
        *,
        color: tuple[int, int, int] = (255, 64, 64),
        opacity: int = 110,
    ) -> None:
        if self._array is None or np.asarray(mask).shape != self._array.shape[:2]:
            raise ValueError("Mask shape must match the displayed image.")
        binary = np.asarray(mask) != 0
        rgba = np.zeros((*binary.shape, 4), dtype=np.uint8)
        rgba[binary, :3] = color
        rgba[binary, 3] = np.uint8(opacity)
        qimage = array_to_qimage(rgba)
        item = QGraphicsPixmapItem(QPixmap.fromImage(qimage))
        item.setTransform(
            QTransform.fromScale(float(self._pixel_spacing[0]), float(self._pixel_spacing[1]))
        )
        item.setZValue(10)
        self._scene.addItem(item)
        self._overlay_items.append(item)

    def set_box_overlays(
        self,
        boxes: np.ndarray,
        *,
        labels: list[str] | None = None,
    ) -> None:
        values = np.asarray(boxes, dtype=float).reshape(-1, 4)
        pen = _screen_pen("#22d3ee", 1.5)
        for index, (x1, y1, x2, y2) in enumerate(values):
            sx, sy = self._pixel_spacing
            rectangle = QGraphicsRectItem(QRectF(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy))
            rectangle.setPen(pen)
            rectangle.setZValue(20)
            self._scene.addItem(rectangle)
            self._overlay_items.append(rectangle)
            if labels and index < len(labels):
                text = QGraphicsSimpleTextItem(labels[index])
                text.setBrush(QColor("#22d3ee"))
                text.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
                text.setPos(x1 * sx, y1 * sy)
                text.setZValue(21)
                self._scene.addItem(text)
                self._overlay_items.append(text)

    def set_point_overlays(
        self,
        points: np.ndarray,
        *,
        labels: list[str] | None = None,
    ) -> None:
        """Draw canonical pixel-coordinate points without changing image data."""

        values = np.asarray(points, dtype=float).reshape(-1, 2)
        pen = _screen_pen("#a78bfa", 1.5)
        sx, sy = self._pixel_spacing
        radius = max(sx, sy) * 3.0
        for index, (x, y) in enumerate(values):
            marker = QGraphicsEllipseItem(
                x * sx - radius,
                y * sy - radius,
                radius * 2,
                radius * 2,
            )
            marker.setPen(pen)
            marker.setZValue(20)
            self._scene.addItem(marker)
            self._overlay_items.append(marker)
            if labels and index < len(labels) and labels[index]:
                text = QGraphicsSimpleTextItem(labels[index])
                text.setBrush(QColor("#a78bfa"))
                text.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
                text.setPos(x * sx + radius, y * sy + radius)
                text.setZValue(21)
                self._scene.addItem(text)
                self._overlay_items.append(text)

    def set_vector_field_overlay(
        self,
        vector_field: np.ndarray,
        *,
        max_vectors: int = 400,
    ) -> None:
        """Draw a bounded sampling of a 2-D displacement field.

        Component-last ``H x W x 2`` and component-first ``2 x H x W``
        arrays are accepted.  Coordinates and vector components use displayed
        pixel units; sampling is bounded so a dense registration field cannot
        flood the Qt scene with millions of graphics items.
        """

        if self._array is None:
            raise ValueError("Set a source or magnitude image before adding a vector field.")
        if isinstance(max_vectors, bool) or int(max_vectors) <= 0:
            raise ValueError("max_vectors must be a positive integer.")
        field = np.asarray(vector_field, dtype=np.float64)
        spatial_shape = self._array.shape[:2]
        if (
            field.ndim == 3
            and field.shape[:2] != spatial_shape
            and field.shape[0] >= 2
            and field.shape[1:] == spatial_shape
        ):
            field = np.moveaxis(field, 0, -1)
        if field.ndim != 3 or field.shape[2] < 2:
            raise ValueError("Vector field must have shape H x W x 2 or 2 x H x W.")
        if field.shape[:2] != spatial_shape:
            raise ValueError("Vector field spatial shape must match the displayed image.")
        height, width = field.shape[:2]
        stride = max(1, int(np.ceil(np.sqrt((height * width) / int(max_vectors)))))
        pen = _screen_pen("#34d399", 1.25)
        sx, sy = self._pixel_spacing
        for y in range(stride // 2, height, stride):
            for x in range(stride // 2, width, stride):
                dx, dy = field[y, x, :2]
                if not np.isfinite(dx) or not np.isfinite(dy):
                    continue
                line = QGraphicsLineItem(
                    x * sx,
                    y * sy,
                    (x + float(dx)) * sx,
                    (y + float(dy)) * sy,
                )
                line.setPen(pen)
                line.setZValue(20)
                self._scene.addItem(line)
                self._overlay_items.append(line)

    def wheelEvent(self, event: object) -> None:
        if self._pixmap_item.pixmap().isNull():
            super().wheelEvent(event)  # type: ignore[arg-type]
            return
        delta = event.angleDelta().y()  # type: ignore[attr-defined]
        if delta == 0:
            super().wheelEvent(event)  # type: ignore[arg-type]
            return
        current_scale = abs(float(self.transform().m11()))
        if current_scale <= 0.0:
            super().wheelEvent(event)  # type: ignore[arg-type]
            return
        steps = max(-60.0, min(60.0, float(delta) / 120.0))
        requested_scale = current_scale * self._ZOOM_FACTOR_PER_STEP**steps
        target_scale = max(self._MINIMUM_SCALE, min(self._MAXIMUM_SCALE, requested_scale))
        factor = target_scale / current_scale
        if factor != 1.0:
            self.scale(factor, factor)
        event.accept()  # type: ignore[attr-defined]

    def keyPressEvent(self, event: object) -> None:
        if event.key() == Qt.Key_Escape and (  # type: ignore[attr-defined]
            self._start is not None or self._preview_item is not None
        ):
            self._cancel_pending_measurement()
            event.accept()  # type: ignore[attr-defined]
            return
        super().keyPressEvent(event)  # type: ignore[arg-type]

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)  # type: ignore[arg-type]
        if self._pixmap_item.pixmap().isNull():
            self._update_scene_geometry()

    def mouseMoveEvent(self, event: object) -> None:
        scene_position = self.mapToScene(event.pos())  # type: ignore[attr-defined]
        pixel = self._scene_to_pixel(scene_position)
        if pixel is not None and self._array is not None:
            x, y = pixel
            self.pixelHovered.emit(x, y, self._array[y, x])
        if self._start is not None and self._preview_item is not None:
            if isinstance(self._preview_item, QGraphicsLineItem):
                self._preview_item.setLine(
                    self._start.x(), self._start.y(), scene_position.x(), scene_position.y()
                )
            else:
                self._preview_item.setRect(QRectF(self._start, scene_position).normalized())
        super().mouseMoveEvent(event)  # type: ignore[arg-type]

    def mousePressEvent(self, event: object) -> None:
        if event.button() == Qt.LeftButton and self._measurement_mode != "none":  # type: ignore[attr-defined]
            point = self.mapToScene(event.pos())  # type: ignore[attr-defined]
            if self._scene_to_pixel(point) is None:
                return
            if self._measurement_mode == "annotation":
                self._add_annotation(point)
                return
            self._start = point
            pen = _screen_pen("#fbbf24", 1.5)
            if self._measurement_mode == "distance":
                item: QGraphicsLineItem | QGraphicsRectItem = QGraphicsLineItem()
            else:
                item = QGraphicsRectItem()
            item.setPen(pen)
            item.setZValue(30)
            self._scene.addItem(item)
            self._preview_item = item
            return
        super().mousePressEvent(event)  # type: ignore[arg-type]

    def mouseReleaseEvent(self, event: object) -> None:
        if (
            event.button() == Qt.LeftButton  # type: ignore[attr-defined]
            and self._start is not None
            and self._preview_item is not None
        ):
            end = self.mapToScene(event.pos())  # type: ignore[attr-defined]
            start_pixel = self._scene_to_pixel(self._start)
            end_pixel = self._scene_to_pixel(end)
            if start_pixel is not None and end_pixel is not None:
                self._measurement_items.append(self._preview_item)
                if self._measurement_mode == "distance":
                    distance = float(np.hypot(end.x() - self._start.x(), end.y() - self._start.y()))
                    payload = {
                        "start": start_pixel,
                        "end": end_pixel,
                        "distance": distance,
                        "unit": "mm" if self._physical_units else "px",
                    }
                else:
                    x1, y1 = start_pixel
                    x2, y2 = end_pixel
                    left, right = sorted((x1, x2))
                    top, bottom = sorted((y1, y2))
                    width = right - left + 1
                    height = bottom - top + 1
                    area = width * height * self._pixel_spacing[0] * self._pixel_spacing[1]
                    payload = {
                        "rectangle": (left, top, width, height),
                        "area": float(area),
                        "unit": "mm²" if self._physical_units else "px²",
                    }
                    if self._array is not None:
                        roi = self._array[top : bottom + 1, left : right + 1]
                        if roi.size and np.issubdtype(roi.dtype, np.number):
                            payload["mean"] = float(np.mean(roi))
                            payload["std"] = float(np.std(roi))
                self.measurementCompleted.emit(self._measurement_mode, payload)
            else:
                self._scene.removeItem(self._preview_item)
            self._start = None
            self._preview_item = None
            return
        super().mouseReleaseEvent(event)  # type: ignore[arg-type]

    def _cancel_pending_measurement(self) -> None:
        """Remove an unfinished measurement without touching committed marks."""

        if self._preview_item is not None and self._preview_item.scene() is self._scene:
            self._scene.removeItem(self._preview_item)
        self._preview_item = None
        self._start = None

    def _update_scene_geometry(self) -> None:
        """Keep the title reachable without scanning every graphics item."""

        if not self._pixmap_item.pixmap().isNull():
            image_rect = self._pixmap_item.sceneBoundingRect()
            self._title_item.setPos(image_rect.topLeft())
            # The screen-space title is an overlay and must not enlarge the image's
            # scrollable geometry, especially with long Chinese translations.
            self._scene.setSceneRect(image_rect)
            return

        margin = self._EMPTY_SCENE_MARGIN
        self._title_item.setPos(margin, margin)
        viewport_size = self.viewport().size()
        # Leave a small tolerance for style borders and fractional-DPI rounding so
        # an empty view never sprouts meaningless scroll bars.
        width = float(max(1, viewport_size.width() - 4))
        height = float(max(1, viewport_size.height() - 4))
        self._scene.setSceneRect(0.0, 0.0, width, height)

    def _scene_to_pixel(self, point: QPointF) -> tuple[int, int] | None:
        if self._array is None:
            return None
        x = int(np.floor(point.x() / self._pixel_spacing[0]))
        y = int(np.floor(point.y() / self._pixel_spacing[1]))
        if 0 <= y < self._array.shape[0] and 0 <= x < self._array.shape[1]:
            return x, y
        return None

    def _add_annotation(self, point: QPointF) -> None:
        radius = max(self._pixel_spacing) * 2.5
        marker = QGraphicsEllipseItem(
            point.x() - radius,
            point.y() - radius,
            radius * 2,
            radius * 2,
        )
        marker.setPen(_screen_pen("#a78bfa", 1.5))
        marker.setZValue(30)
        self._scene.addItem(marker)
        self._measurement_items.append(marker)
        pixel = self._scene_to_pixel(point)
        if pixel is not None:
            self.measurementCompleted.emit("annotation", {"point": pixel})
