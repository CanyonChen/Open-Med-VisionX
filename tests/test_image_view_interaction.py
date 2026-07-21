from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt5.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PyQt5.QtWidgets import (
    QApplication,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
)

from dicom_viewer.ui.image_view import ImageView


def _wheel_event(angle_delta: QPoint) -> QWheelEvent:
    return QWheelEvent(
        QPointF(12.0, 12.0),
        QPointF(12.0, 12.0),
        QPoint(),
        angle_delta,
        Qt.NoButton,
        Qt.NoModifier,
        Qt.NoScrollPhase,
        False,
    )


def _add_pending_measurement(view: ImageView) -> QGraphicsLineItem:
    item = QGraphicsLineItem()
    view._scene.addItem(item)
    view._start = QPointF(2.0, 3.0)
    view._preview_item = item
    return item


@pytest.mark.parametrize(
    ("delta", "expected_scale"),
    (
        (60, 1.2**0.5),
        (240, 1.2**2),
        (-240, 1.2**-2),
    ),
)
def test_wheel_zoom_uses_fractional_and_multi_step_delta(
    qtbot, delta: int, expected_scale: float
) -> None:
    view = ImageView()
    qtbot.addWidget(view)
    view.set_array(np.arange(256, dtype=np.uint8).reshape(16, 16))
    event = _wheel_event(QPoint(0, delta))
    event.ignore()

    view.wheelEvent(event)

    assert view.transform().m11() == pytest.approx(expected_scale)
    assert view.transform().m22() == pytest.approx(expected_scale)
    assert event.isAccepted()


def test_horizontal_wheel_delta_does_not_zoom(qtbot) -> None:
    view = ImageView()
    qtbot.addWidget(view)
    view.set_array(np.arange(256, dtype=np.uint8).reshape(16, 16))
    before = view.transform()

    view.wheelEvent(_wheel_event(QPoint(120, 0)))

    assert view.transform() == before


def test_mode_change_and_clear_image_cancel_only_pending_measurement(qtbot) -> None:
    view = ImageView()
    qtbot.addWidget(view)
    committed = QGraphicsEllipseItem(0.0, 0.0, 2.0, 2.0)
    view._scene.addItem(committed)
    view._measurement_items.append(committed)
    pending = _add_pending_measurement(view)

    view.set_measurement_mode("area")

    assert pending.scene() is None
    assert committed.scene() is view._scene
    assert view._start is None
    assert view._preview_item is None

    second_pending = _add_pending_measurement(view)
    view.clear_image()

    assert second_pending.scene() is None
    assert committed.scene() is None
    assert view._start is None
    assert view._preview_item is None


def test_escape_cancels_pending_measurement_and_accepts_key(qtbot) -> None:
    view = ImageView()
    qtbot.addWidget(view)
    pending = _add_pending_measurement(view)
    event = QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier)
    event.ignore()

    view.keyPressEvent(event)

    assert pending.scene() is None
    assert view._start is None
    assert view._preview_item is None
    assert event.isAccepted()


def test_titles_labels_and_overlay_pens_remain_screen_stable(qtbot) -> None:
    view = ImageView()
    qtbot.addWidget(view)
    view.set_array(np.arange(64, dtype=np.uint8).reshape(8, 8))
    assert view._title_item.flags() & QGraphicsItem.ItemIgnoresTransformations

    view.set_box_overlays(np.array([[1, 1, 5, 5]]), labels=["box"])
    rectangle, box_label = view._overlay_items[-2:]
    assert isinstance(rectangle, QGraphicsRectItem)
    assert rectangle.pen().isCosmetic()
    assert isinstance(box_label, QGraphicsSimpleTextItem)
    assert box_label.flags() & QGraphicsItem.ItemIgnoresTransformations

    view.set_point_overlays(np.array([[2, 3]]), labels=["point"])
    marker, point_label = view._overlay_items[-2:]
    assert isinstance(marker, QGraphicsEllipseItem)
    assert marker.pen().isCosmetic()
    assert isinstance(point_label, QGraphicsSimpleTextItem)
    assert point_label.flags() & QGraphicsItem.ItemIgnoresTransformations

    vector_field = np.zeros((8, 8, 2), dtype=float)
    view.set_vector_field_overlay(vector_field, max_vectors=1)
    vector = view._overlay_items[-1]
    assert isinstance(vector, QGraphicsLineItem)
    assert vector.pen().isCosmetic()


def test_measurement_preview_uses_a_cosmetic_pen(qtbot) -> None:
    view = ImageView()
    qtbot.addWidget(view)
    view.resize(240, 180)
    view.show()
    QApplication.processEvents()
    view.set_array(np.arange(100, dtype=np.uint8).reshape(10, 10))
    view.set_measurement_mode("distance")
    viewport_position = view.mapFromScene(QPointF(4.0, 4.0))
    event = QMouseEvent(
        QEvent.MouseButtonPress,
        QPointF(viewport_position),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )

    view.mousePressEvent(event)

    assert isinstance(view._preview_item, QGraphicsLineItem)
    assert view._preview_item.pen().isCosmetic()


def test_empty_view_keeps_title_and_non_empty_scene_rect_across_resize(qtbot) -> None:
    view = ImageView("Image")
    qtbot.addWidget(view)
    view.resize(320, 180)
    view.show()
    QApplication.processEvents()
    initial_rect = view.sceneRect()

    assert not initial_rect.isEmpty()
    assert view._title_item.scene() is view._scene
    assert view.viewport().rect().contains(view.mapFromScene(view._title_item.pos()))

    view.resize(520, 300)
    QApplication.processEvents()

    assert view.sceneRect().width() > initial_rect.width()
    assert view.sceneRect().height() > initial_rect.height()

    view.set_array(np.arange(64, dtype=np.uint8).reshape(8, 8))
    view.scale(3.0, 3.0)
    view.clear_image()

    assert not view.sceneRect().isEmpty()
    assert view.transform().m11() == pytest.approx(1.0)
    assert view.viewport().rect().contains(view.mapFromScene(view._title_item.pos()))


def test_accessible_description_tracks_language(qtbot) -> None:
    view = ImageView("Image")
    qtbot.addWidget(view)
    assert view.accessibleDescription() == "Zoomable image view with pan and measurement tools"

    view.set_language("zh_CN")

    assert view.accessibleName() == "图像"
    assert view.accessibleDescription() == "支持平移、缩放和测量工具的图像视图"

    view.set_language("en")
    assert view.accessibleDescription() == "Zoomable image view with pan and measurement tools"
