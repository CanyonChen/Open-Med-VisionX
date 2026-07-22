from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QSignalSpy
from PyQt5.QtWidgets import QAbstractButton, QGroupBox, QLabel

from workbench.domain.images import ImageVolume, IntensitySemantics, SourceType
from workbench.domain.studies import (
    DeidentificationStatus,
    ImageSeries,
    ImageStudy,
    LabelDefinition,
    LayerPresentation,
    LayerReference,
    LayerValidationState,
    SegmentationLayer,
    SegmentationValueType,
    SourceFormat,
    SourceReference,
    SpatialGeometry,
    VolumeLayer,
)
from workbench.ui.study_layers import StudyLayersPanel

STUDY_UID = "1.2.826.0.1.3680043.10.543.1"
SERIES_UID = "1.2.826.0.1.3680043.10.543.2"
FRAME_UID = "1.2.826.0.1.3680043.10.543.3"


def _study(*, locked_segmentation: bool = False) -> ImageStudy:
    geometry = SpatialGeometry((2, 3, 4), np.eye(4))
    image_source = SourceReference(
        source_id="1.2.826.0.1.3680043.10.543.100",
        source_type=SourceType.DICOM,
        source_format=SourceFormat.DICOM,
    )
    volume = ImageVolume(
        np.zeros(geometry.shape_zyx, dtype=np.float32),
        SourceType.DICOM,
        IntensitySemantics.ARBITRARY_SIGNAL,
        affine=np.eye(4),
        modality="MR",
    )
    base = VolumeLayer(
        layer_id="base-local",
        series_id="series-local",
        name=r"C:\private\PatientName=Example\1.2.840.10008.dcm",
        source=image_source,
        volume=volume,
        is_base_image=True,
        validation_state=LayerValidationState.VALIDATED,
    )
    segmentation_source = SourceReference(
        source_id="seg-local",
        source_type=SourceType.GENERATED,
        source_format=SourceFormat.GENERATED,
    )
    segmentation = SegmentationLayer(
        layer_id="seg-local",
        series_id="series-local",
        name="Tissue segmentation",
        source=segmentation_source,
        presentation=LayerPresentation(
            visible=True,
            locked=locked_segmentation,
            opacity=0.65,
            color="#58A6FF",
        ),
        validation_state=LayerValidationState.VALIDATED,
        array=np.asarray(
            [
                [[0, 1, 1, 0], [0, 2, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]],
            ],
            dtype=np.uint8,
        ),
        geometry=geometry,
        value_type=SegmentationValueType.DISCRETE,
        reference=LayerReference(local_series_id="series-local"),
        labels=(
            LabelDefinition(1, "Core", "#FF375F", True),
            LabelDefinition(2, "Edema", "#34C759", False),
        ),
        reference_geometry=geometry,
    )
    series = ImageSeries(
        series_id="series-local",
        modality="MR",
        source=image_source,
        geometry=geometry,
        intensity_semantics=IntensitySemantics.ARBITRARY_SIGNAL,
        layers=(base, segmentation),
        study_instance_uid=STUDY_UID,
        series_instance_uid=SERIES_UID,
        frame_of_reference_uid=FRAME_UID,
    )
    return ImageStudy(
        study_id="study-local",
        source_type=SourceType.DICOM,
        series=(series,),
        deidentification_status=DeidentificationStatus.VERIFIED,
        provenance={"PatientName": "removed", "importer": "test"},
    )


def _tree_texts(panel: StudyLayersPanel) -> list[str]:
    texts: list[str] = []

    def collect(item) -> None:
        for column in range(panel.tree.columnCount()):
            texts.extend((item.text(column), item.toolTip(column)))
        for index in range(item.childCount()):
            collect(item.child(index))

    for index in range(panel.tree.topLevelItemCount()):
        collect(panel.tree.topLevelItem(index))
    return texts


def _visible_and_accessible_text(panel: StudyLayersPanel) -> str:
    texts = _tree_texts(panel)
    for label in panel.findChildren(QLabel):
        texts.extend((label.text(), label.accessibleName(), label.accessibleDescription()))
    for group in panel.findChildren(QGroupBox):
        texts.extend((group.title(), group.accessibleName(), group.accessibleDescription()))
    for button in panel.findChildren(QAbstractButton):
        texts.extend(
            (
                button.text(),
                button.toolTip(),
                button.accessibleName(),
                button.accessibleDescription(),
            )
        )
    texts.extend((panel.tree.accessibleName(), panel.tree.accessibleDescription()))
    for row in range(panel.label_list.count()):
        item = panel.label_list.item(row)
        texts.extend((item.text(), item.toolTip()))
    return "\n".join(texts)


def test_empty_state_is_clear_bilingual_and_noninteractive(qtbot) -> None:
    panel = StudyLayersPanel()
    qtbot.addWidget(panel)

    assert panel.study is None
    assert panel.active_layer is None
    assert panel.tree.topLevelItem(0).text(0) == "No study loaded"
    assert not panel.visible_checkbox.isEnabled()
    assert not panel.opacity_slider.isEnabled()
    assert not panel.labels_group.isVisible()

    panel.set_language("zh_CN")
    assert panel.title_label.text() == "检查与图层"
    assert panel.tree.topLevelItem(0).text(0) == "尚未加载检查"
    assert panel.tree.accessibleName() == "检查、序列与图层浏览器"

    with pytest.raises(ValueError, match="Unsupported UI language"):
        panel.set_language("fr")  # type: ignore[arg-type]


def test_tree_and_detail_never_render_uid_path_hash_or_phi_fields(qtbot) -> None:
    panel = StudyLayersPanel()
    qtbot.addWidget(panel)
    study = _study()
    panel.set_study(study)

    rendered = _visible_and_accessible_text(panel)
    assert STUDY_UID not in rendered
    assert SERIES_UID not in rendered
    assert FRAME_UID not in rendered
    assert "1.2.826.0.1.3680043.10.543.100" not in rendered
    assert "C:\\private" not in rendered
    assert "PatientName=Example" not in rendered
    assert "PatientName" not in rendered
    assert "importer" not in rendered
    assert "Layer 1" in rendered


def test_layer_selection_and_controls_emit_requests_without_mutating_study(qtbot) -> None:
    study = _study()
    original_segmentation = study.series[0].layers[1]
    panel = StudyLayersPanel()
    qtbot.addWidget(panel)
    panel.set_study(study)

    active_spy = QSignalSpy(panel.activeLayerChanged)
    visibility_spy = QSignalSpy(panel.layerVisibilityChangeRequested)
    lock_spy = QSignalSpy(panel.layerLockChangeRequested)
    opacity_spy = QSignalSpy(panel.layerOpacityChangeRequested)

    assert panel.set_active_layer("series-local", "seg-local", emit_signal=True)
    assert list(active_spy[-1]) == ["series-local", "seg-local"]
    assert panel.active_layer is original_segmentation
    assert panel.visible_checkbox.isChecked()
    assert not panel.locked_checkbox.isChecked()
    assert panel.opacity_slider.value() == 65

    panel.visible_checkbox.click()
    panel.locked_checkbox.click()
    panel.opacity_slider.setValue(35)

    assert list(visibility_spy[-1]) == ["series-local", "seg-local", False]
    assert list(lock_spy[-1]) == ["series-local", "seg-local", True]
    assert list(opacity_spy[-1]) == ["series-local", "seg-local", pytest.approx(0.35)]
    assert original_segmentation.presentation.visible
    assert not original_segmentation.presentation.locked
    assert original_segmentation.presentation.opacity == pytest.approx(0.65)


def test_segmentation_labels_have_checkable_color_swatches_and_pure_signal(qtbot) -> None:
    study = _study()
    original_labels = study.series[0].layers[1].labels
    panel = StudyLayersPanel()
    qtbot.addWidget(panel)
    panel.set_study(study)
    panel.set_active_layer("series-local", "seg-local")
    label_spy = QSignalSpy(panel.labelVisibilityChangeRequested)

    assert panel.label_list.count() == 2
    first = panel.label_list.item(0)
    second = panel.label_list.item(1)
    assert not first.icon().isNull()
    assert "#FF375F" in first.toolTip()
    assert first.checkState() == Qt.Checked
    assert second.checkState() == Qt.Unchecked

    second.setCheckState(Qt.Checked)

    assert list(label_spy[-1]) == ["series-local", "seg-local", 2, True]
    assert original_labels[1].visible is False


def test_locked_layer_disables_editing_controls_until_parent_supplies_new_revision(qtbot) -> None:
    panel = StudyLayersPanel()
    qtbot.addWidget(panel)
    panel.set_study(_study(locked_segmentation=True))
    panel.set_active_layer("series-local", "seg-local")
    lock_spy = QSignalSpy(panel.layerLockChangeRequested)

    assert panel.locked_checkbox.isChecked()
    assert not panel.opacity_slider.isEnabled()
    assert not panel.label_list.isEnabled()
    assert panel.visible_checkbox.isEnabled()

    panel.locked_checkbox.click()
    assert list(lock_spy[-1]) == ["series-local", "seg-local", False]
    assert not panel.opacity_slider.isEnabled()


def test_layout_reflows_and_native_controls_remain_keyboard_reachable(qtbot) -> None:
    panel = StudyLayersPanel()
    qtbot.addWidget(panel)
    panel.set_study(_study())
    panel.set_active_layer("series-local", "seg-local")
    panel.resize(920, 620)
    panel.show()
    qtbot.wait(10)
    assert panel.splitter.orientation() == Qt.Horizontal

    panel.resize(560, 720)
    qtbot.wait(10)
    assert panel.splitter.orientation() == Qt.Vertical
    assert panel.minimumSizeHint().width() == 0

    panel.tree.setFocus()
    qtbot.keyClick(panel.tree, Qt.Key_Tab)
    assert panel.visible_checkbox.hasFocus()
    qtbot.keyClick(panel.visible_checkbox, Qt.Key_Tab)
    assert panel.locked_checkbox.hasFocus()
