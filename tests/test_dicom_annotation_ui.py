from __future__ import annotations

from datetime import datetime, timezone
from threading import Event

import numpy as np
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox

from workbench.domain import (
    CompressionKind,
    ImageSeries,
    ImageStudy,
    ImageVolume,
    IntensitySemantics,
    InterpolationMode,
    LabelDefinition,
    LayerReference,
    LayerValidationState,
    SegmentationLayer,
    SegmentationValueType,
    SourceFormat,
    SourceReference,
    SourceType,
    SpatialGeometry,
    VolumeLayer,
)
from workbench.io import DicomAnnotationImport, DicomAnnotationKind
from workbench.services import LoadedStudy
from workbench.ui.i18n import translate
from workbench.ui.main_window import ViewerPage

_CLINICAL_IMPORT_STATUS_MESSAGES = (
    "The label-map request was discarded because the study changed.",
    "Reading the selected clinical label map",
    "Label-map validation complete",
    "Label-map volume selection cancelled; the study is unchanged.",
    "The label-map result was discarded because the study changed.",
    "Label-map import cancelled before any layer changed.",
    "Resampling a confirmed label-map preview with nearest neighbour",
    "Confirmed label-map preview is ready",
    "Reading the selected DICOM annotation",
    "DICOM annotation validation complete",
    "The annotation result was discarded because the active study changed.",
    "DICOM annotation import cancelled before any layer changed.",
    "Resampling a confirmed annotation preview",
    "Confirmed annotation previews are ready",
    "Local operation cancelled safely; no pending result was applied.",
)


def _loaded_dicom_study() -> LoadedStudy:
    volume = ImageVolume(
        np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4),
        SourceType.DICOM,
        IntensitySemantics.HOUNSFIELD_UNIT,
        affine=np.eye(4),
        modality="CT",
    )
    source = SourceReference(
        source_id="source-1",
        source_type=SourceType.DICOM,
        source_format=SourceFormat.DICOM,
    )
    series = ImageSeries(
        series_id="series-1",
        modality="CT",
        source=source,
        geometry=SpatialGeometry.from_volume(volume),
        intensity_semantics=volume.intensity_semantics,
        layers=(
            VolumeLayer(
                layer_id="base-1",
                series_id="series-1",
                name="Base image",
                source=source,
                validation_state=LayerValidationState.VALIDATED,
                volume=volume,
                is_base_image=True,
            ),
        ),
        study_instance_uid="1.2.3",
        series_instance_uid="1.2.3.4",
        frame_of_reference_uid="1.2.3.5",
    )
    domain_study = ImageStudy(
        study_id="study-1",
        source_type=SourceType.DICOM,
        series=(series,),
    )
    return LoadedStudy(
        image=volume,
        display_image=volume,
        imported_at=datetime.now(timezone.utc),
        source_kind="DICOM",
        domain_study=domain_study,
    )


def _segmentation(
    reference: ImageSeries,
    *,
    geometry: SpatialGeometry | None = None,
) -> SegmentationLayer:
    active_geometry = geometry or reference.geometry
    values = np.zeros(active_geometry.shape_zyx, dtype=np.uint8)
    values[(0,) * 3] = 1
    return SegmentationLayer(
        layer_id="seg-1",
        series_id=reference.series_id,
        name="Validated region",
        source=SourceReference(
            source_id="seg-source",
            source_type=SourceType.DICOM,
            source_format=SourceFormat.DICOM_SEG,
            compression=CompressionKind.LOSSLESS,
        ),
        validation_state=LayerValidationState.VALIDATED,
        array=values,
        geometry=active_geometry,
        value_type=SegmentationValueType.BINARY,
        reference=LayerReference(
            local_series_id=reference.series_id,
            dicom_series_uid=reference.series_instance_uid,
            frame_of_reference_uid=reference.frame_of_reference_uid,
            referenced_sop_instance_uids=("1.2.3.4.1",),
        ),
        labels=(LabelDefinition(1, "Validated region", "#FF4040"),),
        reference_geometry=reference.geometry,
    )


def _label_map(reference: ImageSeries) -> SegmentationLayer:
    values = np.zeros(reference.geometry.shape_zyx, dtype=np.uint8)
    values[0, 1, 1] = 3
    return SegmentationLayer(
        layer_id="label-map-1",
        series_id=reference.series_id,
        name="Imported label map",
        source=SourceReference(
            source_id="label-map-source",
            source_type=SourceType.RASTER,
            source_format=SourceFormat.PNG,
            compression=CompressionKind.LOSSLESS,
        ),
        validation_state=LayerValidationState.VALIDATED,
        array=values,
        geometry=reference.geometry,
        value_type=SegmentationValueType.DISCRETE,
        reference=LayerReference(
            local_series_id=reference.series_id,
            dicom_series_uid=reference.series_instance_uid,
            frame_of_reference_uid=reference.frame_of_reference_uid,
        ),
        labels=(LabelDefinition(3, "Reviewed label", "#30D158"),),
        reference_geometry=reference.geometry,
    )


def test_matching_dicom_seg_is_committed_and_rendered_as_a_layer(qtbot) -> None:
    page = ViewerPage()
    qtbot.addWidget(page)
    loaded = _loaded_dicom_study()
    page._loaded(loaded)
    reference = loaded.reference_series
    assert reference is not None
    imported = DicomAnnotationImport(
        DicomAnnotationKind.SEGMENTATION,
        (_segmentation(reference),),
    )

    page._dicom_annotation_ready(imported, page.study)

    current = page.study
    assert current is not None and current.reference_series is not None
    assert len(current.reference_series.layers) == 2
    assert current.reference_series.layers[-1].name == "Validated region"
    assert len(page.axial_view._overlay_items) >= 3  # two crosshair lines and one mask
    assert "1.2.3.4" not in page.pairing_status.text()
    assert "No source file was changed or uploaded" in page.pairing_status.text()
    page.close()


def test_geometry_mismatch_can_be_kept_hidden_without_resampling(qtbot, monkeypatch) -> None:
    page = ViewerPage()
    qtbot.addWidget(page)
    loaded = _loaded_dicom_study()
    page._loaded(loaded)
    reference = loaded.reference_series
    assert reference is not None
    mismatch = SpatialGeometry(
        shape_zyx=(1, 2, 2),
        affine_ras=np.diag((2.0, 2.0, 2.0, 1.0)),
    )
    monkeypatch.setattr(page, "_confirm_segmentation_resampling", lambda *_: "keep")
    imported = DicomAnnotationImport(
        DicomAnnotationKind.SEGMENTATION,
        (_segmentation(reference, geometry=mismatch),),
    )

    page._dicom_annotation_ready(imported, page.study)

    current = page.study
    assert current is not None and current.reference_series is not None
    layer = current.reference_series.layers[-1]
    assert isinstance(layer, SegmentationLayer)
    assert not layer.presentation.visible
    assert not layer.transform_chain
    assert len(page.axial_view._overlay_items) == 2  # linked crosshair only
    page.close()


def test_confirmed_geometry_resampling_creates_a_visible_immutable_derivative(
    qtbot, monkeypatch
) -> None:
    page = ViewerPage()
    qtbot.addWidget(page)
    loaded = _loaded_dicom_study()
    page._loaded(loaded)
    reference = loaded.reference_series
    assert reference is not None
    mismatch = SpatialGeometry(
        shape_zyx=(1, 2, 2),
        affine_ras=np.diag((2.0, 2.0, 2.0, 1.0)),
    )
    monkeypatch.setattr(page, "_confirm_segmentation_resampling", lambda *_: "resample")
    imported = DicomAnnotationImport(
        DicomAnnotationKind.SEGMENTATION,
        (_segmentation(reference, geometry=mismatch),),
    )

    page._dicom_annotation_ready(imported, page.study)
    qtbot.waitUntil(lambda: page._active_local_task is None, timeout=3_000)

    current = page.study
    assert current is not None and current.reference_series is not None
    original, derived = current.reference_series.layers[-2:]
    assert isinstance(original, SegmentationLayer)
    assert isinstance(derived, SegmentationLayer)
    assert not original.presentation.visible
    assert derived.presentation.visible
    assert derived.derived_from_layer_ids == (original.layer_id,)
    assert derived.transform_chain[-1].user_confirmed
    assert derived.array.shape == reference.geometry.shape_zyx
    page.close()


def test_label_map_and_layer_panel_share_one_immutable_presentation_state(qtbot) -> None:
    page = ViewerPage()
    qtbot.addWidget(page)
    loaded = _loaded_dicom_study()
    page._loaded(loaded)
    reference = loaded.reference_series
    assert reference is not None
    label_map = _label_map(reference)

    page._label_map_ready(label_map, page.study)

    current = page.study
    assert current is not None and current.reference_series is not None
    assert page.study_layers.study is current.domain_study
    assert page.study_layers.active_layer_key == (reference.series_id, label_map.layer_id)
    assert len(page.axial_view._overlay_items) >= 3

    page.study_layers.opacity_slider.setValue(20)
    current = page.study
    assert current is not None and current.reference_series is not None
    updated = current.reference_series.layers[-1]
    assert isinstance(updated, SegmentationLayer)
    assert updated.array is label_map.array
    assert updated.presentation.opacity == pytest.approx(0.2)

    label_item = page.study_layers.label_list.item(0)
    label_item.setCheckState(Qt.Unchecked)
    current = page.study
    assert current is not None and current.reference_series is not None
    updated = current.reference_series.layers[-1]
    assert isinstance(updated, SegmentationLayer)
    assert not updated.labels[0].visible
    assert len(page.axial_view._overlay_items) == 2  # linked crosshair only
    page.close()


def test_clinical_import_status_states_switch_between_english_and_chinese(qtbot) -> None:
    page = ViewerPage()
    qtbot.addWidget(page)

    page.set_language("zh_CN")
    for source in _CLINICAL_IMPORT_STATUS_MESSAGES:
        page._set_status(source)
        assert page.status.text() == translate(source, "zh_CN")
        assert page.status.text() != source

    page.set_language("en")
    for source in _CLINICAL_IMPORT_STATUS_MESSAGES:
        page._set_status(source)
        assert page.status.text() == source
    page.close()


def test_background_clinical_import_cancel_is_localised_and_commits_nothing(qtbot) -> None:
    page = ViewerPage()
    qtbot.addWidget(page)
    loaded = _loaded_dicom_study()
    page._loaded(loaded)
    page.set_language("zh_CN")
    started = Event()

    def cancellable_import(context):
        started.set()
        context.token.wait(2.0)
        context.raise_if_cancelled()
        return "must not commit"

    committed: list[object] = []
    original_study = page.study
    page._start_local_operation(cancellable_import, committed.append)
    assert started.wait(1.0)

    page.cancel_local_button.click()
    qtbot.waitUntil(lambda: page._active_local_task is None, timeout=3_000)

    assert page.status.text() == "本地操作已安全取消；没有应用任何待处理结果。"
    assert "Background operation was cancelled" not in page.status.text()
    assert committed == []
    assert page.study is original_study
    page.close()


@pytest.mark.parametrize(
    ("language", "expected_details", "expected_cancel"),
    (
        (
            "en",
            (
                "Layer: Validated region",
                "Source shape ZYX:",
                "Display shape ZYX:",
                "Differences: shape",
                "Proposed interpolation: nearest",
                "Outside value: 0",
                "The imported source layer remains immutable.",
            ),
            "Cancel",
        ),
        (
            "zh_CN",
            (
                "图层：Validated region",
                "源形状 ZYX：",
                "显示形状 ZYX：",
                "差异：体素形状",
                "建议插值方式：最近邻",
                "范围外取值：0",
                "导入的源图层保持不可变。",
            ),
            "取消",
        ),
    ),
)
def test_resampling_dialog_details_are_bilingual(
    qtbot,
    monkeypatch,
    language: str,
    expected_details: tuple[str, ...],
    expected_cancel: str,
) -> None:
    page = ViewerPage()
    qtbot.addWidget(page)
    loaded = _loaded_dicom_study()
    page._loaded(loaded)
    page.set_language(language)
    reference = loaded.reference_series
    assert reference is not None
    mismatch = SpatialGeometry(
        shape_zyx=(1, 2, 2),
        affine_ras=np.diag((2.0, 2.0, 2.0, 1.0)),
    )
    layer = _segmentation(reference, geometry=mismatch)
    captured: dict[str, object] = {}

    def capture_dialog(dialog: QMessageBox) -> int:
        captured["details"] = dialog.detailedText()
        captured["buttons"] = tuple(button.text().replace("&", "") for button in dialog.buttons())
        return 0

    monkeypatch.setattr(QMessageBox, "exec_", capture_dialog)

    decision = page._confirm_segmentation_resampling(
        layer,
        reference.geometry,
        InterpolationMode.NEAREST,
    )

    assert decision == "cancel"
    details = str(captured["details"])
    assert all(expected in details for expected in expected_details)
    assert expected_cancel in captured["buttons"]
    page.close()
