from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtGui import QTransform

import dicom_viewer.ui.main_window as main_window_module
from dicom_viewer.domain import (
    Capability,
    ColorSpace,
    ImageSequence2D,
    ImageVolume,
    IntensitySemantics,
    RasterImage2D,
    SourceType,
    TransformRecord,
)
from dicom_viewer.inference import VisualizationArtifact, VisualizationKind
from dicom_viewer.llm import RenderedPreview
from dicom_viewer.services import LoadedStudy
from dicom_viewer.ui.main_window import (
    AssistantPage,
    ModelPage,
    OpenMedVisionXWindow,
    ReconstructionPage,
    TeachingExperimentPage,
    ViewerPage,
)


def test_ui_module_does_not_import_concrete_model_or_provider_adapters() -> None:
    forbidden = {
        "AnthropicProvider",
        "DeepSeekProvider",
        "DisabledTransport",
        "GLMProvider",
        "KimiProvider",
        "OnnxModelPlugin",
        "OpenAICompatibleProvider",
        "OpenAIProvider",
        "PythonAdapterProxy",
        "TorchScriptModelPlugin",
        "UrllibTransport",
        "load_manifest",
    }
    assert forbidden.isdisjoint(vars(main_window_module))


def _raster_study() -> LoadedStudy:
    image = RasterImage2D(
        np.arange(64, dtype=np.uint16).reshape(8, 8),
        SourceType.RASTER,
        IntensitySemantics.GRAYSCALE,
        bit_depth=16,
        runtime_metadata={"format": "PNG", "lossy_compression": False},
    )
    return LoadedStudy(
        image=image,
        display_image=image,
        imported_at=datetime.now(timezone.utc),
        source_kind="PNG",
    )


def test_window_constructs_offscreen_and_has_product_identity(qtbot) -> None:
    window = OpenMedVisionXWindow()
    qtbot.addWidget(window)
    assert "OpenMedVisionX" in window.windowTitle()
    assert window.tabs.count() == 5
    assert window.tabs.tabText(3) == "Learn"
    window.close()


def test_window_switches_between_english_and_chinese_without_rebuilding_pages(qtbot) -> None:
    window = OpenMedVisionXWindow()
    qtbot.addWidget(window)
    viewer = window.viewer
    viewer._loaded(_raster_study())
    loaded_image = viewer.image
    window.tabs.setCurrentIndex(2)
    window.viewer.measurement_combo.setCurrentIndex(2)
    algorithm_values = [
        window.reconstruction.algorithm.itemText(index)
        for index in range(window.reconstruction.algorithm.count())
    ]

    assert window.language == "en"
    assert window.language_button.text() == "中文"
    assert window.tabs.tabText(0) == "Images"

    window.language_button.click()

    assert window.language == "zh_CN"
    assert window.language_button.text() == "English"
    assert window.tabs.tabText(0) == "影像"
    assert window.tabs.tabText(3) == "教学"
    assert window.viewer.open_file_button.text() == "打开影像 / DICOM ZIP"
    assert window.learning.experiment_selector.itemText(0) == "二维像素、位深与插值"
    assert "光栅图像是采样信号" in window.learning.section_labels["Principle"].text()
    assert window.tabs.currentIndex() == 2
    assert window.viewer is viewer
    assert window.viewer.image is loaded_image
    assert window.viewer.info.toPlainText().startswith("类型:")
    assert window.viewer.axial_view._title_item.text() == "二维光栅图像"
    assert window.viewer.measurement_combo.currentIndex() == 2
    assert [
        window.reconstruction.algorithm.itemText(index)
        for index in range(window.reconstruction.algorithm.count())
    ] == algorithm_values

    window.language_button.click()

    assert window.language == "en"
    assert window.language_button.text() == "中文"
    assert window.tabs.tabText(0) == "Images"
    assert window.viewer.open_file_button.text() == "Open image / DICOM ZIP"
    assert window.learning.experiment_selector.itemText(0).startswith("2-D pixels")
    assert window.viewer.image is loaded_image
    assert window.viewer.axial_view._title_item.text() == "2-D raster image"
    assert window.tabs.currentIndex() == 2
    window.close()


def test_language_switch_retranslates_owned_status_but_preserves_provider_content(qtbot) -> None:
    window = OpenMedVisionXWindow()
    qtbot.addWidget(window)
    window.viewer._measurement_completed("distance", {"distance": 2.5, "unit": "px"})
    window.assistant._failed(RuntimeError("Image"))

    window.set_language("zh_CN")
    assert window.viewer.status.text().startswith("距离:")
    assert window.assistant.answer.toPlainText().startswith("请求已安全失败：")
    assert window.assistant.answer.toPlainText().endswith("Image")

    window.set_language("en")
    assert window.viewer.status.text().startswith("Distance:")
    assert window.assistant.answer.toPlainText() == "Request failed safely: Image"

    window.assistant._answered(SimpleNamespace(content="Image"))
    window.set_language("zh_CN")
    assert window.assistant.answer.toPlainText() == "Image"
    window.close()


def test_reconstruction_language_switch_preserves_view_transform_and_process_key(qtbot) -> None:
    page = ReconstructionPage(lambda: np.zeros((8, 8)))
    qtbot.addWidget(page)
    page._sinogram_ready(
        SimpleNamespace(
            sinogram=np.zeros((8, 4)),
            theta_degrees=np.arange(4),
            circle=True,
            intermediate={"projections_2": np.zeros((8, 2))},
        )
    )
    page.error_view.setTransform(QTransform.fromScale(2.75, 2.75))

    page.set_language("zh_CN")
    assert page.intermediate_combo.currentData() == "Radon projection 2/4"
    assert page.intermediate_combo.currentText() == "Radon 投影 2/4"
    assert page.error_view.transform().m11() == pytest.approx(2.75)

    page.set_language("en")
    assert page.intermediate_combo.currentText() == "Radon projection 2/4"
    assert page.error_view.transform().m11() == pytest.approx(2.75)
    page.close()


def test_learning_page_is_branded_and_every_experiment_has_complete_structure(qtbot) -> None:
    page = TeachingExperimentPage()
    qtbot.addWidget(page)
    logo = page.brand_logo.pixmap()
    assert (logo is not None and not logo.isNull()) or "OpenMedVisionX" in page.brand_logo.text()
    assert set(page.section_labels) == {
        "Principle",
        "Formula",
        "Parameter explanation",
        "Steps",
        "Expected observation",
        "Common mistakes",
        "Reflection question",
    }
    for index in range(page.experiment_selector.count()):
        page.experiment_selector.setCurrentIndex(index)
        assert all(label.text().strip() for label in page.section_labels.values())


def test_window_layout_fits_a_common_desktop_width(qtbot) -> None:
    window = OpenMedVisionXWindow()
    qtbot.addWidget(window)
    assert window.minimumSizeHint().width() <= 1280
    assert max(window.tabs.widget(index).minimumSizeHint().width() for index in range(5)) <= 1100
    window.close()


def test_reconstruction_parameters_progressively_disclose_and_invalidate_results(qtbot) -> None:
    page = ReconstructionPage(lambda: np.zeros((8, 8)))
    qtbot.addWidget(page)
    page.sinogram_result = SimpleNamespace(
        theta_degrees=np.arange(4),
        intermediate={},
    )
    page.reconstruction_result = SimpleNamespace(intermediate={})
    page.export_result_button.setEnabled(True)

    page.algorithm.setCurrentText("DFR")

    assert page.algorithm_options.currentIndex() == 2
    assert page.sinogram_result is not None
    assert page.reconstruction_result is None
    assert not page.export_result_button.isEnabled()

    page.angle_range.setCurrentText("360")
    assert page.sinogram_result is None
    assert not page.reconstruct_button.isEnabled()


def test_model_page_exposes_official_sam_setup_without_downloading(qtbot) -> None:
    page = ModelPage(lambda: np.zeros((8, 8)))
    qtbot.addWidget(page)
    assert "github.com/facebookresearch/segment-anything" in page.resource_status.text()
    assert "compatible plugin manifest" in page.resource_status.text()


def test_model_page_reports_missing_weights_without_opening_untrusted_sources(qtbot) -> None:
    page = ModelPage(lambda: np.zeros((8, 8)))
    qtbot.addWidget(page)
    example = (
        Path(main_window_module.__file__).parents[1]
        / "inference"
        / "examples"
        / "manifest.yaml"
    )
    page.manifest = page.service.inspect_manifest(example)

    page._refresh_model_resource_status()

    assert "example_classifier.onnx" in page.resource_status.text()
    assert "example.invalid" not in page.resource_status.text()
    assert "No GitHub source" in page.resource_status.text()


def test_raster_capabilities_drive_views_and_privacy_filtered_preview(qtbot) -> None:
    page = ViewerPage()
    qtbot.addWidget(page)
    page._loaded(_raster_study())
    assert page.image is not None
    assert Capability.ORTHOGONAL_VIEWS not in page.image.capabilities
    assert not page.coronal_view.isVisible()
    assert not page.sagittal_view.isVisible()
    assert page.navigation_group.isHidden()
    preview = RenderedPreview.from_png(page.rendered_preview_png())
    assert preview.width == 8
    assert preview.height == 8
    page.close()


def test_sequence_and_volume_controls_follow_declared_capabilities(qtbot) -> None:
    page = ViewerPage()
    qtbot.addWidget(page)
    imported_at = datetime.now(timezone.utc)
    sequence = ImageSequence2D(
        np.zeros((3, 8, 8), dtype=np.uint8),
        SourceType.RASTER,
        IntensitySemantics.GRAYSCALE,
    )
    page._loaded(LoadedStudy(sequence, sequence, imported_at=imported_at, source_kind="TIFF"))
    assert not page.navigation_group.isHidden()
    assert not page.frame_slider.isHidden()
    assert not page.play_button.isHidden()
    assert page.coronal_slider.isHidden()
    assert page.sagittal_slider.isHidden()

    volume = ImageVolume(
        np.zeros((4, 5, 6), dtype=np.float32),
        SourceType.NIFTI,
        IntensitySemantics.QUANTITATIVE,
        affine=np.eye(4),
    )
    projection = np.max(volume.array, axis=0)
    page._loaded(
        LoadedStudy(
            volume,
            volume,
            imported_at=imported_at,
            source_kind="NIfTI",
            volume_projection=projection,
        )
    )
    assert Capability.ORTHOGONAL_VIEWS in volume.capabilities
    assert not page.coronal_view.isHidden()
    assert not page.sagittal_view.isHidden()
    assert not page.coronal_slider.isHidden()
    assert not page.sagittal_slider.isHidden()
    assert page.play_button.isHidden()
    assert not page.aux_view.isHidden()
    np.testing.assert_array_equal(page.aux_view.array, projection)
    page.close()


def test_histogram_action_uses_valid_dialog_signature(qtbot) -> None:
    page = ViewerPage()
    qtbot.addWidget(page)
    page._loaded(_raster_study())
    with patch("dicom_viewer.ui.main_window.HistogramDialog.exec_", return_value=0) as execute:
        page._show_histogram()
    execute.assert_called_once()
    page.close()


def test_photometric_inversion_is_preserved_when_window_changes(qtbot) -> None:
    page = ViewerPage()
    qtbot.addWidget(page)
    study = _raster_study()
    inverted = RasterImage2D(
        study.image.array,
        SourceType.RASTER,
        IntensitySemantics.GRAYSCALE,
        bit_depth=16,
        runtime_metadata={"format": "test", "display_inverted": True},
    )
    page._loaded(
        LoadedStudy(
            image=inverted,
            display_image=inverted,
            imported_at=study.imported_at,
            source_kind="test",
        )
    )
    assert page._gray_mapping is not None and page._gray_mapping.invert
    page.lower_spin.setValue(page.lower_spin.value() - 1.0)
    page._display_range_changed()
    assert page._gray_mapping is not None and page._gray_mapping.invert
    page.close()


def test_rgb_display_uses_brightness_contrast_and_gamma_controls(qtbot) -> None:
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[:, :, 0] = 96
    image = RasterImage2D(
        pixels,
        SourceType.RASTER,
        IntensitySemantics.COLOR,
        bit_depth=8,
        color_space=ColorSpace.RGB,
    )
    page = ViewerPage()
    qtbot.addWidget(page)
    page._loaded(
        LoadedStudy(
            image=image,
            display_image=image,
            imported_at=datetime.now(timezone.utc),
            source_kind="PNG",
        )
    )

    assert not page.lower_spin.isEnabled()
    assert page.brightness_spin.isEnabled()
    assert page.contrast_spin.isEnabled()
    assert page.gamma_spin.isEnabled()
    page.brightness_spin.setValue(0.2)
    page.contrast_spin.setValue(1.3)
    page.gamma_spin.setValue(1.5)
    assert page._color_mapping.brightness == pytest.approx(0.2)
    assert page._color_mapping.contrast == pytest.approx(1.3)
    assert page._color_mapping.gamma == pytest.approx(1.5)
    page.close()


def test_reconstruction_exposes_signed_difference_and_current_roi_metrics(qtbot) -> None:
    reference = np.arange(64, dtype=np.float64).reshape(8, 8)
    page = ReconstructionPage(
        lambda: reference,
        roi_source=lambda: {"viewer_roi": (2, 2, 4, 4)},
    )
    qtbot.addWidget(page)
    page.reference = page._prepare_source(reference)
    page._reference_rois = page._prepare_rois(reference.shape, page.roi_source())
    reconstruction = reference.copy()
    reconstruction[3, 3] += 1.0
    page._reconstruction_ready(
        SimpleNamespace(image=reconstruction, intermediate={"backprojection": reference})
    )

    entries = [page.intermediate_combo.itemText(i) for i in range(page.intermediate_combo.count())]
    assert "Signed normalized difference" in entries
    assert "Absolute error heatmap" in entries
    assert "ROI viewer_roi" in page.metrics_label.text()
    assert "roi_viewer_roi_mse" in page._latest_metrics
    page.intermediate_combo.setCurrentText("Signed normalized difference")
    assert page.error_view._title_item.text() == "Intermediate: Signed normalized difference"
    page.close()


def test_reconstruction_exposes_every_radon_backprojection_and_sart_snapshot(qtbot) -> None:
    reference = np.arange(64, dtype=np.float64).reshape(8, 8)
    page = ReconstructionPage(lambda: reference)
    qtbot.addWidget(page)
    page.reference = reference
    sinogram = np.arange(32, dtype=np.float64).reshape(8, 4)
    page._sinogram_ready(
        SimpleNamespace(
            sinogram=sinogram,
            theta_degrees=np.arange(4, dtype=float),
            circle=True,
            intermediate={
                "projections_1": sinogram[:, :1],
                "projections_2": sinogram[:, :2],
                "projections_4": sinogram,
            },
        )
    )
    entries = [page.intermediate_combo.itemText(i) for i in range(page.intermediate_combo.count())]
    assert entries[:3] == [
        "Radon projection 1/4",
        "Radon projection 2/4",
        "Radon projection 4/4",
    ]

    backprojections = tuple(np.full((8, 8), value, dtype=float) for value in (1, 2, 3))
    page._reconstruction_ready(
        SimpleNamespace(
            image=backprojections[-1],
            algorithm="fbp",
            intermediate={"backprojection_steps": backprojections},
        )
    )
    entries = [page.intermediate_combo.itemText(i) for i in range(page.intermediate_combo.count())]
    assert "FBP progress 1/3" in entries
    assert "FBP progress 2/3" in entries
    assert "FBP progress 3/3" in entries
    page.intermediate_combo.setCurrentText("FBP progress 2/3")
    np.testing.assert_array_equal(page.error_view.array, backprojections[1])

    iterations = tuple(np.full((8, 8), value, dtype=float) for value in (4, 5))
    page._reconstruction_ready(
        SimpleNamespace(
            image=iterations[-1],
            algorithm="sart",
            intermediate={"iteration_images": iterations},
        )
    )
    entries = [page.intermediate_combo.itemText(i) for i in range(page.intermediate_combo.count())]
    assert "SART iteration 1/2" in entries
    assert "SART iteration 2/2" in entries
    page.close()


def test_assistant_image_authorization_is_destination_scoped_and_revocable(qtbot) -> None:
    page = AssistantPage(lambda: b"unused")
    qtbot.addWidget(page)
    page.network_enabled.setChecked(True)
    page.send_preview.setChecked(True)
    openai_destination = page._authorization_key()
    assert "confirmation required" in page.cloud_status.text()
    page._authorized_image_destinations.add(openai_destination)
    page._update_cloud_status()
    assert "destination authorized" in page.cloud_status.text()
    assert page.revoke_button.isEnabled()
    page.send_preview.setChecked(False)
    assert "authorization retained" in page.cloud_status.text()
    page.send_preview.setChecked(True)
    page.provider.setCurrentText("Anthropic")
    assert page._authorization_key() != openai_destination
    assert "confirmation required" in page.cloud_status.text()
    page.provider.setCurrentText("OpenAI")
    page._revoke_image_authorization()
    assert openai_destination not in page._authorized_image_destinations
    assert not page.revoke_button.isEnabled()
    page.close()


def test_assistant_disables_image_controls_for_deepseek(qtbot) -> None:
    page = AssistantPage(lambda: b"unused")
    qtbot.addWidget(page)
    page.vision.setChecked(True)
    page.send_preview.setChecked(True)

    page.provider.setCurrentText("DeepSeek")

    assert not page.vision.isChecked()
    assert not page.send_preview.isChecked()
    assert not page.vision.isEnabled()
    assert not page.send_preview.isEnabled()
    assert "deepseek-v4-flash" in page.model.placeholderText()
    page.close()


def test_assistant_cancel_control_cancels_active_task(qtbot) -> None:
    page = AssistantPage(lambda: b"unused")
    qtbot.addWidget(page)
    task = Mock()
    task.cancel.return_value = True
    page._active_task = task
    page.cancel_request_button.setEnabled(True)
    page._cancel_request()
    task.cancel.assert_called_once_with()
    assert not page.cancel_request_button.isEnabled()
    page._active_task = None
    page.close()


def test_model_page_renders_typed_mask_artifact_over_source_pixels(qtbot) -> None:
    source = np.arange(64, dtype=np.uint8).reshape(8, 8)
    page = ModelPage(lambda: source)
    qtbot.addWidget(page)
    page._last_input = source
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 3:7] = 1
    page._render_artifacts(
        (
            VisualizationArtifact(
                kind=VisualizationKind.MASK_OVERLAY,
                title="Source-aligned mask",
                payload=mask,
                coordinate_system="source-pixel",
            ),
        )
    )
    np.testing.assert_array_equal(page.model_output_view.array, source)
    assert page.model_output_view._overlay_items
    assert page.model_output_view._title_item.text() == "Source-aligned mask"
    page.close()


def test_model_page_keeps_all_visual_artifacts_in_comparison_tabs(qtbot) -> None:
    source = np.arange(64, dtype=np.uint8).reshape(8, 8)
    page = ModelPage(lambda: source)
    qtbot.addWidget(page)
    page._last_input = source
    image = np.full((8, 8), 2.0)
    heatmap = np.eye(8)
    page._render_artifacts(
        (
            VisualizationArtifact(
                kind=VisualizationKind.IMAGE,
                title="Intermediate feature",
                payload=image,
            ),
            VisualizationArtifact(
                kind=VisualizationKind.HEATMAP,
                title="Final heatmap",
                payload=heatmap,
            ),
        )
    )

    assert page.model_output_tabs.count() == 2
    assert page.model_output_tabs.tabText(0) == "Intermediate feature"
    assert page.model_output_tabs.tabText(1) == "Final heatmap"
    np.testing.assert_array_equal(page.model_output_view.array, image)
    np.testing.assert_array_equal(page.model_output_tabs.widget(1).array, heatmap)
    page.close()


def test_model_language_switch_preserves_plugin_text_payload(qtbot) -> None:
    page = ModelPage(lambda: np.zeros((8, 8)))
    qtbot.addWidget(page)
    payload = "Image\nDistance\nnone"
    page._render_artifacts(
        (
            VisualizationArtifact(
                kind=VisualizationKind.TEXT,
                title="Plugin text",
                payload=payload,
            ),
        )
    )

    page.set_language("zh_CN")
    assert f"Plugin text:\n{payload}" in page.summary.toPlainText()
    page.set_language("en")
    assert f"Plugin text:\n{payload}" in page.summary.toPlainText()
    page.close()


def test_exif_source_coordinates_are_preserved_for_models_and_mapped_to_display(qtbot) -> None:
    transform = TransformRecord.from_exif_orientation(6, (3, 5))
    canonical = np.arange(15, dtype=np.uint8).reshape(5, 3)
    image = RasterImage2D(
        canonical,
        SourceType.RASTER,
        IntensitySemantics.GRAYSCALE,
        bit_depth=8,
        transform_record=transform,
    )
    viewer = ViewerPage()
    qtbot.addWidget(viewer)
    viewer._loaded(
        LoadedStudy(
            image=image,
            display_image=image,
            imported_at=datetime.now(timezone.utc),
            source_kind="JPEG",
        )
    )
    assert viewer.current_model_input() is image

    page = ModelPage(viewer.current_model_input)
    qtbot.addWidget(page)
    page._last_input = canonical
    page._display_transform = transform
    source_mask = np.zeros((3, 5), dtype=np.uint8)
    source_mask[0, 0] = 1
    display_mask = page._source_array_to_display(source_mask, discrete=True)
    assert display_mask.shape == canonical.shape
    assert display_mask[0, 2] == 1
    page.close()
    viewer.close()


def test_new_image_signal_resets_reconstruction_state(qtbot) -> None:
    window = OpenMedVisionXWindow()
    qtbot.addWidget(window)
    window.reconstruction.reference = np.ones((4, 4))
    window.reconstruction.sinogram_result = object()
    window.viewer.imageChanged.emit(None)
    assert window.reconstruction.reference is None
    assert window.reconstruction.sinogram_result is None
    assert not window.reconstruction.reconstruct_button.isEnabled()
    window.close()
