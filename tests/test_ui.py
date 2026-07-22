from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTransform
from PyQt5.QtWidgets import QApplication, QFormLayout, QMessageBox, QPlainTextEdit

import workbench.ui.main_window as main_window_module
from workbench.domain import (
    Capability,
    ColorSpace,
    ImageSequence2D,
    ImageVolume,
    IntensitySemantics,
    RasterImage2D,
    SourceType,
    TransformRecord,
)
from workbench.errors import ViewerError
from workbench.inference import (
    Dimensionality,
    VisualizationArtifact,
    VisualizationKind,
)
from workbench.llm import (
    ArtifactValidationStatus,
    LLMArtifactResponse,
    LLMArtifactType,
    LLMInputKind,
    LLMInputReference,
    LLMTaskKind,
    LLMTaskRequest,
    ProviderResponseMetadata,
    RenderedPreview,
    TextArtifact,
)
from workbench.services import LoadedStudy
from workbench.ui.main_window import (
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
        imported_at=datetime.now(UTC),
        source_kind="PNG",
    )


def test_window_constructs_offscreen_and_has_product_identity(qtbot) -> None:
    window = OpenMedVisionXWindow()
    qtbot.addWidget(window)
    assert "OpenMedVisionX" in window.windowTitle()
    assert window.tabs.count() == 6
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
    assert window.viewer.view_context.text().startswith("活动视图")
    assert "合成仿体" in window.reconstruction.action_status.text()
    assert "经过审查的清单" in window.models.capability_status.text()
    assert window.learning.first_experiment_button.text() == "开始首个实验"
    assert "下一步" in window.assistant.send_reason.text()
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
    assert "public LoDoPaB benchmark teaching case" in page.introduction.text()
    assert "no DICOM metadata" in page.introduction.text()
    assert "patient identifiers" in page.introduction.text()
    assert "clinical case narrative" in page.introduction.text()
    assert "No medical dataset is bundled" not in page.introduction.text()
    page.set_language("zh_CN")
    assert "公开的 LoDoPaB 基准教学案例" in page.introduction.text()
    assert "患者标识符" in page.introduction.text()


def test_window_layout_fits_a_common_desktop_width(qtbot) -> None:
    window = OpenMedVisionXWindow()
    qtbot.addWidget(window)
    window.resize(1024, 680)
    window.show()
    QApplication.processEvents()

    assert window.minimumSizeHint().width() <= 1024
    assert window.minimumSizeHint().height() <= 680
    assert max(window.tabs.widget(index).minimumSizeHint().width() for index in range(5)) <= 1024
    window.tabs.setCurrentWidget(window.reconstruction)
    QApplication.processEvents()
    assert (
        window.reconstruction.generate_button.sizeHint().width()
        <= window.reconstruction.generate_button.width()
    )
    window.tabs.setCurrentWidget(window.models)
    QApplication.processEvents()
    model_actions = (
        window.models.manifest_button,
        window.models.load_button,
        window.models.run_button,
        window.models.cancel_button,
    )
    assert all(button.sizeHint().width() <= button.width() for button in model_actions)
    window.resize(1600, 900)
    QApplication.processEvents()
    assert len({button.y() for button in model_actions}) == 1
    assert window.assistant.config_group.layout().rowWrapPolicy() == QFormLayout.WrapAllRows
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


def test_ct_lab_defaults_to_one_click_synthetic_phantom_without_image(qtbot) -> None:
    page = ReconstructionPage(lambda: np.zeros((8, 8)))
    qtbot.addWidget(page)
    page.reset(None)

    assert page.source_mode.currentData() == "phantom"
    assert page.generate_button.isEnabled()
    assert "mm⁻¹" in page.source_provenance.text()
    assert "one click" in page.action_status.text()

    page._set_reconstruction_controls_enabled(False)
    assert not page.source_mode.isEnabled()
    assert not page.phantom_size.isEnabled()
    page._set_reconstruction_controls_enabled(True)
    assert page.source_mode.isEnabled()
    assert page.phantom_size.isEnabled()

    page.source_mode.setCurrentIndex(page.source_mode.findData("image"))

    assert not page.generate_button.isEnabled()
    assert "open an image" in page.action_status.text().lower()
    assert "open an image" in page.generate_button.toolTip().lower()
    assert "open an image" in page.generate_button.accessibleDescription().lower()
    assert "not scanner projections" in page.source_provenance.text()
    page._set_reconstruction_controls_enabled(False)
    page._set_reconstruction_controls_enabled(True)
    assert not page.phantom_size.isEnabled()
    page.close()


def test_first_experiment_runs_the_synthetic_phantom_end_to_end(qtbot) -> None:
    source = Mock(side_effect=AssertionError("The offline phantom must not read patient data."))
    page = ReconstructionPage(source)
    qtbot.addWidget(page)
    page.phantom_size.setValue(64)
    page.projections.setValue(24)

    page.start_first_experiment()

    qtbot.waitUntil(lambda: page.reconstruction_result is not None, timeout=15_000)
    source.assert_not_called()
    assert page.reference is not None
    assert np.min(page.reference) >= 0.0
    assert page._evaluation_range == pytest.approx((0.0, 0.03))
    assert page.export_result_button.isEnabled()
    assert page.input_view.value_scale == (0.0, 0.03, "mm^-1", False)
    assert page.result_view.value_scale == (0.0, 0.03, "mm^-1", False)
    assert page.error_view.value_scale == (0.0, 1.0, "normalized |error|", False)
    assert "mm⁻¹" in page.source_provenance.text()
    page.close()


def test_signed_difference_uses_a_fixed_diverging_display_map() -> None:
    rgb = ReconstructionPage._diverging_difference_rgb(
        np.asarray([[-1.0, 0.0, 1.0, 4.0]], dtype=np.float64)
    )

    np.testing.assert_array_equal(rgb[0, 0], (37, 99, 235))
    np.testing.assert_array_equal(rgb[0, 1], (249, 250, 251))
    np.testing.assert_array_equal(rgb[0, 2], (220, 38, 38))
    np.testing.assert_array_equal(rgb[0, 3], (220, 38, 38))


def test_model_page_uses_a_model_neutral_manifest_empty_state(qtbot) -> None:
    page = ModelPage(lambda: np.zeros((8, 8)))
    qtbot.addWidget(page)
    assert "segment-anything" not in page.resource_status.text().lower()
    assert "reviewed model manifest" in page.resource_status.text()
    assert "No model code runs" in page.capability_status.text()


def test_model_page_reports_missing_weights_without_opening_untrusted_sources(qtbot) -> None:
    page = ModelPage(lambda: np.zeros((8, 8)))
    qtbot.addWidget(page)
    example = (
        Path(main_window_module.__file__).parents[1] / "inference" / "examples" / "manifest.yaml"
    )
    page.manifest = page.service.inspect_manifest(example)

    page._refresh_model_resource_status()
    page._update_capability_gate()

    assert "example_classifier.onnx" in page.resource_status.text()
    assert "example.invalid" not in page.resource_status.text()
    assert "No GitHub source" in page.resource_status.text()
    assert not page.load_button.isEnabled()
    assert "example_classifier.onnx" in page.load_button.toolTip()
    assert "example_classifier.onnx" in page.load_button.accessibleDescription()
    assert "required local weight files" in page.capability_status.text()


def test_model_page_blocks_silent_dimensional_downgrade_before_execution(qtbot) -> None:
    page = ModelPage(
        lambda: np.zeros((8, 8)),
        input_context=lambda: {"modality": "generic-image", "spacing": None},
    )
    qtbot.addWidget(page)
    example = (
        Path(main_window_module.__file__).parents[1] / "inference" / "examples" / "manifest.yaml"
    )
    manifest = page.service.inspect_manifest(example)
    page.manifest = replace(
        manifest,
        inputs=(replace(manifest.inputs[0], dimensionality=Dimensionality.THREE_D),),
    )

    page._update_capability_gate()

    assert not page.run_button.isEnabled()
    assert "supports declared 2-D" in page.capability_status.text()
    assert "silent dimensional downgrade" in page.capability_status.text()


def test_model_page_blocks_preview_pixels_until_explicit_override(qtbot, monkeypatch) -> None:
    context = {
        "modality": "generic-image",
        "spacing": None,
        "preview_only": True,
        "source_shape": (4096, 4096),
        "display_shape": (1024, 1024),
    }
    page = ModelPage(lambda: np.zeros((1024, 1024)), input_context=lambda: context)
    qtbot.addWidget(page)
    example = (
        Path(main_window_module.__file__).parents[1] / "inference" / "examples" / "manifest.yaml"
    )
    page.manifest = page.service.inspect_manifest(example)
    monkeypatch.setattr(page, "_missing_weight_specs", lambda: ())

    page._update_capability_gate()

    assert not page.preview_override.isHidden()
    assert not page.run_button.isEnabled()
    assert "reduced-resolution preview" in page.capability_status.text()
    assert "4096" in page.capability_status.text()

    page.preview_override.setChecked(True)
    assert "final confirmation" in page.capability_status.text()
    page.close()


def test_preview_override_still_requires_final_model_specific_confirmation(
    qtbot, monkeypatch
) -> None:
    source = Mock(return_value=np.zeros((8, 8), dtype=np.uint8))
    context = {
        "modality": "generic-image",
        "spacing": None,
        "preview_only": True,
        "source_shape": (64, 64),
        "display_shape": (8, 8),
    }
    page = ModelPage(source, input_context=lambda: context)
    qtbot.addWidget(page)
    example = (
        Path(main_window_module.__file__).parents[1] / "inference" / "examples" / "manifest.yaml"
    )
    page.manifest = page.service.inspect_manifest(example)
    monkeypatch.setattr(page, "_missing_weight_specs", lambda: ())
    page.service._plugin = object()  # type: ignore[assignment]
    page.service._loaded = True
    page.preview_override.setChecked(True)
    confirmation = Mock(return_value=False)
    monkeypatch.setattr(page, "_confirm_preview_inference", confirmation)

    page._run_plugin()

    confirmation.assert_called_once_with(context)
    source.assert_not_called()
    assert page._active_task is None
    page.close()


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


def test_viewer_marks_bounded_thumbnail_as_preview_only_model_input(qtbot) -> None:
    image = RasterImage2D(
        np.zeros((8, 10), dtype=np.uint8),
        SourceType.RASTER,
        IntensitySemantics.GRAYSCALE,
        runtime_metadata={
            "access": "thumbnail",
            "source_shape": (80, 100),
        },
    )
    page = ViewerPage()
    qtbot.addWidget(page)
    page._loaded(
        LoadedStudy(
            image,
            image,
            imported_at=datetime.now(UTC),
            source_kind="PNG",
        )
    )

    context = page.current_model_input_context()

    assert context["preview_only"] is True
    assert context["source_shape"] == (80, 100)
    assert context["display_shape"] == (8, 10)
    assert "bounded thumbnail" in page.warning_label.text()
    page.close()


def test_sequence_and_volume_controls_follow_declared_capabilities(qtbot) -> None:
    page = ViewerPage()
    qtbot.addWidget(page)
    imported_at = datetime.now(UTC)
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
    assert page.axial_view.orientation_labels == ("P", "R", "A", "L")
    assert page.coronal_view.orientation_labels == ("I", "R", "S", "L")
    assert page.sagittal_view.orientation_labels == ("I", "A", "S", "P")
    assert page.aux_view.orientation_labels == ("P", "R", "A", "L")
    np.testing.assert_array_equal(page.aux_view.array, projection)
    page.close()


def test_active_volume_view_drives_plane_and_world_coordinate_feedback(qtbot) -> None:
    page = ViewerPage()
    qtbot.addWidget(page)
    volume = ImageVolume(
        np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6),
        SourceType.NIFTI,
        IntensitySemantics.QUANTITATIVE,
        affine=np.eye(4),
    )
    page._loaded(
        LoadedStudy(
            volume,
            volume,
            imported_at=datetime.now(UTC),
            source_kind="NIfTI",
        )
    )

    assert "Loaded NIfTI" in page.status.text()
    page._set_active_view("coronal")
    np.testing.assert_array_equal(
        page.current_plane(),
        volume.coronal(page.coronal_slider.value()),
    )
    page._pixel_hovered(2, 1, page.current_plane()[1, 2])

    assert page.coronal_view.property("active") is True
    assert page.axial_view.property("active") is False
    assert "Coronal" in page.view_context.text()
    assert "RAS+ hover (Coronal): R 2.00" in page.view_context.text()
    assert f"A {page.coronal_slider.value():.2f}" in page.view_context.text()
    page.close()


def test_volume_navigation_links_all_mpr_planes_without_changing_data(qtbot) -> None:
    page = ViewerPage()
    qtbot.addWidget(page)
    volume = ImageVolume(
        np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6),
        SourceType.NIFTI,
        IntensitySemantics.QUANTITATIVE,
        affine=np.eye(4),
    )
    page._loaded(
        LoadedStudy(
            volume,
            volume,
            imported_at=datetime.now(UTC),
            source_kind="NIfTI",
        )
    )

    page._navigate_from_view("coronal", 4, 3)

    assert page.sagittal_slider.value() == 4
    assert page.frame_slider.value() == 3
    assert "Double-click a view to link all MPR planes" in page.view_context.text()
    assert len(page.axial_view._overlay_items) >= 2
    assert len(page.coronal_view._overlay_items) >= 2
    assert len(page.sagittal_view._overlay_items) >= 2
    np.testing.assert_array_equal(page.image.array, volume.array)
    page.close()


def test_hover_feedback_names_a_non_active_volume_view(qtbot) -> None:
    page = ViewerPage()
    qtbot.addWidget(page)
    volume = ImageVolume(
        np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6),
        SourceType.NIFTI,
        IntensitySemantics.QUANTITATIVE,
        affine=np.eye(4),
    )
    page._loaded(
        LoadedStudy(
            volume,
            volume,
            imported_at=datetime.now(UTC),
            source_kind="NIfTI",
        )
    )
    page._set_active_view("axial")

    page.coronal_view.pixelHovered.emit(2, 1, page.coronal_view.array[1, 2])

    assert "Active view: Axial" in page.view_context.text()
    assert "RAS+ hover (Coronal)" in page.view_context.text()
    page.close()


def test_histogram_action_uses_valid_dialog_signature(qtbot) -> None:
    page = ViewerPage()
    qtbot.addWidget(page)
    page._loaded(_raster_study())
    with patch("workbench.ui.main_window.HistogramDialog.exec_", return_value=0) as execute:
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
            imported_at=datetime.now(UTC),
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
    assert page.intermediate_combo.currentData() == "Absolute error heatmap"
    assert page.error_view._title_item.text() == "Absolute error heatmap"
    assert "ROI viewer_roi" in page.metrics_label.text()
    assert "roi_viewer_roi_mse" in page._latest_metrics
    assert "Evaluation range" in page.metrics_label.text()
    page.intermediate_combo.setCurrentText("Signed normalized difference")
    assert page.error_view._title_item.text() == "Intermediate: Signed normalized difference"
    page.set_language("zh_CN")
    assert "评估范围" in page.metrics_label.text()
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


def test_assistant_image_authorization_is_exact_and_one_time(qtbot) -> None:
    page = AssistantPage(lambda: b"unused")
    qtbot.addWidget(page)
    page.vision.setChecked(True)
    page.network_enabled.setChecked(True)
    page.send_preview.setChecked(True)
    assert "one-time review required" in page.cloud_status.text()
    page.send_preview.setChecked(False)
    assert page.cloud_status.text().endswith("OFF")
    page.send_preview.setChecked(True)
    assert "one-time review required" in page.cloud_status.text()
    assert not hasattr(page, "_authorized_image_destinations")
    page.close()


def test_assistant_allows_none_credentials_only_for_loopback(qtbot) -> None:
    page = AssistantPage(lambda: b"unused")
    qtbot.addWidget(page)
    page.network_enabled.setChecked(True)
    page.model.setText("local-teaching-model")
    page.prompt.setPlainText("Explain windowing")
    page.endpoint.setText("http://127.0.0.1:8080/v1/chat/completions")
    page.credential.setText("none")

    assert page.send_button.isEnabled()
    assert page.credential.property("validation") == "normal"

    page.endpoint.setText("https://example.org/v1/chat/completions")
    assert not page.send_button.isEnabled()
    assert page.credential.property("validation") == "error"
    assert "loopback" in page.send_reason.text()
    page.close()


def test_assistant_transfer_confirmation_shows_preview_and_exact_plan(qtbot) -> None:
    viewer = ViewerPage()
    qtbot.addWidget(viewer)
    viewer._loaded(_raster_study())
    preview = RenderedPreview.from_png(viewer.rendered_preview_png())
    page = AssistantPage(lambda: preview.data)
    qtbot.addWidget(page)
    page.model.setText("<b>vision-model-v1</b>")
    page.vision.setChecked(True)
    provider = page._make_provider()
    plan = page.service.plan_image_transfer(
        provider,
        "Explain this visible slice",
        preview,
        task="teaching-explanation",
    )

    dialog = page._build_image_transfer_confirmation(plan, preview)
    text = dialog.informativeText()
    assert dialog.iconPixmap() is not None
    assert not dialog.iconPixmap().isNull()
    assert plan.provider_name in text
    assert plan.endpoint_host in text
    assert plan.endpoint in text
    assert plan.model_id in text
    assert plan.task in text
    assert str(plan.total_bytes) in text
    assert plan.items[0].name in text
    assert plan.items[0].mime_type in text
    assert f"{preview.width} × {preview.height}" in text
    assert str(len(preview.data)) in text
    assert plan.prompt_sha256 in text
    assert preview.sha256 in text
    assert len(plan.prompt_sha256) == 64
    assert len(preview.sha256) == 64
    assert plan.items[0].transform in text
    assert plan.items[0].deidentification_actions[0] in text
    assert plan.items[0].burned_in_text_review in text
    assert "Burned-in text" in text
    assert "Cancellation cannot recall" in text
    assert "full prompt text" in text
    details_view = dialog.findChild(QPlainTextEdit, "transferPlanDetails")
    assert details_view is not None
    assert details_view.toPlainText() == text
    assert details_view.lineWrapMode() == QPlainTextEdit.WidgetWidth
    details_view.selectAll()
    assert details_view.textCursor().selectedText()
    assert "<b>vision-model-v1</b>" in details_view.toPlainText()
    assert dialog.textFormat() == Qt.PlainText
    assert dialog.button(QMessageBox.Yes).text() == "Authorize once and send"
    dialog.show()
    QApplication.processEvents()
    assert dialog.button(QMessageBox.No).isDefault()
    assert not dialog.button(QMessageBox.Yes).isDefault()
    dialog.close()

    mismatched_plan = replace(
        plan,
        items=(replace(plan.items[0], sha256="0" * 64),),
    )
    with pytest.raises(ViewerError, match="does not match"):
        page._build_image_transfer_confirmation(mismatched_plan, preview)

    page.set_language("zh_CN")
    translated = page._build_image_transfer_confirmation(plan, preview)
    assert "目标主机" in translated.informativeText()
    assert "剩余风险" in translated.informativeText()
    assert "去标识与载荷最小化" in translated.informativeText()
    assert "外发总字节数" in translated.informativeText()
    assert "完整提示词" in translated.informativeText()
    assert translated.button(QMessageBox.Yes).text() == "授权一次并发送"
    translated.close()
    page.close()
    viewer.close()


def test_assistant_rejection_does_not_authorize_or_dispatch(qtbot) -> None:
    viewer = ViewerPage()
    qtbot.addWidget(viewer)
    viewer._loaded(_raster_study())
    service = Mock()
    service.provider_names = ("OpenAI",)
    service.provider_defaults.return_value = SimpleNamespace(
        endpoint="https://api.openai.com/v1/responses",
        credential_ref="env:OPENAI_API_KEY",
        timeout=30.0,
    )
    provider = object()
    service.create_provider.return_value = provider
    service.plan_image_transfer.return_value = object()
    page = AssistantPage(viewer.rendered_preview_png, service=service)
    qtbot.addWidget(page)
    page.model.setText("vision-model-v1")
    page.network_enabled.setChecked(True)
    page.vision.setChecked(True)
    page.send_preview.setChecked(True)
    page.prompt.setPlainText("Explain this active plane")

    with (
        patch.object(page, "_confirm_image_transfer", return_value=False),
        patch.object(page.runner, "submit") as submit,
    ):
        page._send()

    service.authorize_image_transfer.assert_not_called()
    service.chat.assert_not_called()
    submit.assert_not_called()
    assert page._active_task is None
    page.close()
    viewer.close()


def test_assistant_uses_reviewed_one_shot_transfer_plan_for_chat(qtbot) -> None:
    viewer = ViewerPage()
    qtbot.addWidget(viewer)
    viewer._loaded(_raster_study())
    service = Mock()
    service.provider_names = ("OpenAI",)
    service.provider_defaults.return_value = SimpleNamespace(
        endpoint="https://api.openai.com/v1/responses",
        credential_ref="env:OPENAI_API_KEY",
        timeout=30.0,
    )
    provider = object()
    service.create_provider.return_value = provider
    transfer_plan = object()
    service.plan_image_transfer.return_value = transfer_plan
    service.chat.return_value = SimpleNamespace(text="A teaching explanation")
    page = AssistantPage(viewer.rendered_preview_png, service=service)
    qtbot.addWidget(page)
    page.model.setText("vision-model-v1")
    page.network_enabled.setChecked(True)
    page.vision.setChecked(True)
    page.send_preview.setChecked(True)
    page.prompt.setPlainText("Explain this visible slice")
    token = object()
    task = Mock()

    def run_now(operation):
        operation(SimpleNamespace(raise_if_cancelled=Mock(), token=token))
        return task

    with (
        patch.object(page, "_confirm_image_transfer", return_value=True) as confirm,
        patch.object(page.runner, "submit", side_effect=run_now),
        patch.object(page.watcher, "watch") as watch,
    ):
        page._send()

    plan_call = service.plan_image_transfer.call_args
    assert plan_call.args[:2] == (provider, "Explain this visible slice")
    assert isinstance(plan_call.args[2], RenderedPreview)
    assert plan_call.kwargs == {"task": "teaching-explanation"}
    preview = plan_call.args[2]
    confirm.assert_called_once_with(transfer_plan, preview)
    service.authorize_image_transfer.assert_called_once_with(provider, transfer_plan)
    service.chat.assert_called_once_with(
        provider,
        "Explain this visible slice",
        preview=preview,
        transfer_plan=transfer_plan,
        cancellation_token=token,
    )
    watch.assert_called_once()
    assert "reviewed one-time request in progress" in page.cloud_status.text()
    page._active_task = None
    page.close()
    viewer.close()


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


def test_assistant_separates_chat_from_structured_artifacts_and_reviews_locally(qtbot) -> None:
    request = LLMTaskRequest(
        request_id="request-local",
        task=LLMTaskKind.CLASSIFY,
        inputs=(
            LLMInputReference(
                input_id="input-local",
                kind=LLMInputKind.IMAGE_2D,
                payload_sha256="a" * 64,
            ),
        ),
        transfer_plan_sha256="b" * 64,
        prompt_sha256="c" * 64,
        requested_artifact_types=(LLMArtifactType.CLASS_SCORES,),
    )
    response = LLMArtifactResponse.from_normalized_payload(
        artifact_id="artifact-local",
        request=request,
        artifact_type=LLMArtifactType.TEXT,
        payload=TextArtifact(text="Teaching explanation", language="en"),
        provider=ProviderResponseMetadata(
            provider_id="provider-local",
            model_id="registry/model-local",
            response_id="response-local",
            authenticated=True,
        ),
    )
    page = AssistantPage(lambda: b"unused")
    qtbot.addWidget(page)

    page.set_structured_artifact_context(request, response)

    assert page.assistant_workspace_tabs.count() == 2
    assert page.assistant_workspace_tabs.currentWidget() is page.artifact_scroll
    assert page.artifact_workbench.binding_valid
    page.artifact_workbench.confirm_button.click()
    assert response.validation_status is ArtifactValidationStatus.UNVERIFIED
    assert (
        page.artifact_workbench.artifact.validation_status
        is ArtifactValidationStatus.USER_CONFIRMED
    )

    page.set_language("zh_CN")
    assert page.assistant_workspace_tabs.tabText(0) == "教学对话"
    assert page.assistant_workspace_tabs.tabText(1) == "结构化产物 · API 预览"
    assert page.artifact_workbench.title_label.text() == "结构化产物契约"
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
    assert page.model_output_view.value_scale is not None
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
    assert page._source_grayscale_mapping is not None
    assert page.model_output_view.value_scale == (
        page._source_grayscale_mapping.lower,
        page._source_grayscale_mapping.upper,
        "",
        False,
    )
    assert page.model_output_tabs.widget(1).value_scale == (0.0, 1.0, "", False)
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
            imported_at=datetime.now(UTC),
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
