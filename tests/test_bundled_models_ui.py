from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from workbench.models import ModelRunResult
from workbench.services.brats_segmentation import (
    BraTSRegionEvaluation,
    BraTSSegmentationResult,
    ChannelNormalization,
)
from workbench.services.bundled_models import BundledDemoResult, TorchDeviceStatus
from workbench.ui.bundled_models import BundledModelsPanel
from workbench.ui.main_window import ModelPage, OpenMedVisionXWindow

_BUNDLE_IDS = (
    "dival-lodopab-fbpunet",
    "deepinv-mri-modl",
    "monai-brats-segmentation",
)
_RUN_THREAD_IDS: list[int] = []
_DEMO_THREAD_IDS: list[int] = []
_BRATS_THREAD_IDS: list[int] = []


def _record(bundle_id: str, *, verified: bool = False) -> SimpleNamespace:
    bundle_root = (
        Path(__file__).parents[1] / "src" / "workbench" / "resources" / "model_bundles" / bundle_id
    )
    names = {
        "dival-lodopab-fbpunet": (
            "DIVal LoDoPaB FBP-U-Net",
            "DIVal LoDoPaB FBP-U-Net 后处理模型",
        ),
        "deepinv-mri-modl": (
            "DeepInverse MRI MoDL Teaching Model",
            "DeepInverse MRI MoDL 教学模型",
        ),
        "monai-brats-segmentation": (
            "MONAI BraTS 3D Segmentation",
            "MONAI BraTS 三维分割模型",
        ),
    }
    tasks = {
        "dival-lodopab-fbpunet": "ct-reconstruction-postprocessing",
        "deepinv-mri-modl": "mri-reconstruction",
        "monai-brats-segmentation": "segmentation",
    }
    modalities = {
        "dival-lodopab-fbpunet": ("FBP image",),
        "deepinv-mri-modl": ("real", "imaginary"),
        "monai-brats-segmentation": ("T1ce", "T1", "T2", "FLAIR"),
    }
    name_en, name_zh = names[bundle_id]
    task = tasks[bundle_id]
    return SimpleNamespace(
        bundle_id=bundle_id,
        display_name=name_en,
        display_name_zh=name_zh,
        description_en=f"Reviewed {name_en} description.",
        description_zh=f"经过审查的 {name_zh} 说明。",
        task=task,
        dimensionality="3d" if bundle_id == "monai-brats-segmentation" else "2d",
        artifact_sha256="a" * 64,
        license="reviewed-test-license",
        modalities=modalities[bundle_id],
        task_contract={
            "task": task,
            "input_semantics": "Explicit reviewed tensor layout",
            "spacing_orientation_intensity": "Explicit reviewed intensity domain",
            "labels_and_output_semantics": "Explicit typed output",
            "preprocessing_postprocessing": "Explicit preprocessing and postprocessing",
        },
        resources={
            "gpu_backend": "PyTorch CUDA preferred with explicit CPU fallback",
            "expected_runtime_cpu_gpu": "Elapsed time and fallback are visible",
        },
        runtime="PyTorch CUDA preferred with explicit CPU fallback",
        source_url="https://example.test/reviewed-source",
        limitations=f"Reviewed limitations for {name_en}.",
        verified=verified,
        manifest_path=bundle_root / "bundle.yaml",
        record_path=bundle_root / "bundle.yaml",
    )


def _case() -> SimpleNamespace:
    observation = np.linspace(0.0, 1.0, 1000 * 513, dtype=np.float32).reshape(1000, 513)
    fbp = np.linspace(0.0, 1.0, 362 * 362, dtype=np.float32).reshape(362, 362)
    ground_truth = np.clip(fbp * 0.95 + 0.01, 0.0, 1.0).astype(np.float32)
    record = SimpleNamespace(
        case_id="lodopab-ct-test-03456",
        artifact_sha256="b" * 64,
        arrays={"fbp": {"sha256": "c" * 64}},
    )
    return SimpleNamespace(
        record=record,
        observation=observation,
        fbp=fbp,
        ground_truth=ground_truth,
        metadata=MappingProxyType({"case_id": record.case_id}),
    )


def _run_dival(value: np.ndarray, *, device: str) -> ModelRunResult:
    _RUN_THREAD_IDS.append(threading.get_ident())
    output = np.asarray(value, dtype=np.float32) + np.float32(0.005)
    return ModelRunResult(
        bundle_id="dival-lodopab-fbpunet",
        device="cuda" if device == "auto" else device,
        elapsed_seconds=0.012,
        outputs=MappingProxyType({"reconstruction": output}),
        output_channels=("FBP post-processing reconstruction",),
        fallback_reason=None,
        warnings=(
            "Input must come from the documented LoDoPaB FBP operator; it is not HU.",
            "Education and research only; not for diagnosis.",
        ),
    )


def _run_deepinv_demo(_context, *, device: str) -> BundledDemoResult:
    _DEMO_THREAD_IDS.append(threading.get_ident())
    source = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)
    prepared = np.asarray(source * np.float32(0.72), dtype=np.float32)
    output = np.asarray(source * np.float32(0.94) + np.float32(0.01), dtype=np.float32)
    return BundledDemoResult(
        bundle_id="deepinv-mri-modl",
        source=source,
        prepared_input=prepared,
        output=output,
        device="cuda" if device == "auto" else device,
        elapsed_seconds=0.008,
        fallback_reason=None,
        warnings=(
            "Synthetic phantom -> single-coil FFT -> deterministic undersampling mask -> "
            "reviewed DeepInverse task adapter.",
            "The k-space is simulated from an image phantom; it is not scanner raw data.",
        ),
    )


def _run_brats(_context, _directory: Path, *, config) -> BraTSSegmentationResult:
    _BRATS_THREAD_IDS.append(threading.get_ident())
    probabilities = np.zeros((3, 4, 5, 6), dtype=np.float32)
    probabilities[0, 2, 1:4, 1:5] = 0.8
    probabilities[1, 2, 2:4, 2:5] = 0.9
    probabilities[2, 2, 2:3, 3:4] = 0.95
    masks = probabilities >= 0.5
    normalizations = tuple(
        ChannelNormalization(modality, 100, 2.0, 0.5) for modality in ("T1ce", "T1", "T2", "FLAIR")
    )
    evaluations = MappingProxyType(
        {
            region: BraTSRegionEvaluation(
                region=region,
                ground_truth_labels=labels,
                dice=0.8,
                hd95_mm=1.5,
                prediction_volume_ml=1.2,
                ground_truth_volume_ml=1.0,
                signed_volume_error_ml=0.2,
                absolute_volume_error_ml=0.2,
            )
            for region, labels in {
                "WT": (1, 2, 4),
                "TC": (1, 4),
                "ET": (4,),
            }.items()
        }
    )
    return BraTSSegmentationResult(
        case_alias="brats-2021-0123456789abcdef",
        probabilities=probabilities,
        masks=masks,
        source_affine=np.eye(4),
        canonical_affine=np.eye(4),
        source_orientation=("R", "A", "S"),
        input_channels=("T1ce", "T1", "T2", "FLAIR"),
        native_output_regions=("TC", "WT", "ET"),
        output_regions=("WT", "TC", "ET"),
        normalizations=normalizations,
        evaluations=evaluations,
        threshold=0.5,
        patch_size=(96, 96, 96),
        overlap=0.5,
        patch_count=8,
        device="cuda" if config.device == "auto" else config.device,
        model_elapsed_seconds=0.25,
        elapsed_seconds=0.5,
        model_sha256="b" * 64,
        fallback_reason=None,
        warnings=(
            "Domain shift: model training domain is BraTS 2018; the selected case and "
            "evaluation domain are BraTS 2021 Task 1.",
            "Education and research only; not for diagnosis.",
        ),
        input_artifact_sha256=tuple(
            (modality, str(index) * 64)
            for index, modality in enumerate(("T1ce", "T1", "T2", "FLAIR"), start=1)
        ),
        ground_truth_artifact_sha256="5" * 64,
        requested_device=config.device,
    )


@pytest.fixture
def bundled_panel(qtbot):
    panel = BundledModelsPanel(
        record_loader=_record,
        bundle_verifier=lambda bundle_id: _record(bundle_id, verified=True),
        device_probe=lambda _context: TorchDeviceStatus(
            torch_version="test",
            cuda_available=True,
            device_name="Test CUDA GPU",
            total_vram_gib=8.0,
        ),
        case_loader=_case,
        dival_runner=_run_dival,
        demo_operations={"deepinv-mri-modl": _run_deepinv_demo},
    )
    qtbot.addWidget(panel)
    yield panel
    panel.shutdown()


def test_catalog_prioritizes_three_models_and_exposes_complete_contracts(
    bundled_panel: BundledModelsPanel,
) -> None:
    panel = bundled_panel
    assert panel.catalog.count() == 3
    assert [panel.catalog.item(index).data(Qt.UserRole) for index in range(3)] == list(_BUNDLE_IDS)
    contract = panel.contract_details.toPlainText()
    for label in (
        "Input layout & semantics",
        "Orientation, spacing & intensity range",
        "Preprocessing & postprocessing",
        "Output semantics",
        "Device policy",
        "Weights license",
        "Source",
        "Limitations",
        "Verified",
        "Weight SHA-256",
    ):
        assert label in contract
    assert "a" * 64 in contract
    assert panel.run_button.isEnabled()

    panel.catalog.setCurrentRow(1)
    assert panel.run_button.isEnabled()
    assert panel.run_button.text() == "Run synthetic MRI demo"
    assert "deterministic image-derived MRI simulation" in panel.run_button.accessibleName()
    assert "deterministic digital phantom" in panel.run_button.accessibleDescription()
    assert "image-derived simulation" in panel.limitation.text()
    assert "never scanner raw data" in panel.limitation.text()
    assert "real, imaginary" in panel.contract_details.toPlainText()
    assert "LoDoPaB or BraTS" in panel.export_button.toolTip()

    panel.catalog.setCurrentRow(2)
    assert panel.run_button.isEnabled()
    assert panel.run_button.text() == "Set up local BraTS 2021…"
    assert "Validate a user-managed" in panel.run_button.accessibleName()
    assert "no data will be downloaded" in panel.run_button.accessibleDescription()
    assert "four co-registered" in panel.limitation.text()
    assert "T1ce, T1, T2, FLAIR" in panel.contract_details.toPlainText()
    assert "BraTS 2018" not in panel.limitation.text()  # injected record stays authoritative
    assert "Validate a local BraTS 2021 case" in panel.export_button.toolTip()


def test_validated_brats_case_runs_in_background_and_renders_three_regions(
    qtbot, tmp_path: Path
) -> None:
    panel = BundledModelsPanel(
        record_loader=_record,
        demo_operations={"deepinv-mri-modl": _run_deepinv_demo},
        brats_runner=_run_brats,
    )
    qtbot.addWidget(panel)
    panel._brats_case_directory = tmp_path
    panel._brats_validation_report = SimpleNamespace(is_valid=True)
    panel.catalog.setCurrentRow(2)
    panel._refresh_selection()
    gui_thread = threading.get_ident()
    _BRATS_THREAD_IDS.clear()

    assert panel.run_button.text() == "Run BraTS segmentation"
    assert not panel.configure_brats_button.isHidden()
    panel.run_button.click()

    assert panel.active_task is not None
    assert not panel.catalog.isEnabled()
    qtbot.waitUntil(lambda: panel.active_task is None, timeout=3_000)

    assert panel.last_brats_result is not None
    assert panel.observation_view.array.shape == (5, 6)
    assert panel.fbp_view.array.shape == (5, 6)
    assert panel.output_view.array.shape == (5, 6)
    assert panel.ground_truth_view.array.shape == (5, 6, 3)
    assert "TC/WT/ET → WT/TC/ET" in panel.result_details.text()
    assert "Dice 0.8000" in panel.result_details.text()
    assert "BraTS 2018" in panel.result_details.text()
    assert panel.export_button.isEnabled()
    assert _BRATS_THREAD_IDS and _BRATS_THREAD_IDS[-1] != gui_thread

    destination = tmp_path / "brats-experiment.json"
    panel.export_experiment(destination)
    qtbot.waitUntil(lambda: panel.active_task is None, timeout=3_000)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema"] == "openmedvisionx-experiment-record/v1"
    assert payload["task"] == "brats-2021-task-1-segmentation"
    assert payload["model"]["id"] == "monai-brats-segmentation"
    assert payload["contains_phi"] is False
    assert str(tmp_path) not in destination.read_text(encoding="utf-8")
    panel.shutdown()


def test_lodopab_one_click_experiment_uses_fixed_reference_scales(
    bundled_panel: BundledModelsPanel,
    qtbot,
) -> None:
    panel = bundled_panel
    gui_thread = threading.get_ident()
    _RUN_THREAD_IDS.clear()

    panel.run_button.click()

    assert panel.active_task is not None
    assert not panel.catalog.isEnabled()
    qtbot.waitUntil(lambda: panel.active_task is None, timeout=3_000)

    assert panel.observation_view.array is not None
    assert panel.observation_view.array.shape == (1000, 513)
    assert panel.fbp_view.array is not None
    assert panel.output_view.array is not None
    assert panel.ground_truth_view.array is not None
    assert panel.lodopab_error_view.array is not None
    assert panel.last_experiment is not None
    ground_truth = np.asarray(panel.last_experiment.case.ground_truth)
    expected_lower = float(np.min(ground_truth))
    expected_upper = float(np.max(ground_truth))
    expected_comparison_scale = (
        expected_lower,
        expected_upper,
        "normalized attenuation",
        False,
    )
    assert panel.fbp_view.value_scale == expected_comparison_scale
    assert panel.output_view.value_scale == expected_comparison_scale
    assert panel.ground_truth_view.value_scale == expected_comparison_scale
    assert panel.lodopab_error_view.value_scale == (
        0.0,
        expected_upper - expected_lower,
        "normalized attenuation",
        False,
    )
    np.testing.assert_allclose(
        panel.lodopab_error_view.array,
        np.abs(panel.output_view.array - ground_truth),
    )
    assert "lodopab-ct-test-03456" in panel.result_details.text()
    assert "MAE" in panel.result_details.text()
    assert "RMSE" in panel.result_details.text()
    assert "PSNR" in panel.result_details.text()
    assert "auto → cuda" in panel.result_details.text()
    assert panel.export_button.isEnabled()
    assert panel.operation_status.property("state") == "ready"
    assert "Completed locally" in panel.operation_status.text()
    assert "Fixed display scale" in panel.result_details.text()
    assert _RUN_THREAD_IDS and _RUN_THREAD_IDS[-1] != gui_thread

    panel.set_language("zh_CN")
    assert "已在本地" in panel.operation_status.text()
    assert "仅供学习与研究" in panel.result_details.text()
    assert panel.observation_view.accessibleName() == "LoDoPaB 观测值"
    assert panel.lodopab_error_view.accessibleName() == "绝对误差 · 固定参考量程"
    assert panel.lodopab_error_view.value_scale is not None
    assert panel.lodopab_error_view.value_scale[2] == "归一化衰减"


def test_lodopab_extreme_output_cannot_change_fixed_display_or_error_scale(
    qtbot,
) -> None:
    extreme_value = np.float32(1_000_000.0)

    def extreme_runner(value: np.ndarray, *, device: str) -> ModelRunResult:
        output = np.full_like(value, extreme_value)
        output[0, 0] = -extreme_value
        return ModelRunResult(
            bundle_id="dival-lodopab-fbpunet",
            device="cuda" if device == "auto" else device,
            elapsed_seconds=0.001,
            outputs=MappingProxyType({"reconstruction": output}),
            output_channels=("FBP post-processing reconstruction",),
            fallback_reason=None,
            warnings=("Education and research only; not for diagnosis.",),
        )

    panel = BundledModelsPanel(
        record_loader=_record,
        bundle_verifier=lambda bundle_id: _record(bundle_id, verified=True),
        case_loader=_case,
        dival_runner=extreme_runner,
        demo_operations={"deepinv-mri-modl": _run_deepinv_demo},
    )
    qtbot.addWidget(panel)
    try:
        panel.run_button.click()
        qtbot.waitUntil(lambda: panel.active_task is None, timeout=3_000)

        reference = np.asarray(panel.last_experiment.case.ground_truth)
        lower = float(np.min(reference))
        upper = float(np.max(reference))
        reference_span = upper - lower
        comparison_scale = (lower, upper, "normalized attenuation", False)

        assert panel.fbp_view.value_scale == comparison_scale
        assert panel.output_view.value_scale == comparison_scale
        assert panel.ground_truth_view.value_scale == comparison_scale
        assert panel.lodopab_error_view.value_scale == (
            0.0,
            reference_span,
            "normalized attenuation",
            False,
        )
        assert float(np.max(panel.output_view.array)) == float(extreme_value)
        assert float(np.max(panel.lodopab_error_view.array)) > reference_span
        np.testing.assert_array_equal(
            panel.lodopab_error_view.array,
            np.abs(panel.output_view.array - reference),
        )
    finally:
        panel.shutdown()


def test_deepinv_demo_runs_in_background_renders_four_views_and_retranslates(
    bundled_panel: BundledModelsPanel,
    qtbot,
) -> None:
    panel = bundled_panel
    gui_thread = threading.get_ident()
    _DEMO_THREAD_IDS.clear()
    panel.catalog.setCurrentRow(1)
    assert panel.lodopab_error_view.isHidden()

    panel.run_button.click()

    assert panel.active_task is not None
    assert not panel.catalog.isEnabled()
    qtbot.waitUntil(lambda: panel.active_task is None, timeout=3_000)

    assert panel.last_demo is not None
    source = panel.observation_view.array
    measurement = panel.fbp_view.array
    reconstruction = panel.output_view.array
    absolute_error = panel.ground_truth_view.array
    assert source is not None and source.shape == (32, 32)
    assert measurement is not None and measurement.shape == source.shape
    assert reconstruction is not None and reconstruction.shape == source.shape
    assert absolute_error is not None
    np.testing.assert_allclose(absolute_error, np.abs(reconstruction - source))
    assert "auto → cuda" in panel.result_details.text()
    assert "Model time" in panel.result_details.text()
    assert "Total time" in panel.result_details.text()
    assert "MAE" in panel.result_details.text()
    assert "RMSE" in panel.result_details.text()
    assert "PSNR" in panel.result_details.text()
    assert "Warnings" in panel.result_details.text()
    assert "not scanner raw data" in panel.result_details.text()
    assert not panel.export_button.isEnabled()
    assert "LoDoPaB or BraTS" in panel.export_button.toolTip()
    assert panel.operation_status.property("state") == "ready"
    assert "Completed synthetic MRI locally" in panel.operation_status.text()
    assert _DEMO_THREAD_IDS and _DEMO_THREAD_IDS[-1] != gui_thread

    panel.set_language("zh_CN")
    assert "请求设备 → 实际设备" in panel.result_details.text()
    assert "相对数字模体的指标" in panel.result_details.text()
    assert "并非扫描仪原始数据" in panel.result_details.text()
    assert panel.observation_view.accessibleName() == "数字模体"
    assert panel.fbp_view.accessibleName() == "零填充 IFFT"
    assert panel.output_view.accessibleName() == "MoDL 幅值图"
    assert panel.ground_truth_view.accessibleName() == "绝对误差"

    panel.catalog.setCurrentRow(0)
    assert not panel.lodopab_error_view.isHidden()
    assert panel.observation_view.array is None
    assert panel.fbp_view.array is None
    assert panel.output_view.array is None
    assert panel.ground_truth_view.array is None
    assert panel.lodopab_error_view.array is None
    assert panel.result_details.text() == ""
    assert panel.operation_status.property("state") == "ready"
    assert "合成 MRI" not in panel.operation_status.text()


def test_reproducible_export_is_atomic_complete_and_pixel_free(
    bundled_panel: BundledModelsPanel,
    qtbot,
    tmp_path: Path,
) -> None:
    panel = bundled_panel
    panel.run_button.click()
    qtbot.waitUntil(lambda: panel.active_task is None, timeout=3_000)
    destination = tmp_path / "experiment.json"

    panel.export_experiment(destination)
    qtbot.waitUntil(lambda: panel.active_task is None, timeout=3_000)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema"] == "openmedvisionx.bundled-experiment.v1"
    assert payload["application"]["version"]
    assert payload["timestamp_utc"].endswith("Z")
    assert payload["model"]["id"] == "dival-lodopab-fbpunet"
    assert payload["model"]["weight_sha256"] == "a" * 64
    manifest = _record("dival-lodopab-fbpunet").manifest_path.read_bytes()
    assert payload["model"]["manifest_sha256"] == hashlib.sha256(manifest).hexdigest()
    assert payload["case"] == {
        "id": "lodopab-ct-test-03456",
        "artifact_sha256": "b" * 64,
        "fbp_sha256": "c" * 64,
    }
    assert payload["execution"]["requested_device"] == "auto"
    assert payload["execution"]["actual_device"] == "cuda"
    assert payload["metrics"]["mae"] >= 0.0
    assert payload["privacy"] == {
        "contains_dicom_metadata": False,
        "contains_image_pixels": False,
        "contains_original_file_path": False,
    }
    exported_text = destination.read_text(encoding="utf-8")
    assert str(Path(__file__).parents[1]) not in exported_text
    assert "observation.npy" not in exported_text
    assert not list(tmp_path.glob("*.tmp"))
    assert "saved atomically" in panel.operation_status.text()


def test_integrity_and_device_check_marks_each_model_verified(
    bundled_panel: BundledModelsPanel,
    qtbot,
) -> None:
    panel = bundled_panel

    panel.verify_button.click()
    qtbot.waitUntil(lambda: panel.active_task is None, timeout=3_000)

    assert "3/3" in panel.integrity_status.text()
    assert "CUDA available" in panel.device_status.text()
    assert "Test CUDA GPU" in panel.device_status.text()
    assert all("verified" in panel.catalog.item(index).text() for index in range(3))
    assert "Verified: yes" in panel.contract_details.toPlainText()


def test_panel_reflows_and_retranslates_at_narrow_width(
    bundled_panel: BundledModelsPanel,
) -> None:
    panel = bundled_panel
    panel.resize(580, 620)
    panel.show()
    QApplication.processEvents()

    assert panel.main_splitter.orientation() == Qt.Vertical
    assert panel.demo_top_row.orientation() == Qt.Vertical
    assert panel.demo_bottom_row.orientation() == Qt.Vertical

    panel.catalog.setCurrentRow(0)
    panel.set_language("zh_CN")
    assert panel.catalog_title.text() == "随包模型目录"
    assert panel.run_button.text() == "运行 LoDoPaB 案例"
    assert "输入布局与语义" in panel.contract_details.toPlainText()
    assert "不接受临床 HU" in panel.limitation.text()


def test_models_workspace_keeps_bundled_and_external_workflows_separate(qtbot) -> None:
    page = ModelPage(lambda: np.zeros((8, 8), dtype=np.float32))
    qtbot.addWidget(page)

    assert page.model_workspace_tabs.count() == 2
    assert page.model_workspace_tabs.tabText(0) == "Bundled models"
    assert page.model_workspace_tabs.tabText(1) == "External manifests"
    assert page.model_workspace_tabs.currentWidget() is page.bundled_models_scroll
    assert page.manifest_button.parentWidget() is not page.bundled_models

    page.close()


def test_window_models_workspace_fits_1024_by_680_without_horizontal_overflow(qtbot) -> None:
    window = OpenMedVisionXWindow()
    qtbot.addWidget(window)
    window.resize(1024, 680)
    window.show()
    window.tabs.setCurrentWidget(window.models)
    QApplication.processEvents()

    scroll = window.models.bundled_models_scroll
    assert scroll.horizontalScrollBar().maximum() == 0
    assert window.models.size().width() <= window.tabs.contentsRect().width()
    assert window.models.size().height() <= window.tabs.contentsRect().height()

    window.close()
