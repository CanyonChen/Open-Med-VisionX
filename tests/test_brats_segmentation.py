from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from workbench.cases.brats import BraTS2021CaseLoadError
from workbench.errors import OperationCancelled, ResourceLimitError
from workbench.services.brats_segmentation import (
    BRATS_INPUT_CHANNELS,
    BRATS_NATIVE_REGIONS,
    BRATS_OUTPUT_REGIONS,
    BraTSSegmentationConfig,
    BraTSSegmentationService,
)

nib = pytest.importorskip("nibabel")


class _Runtime:
    def __init__(self, device: str) -> None:
        self.device = "cuda" if device == "auto" else device
        self.fallback_reason = None
        self.record = SimpleNamespace(artifact_sha256="a" * 64)


class _RecordingContext:
    def __init__(self, *, cancel_after: int | None = None) -> None:
        self.cancel_after = cancel_after
        self.check_count = 0
        self.progress: list[tuple[float, str, int | None, int | None]] = []

    def raise_if_cancelled(self) -> None:
        self.check_count += 1
        if self.cancel_after is not None and self.check_count >= self.cancel_after:
            raise OperationCancelled("cancelled by test")

    def report_progress(
        self,
        fraction: float | None = None,
        *,
        message: str = "",
        current: int | None = None,
        total: int | None = None,
    ) -> None:
        assert fraction is not None
        self.progress.append((fraction, message, current, total))


def _write_volume(path: Path, values: np.ndarray, affine: np.ndarray) -> None:
    image = nib.Nifti1Image(values, affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, path)


def _make_case(
    root: Path,
    *,
    shape: tuple[int, int, int] = (16, 16, 16),
    affine: np.ndarray | None = None,
    include_segmentation: bool = True,
) -> tuple[Path, dict[str, np.ndarray], np.ndarray | None]:
    root.mkdir()
    active_affine = np.eye(4, dtype=np.float64) if affine is None else affine
    first, second, third = np.indices(shape, dtype=np.float32)
    modalities = {
        "T1ce": 1.0 + first + 0.25 * second,
        "T1": 1.0 + second + 0.125 * third,
        "T2": 1.0 + third + 0.5 * first,
        "FLAIR": 1.0 + first + 2.0 * second + 3.0 * third,
    }
    for modality, values in modalities.items():
        _write_volume(root / f"case_{modality.casefold()}.nii.gz", values, active_affine)
    segmentation: np.ndarray | None = None
    if include_segmentation:
        segmentation = np.zeros(shape, dtype=np.int16)
        segmentation[1:5, 2:6, 3:7] = 2
        segmentation[4:9, 5:10, 6:11] = 1
        segmentation[7:10, 8:11, 9:12] = 4
        _write_volume(root / "case_seg.nii.gz", segmentation, active_affine)
    return root, modalities, segmentation


def _zscore_nonzero(values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float32)
    selected = values != 0
    foreground = values[selected]
    result[selected] = (foreground - foreground.mean(dtype=np.float64)) / foreground.std(
        dtype=np.float64
    )
    return result


def test_channel_order_ras_preprocessing_region_mapping_and_task1_metrics(
    tmp_path: Path,
) -> None:
    source_affine = np.asarray(
        [
            [-1.0, 0.0, 0.0, 15.0],
            [0.0, -1.0, 0.0, 15.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    case, modalities, segmentation = _make_case(
        tmp_path / "case",
        affine=source_affine,
    )
    assert segmentation is not None
    canonical_segmentation = np.flip(segmentation, axis=(0, 1))
    calls: list[np.ndarray] = []

    def infer_patch(_runtime: Any, patch: np.ndarray) -> tuple[np.ndarray, float]:
        calls.append(np.array(patch, copy=True))
        whole_tumour = np.isin(canonical_segmentation, (1, 2, 4))
        tumour_core = np.isin(canonical_segmentation, (1, 4))
        enhancing_tumour = canonical_segmentation == 4
        native = np.stack((tumour_core, whole_tumour, enhancing_tumour))
        return np.where(native, 20.0, -20.0).astype(np.float32), 0.01

    factory_calls: list[str] = []

    def runtime_factory(device: str) -> _Runtime:
        factory_calls.append(device)
        return _Runtime(device)

    service = BraTSSegmentationService(
        runtime_factory=runtime_factory,
        patch_inferer=infer_patch,
    )
    context = _RecordingContext()
    result = service.segment_directory(
        case,
        config=BraTSSegmentationConfig(patch_size=(16, 16, 16)),
        context=context,
    )

    assert result.input_channels == BRATS_INPUT_CHANNELS == ("T1ce", "T1", "T2", "FLAIR")
    assert result.native_output_regions == BRATS_NATIVE_REGIONS == ("TC", "WT", "ET")
    assert result.output_regions == BRATS_OUTPUT_REGIONS == ("WT", "TC", "ET")
    assert len(calls) == 1
    for index, modality in enumerate(BRATS_INPUT_CHANNELS):
        expected = _zscore_nonzero(np.flip(modalities[modality], axis=(0, 1)))
        np.testing.assert_allclose(calls[0][index], expected, atol=1e-6)
    assert tuple(record.modality for record in result.normalizations) == BRATS_INPUT_CHANNELS
    assert all(record.background_zero_preserved for record in result.normalizations)

    np.testing.assert_array_equal(result.mask_for("WT"), np.isin(segmentation, (1, 2, 4)))
    np.testing.assert_array_equal(result.mask_for("TC"), np.isin(segmentation, (1, 4)))
    np.testing.assert_array_equal(result.mask_for("ET"), segmentation == 4)
    assert set(result.evaluations) == {"WT", "TC", "ET"}
    assert all(metric.dice == pytest.approx(1.0) for metric in result.evaluations.values())
    assert all(metric.hd95_mm == pytest.approx(0.0) for metric in result.evaluations.values())
    assert all(
        metric.absolute_volume_error_ml == pytest.approx(0.0)
        for metric in result.evaluations.values()
    )
    assert result.evaluations["WT"].ground_truth_labels == (1, 2, 4)
    assert result.evaluations["TC"].ground_truth_labels == (1, 4)
    assert result.evaluations["ET"].ground_truth_labels == (4,)
    np.testing.assert_array_equal(result.source_affine, source_affine)
    assert result.source_orientation == ("L", "P", "S")
    assert result.device == "cuda"
    assert result.model_sha256 == "a" * 64
    assert result.requested_device == "auto"
    input_hashes = dict(result.input_artifact_sha256)
    assert tuple(input_hashes) == BRATS_INPUT_CHANNELS
    for modality in BRATS_INPUT_CHANNELS:
        source = case / f"case_{modality.casefold()}.nii.gz"
        assert input_hashes[modality] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert (
        result.ground_truth_artifact_sha256
        == hashlib.sha256((case / "case_seg.nii.gz").read_bytes()).hexdigest()
    )
    assert "BraTS 2018" in result.warnings[0] and "BraTS 2021" in result.warnings[0]
    assert factory_calls == ["auto"]
    assert context.progress[-1][0] == 1.0
    assert not result.probabilities.flags.writeable
    assert not result.masks.flags.writeable
    assert not result.source_affine.flags.writeable


def test_sliding_window_covers_edges_blends_overlap_and_reuses_runtime(tmp_path: Path) -> None:
    case, _, _ = _make_case(
        tmp_path / "case",
        shape=(20, 22, 24),
        include_segmentation=False,
    )
    infer_shapes: list[tuple[int, ...]] = []
    factory_calls: list[str] = []

    def infer_patch(_runtime: Any, patch: np.ndarray) -> tuple[np.ndarray, float]:
        infer_shapes.append(patch.shape)
        return np.full((3, 16, 16, 16), 2.0, dtype=np.float32), 0.001

    def runtime_factory(device: str) -> _Runtime:
        factory_calls.append(device)
        return _Runtime(device)

    service = BraTSSegmentationService(
        runtime_factory=runtime_factory,
        patch_inferer=infer_patch,
    )
    config = BraTSSegmentationConfig(
        patch_size=(16, 16, 16),
        overlap=0.5,
        threshold=0.75,
    )
    first = service.segment_directory(case, config=config)
    second = service.segment_directory(case, config=config)

    assert first.patch_count == 8
    assert len(infer_shapes) == 16
    assert set(infer_shapes) == {(4, 16, 16, 16)}
    assert factory_calls == ["auto"]
    assert first.masks.shape == (3, 20, 22, 24)
    assert np.all(first.masks)
    assert np.all(first.masks[:, -1, -1, -1])
    np.testing.assert_allclose(first.probabilities, second.probabilities)
    assert not first.evaluations
    assert any("SEG was not present" in warning for warning in first.warnings)


def test_default_inferer_runs_the_reviewed_torchscript_on_cpu(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    case, _, _ = _make_case(tmp_path / "case", include_segmentation=False)

    result = BraTSSegmentationService().segment_directory(
        case,
        config=BraTSSegmentationConfig(
            patch_size=(16, 16, 16),
            device="cpu",
        ),
    )

    assert result.device == "cpu"
    assert result.probabilities.shape == (3, 16, 16, 16)
    assert np.isfinite(result.probabilities).all()
    assert result.model_elapsed_seconds > 0.0
    assert result.model_sha256 == "729980a0bd9347bf2397701eb329e12517918dc282a2d09c40458e95b24ceed9"


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        ("geometry", "affine_mismatch"),
        ("labels", "seg_invalid_labels"),
    ],
)
def test_invalid_geometry_and_task1_labels_stop_before_model(
    tmp_path: Path,
    failure: str,
    expected_code: str,
) -> None:
    case, _, segmentation = _make_case(tmp_path / "case")
    if failure == "geometry":
        translated = np.eye(4)
        translated[0, 3] = 2.0
        values = np.ones((16, 16, 16), dtype=np.float32)
        _write_volume(case / "case_t2.nii.gz", values, translated)
    else:
        assert segmentation is not None
        segmentation[0, 0, 0] = 3
        _write_volume(case / "case_seg.nii.gz", segmentation, np.eye(4))

    model_calls = 0

    def runtime_factory(device: str) -> _Runtime:
        nonlocal model_calls
        model_calls += 1
        return _Runtime(device)

    service = BraTSSegmentationService(runtime_factory=runtime_factory)
    with pytest.raises(BraTS2021CaseLoadError, match=expected_code):
        service.segment_directory(
            case,
            config=BraTSSegmentationConfig(patch_size=(16, 16, 16)),
        )
    assert model_calls == 0


def test_cancellation_is_checked_between_sliding_window_patches(tmp_path: Path) -> None:
    case, _, _ = _make_case(
        tmp_path / "case",
        shape=(20, 22, 24),
        include_segmentation=False,
    )
    infer_calls = 0

    def infer_patch(_runtime: Any, _patch: np.ndarray) -> tuple[np.ndarray, float]:
        nonlocal infer_calls
        infer_calls += 1
        return np.zeros((3, 16, 16, 16), dtype=np.float32), 0.001

    service = BraTSSegmentationService(
        runtime_factory=_Runtime,
        patch_inferer=infer_patch,
    )
    context = _RecordingContext(cancel_after=5)

    with pytest.raises(OperationCancelled):
        service.segment_directory(
            case,
            config=BraTSSegmentationConfig(patch_size=(16, 16, 16)),
            context=context,
        )

    assert infer_calls == 1
    assert context.progress
    assert context.progress[-1][0] < 1.0


def test_patch_count_resource_limit_stops_before_model_loading(tmp_path: Path) -> None:
    case, _, _ = _make_case(
        tmp_path / "case",
        shape=(20, 22, 24),
        include_segmentation=False,
    )
    model_calls = 0

    def runtime_factory(device: str) -> _Runtime:
        nonlocal model_calls
        model_calls += 1
        return _Runtime(device)

    service = BraTSSegmentationService(runtime_factory=runtime_factory)
    with pytest.raises(ResourceLimitError, match="patches"):
        service.segment_directory(
            case,
            config=BraTSSegmentationConfig(
                patch_size=(16, 16, 16),
                max_patch_count=1,
            ),
        )
    assert model_calls == 0
