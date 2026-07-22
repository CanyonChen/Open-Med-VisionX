"""Auditable full-volume inference for the reviewed MONAI BraTS bundle.

The service is deliberately local-only.  A caller supplies a directory for one
run, the case is revalidated and decoded, and no source path, filename, manifest
entry, or input voxel array is retained in the result.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from ..cases.brats import BraTS2021LocalCase, load_brats2021_local_case
from ..errors import ResourceLimitError, ValidationError
from ..evaluation.metrics import compute_segmentation_metrics
from ..models.bundled import load_bundled_model
from ..models.tasks import DeviceRequest

BRATS_INPUT_CHANNELS = ("T1ce", "T1", "T2", "FLAIR")
BRATS_NATIVE_REGIONS = ("TC", "WT", "ET")
BRATS_OUTPUT_REGIONS = ("WT", "TC", "ET")
_NATIVE_TO_OUTPUT = (1, 0, 2)
_GROUND_TRUTH_LABELS = MappingProxyType(
    {
        "WT": frozenset((1, 2, 4)),
        "TC": frozenset((1, 4)),
        "ET": frozenset((4,)),
    }
)
_ESTIMATED_WORKING_BYTES_PER_VOXEL = 96
_ESTIMATED_PATCH_BYTES_PER_VOXEL = 32


class InferenceContext(Protocol):
    """Minimal TaskContext-compatible cancellation and progress surface."""

    def raise_if_cancelled(self) -> None: ...

    def report_progress(
        self,
        fraction: float | None = None,
        *,
        message: str = "",
        current: int | None = None,
        total: int | None = None,
    ) -> None: ...


class PatchInferer(Protocol):
    """Injectable single-patch model call used by fast deterministic tests."""

    def __call__(self, runtime: Any, patch: np.ndarray) -> tuple[np.ndarray, float]: ...


@dataclass(frozen=True, slots=True)
class BraTSSegmentationConfig:
    """Visible preprocessing, inference, threshold, and resource controls."""

    patch_size: tuple[int, int, int] = (96, 96, 96)
    overlap: float = 0.5
    threshold: float = 0.5
    device: DeviceRequest = "auto"
    max_volume_voxels: int = 32_000_000
    max_patch_voxels: int = 2_097_152
    max_patch_count: int = 4096
    max_working_bytes: int = 1_500_000_000

    def __post_init__(self) -> None:
        if len(self.patch_size) != 3 or any(
            isinstance(size, bool) or not isinstance(size, int) for size in self.patch_size
        ):
            raise ValidationError("patch_size must contain exactly three integer dimensions.")
        if any(size < 16 or size % 8 for size in self.patch_size):
            raise ValidationError("Each patch dimension must be at least 16 and divisible by 8.")
        if not np.isfinite(self.overlap) or not 0.0 <= self.overlap < 1.0:
            raise ValidationError("overlap must be finite and in the interval [0, 1).")
        if not np.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValidationError("threshold must be finite and between 0 and 1.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValidationError("device must be 'auto', 'cpu', or 'cuda'.")
        for name in (
            "max_volume_voxels",
            "max_patch_voxels",
            "max_patch_count",
            "max_working_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValidationError(f"{name} must be a positive integer.")
        patch_voxels = math.prod(self.patch_size)
        if patch_voxels > self.max_patch_voxels:
            raise ResourceLimitError(
                "Configured patch_size exceeds the configured patch voxel limit."
            )


@dataclass(frozen=True, slots=True)
class ChannelNormalization:
    """Audit record for one nonzero-voxel z-score transformation."""

    modality: str
    nonzero_voxels: int
    nonzero_mean: float
    nonzero_standard_deviation: float
    background_zero_preserved: bool = True
    formula: str = "nonzero voxels: (x - mean) / standard_deviation; zero voxels remain zero"


@dataclass(frozen=True, slots=True)
class BraTSRegionEvaluation:
    """Task 1 region metrics in physical units where applicable."""

    region: str
    ground_truth_labels: tuple[int, ...]
    dice: float
    hd95_mm: float | None
    prediction_volume_ml: float
    ground_truth_volume_ml: float
    signed_volume_error_ml: float
    absolute_volume_error_ml: float


@dataclass(frozen=True, slots=True, eq=False)
class BraTSSegmentationResult:
    """Immutable source-grid results and explicit scientific provenance."""

    case_alias: str
    probabilities: np.ndarray = dataclass_field(repr=False)
    masks: np.ndarray = dataclass_field(repr=False)
    source_affine: np.ndarray
    canonical_affine: np.ndarray
    source_orientation: tuple[str, str, str]
    input_channels: tuple[str, ...]
    native_output_regions: tuple[str, ...]
    output_regions: tuple[str, ...]
    normalizations: tuple[ChannelNormalization, ...]
    evaluations: Mapping[str, BraTSRegionEvaluation]
    threshold: float
    patch_size: tuple[int, int, int]
    overlap: float
    patch_count: int
    device: str
    model_elapsed_seconds: float
    elapsed_seconds: float
    model_sha256: str
    fallback_reason: str | None
    warnings: tuple[str, ...]
    array_space: str = "source voxel grid"
    input_artifact_sha256: tuple[tuple[str, str], ...] = ()
    ground_truth_artifact_sha256: str | None = None
    requested_device: DeviceRequest = "auto"

    def probability_for(self, region: str) -> np.ndarray:
        """Return one read-only probability volume by WT, TC, or ET name."""

        normalized = str(region).upper()
        try:
            index = self.output_regions.index(normalized)
        except ValueError as exc:
            raise KeyError(f"Unknown BraTS region: {region!r}") from exc
        return self.probabilities[index]

    def mask_for(self, region: str) -> np.ndarray:
        """Return one read-only binary mask by WT, TC, or ET name."""

        normalized = str(region).upper()
        try:
            index = self.output_regions.index(normalized)
        except ValueError as exc:
            raise KeyError(f"Unknown BraTS region: {region!r}") from exc
        return self.masks[index]


def _check_cancelled(context: InferenceContext | None) -> None:
    if context is not None:
        context.raise_if_cancelled()


def _report_progress(
    context: InferenceContext | None,
    fraction: float,
    message: str,
    *,
    current: int | None = None,
    total: int | None = None,
) -> None:
    if context is not None:
        context.report_progress(
            fraction,
            message=message,
            current=current,
            total=total,
        )


def _readonly(value: Any, *, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _normalize_modalities(
    local_case: BraTS2021LocalCase,
) -> tuple[np.ndarray, tuple[ChannelNormalization, ...]]:
    shape = local_case.modality(BRATS_INPUT_CHANNELS[0]).shape
    normalized = np.empty((len(BRATS_INPUT_CHANNELS), *shape), dtype=np.float32)
    records: list[ChannelNormalization] = []
    for index, modality in enumerate(BRATS_INPUT_CHANNELS):
        channel = np.asarray(local_case.modality(modality), dtype=np.float32)
        if channel.shape != shape:
            raise ValidationError("BraTS modalities must have identical canonical geometry.")
        foreground = channel != 0
        count = int(np.count_nonzero(foreground))
        if count == 0:
            raise ValidationError(f"BraTS {modality} has no nonzero voxels.")
        values = channel[foreground]
        mean = float(values.mean(dtype=np.float64))
        deviation = float(values.std(dtype=np.float64))
        if not np.isfinite(mean) or not np.isfinite(deviation):
            raise ValidationError(f"BraTS {modality} normalization statistics are not finite.")
        if deviation <= float(np.finfo(np.float32).eps):
            raise ValidationError(f"BraTS {modality} nonzero voxels have zero variance.")
        target = normalized[index]
        target.fill(0.0)
        target[foreground] = (values - mean) / deviation
        records.append(
            ChannelNormalization(
                modality=modality,
                nonzero_voxels=count,
                nonzero_mean=mean,
                nonzero_standard_deviation=deviation,
            )
        )
    return np.ascontiguousarray(normalized), tuple(records)


def _axis_starts(size: int, patch: int, overlap: float) -> tuple[int, ...]:
    if size <= patch:
        return (0,)
    stride = max(1, int(patch * (1.0 - overlap)))
    last = size - patch
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return tuple(starts)


def _importance_map(patch_size: tuple[int, int, int]) -> np.ndarray:
    axes: list[np.ndarray] = []
    for size in patch_size:
        # Removing the zero-valued endpoints keeps outer source voxels covered.
        axis = np.hanning(size + 2)[1:-1].astype(np.float32)
        axis /= float(axis.max())
        axes.append(np.maximum(axis, np.float32(1e-3)))
    importance = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return np.ascontiguousarray(importance, dtype=np.float32)


def _default_patch_inferer(runtime: Any, patch: np.ndarray) -> tuple[np.ndarray, float]:
    outputs, elapsed = runtime.run(np.ascontiguousarray(patch[None]))
    if len(outputs) != 1:
        raise ValidationError("The reviewed BraTS model must produce exactly one tensor.")
    raw = outputs[0]
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().numpy()
    logits = np.asarray(raw)
    if logits.ndim == 5 and logits.shape[0] == 1:
        logits = logits[0]
    expected_shape = (len(BRATS_NATIVE_REGIONS), *patch.shape[-3:])
    if logits.shape != expected_shape:
        raise ValidationError(
            f"The reviewed BraTS model returned {logits.shape}; expected {expected_shape}."
        )
    if not np.issubdtype(logits.dtype, np.number) or not np.isfinite(logits).all():
        raise ValidationError("The reviewed BraTS model returned non-finite logits.")
    return np.ascontiguousarray(logits, dtype=np.float32), float(elapsed)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    probabilities = np.empty_like(logits, dtype=np.float32)
    positive = logits >= 0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    negative_exponential = np.exp(logits[~positive])
    probabilities[~positive] = negative_exponential / (1.0 + negative_exponential)
    return probabilities


def _to_source_orientation(
    values: np.ndarray,
    transform: np.ndarray,
    *,
    dtype: Any,
) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - local case loading requires it first
        raise RuntimeError("BraTS source-grid mapping requires nibabel.") from exc
    mapped = np.stack(
        [nib.orientations.apply_orientation(channel, transform) for channel in values],
        axis=0,
    )
    return _readonly(mapped, dtype=dtype)


def _evaluate_regions(
    masks: np.ndarray,
    segmentation: np.ndarray,
    canonical_affine: np.ndarray,
    *,
    context: InferenceContext | None,
) -> Mapping[str, BraTSRegionEvaluation]:
    spacing = tuple(float(value) for value in np.linalg.norm(canonical_affine[:3, :3], axis=0))
    voxel_volume_ml = abs(float(np.linalg.det(canonical_affine[:3, :3]))) / 1000.0
    evaluations: dict[str, BraTSRegionEvaluation] = {}
    for index, region in enumerate(BRATS_OUTPUT_REGIONS):
        _check_cancelled(context)
        labels = _GROUND_TRUTH_LABELS[region]
        reference = np.isin(segmentation, tuple(labels))
        prediction = masks[index]
        metrics = compute_segmentation_metrics(reference, prediction, spacing=spacing)
        prediction_volume = metrics.prediction_voxels * voxel_volume_ml
        reference_volume = metrics.reference_voxels * voxel_volume_ml
        signed_error = prediction_volume - reference_volume
        evaluations[region] = BraTSRegionEvaluation(
            region=region,
            ground_truth_labels=tuple(sorted(labels)),
            dice=metrics.dice,
            hd95_mm=metrics.hd95,
            prediction_volume_ml=float(prediction_volume),
            ground_truth_volume_ml=float(reference_volume),
            signed_volume_error_ml=float(signed_error),
            absolute_volume_error_ml=float(abs(signed_error)),
        )
    return MappingProxyType(evaluations)


class BraTSSegmentationService:
    """Cache one runtime per device request and serialize bounded model access."""

    def __init__(
        self,
        *,
        runtime_factory: Callable[[DeviceRequest], Any] | None = None,
        patch_inferer: PatchInferer | None = None,
    ) -> None:
        self._runtime_factory = runtime_factory or self._load_runtime
        self._patch_inferer = patch_inferer or _default_patch_inferer
        self._runtimes: dict[DeviceRequest, Any] = {}
        self._cache_lock = RLock()
        self._inference_lock = RLock()

    @staticmethod
    def _load_runtime(device: DeviceRequest) -> Any:
        return load_bundled_model("monai-brats-segmentation", device=device)

    def _runtime_for(self, device: DeviceRequest) -> Any:
        with self._cache_lock:
            runtime = self._runtimes.get(device)
            if runtime is None:
                runtime = self._runtime_factory(device)
                self._runtimes[device] = runtime
            return runtime

    def segment_directory(
        self,
        directory: str | Path,
        *,
        config: BraTSSegmentationConfig | None = None,
        context: InferenceContext | None = None,
    ) -> BraTSSegmentationResult:
        """Revalidate a selected directory and segment its complete source volume."""

        active = config or BraTSSegmentationConfig()
        started = time.perf_counter()
        _check_cancelled(context)
        _report_progress(context, 0.01, "Revalidating the selected local BraTS case")
        patch_voxels = math.prod(active.patch_size)
        patch_buffer_bytes = patch_voxels * _ESTIMATED_PATCH_BYTES_PER_VOXEL
        if patch_buffer_bytes >= active.max_working_bytes:
            raise ResourceLimitError(
                "Configured patch buffers exceed the configured working-memory limit."
            )
        memory_voxel_limit = max(
            1,
            (active.max_working_bytes - patch_buffer_bytes) // _ESTIMATED_WORKING_BYTES_PER_VOXEL,
        )
        local_case = load_brats2021_local_case(
            directory,
            require_segmentation=False,
            max_voxels=min(active.max_volume_voxels, memory_voxel_limit),
        )
        _check_cancelled(context)
        spacing = np.linalg.norm(local_case.canonical_affine[:3, :3], axis=0)
        if not np.allclose(spacing, (1.0, 1.0, 1.0), rtol=0.0, atol=1e-3):
            raise ValidationError(
                "The reviewed BraTS model requires aligned 1 mm voxels; no hidden resampling "
                "is performed."
            )
        spatial_shape = local_case.modality(BRATS_INPUT_CHANNELS[0]).shape
        volume_voxels = math.prod(spatial_shape)
        if volume_voxels > active.max_volume_voxels:
            raise ResourceLimitError("BraTS volume exceeds the configured voxel limit.")
        padded_shape = tuple(
            max(size, patch) for size, patch in zip(spatial_shape, active.patch_size, strict=True)
        )
        padded_voxels = math.prod(padded_shape)
        # Includes decoded inputs, normalized input, fusion buffers, final arrays,
        # and one model patch.  Framework-owned model memory is reported separately
        # by the bundle and intentionally not guessed here.
        estimated_working_bytes = (
            padded_voxels * _ESTIMATED_WORKING_BYTES_PER_VOXEL + patch_buffer_bytes
        )
        if estimated_working_bytes > active.max_working_bytes:
            raise ResourceLimitError(
                "BraTS inference would exceed the configured working-memory limit "
                f"({estimated_working_bytes} estimated bytes)."
            )

        starts = tuple(
            _axis_starts(size, patch, active.overlap)
            for size, patch in zip(padded_shape, active.patch_size, strict=True)
        )
        patch_count = math.prod(len(axis) for axis in starts)
        if patch_count > active.max_patch_count:
            raise ResourceLimitError(
                f"BraTS inference requires {patch_count} patches, over the configured limit."
            )

        _report_progress(context, 0.12, "Applying auditable nonzero-voxel z-score normalization")
        normalized, normalization_records = _normalize_modalities(local_case)
        _check_cancelled(context)
        padded = np.zeros((len(BRATS_INPUT_CHANNELS), *padded_shape), dtype=np.float32)
        source_slices = tuple(slice(0, size) for size in spatial_shape)
        padded[(slice(None), *source_slices)] = normalized
        del normalized
        importance = _importance_map(active.patch_size)
        logits_sum = np.zeros((len(BRATS_NATIVE_REGIONS), *padded_shape), dtype=np.float32)
        weight_sum = np.zeros(padded_shape, dtype=np.float32)
        model_elapsed = 0.0

        _report_progress(context, 0.18, "Loading the reviewed local MONAI model")
        with self._inference_lock:
            runtime = self._runtime_for(active.device)
            patch_index = 0
            for first in starts[0]:
                for second in starts[1]:
                    for third in starts[2]:
                        _check_cancelled(context)
                        patch_slices = (
                            slice(first, first + active.patch_size[0]),
                            slice(second, second + active.patch_size[1]),
                            slice(third, third + active.patch_size[2]),
                        )
                        patch = np.ascontiguousarray(padded[(slice(None), *patch_slices)])
                        raw_logits, elapsed = self._patch_inferer(runtime, patch)
                        logits = np.asarray(raw_logits)
                        expected = (len(BRATS_NATIVE_REGIONS), *active.patch_size)
                        if logits.shape != expected or not np.issubdtype(logits.dtype, np.number):
                            raise ValidationError(
                                f"BraTS patch inferer returned {logits.shape}; expected {expected} "
                                "with numeric values."
                            )
                        if not np.isfinite(logits).all():
                            raise ValidationError("BraTS patch inferer returned non-finite logits.")
                        elapsed_value = float(elapsed)
                        if not np.isfinite(elapsed_value) or elapsed_value < 0.0:
                            raise ValidationError(
                                "BraTS patch inferer returned an invalid elapsed time."
                            )
                        logits_sum[(slice(None), *patch_slices)] += (
                            np.asarray(logits, dtype=np.float32) * importance
                        )
                        weight_sum[patch_slices] += importance
                        model_elapsed += elapsed_value
                        patch_index += 1
                        _report_progress(
                            context,
                            0.20 + 0.60 * (patch_index / patch_count),
                            "Running full-volume sliding-window inference",
                            current=patch_index,
                            total=patch_count,
                        )
            _check_cancelled(context)

        if np.any(weight_sum <= 0.0):
            raise RuntimeError("Sliding-window coverage left uncovered source voxels.")
        logits_native = logits_sum[(slice(None), *source_slices)] / weight_sum[source_slices]
        del logits_sum, weight_sum, padded
        _report_progress(context, 0.84, "Reordering TC/WT/ET to WT/TC/ET and thresholding")
        probabilities_canonical = _sigmoid(logits_native)[list(_NATIVE_TO_OUTPUT)]
        masks_canonical = np.asarray(probabilities_canonical >= active.threshold, dtype=np.bool_)
        _check_cancelled(context)

        evaluations: Mapping[str, BraTSRegionEvaluation] = MappingProxyType({})
        if local_case.segmentation is not None:
            _report_progress(context, 0.88, "Evaluating Task 1 WT, TC, and ET regions")
            evaluations = _evaluate_regions(
                masks_canonical,
                local_case.segmentation,
                local_case.canonical_affine,
                context=context,
            )
        _check_cancelled(context)
        probabilities = _to_source_orientation(
            probabilities_canonical,
            local_case.canonical_to_source_orientation,
            dtype=np.float32,
        )
        masks = _to_source_orientation(
            masks_canonical,
            local_case.canonical_to_source_orientation,
            dtype=np.bool_,
        )
        source_affine = _readonly(local_case.source_affine, dtype=np.float64)
        canonical_affine = _readonly(local_case.canonical_affine, dtype=np.float64)
        runtime_record = getattr(runtime, "record", None)
        model_sha256 = str(getattr(runtime_record, "artifact_sha256", ""))
        if len(model_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in model_sha256.lower()
        ):
            raise ValidationError("The BraTS runtime did not expose a valid model SHA-256.")
        device = str(getattr(runtime, "device", "unknown"))
        fallback_reason_value = getattr(runtime, "fallback_reason", None)
        fallback_reason = None if fallback_reason_value is None else str(fallback_reason_value)
        warnings = [
            "Domain shift: model training domain is BraTS 2018; the selected case and "
            "evaluation domain are BraTS 2021 Task 1.",
            "Education and research only; not for diagnosis.",
        ]
        if local_case.segmentation is None:
            warnings.append(
                "Task 1 SEG was not present; Dice, HD95, and volume errors were not computed."
            )
        _report_progress(context, 1.0, "BraTS full-volume segmentation complete")
        return BraTSSegmentationResult(
            case_alias=local_case.case_alias,
            probabilities=probabilities,
            masks=masks,
            source_affine=source_affine,
            canonical_affine=canonical_affine,
            source_orientation=local_case.source_orientation,
            input_channels=BRATS_INPUT_CHANNELS,
            native_output_regions=BRATS_NATIVE_REGIONS,
            output_regions=BRATS_OUTPUT_REGIONS,
            normalizations=normalization_records,
            evaluations=evaluations,
            threshold=active.threshold,
            patch_size=active.patch_size,
            overlap=active.overlap,
            patch_count=patch_count,
            device=device,
            model_elapsed_seconds=float(model_elapsed),
            elapsed_seconds=float(time.perf_counter() - started),
            model_sha256=model_sha256.lower(),
            fallback_reason=fallback_reason,
            warnings=tuple(warnings),
            input_artifact_sha256=tuple(
                (modality, local_case.artifact_sha256[modality.upper()])
                for modality in BRATS_INPUT_CHANNELS
            ),
            ground_truth_artifact_sha256=local_case.artifact_sha256.get("SEG"),
            requested_device=active.device,
        )


_DEFAULT_SERVICE = BraTSSegmentationService()


def run_brats2021_segmentation(
    context: InferenceContext,
    directory: str | Path,
    *,
    config: BraTSSegmentationConfig | None = None,
) -> BraTSSegmentationResult:
    """TaskRunner-compatible entry point backed by the process-local model cache."""

    return _DEFAULT_SERVICE.segment_directory(directory, config=config, context=context)


__all__ = [
    "BRATS_INPUT_CHANNELS",
    "BRATS_NATIVE_REGIONS",
    "BRATS_OUTPUT_REGIONS",
    "BraTSRegionEvaluation",
    "BraTSSegmentationConfig",
    "BraTSSegmentationResult",
    "BraTSSegmentationService",
    "ChannelNormalization",
    "run_brats2021_segmentation",
]
