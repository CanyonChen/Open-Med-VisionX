"""Typed, pixel-free experiment records for local BraTS segmentation runs."""

from __future__ import annotations

import hashlib
import re
import struct
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import __version__
from ..errors import ValidationError
from .records import ArtifactReference, ExperimentRecord, TransformStep

if TYPE_CHECKING:
    from ..services.brats_segmentation import BraTSSegmentationResult


_INPUT_CHANNELS = ("T1ce", "T1", "T2", "FLAIR")
_NATIVE_REGIONS = ("TC", "WT", "ET")
_OUTPUT_REGIONS = ("WT", "TC", "ET")
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_DOMAIN_SHIFT_WARNING = (
    "Domain shift: model training domain is BraTS 2018; the selected case and "
    "evaluation domain are BraTS 2021 Task 1."
)
_EDUCATION_WARNING = "Education and research only; not for diagnosis."
_NO_REFERENCE_WARNING = (
    "Task 1 SEG was not present; Dice, HD95, and volume errors were not computed."
)
_ALLOWED_WARNINGS = frozenset({_DOMAIN_SHIFT_WARNING, _EDUCATION_WARNING, _NO_REFERENCE_WARNING})
_CASE_ALIAS = re.compile(r"^brats-2021-[0-9a-f]{16}$")
_REGION_LABELS: dict[str, tuple[int, ...]] = {
    "WT": (1, 2, 4),
    "TC": (1, 4),
    "ET": (4,),
}
_REGION_MEANINGS = {
    "WT": "whole_tumor",
    "TC": "tumor_core",
    "ET": "enhancing_tumor",
}


def _require_sha256(value: object, *, field_name: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in _SHA256_CHARACTERS for character in digest):
        raise ValidationError(f"{field_name} must be a lowercase SHA-256 digest.")
    return digest


def _output_digest(region: str, mask: np.ndarray) -> str:
    """Hash a versioned, deterministic uint8 C-order binary-mask representation."""

    values = np.asarray(mask)
    if values.ndim != 3:
        raise ValidationError(f"BraTS {region} output must be a 3-D mask.")
    if values.dtype != np.bool_ and not np.isin(values, (0, 1)).all():
        raise ValidationError(f"BraTS {region} output must be binary.")
    encoded = np.ascontiguousarray(values, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(b"openmedvisionx-binary-mask-v1\0")
    digest.update(struct.pack("<3Q", *encoded.shape))
    digest.update(encoded.tobytes(order="C"))
    return digest.hexdigest()


def _fallback_code(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).casefold()
    if "cuda is unavailable" in normalized:
        return "cuda_unavailable"
    if "cuda load failed" in normalized:
        return "cuda_load_failed"
    if "cuda execution failed" in normalized:
        return "cuda_execution_failed"
    return "runtime_fallback"


def _input_hashes(result: BraTSSegmentationResult) -> dict[str, str]:
    try:
        pairs = tuple(result.input_artifact_sha256)
    except (AttributeError, TypeError) as exc:
        raise ValidationError("BraTS result does not contain input artifact hashes.") from exc
    normalized: dict[str, str] = {}
    for pair in pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValidationError("BraTS input hashes must contain modality/digest pairs.")
        modality, digest = pair
        name = str(modality)
        if name not in _INPUT_CHANNELS or name in normalized:
            raise ValidationError("BraTS input hashes contain an unknown or duplicate modality.")
        normalized[name] = _require_sha256(digest, field_name=f"{name} input hash")
    if tuple(name for name in _INPUT_CHANNELS if name in normalized) != _INPUT_CHANNELS:
        raise ValidationError("BraTS result must retain hashes for T1ce, T1, T2, and FLAIR.")
    return normalized


def _metric_payload(result: BraTSSegmentationResult) -> dict[str, dict[str, Any]]:
    evaluations = dict(result.evaluations)
    if not evaluations:
        return {}
    if set(evaluations) != set(_OUTPUT_REGIONS):
        raise ValidationError("BraTS evaluation must contain WT, TC, and ET together.")
    payload: dict[str, dict[str, Any]] = {}
    for region in _OUTPUT_REGIONS:
        item = evaluations[region]
        if tuple(item.ground_truth_labels) != _REGION_LABELS[region]:
            raise ValidationError(f"BraTS {region} ground-truth label semantics are invalid.")
        payload[region] = {
            "ground_truth_labels": tuple(item.ground_truth_labels),
            "dice": item.dice,
            "hd95_mm": item.hd95_mm,
            "prediction_volume_ml": item.prediction_volume_ml,
            "ground_truth_volume_ml": item.ground_truth_volume_ml,
            "signed_volume_error_ml": item.signed_volume_error_ml,
            "absolute_volume_error_ml": item.absolute_volume_error_ml,
        }
    return payload


def _normalization_steps(result: BraTSSegmentationResult) -> tuple[TransformStep, ...]:
    source_records = tuple(result.normalizations)
    records = {item.modality: item for item in source_records}
    if len(source_records) != 4 or tuple(records) != _INPUT_CHANNELS:
        raise ValidationError("BraTS result must contain one normalization record per modality.")
    steps = []
    for modality in _INPUT_CHANNELS:
        item = records[modality]
        if (
            isinstance(item.nonzero_voxels, bool)
            or not isinstance(item.nonzero_voxels, (int, np.integer))
            or int(item.nonzero_voxels) <= 0
            or not np.isfinite(item.nonzero_mean)
            or not np.isfinite(item.nonzero_standard_deviation)
            or item.nonzero_standard_deviation <= 0.0
            or item.background_zero_preserved is not True
        ):
            raise ValidationError(f"BraTS {modality} normalization provenance is invalid.")
        steps.append(
            TransformStep(
                name=f"nonzero-zscore-{modality.casefold()}",
                version="1",
                parameters={
                    "modality": modality,
                    "nonzero_voxels": item.nonzero_voxels,
                    "mean": item.nonzero_mean,
                    "standard_deviation": item.nonzero_standard_deviation,
                    "background_zero_preserved": item.background_zero_preserved,
                    "formula": item.formula,
                },
            )
        )
    steps.extend(
        (
            TransformStep(
                name="sliding-window-inference",
                version="1",
                parameters={
                    "patch_size": tuple(result.patch_size),
                    "overlap": result.overlap,
                    "patch_count": result.patch_count,
                },
            ),
            TransformStep(
                name="region-reorder-and-threshold",
                version="1",
                parameters={
                    "native_order": tuple(result.native_output_regions),
                    "recorded_order": tuple(result.output_regions),
                    "sigmoid_threshold": result.threshold,
                },
            ),
            TransformStep(
                name="restore-source-grid",
                version="1",
                parameters={
                    "source_orientation": tuple(result.source_orientation),
                    "array_space": result.array_space,
                },
            ),
        )
    )
    return tuple(steps)


def create_brats_experiment_record(
    result: BraTSSegmentationResult,
    *,
    record_id: str | None = None,
    created_at: datetime | None = None,
    application_version: str = __version__,
    code_revision: str = "unrecorded",
    dataset_manifest_id: str | None = None,
) -> ExperimentRecord:
    """Create an immutable BraTS record containing hashes and metadata, never pixels.

    The four model inputs are referenced by the hashes produced during the same
    revalidation/load boundary used for inference. Output hashes cover a
    versioned uint8 C-order representation of each binary source-grid mask.
    """

    input_hashes = _input_hashes(result)
    if not _CASE_ALIAS.fullmatch(str(result.case_alias)):
        raise ValidationError("BraTS case_alias must be the anonymous content-derived alias.")
    if tuple(result.input_channels) != _INPUT_CHANNELS:
        raise ValidationError("BraTS input channel order must be T1ce, T1, T2, FLAIR.")
    if tuple(result.native_output_regions) != _NATIVE_REGIONS:
        raise ValidationError("BraTS native output order must be TC, WT, ET.")
    if tuple(result.output_regions) != _OUTPUT_REGIONS:
        raise ValidationError("BraTS recorded output order must be WT, TC, ET.")
    if result.array_space != "source voxel grid":
        raise ValidationError("BraTS result arrays must be recorded on the source voxel grid.")
    orientation = tuple(result.source_orientation)
    if (
        len(orientation) != 3
        or any(axis not in {"L", "R", "P", "A", "I", "S"} for axis in orientation)
        or len(
            {
                "x" if axis in {"L", "R"} else "y" if axis in {"P", "A"} else "z"
                for axis in orientation
            }
        )
        != 3
    ):
        raise ValidationError("BraTS source orientation is invalid.")

    masks = np.asarray(result.masks)
    if masks.ndim != 4 or masks.shape[0] != len(_OUTPUT_REGIONS):
        raise ValidationError("BraTS result must contain three 3-D masks in WT, TC, ET order.")
    spatial_shape = tuple(int(value) for value in masks.shape[1:])
    affine = np.asarray(result.source_affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValidationError("BraTS result must contain a finite 4x4 source affine.")
    model_hash = _require_sha256(result.model_sha256, field_name="MONAI model hash")
    metrics = _metric_payload(result)

    inputs = [
        ArtifactReference(
            artifact_id=f"input-{modality.casefold()}",
            sha256=input_hashes[modality],
            kind="mri-volume",
            media_type="application/x-nifti",
            shape=spatial_shape,
            coordinate_system="RAS+",
        )
        for modality in _INPUT_CHANNELS
    ]
    ground_truth_hash = getattr(result, "ground_truth_artifact_sha256", None)
    if metrics:
        if ground_truth_hash is None:
            raise ValidationError("Evaluated BraTS results must retain the SEG artifact hash.")
        inputs.append(
            ArtifactReference(
                artifact_id="evaluation-reference-seg",
                sha256=_require_sha256(
                    ground_truth_hash,
                    field_name="BraTS SEG ground-truth hash",
                ),
                kind="segmentation-reference",
                media_type="application/x-nifti",
                shape=spatial_shape,
                coordinate_system="RAS+",
                label_schema={
                    "labels": {
                        "0": "background",
                        "1": "necrotic_core",
                        "2": "edema",
                        "4": "enhancing_tumor",
                    },
                    "evaluation_regions": {
                        region: labels for region, labels in _REGION_LABELS.items()
                    },
                },
            )
        )

    outputs = tuple(
        ArtifactReference(
            artifact_id=f"output-{region.casefold()}-mask",
            sha256=_output_digest(region, masks[index]),
            kind="segmentation-mask",
            media_type="application/vnd.openmedvisionx.binary-mask",
            shape=spatial_shape,
            coordinate_system="RAS+",
            label_schema={
                "region": region,
                "meaning": _REGION_MEANINGS[region],
                "labels": {"0": "background", "1": _REGION_MEANINGS[region]},
                "ground_truth_labels": _REGION_LABELS[region],
            },
        )
        for index, region in enumerate(_OUTPUT_REGIONS)
    )

    timestamp = datetime.now(timezone.utc) if created_at is None else created_at
    if record_id is None:
        identifier_seed = f"{result.case_alias}|{model_hash}|{timestamp.isoformat()}|" + "|".join(
            item.sha256 for item in outputs
        )
        record_id = f"brats-{hashlib.sha256(identifier_seed.encode('utf-8')).hexdigest()[:20]}"
    warnings = tuple(result.warnings)
    if any(warning not in _ALLOWED_WARNINGS for warning in warnings):
        raise ValidationError("BraTS result contains an unrecognized warning value.")
    if not any("BraTS 2018" in warning and "BraTS 2021" in warning for warning in warnings):
        warnings = (*warnings, _DOMAIN_SHIFT_WARNING)

    requested_device = str(getattr(result, "requested_device", "auto"))
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValidationError("BraTS requested device must be auto, cpu, or cuda.")
    if result.device not in {"cpu", "cuda"}:
        raise ValidationError("BraTS execution device must be cpu or cuda.")
    fallback_code = _fallback_code(result.fallback_reason)
    return ExperimentRecord(
        record_id=record_id,
        created_at=timestamp,
        application_version=application_version,
        code_revision=code_revision,
        dataset_manifest_id=dataset_manifest_id,
        task="brats-2021-task-1-segmentation",
        model_id="monai-brats-segmentation",
        model_version="bundle-1",
        inputs=tuple(inputs),
        transforms=_normalization_steps(result),
        outputs=outputs,
        metrics=metrics,
        parameters={
            "case_alias": result.case_alias,
            "input_channel_order": _INPUT_CHANNELS,
            "native_output_region_order": _NATIVE_REGIONS,
            "output_region_order": _OUTPUT_REGIONS,
            "model_sha256": model_hash,
            "model_training_domain": "BraTS 2018",
            "evaluation_domain": "BraTS 2021 Task 1",
            "patch_size": tuple(result.patch_size),
            "overlap": result.overlap,
            "threshold": result.threshold,
            "patch_count": result.patch_count,
            "requested_device": requested_device,
            "execution_device": result.device,
            "fallback_applied": fallback_code is not None,
            "fallback_reason": fallback_code,
            "model_elapsed_seconds": result.model_elapsed_seconds,
            "source_orientation": tuple(result.source_orientation),
            "source_affine_ras_mm": tuple(tuple(float(value) for value in row) for row in affine),
            "array_space": result.array_space,
            "output_hash_encoding": "openmedvisionx-binary-mask-v1",
        },
        warnings=warnings,
        duration_ms=float(result.elapsed_seconds) * 1000.0,
    )


__all__ = ["create_brats_experiment_record"]
