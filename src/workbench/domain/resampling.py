"""Explicit, provenance-preserving resampling for segmentation layers."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..errors import MissingDependencyError, ResourceLimitError, ValidationError
from .studies import (
    InterpolationMode,
    SegmentationLayer,
    SegmentationValueType,
    SpatialGeometry,
)

_XYZ_TO_ZYX = np.asarray(
    (
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
    ),
    dtype=np.float64,
)


def _validate_layer_id(value: str) -> str:
    normalized = str(value).strip()
    if not normalized or len(normalized) > 128:
        raise ValidationError("layer_id must contain 1 to 128 characters.")
    if Path(normalized).is_absolute() or any(
        character in normalized for character in ("/", "\\", "\0", "\n", "\r")
    ):
        raise ValidationError("layer_id must be an opaque identifier, not a path.")
    return normalized


def _output_to_input_zyx(
    source: SpatialGeometry,
    target: SpatialGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the SciPy output-index to input-index transform in ZYX order."""

    target_xyz_to_source_xyz = np.linalg.inv(source.affine_ras) @ target.affine_ras
    matrix = _XYZ_TO_ZYX @ target_xyz_to_source_xyz[:3, :3] @ _XYZ_TO_ZYX
    offset = _XYZ_TO_ZYX @ target_xyz_to_source_xyz[:3, 3]
    return matrix, offset


def resample_segmentation_layer(
    layer: SegmentationLayer,
    target_geometry: SpatialGeometry,
    *,
    layer_id: str,
    name: str | None = None,
    interpolation: InterpolationMode,
    user_confirmed: bool,
    max_output_voxels: int = 128_000_000,
) -> SegmentationLayer:
    """Create a resampled derivative while keeping the imported layer unchanged.

    The operation is deliberately impossible without explicit confirmation.  It
    uses nearest-neighbour interpolation for discrete labels and a continuous
    interpolation for fractional/probability layers, matching the public layer
    contract.  Fractional DICOM SEG values are rounded back onto their declared
    native integer scale after interpolation; they are never thresholded.
    """

    if not isinstance(layer, SegmentationLayer):
        raise ValidationError("Only SegmentationLayer values can be resampled.")
    if not isinstance(target_geometry, SpatialGeometry):
        raise ValidationError("target_geometry must be SpatialGeometry.")
    normalized_id = _validate_layer_id(layer_id)
    mode = InterpolationMode(interpolation)
    if not user_confirmed:
        raise ValidationError("Segmentation resampling requires explicit user confirmation.")
    if layer.geometry.matches(target_geometry):
        raise ValidationError("The segmentation already matches the target geometry.")

    discrete = layer.value_type in {
        SegmentationValueType.BINARY,
        SegmentationValueType.DISCRETE,
    }
    if discrete and mode is not InterpolationMode.NEAREST:
        raise ValidationError("Discrete labels require nearest-neighbour interpolation.")
    if not discrete and mode is InterpolationMode.NEAREST:
        raise ValidationError("Fractional and probability layers require continuous interpolation.")
    order = {
        InterpolationMode.NEAREST: 0,
        InterpolationMode.LINEAR: 1,
        InterpolationMode.BSPLINE: 3,
    }[mode]
    output_voxels = math.prod(target_geometry.shape_zyx)
    if output_voxels > int(max_output_voxels):
        raise ResourceLimitError(
            "The requested segmentation grid exceeds the configured voxel limit."
        )

    try:
        from scipy.ndimage import affine_transform
    except ImportError as exc:  # pragma: no cover - required base dependency
        raise MissingDependencyError(
            "Segmentation resampling requires SciPy from the base environment."
        ) from exc

    source_array = np.asarray(layer.array)
    if source_array.ndim == 2:
        source_array = source_array[np.newaxis, ...]
    matrix, offset = _output_to_input_zyx(layer.geometry, target_geometry)
    working = np.asarray(source_array, dtype=np.float64)
    resampled = affine_transform(
        working,
        matrix=matrix,
        offset=offset,
        output_shape=target_geometry.shape_zyx,
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=order > 1,
    )

    if discrete:
        result = np.asarray(np.rint(resampled), dtype=source_array.dtype)
        original_values = set(int(value) for value in np.unique(source_array))
        output_values = set(int(value) for value in np.unique(result))
        if not output_values.issubset(original_values | {0}):
            raise ValidationError("Nearest-neighbour resampling introduced an unknown label.")
    elif layer.value_type is SegmentationValueType.FRACTIONAL:
        assert layer.maximum_fractional_value is not None
        result = np.asarray(
            np.rint(np.clip(resampled, 0.0, layer.maximum_fractional_value)),
            dtype=source_array.dtype,
        )
    else:
        result = np.asarray(np.clip(resampled, 0.0, 1.0), dtype=np.float32)

    return layer.derive_resampled(
        layer_id=normalized_id,
        name=f"{layer.name} (resampled)" if name is None else name,
        array=result,
        target_geometry=target_geometry,
        interpolation=mode,
        user_confirmed=True,
        parameters={
            "coordinate_convention": "RAS+",
            "output_shape_zyx": target_geometry.shape_zyx,
            "outside_value": 0,
            "fractional_native_scale_preserved": (
                layer.value_type is SegmentationValueType.FRACTIONAL
            ),
        },
    )


__all__ = ["resample_segmentation_layer"]
