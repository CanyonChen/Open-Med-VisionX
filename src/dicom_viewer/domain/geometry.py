"""Physical-space helpers for RAS+ medical volumes."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np

from ..errors import MissingDependencyError, ResourceLimitError, ValidationError
from .images import ImageVolume

_ZYX_TO_XYZ = np.array(
    [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0, 0, 0, 1]],
    dtype=np.float64,
)


def resample_to_ras_grid(
    volume: ImageVolume,
    *,
    target_spacing: Sequence[float] | None = None,
    interpolation_order: int = 1,
    max_output_voxels: int = 256_000_000,
) -> ImageVolume:
    """Resample an oblique volume onto axis-aligned RAS+ x/y/z axes.

    ``ImageVolume.affine`` maps input voxel coordinates ``(x, y, z)`` to RAS+
    millimetres, while both input and returned arrays are stored as
    ``(z, y, x)``. ``target_spacing`` is therefore ordered ``(x, y, z)`` and
    the automatically constructed output grid covers every input voxel centre.
    Interpolation order 0 is nearest-neighbour (labels), 1 is linear
    (continuous images), and 3 is cubic. This operation is intended for
    orthogonal viewing and measurement and must run in a background task for
    large inputs.
    """

    try:
        from scipy.ndimage import affine_transform
    except ImportError as exc:  # pragma: no cover - depends on install
        raise MissingDependencyError("Physical volume resampling requires SciPy.") from exc
    if interpolation_order not in {0, 1, 3}:
        raise ValidationError("interpolation_order must be nearest (0), linear (1), or cubic (3).")
    if target_spacing is None:
        isotropic = min(volume.spacing)
        spacing_xyz = (isotropic, isotropic, isotropic)
    else:
        spacing_xyz = tuple(float(item) for item in target_spacing)
        if len(spacing_xyz) != 3 or any(item <= 0 for item in spacing_xyz):
            raise ValidationError("target_spacing must contain three positive x/y/z values.")

    max_z, max_y, max_x = (dimension - 1 for dimension in volume.shape)
    corners_zyx = np.asarray(
        list(itertools.product((0.0, float(max_z)), (0.0, float(max_y)), (0.0, float(max_x))))
    )
    corners_xyz = corners_zyx[:, [2, 1, 0]]
    corners_world = volume.voxel_xyz_to_world_ras(corners_xyz)
    world_min = corners_world.min(axis=0)
    world_max = corners_world.max(axis=0)
    output_xyz = np.ceil((world_max - world_min) / np.asarray(spacing_xyz)).astype(int) + 1
    output_shape = (int(output_xyz[2]), int(output_xyz[1]), int(output_xyz[0]))
    output_voxels = int(np.prod(output_shape, dtype=np.int64))
    if output_voxels > max_output_voxels:
        raise ResourceLimitError(
            f"RAS+ resampling would create {output_voxels:,} voxels, above the safety limit."
        )

    output_affine = np.eye(4, dtype=np.float64)
    output_affine[:3, :3] = np.diag(spacing_xyz)
    output_affine[:3, 3] = world_min
    input_zyx_to_world = volume.affine @ _ZYX_TO_XYZ
    output_zyx_to_world = output_affine @ _ZYX_TO_XYZ
    output_to_input = np.linalg.inv(input_zyx_to_world) @ output_zyx_to_world
    resampled = affine_transform(
        volume.array,
        matrix=output_to_input[:3, :3],
        offset=output_to_input[:3, 3],
        output_shape=output_shape,
        order=interpolation_order,
        mode="constant",
        cval=float(np.min(volume.array)),
        prefilter=interpolation_order > 1,
    )
    metadata = dict(volume.runtime_metadata)
    metadata.update(
        {
            "resampled_to_axis_aligned_ras": True,
            "resampling_order": interpolation_order,
            "source_shape": volume.shape,
        }
    )
    return ImageVolume(
        array=resampled,
        source_type=volume.source_type,
        intensity_semantics=volume.intensity_semantics,
        runtime_metadata=metadata,
        affine=output_affine,
        spacing=spacing_xyz,
        origin=tuple(float(item) for item in world_min),
        direction=np.eye(3),
        modality=volume.modality,
    )
