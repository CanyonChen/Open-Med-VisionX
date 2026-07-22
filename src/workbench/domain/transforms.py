"""Reversible mappings between original and canonical pixel coordinates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from ..errors import ValidationError


@dataclass(frozen=True, slots=True)
class TransformOperation:
    """One auditable step in a preprocessing coordinate transform."""

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("A transform operation must have a name.")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


def _shape_2d(value: Sequence[int], label: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValidationError(f"{label} must be (height, width), got {value!r}.")
    height, width = (int(value[0]), int(value[1]))
    if height <= 0 or width <= 0:
        raise ValidationError(f"{label} dimensions must be positive, got {value!r}.")
    return height, width


@dataclass(frozen=True, slots=True)
class TransformRecord:
    """A reversible 2-D homogeneous transform.

    Coordinates are expressed as ``(x, y)`` pixel coordinates. ``matrix`` maps
    coordinates from ``original_shape`` to ``output_shape``.  Keeping this
    record separate from decoded pixels lets masks, boxes, keypoints, and heat
    maps be mapped back without guessing how preprocessing was performed.
    """

    matrix: np.ndarray
    original_shape: tuple[int, int]
    output_shape: tuple[int, int]
    operations: tuple[TransformOperation, ...] = ()

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValidationError(f"Transform matrix must be 3x3, got {matrix.shape}.")
        if not np.all(np.isfinite(matrix)):
            raise ValidationError("Transform matrix contains NaN or infinity.")
        determinant = float(np.linalg.det(matrix))
        if np.isclose(determinant, 0.0):
            raise ValidationError("Transform matrix must be invertible.")
        matrix = matrix.copy()
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "original_shape", _shape_2d(self.original_shape, "original_shape"))
        object.__setattr__(self, "output_shape", _shape_2d(self.output_shape, "output_shape"))
        object.__setattr__(self, "operations", tuple(self.operations))

    @classmethod
    def identity(cls, shape: Sequence[int]) -> TransformRecord:
        normalized = _shape_2d(shape, "shape")
        return cls(np.eye(3), normalized, normalized)

    @classmethod
    def from_exif_orientation(
        cls,
        orientation: int,
        shape: Sequence[int],
    ) -> TransformRecord:
        """Build the exact EXIF raw-pixel to canonical-pixel mapping."""

        height, width = _shape_2d(shape, "shape")
        matrices: dict[int, np.ndarray] = {
            1: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
            2: np.array([[-1, 0, width - 1], [0, 1, 0], [0, 0, 1]], dtype=float),
            3: np.array([[-1, 0, width - 1], [0, -1, height - 1], [0, 0, 1]], dtype=float),
            4: np.array([[1, 0, 0], [0, -1, height - 1], [0, 0, 1]], dtype=float),
            5: np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float),
            6: np.array([[0, -1, height - 1], [1, 0, 0], [0, 0, 1]], dtype=float),
            7: np.array([[0, -1, height - 1], [-1, 0, width - 1], [0, 0, 1]], dtype=float),
            8: np.array([[0, 1, 0], [-1, 0, width - 1], [0, 0, 1]], dtype=float),
        }
        if orientation not in matrices:
            raise ValidationError(
                f"Unsupported EXIF orientation {orientation!r}; expected 1 through 8."
            )
        output_shape = (width, height) if orientation in {5, 6, 7, 8} else (height, width)
        operations: tuple[TransformOperation, ...] = ()
        if orientation != 1:
            operations = (TransformOperation("exif_orientation", {"orientation": orientation}),)
        return cls(matrices[orientation], (height, width), output_shape, operations)

    @classmethod
    def resize(cls, input_shape: Sequence[int], output_shape: Sequence[int]) -> TransformRecord:
        in_h, in_w = _shape_2d(input_shape, "input_shape")
        out_h, out_w = _shape_2d(output_shape, "output_shape")
        matrix = np.array(
            [[out_w / in_w, 0, 0], [0, out_h / in_h, 0], [0, 0, 1]],
            dtype=float,
        )
        operation = TransformOperation(
            "resize",
            {"input_shape": (in_h, in_w), "output_shape": (out_h, out_w)},
        )
        return cls(matrix, (in_h, in_w), (out_h, out_w), (operation,))

    @classmethod
    def crop(
        cls,
        input_shape: Sequence[int],
        *,
        left: int,
        top: int,
        width: int,
        height: int,
    ) -> TransformRecord:
        in_h, in_w = _shape_2d(input_shape, "input_shape")
        if left < 0 or top < 0 or width <= 0 or height <= 0:
            raise ValidationError("Crop coordinates and dimensions are invalid.")
        if left + width > in_w or top + height > in_h:
            raise ValidationError("Crop rectangle extends outside the input image.")
        matrix = np.array([[1, 0, -left], [0, 1, -top], [0, 0, 1]], dtype=float)
        operation = TransformOperation(
            "crop",
            {"left": left, "top": top, "width": width, "height": height},
        )
        return cls(matrix, (in_h, in_w), (height, width), (operation,))

    @classmethod
    def letterbox(
        cls,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
    ) -> TransformRecord:
        in_h, in_w = _shape_2d(input_shape, "input_shape")
        out_h, out_w = _shape_2d(output_shape, "output_shape")
        scale = min(out_w / in_w, out_h / in_h)
        rendered_w = in_w * scale
        rendered_h = in_h * scale
        offset_x = (out_w - rendered_w) / 2.0
        offset_y = (out_h - rendered_h) / 2.0
        matrix = np.array(
            [[scale, 0, offset_x], [0, scale, offset_y], [0, 0, 1]],
            dtype=float,
        )
        operation = TransformOperation(
            "letterbox",
            {
                "input_shape": (in_h, in_w),
                "output_shape": (out_h, out_w),
                "scale": scale,
                "offset": (offset_x, offset_y),
            },
        )
        return cls(matrix, (in_h, in_w), (out_h, out_w), (operation,))

    def then(self, next_record: TransformRecord) -> TransformRecord:
        """Compose this mapping with a mapping applied after it."""

        if self.output_shape != next_record.original_shape:
            raise ValidationError(
                "Cannot compose transforms with mismatched shapes: "
                f"{self.output_shape} != {next_record.original_shape}."
            )
        return TransformRecord(
            next_record.matrix @ self.matrix,
            self.original_shape,
            next_record.output_shape,
            self.operations + next_record.operations,
        )

    def forward(self, points: Iterable[Sequence[float]] | np.ndarray) -> np.ndarray:
        """Map one point or an array of ``(..., 2)`` points forward."""

        return self._map(points, self.matrix)

    def inverse(self, points: Iterable[Sequence[float]] | np.ndarray) -> np.ndarray:
        """Map canonical/preprocessed points back to original coordinates."""

        return self._map(points, np.linalg.inv(self.matrix))

    @staticmethod
    def _map(points: Iterable[Sequence[float]] | np.ndarray, matrix: np.ndarray) -> np.ndarray:
        array = np.asarray(points, dtype=np.float64)
        if array.shape == (2,):
            target_shape = (2,)
            flat = array.reshape(1, 2)
        elif array.ndim >= 2 and array.shape[-1] == 2:
            target_shape = array.shape
            flat = array.reshape(-1, 2)
        else:
            raise ValidationError(f"Points must have shape (2,) or (..., 2), got {array.shape}.")
        homogeneous = np.column_stack((flat, np.ones(len(flat), dtype=np.float64)))
        mapped = homogeneous @ matrix.T
        mapped = mapped[:, :2] / mapped[:, 2:3]
        return mapped.reshape(target_shape)

    def forward_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """Map axis-aligned ``[x1, y1, x2, y2]`` boxes and re-bound corners."""

        values = np.asarray(boxes, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, 4)
            squeeze = True
        else:
            squeeze = False
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValidationError(f"Boxes must have shape (N, 4), got {values.shape}.")
        corners = np.stack(
            (
                values[:, [0, 1]],
                values[:, [2, 1]],
                values[:, [2, 3]],
                values[:, [0, 3]],
            ),
            axis=1,
        )
        mapped = self.forward(corners)
        result = np.column_stack(
            (
                mapped[:, :, 0].min(axis=1),
                mapped[:, :, 1].min(axis=1),
                mapped[:, :, 0].max(axis=1),
                mapped[:, :, 1].max(axis=1),
            )
        )
        return result[0] if squeeze else result

    def inverse_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """Map axis-aligned ``[x1, y1, x2, y2]`` boxes back to source pixels."""

        values = np.asarray(boxes, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, 4)
            squeeze = True
        else:
            squeeze = False
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValidationError(f"Boxes must have shape (N, 4), got {values.shape}.")
        corners = np.stack(
            (
                values[:, [0, 1]],
                values[:, [2, 1]],
                values[:, [2, 3]],
                values[:, [0, 3]],
            ),
            axis=1,
        )
        mapped = self.inverse(corners)
        result = np.column_stack(
            (
                mapped[:, :, 0].min(axis=1),
                mapped[:, :, 1].min(axis=1),
                mapped[:, :, 0].max(axis=1),
                mapped[:, :, 1].max(axis=1),
            )
        )
        return result[0] if squeeze else result
