"""Dependency-free CT attenuation phantoms with explicit physical semantics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import ValidationError


def _circle_support(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    y, x = np.ogrid[:height, :width]
    radius = min(height, width) // 2
    return (x - width // 2) ** 2 + (y - height // 2) ** 2 <= radius**2


@dataclass(frozen=True, slots=True)
class CTAttenuationPhantom:
    """A synthetic, nonnegative attenuation map for CT teaching experiments."""

    array: np.ndarray
    evaluation_range: tuple[float, float]
    unit: str = "mm^-1"

    def __post_init__(self) -> None:
        if np.iscomplexobj(np.asanyarray(self.array)):
            raise ValidationError("A CT attenuation phantom must be real-valued.")
        try:
            array = np.asarray(self.array, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValidationError("A CT attenuation phantom must contain numeric values.") from exc
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValidationError("A CT attenuation phantom must be a square 2-D array.")
        if not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValidationError(
                "A CT attenuation phantom must contain finite nonnegative values."
            )
        if np.any(array[~_circle_support(array.shape)] != 0.0):
            raise ValidationError(
                "A CT attenuation phantom must be exactly zero outside circular support."
            )
        try:
            range_values = tuple(self.evaluation_range)
            if len(range_values) != 2:
                raise ValueError
            low, high = (float(value) for value in range_values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError(
                "A CT attenuation phantom evaluation range must contain two values."
            ) from exc
        if low != 0.0 or not np.isfinite(high) or high <= 0.0:
            raise ValidationError(
                "A CT attenuation phantom evaluation range must start at zero and be positive."
            )
        if float(array.max()) > high:
            raise ValidationError("Phantom values exceed the declared evaluation range.")
        unit = str(self.unit).strip()
        if not unit:
            raise ValidationError("A CT attenuation phantom must declare a non-empty unit.")
        immutable = np.array(array, copy=True)
        immutable.setflags(write=False)
        object.__setattr__(self, "array", immutable)
        object.__setattr__(self, "evaluation_range", (low, high))
        object.__setattr__(self, "unit", unit)


def generate_ct_phantom(
    size: int = 256,
    *,
    maximum_attenuation: float = 0.03,
) -> CTAttenuationPhantom:
    """Generate a simple material phantom in ``mm^-1`` with exact zero background.

    The geometry is intentionally analytic rather than resized from an image,
    avoiding interpolation undershoot and nonzero values outside the circular
    reconstruction support.  Values are illustrative, not a patient or
    scanner calibration.
    """

    if isinstance(size, (bool, np.bool_)):
        raise ValidationError("CT phantom size must be an integer.")
    try:
        integer_size = int(size)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("CT phantom size must be an integer.") from exc
    if integer_size != size:
        raise ValidationError("CT phantom size must be an integer.")
    size = integer_size
    if np.iscomplexobj(maximum_attenuation):
        raise ValidationError("maximum_attenuation must be a positive finite value.")
    try:
        maximum_attenuation = float(maximum_attenuation)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("maximum_attenuation must be a positive finite value.") from exc
    if size < 16:
        raise ValidationError("CT phantom size must be at least 16 pixels.")
    if not np.isfinite(maximum_attenuation) or maximum_attenuation <= 0.0:
        raise ValidationError("maximum_attenuation must be a positive finite value.")

    coordinate = (np.arange(size, dtype=np.float64) - (size - 1) / 2.0) / (size / 2.0)
    x, y = np.meshgrid(coordinate, coordinate)
    phantom = np.zeros((size, size), dtype=np.float64)

    def ellipse(
        center_x: float,
        center_y: float,
        radius_x: float,
        radius_y: float,
    ) -> np.ndarray:
        return np.square((x - center_x) / radius_x) + np.square((y - center_y) / radius_y) <= 1.0

    # A water-like body with several material inserts.  Assignment rather
    # than addition makes the declared range exact and easy to interpret.
    phantom[ellipse(0.0, 0.0, 0.72, 0.90)] = maximum_attenuation * 0.60
    phantom[ellipse(-0.24, -0.18, 0.13, 0.20)] = maximum_attenuation
    phantom[ellipse(0.25, -0.16, 0.16, 0.12)] = maximum_attenuation * 0.32
    phantom[ellipse(-0.18, 0.30, 0.10, 0.10)] = maximum_attenuation * 0.78
    phantom[ellipse(0.25, 0.28, 0.09, 0.16)] = maximum_attenuation * 0.45

    phantom[~_circle_support(phantom.shape)] = 0.0
    return CTAttenuationPhantom(
        phantom,
        evaluation_range=(0.0, maximum_attenuation),
    )
