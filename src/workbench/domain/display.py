"""Display-only mappings that never alter decoded image data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import ValidationError


def _finite_array(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if not np.issubdtype(values.dtype, np.number):
        raise ValidationError("Display mapping requires numeric pixels.")
    if not np.all(np.isfinite(values)):
        values = np.nan_to_num(values)
    return values


@dataclass(frozen=True, slots=True)
class GrayscaleDisplayMapping:
    """Map grayscale values to uint8 using an explicit intensity window."""

    lower: float
    upper: float
    invert: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.lower) or not np.isfinite(self.upper) or self.upper <= self.lower:
            raise ValidationError("Display lower/upper bounds must be finite and increasing.")

    @classmethod
    def from_window_level(
        cls,
        window_width: float,
        window_level: float,
        *,
        invert: bool = False,
    ) -> GrayscaleDisplayMapping:
        if window_width <= 0:
            raise ValidationError("Window width must be positive.")
        return cls(
            lower=float(window_level) - float(window_width) / 2.0,
            upper=float(window_level) + float(window_width) / 2.0,
            invert=invert,
        )

    @classmethod
    def from_percentiles(
        cls,
        array: np.ndarray,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
        *,
        invert: bool = False,
    ) -> GrayscaleDisplayMapping:
        if not 0 <= lower_percentile < upper_percentile <= 100:
            raise ValidationError("Percentiles must satisfy 0 <= lower < upper <= 100.")
        values = _finite_array(array)
        lower, upper = np.percentile(values, [lower_percentile, upper_percentile])
        if upper <= lower:
            upper = float(lower) + 1.0
        return cls(float(lower), float(upper), invert=invert)

    def map(self, array: np.ndarray) -> np.ndarray:
        values = _finite_array(array).astype(np.float64, copy=False)
        normalized = np.clip((values - self.lower) / (self.upper - self.lower), 0.0, 1.0)
        if self.invert:
            normalized = 1.0 - normalized
        return np.rint(normalized * 255.0).astype(np.uint8)


@dataclass(frozen=True, slots=True)
class ColorDisplayMapping:
    """Display RGB/RGBA data with explicit brightness/contrast/gamma."""

    brightness: float = 0.0
    contrast: float = 1.0
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if not -1.0 <= self.brightness <= 1.0:
            raise ValidationError("brightness must be in [-1, 1].")
        if self.contrast <= 0 or self.gamma <= 0:
            raise ValidationError("contrast and gamma must be positive.")

    def map(self, array: np.ndarray) -> np.ndarray:
        values = _finite_array(array)
        if values.ndim != 3 or values.shape[2] not in {3, 4}:
            raise ValidationError("Color display data must be HxWx3 or HxWx4.")
        color = values[:, :, :3].astype(np.float64)
        if np.issubdtype(values.dtype, np.integer):
            dtype_info = np.iinfo(values.dtype)
            color = (color - dtype_info.min) / (dtype_info.max - dtype_info.min)
        else:
            finite_min = float(np.min(color))
            finite_max = float(np.max(color))
            if finite_max > 1.0 or finite_min < 0.0:
                span = finite_max - finite_min
                color = np.zeros_like(color) if span == 0 else (color - finite_min) / span
        color = (color - 0.5) * self.contrast + 0.5 + self.brightness
        color = np.power(np.clip(color, 0.0, 1.0), 1.0 / self.gamma)
        mapped = np.rint(color * 255.0).astype(np.uint8)
        if values.shape[2] == 4:
            alpha = values[:, :, 3]
            if np.issubdtype(values.dtype, np.integer):
                alpha = (alpha.astype(np.float64) - np.iinfo(values.dtype).min) / (
                    np.iinfo(values.dtype).max - np.iinfo(values.dtype).min
                )
            alpha_u8 = np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
            mapped = np.dstack((mapped, alpha_u8))
        return mapped
