"""Comparable reconstruction metrics computed in one shared intensity range."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from ..errors import MissingDependencyError, ValidationError


@dataclass(frozen=True, slots=True)
class ScalarMetrics:
    mse: float
    psnr: float
    ssim: float


@dataclass(frozen=True, slots=True)
class MetricReport:
    values: ScalarMetrics
    intensity_range: tuple[float, float]
    normalized_reference: np.ndarray
    normalized_reconstruction: np.ndarray
    difference: np.ndarray
    error_heatmap: np.ndarray
    roi: Mapping[str, ScalarMetrics] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "roi", MappingProxyType(dict(self.roi)))


def _scalar_metrics(reference: np.ndarray, reconstruction: np.ndarray) -> ScalarMetrics:
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:  # pragma: no cover - depends on install
        raise MissingDependencyError("Metrics require scikit-image.") from exc
    error = reconstruction - reference
    mse = float(np.mean(np.square(error), dtype=np.float64))
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))
    smallest = min(reference.shape)
    if smallest < 3:
        raise ValidationError("SSIM requires images at least 3x3 pixels.")
    window = min(7, smallest if smallest % 2 else smallest - 1)
    ssim = float(structural_similarity(reference, reconstruction, data_range=1.0, win_size=window))
    return ScalarMetrics(mse=mse, psnr=psnr, ssim=ssim)


def compute_metrics(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    *,
    intensity_range: tuple[float, float] | None = None,
    rois: Mapping[str, tuple[int, int, int, int]] | None = None,
) -> MetricReport:
    """Compute MSE/PSNR/SSIM after one joint linear mapping.

    ``rois`` values are ``(x, y, width, height)`` in reference pixels.  Unlike
    the legacy application, the reconstruction is never independently scaled.
    """

    reference_array = np.asarray(reference, dtype=np.float64)
    reconstruction_array = np.asarray(reconstruction, dtype=np.float64)
    if reference_array.shape != reconstruction_array.shape or reference_array.ndim != 2:
        raise ValidationError(
            "reference and reconstruction must be 2-D arrays with identical shapes."
        )
    if not np.all(np.isfinite(reference_array)) or not np.all(np.isfinite(reconstruction_array)):
        raise ValidationError("Metric inputs contain NaN or infinity.")
    if intensity_range is None:
        low = float(min(reference_array.min(), reconstruction_array.min()))
        high = float(max(reference_array.max(), reconstruction_array.max()))
    else:
        low, high = (float(intensity_range[0]), float(intensity_range[1]))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValidationError("intensity_range must contain finite increasing values.")
    scale = high - low
    normalized_reference = (reference_array - low) / scale
    normalized_reconstruction = (reconstruction_array - low) / scale
    difference = normalized_reconstruction - normalized_reference
    heatmap = np.abs(difference)
    roi_values: dict[str, ScalarMetrics] = {}
    for name, rectangle in (rois or {}).items():
        x, y, width, height = (int(item) for item in rectangle)
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValidationError(f"ROI {name!r} has invalid coordinates {rectangle!r}.")
        if x + width > reference_array.shape[1] or y + height > reference_array.shape[0]:
            raise ValidationError(f"ROI {name!r} extends outside the image.")
        roi_values[str(name)] = _scalar_metrics(
            normalized_reference[y : y + height, x : x + width],
            normalized_reconstruction[y : y + height, x : x + width],
        )
    return MetricReport(
        values=_scalar_metrics(normalized_reference, normalized_reconstruction),
        intensity_range=(low, high),
        normalized_reference=normalized_reference,
        normalized_reconstruction=normalized_reconstruction,
        difference=difference,
        error_heatmap=heatmap,
        roi=roi_values,
    )
