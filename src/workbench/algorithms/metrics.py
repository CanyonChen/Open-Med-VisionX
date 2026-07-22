"""Comparable reconstruction metrics computed in one shared intensity range."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from ..errors import MissingDependencyError, ValidationError


@dataclass(frozen=True, slots=True)
class ScalarMetrics:
    mse: float
    psnr: float
    ssim: float


@dataclass(frozen=True, slots=True)
class RawDomainMetrics:
    """Errors in the unscaled input domain, optionally carrying a unit."""

    mae: float
    rmse: float
    bias: float
    unit: str | None = None


@dataclass(frozen=True, slots=True)
class MetricReport:
    values: ScalarMetrics
    intensity_range: tuple[float, float]
    normalized_reference: np.ndarray
    normalized_reconstruction: np.ndarray
    difference: np.ndarray
    error_heatmap: np.ndarray
    roi: Mapping[str, ScalarMetrics] = field(default_factory=dict)
    raw_values: RawDomainMetrics | None = None
    raw_roi: Mapping[str, RawDomainMetrics] = field(default_factory=dict)
    evaluation_range_source: str = "explicit"

    def __post_init__(self) -> None:
        for name in (
            "normalized_reference",
            "normalized_reconstruction",
            "difference",
            "error_heatmap",
        ):
            array = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if array.ndim != 2 or not np.all(np.isfinite(array)):
                raise ValidationError(f"Metric report {name} must be a finite 2-D array.")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "roi", MappingProxyType(dict(self.roi)))
        object.__setattr__(self, "raw_roi", MappingProxyType(dict(self.raw_roi)))

    @property
    def evaluation_range(self) -> tuple[float, float]:
        """Alias that emphasizes this range is fixed by the evaluation task."""

        return self.intensity_range


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


def _raw_domain_metrics(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    unit: str | None,
) -> RawDomainMetrics:
    with np.errstate(over="ignore", invalid="ignore"):
        error = reconstruction - reference
    if not np.all(np.isfinite(error)):
        raise ValidationError("Raw-domain differences exceed the finite numeric range.")
    absolute_error = np.abs(error)
    maximum_error = float(np.max(absolute_error))
    if maximum_error == 0.0:
        return RawDomainMetrics(mae=0.0, rmse=0.0, bias=0.0, unit=unit)
    scaled_error = error / maximum_error
    return RawDomainMetrics(
        mae=float(maximum_error * np.mean(np.abs(scaled_error), dtype=np.float64)),
        rmse=float(maximum_error * np.sqrt(np.mean(np.square(scaled_error), dtype=np.float64))),
        bias=float(maximum_error * np.mean(scaled_error, dtype=np.float64)),
        unit=unit,
    )


def _real_metric_array(value: object, name: str) -> np.ndarray:
    if np.iscomplexobj(np.asanyarray(value)):
        raise ValidationError(
            "Metric inputs must be real-valued; choose an explicit magnitude or component first."
        )
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must contain numeric values.") from exc
    if array.ndim != 2:
        raise ValidationError(f"{name} must be a 2-D array.")
    if not np.all(np.isfinite(array)):
        raise ValidationError("Metric inputs contain NaN or infinity.")
    return array


def _evaluation_range(value: object) -> tuple[float, float]:
    try:
        raw_values: tuple[Any, ...] = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValidationError("intensity_range must contain exactly two numeric values.") from exc
    if len(raw_values) != 2 or any(np.iscomplexobj(item) for item in raw_values):
        raise ValidationError("intensity_range must contain exactly two real values.")
    try:
        low, high = (float(item) for item in raw_values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("intensity_range must contain exactly two numeric values.") from exc
    scale = high - low
    if not np.isfinite(low) or not np.isfinite(high) or not np.isfinite(scale) or scale <= 0.0:
        raise ValidationError("intensity_range must contain finite increasing values.")
    return low, high


def _roi_rectangle(
    name: object,
    rectangle: object,
) -> tuple[int, int, int, int]:
    try:
        raw_values: tuple[Any, ...] = tuple(rectangle)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValidationError(f"ROI {name!r} must contain four integer coordinates.") from exc
    if len(raw_values) != 4:
        raise ValidationError(f"ROI {name!r} must contain four integer coordinates.")
    coordinates: list[int] = []
    for value in raw_values:
        if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
            raise ValidationError(f"ROI {name!r} must contain four integer coordinates.")
        try:
            numeric = float(value)
            integer = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError(f"ROI {name!r} must contain four integer coordinates.") from exc
        if not np.isfinite(numeric) or numeric != integer:
            raise ValidationError(f"ROI {name!r} must contain four integer coordinates.")
        coordinates.append(integer)
    return tuple(coordinates)  # type: ignore[return-value]


def compute_metrics(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    *,
    intensity_range: tuple[float, float] | None = None,
    rois: Mapping[str, tuple[int, int, int, int]] | None = None,
    unit: str | None = None,
) -> MetricReport:
    """Compute fixed-range normalized metrics and raw-domain error metrics.

    ``rois`` values are ``(x, y, width, height)`` in reference pixels.  Unlike
    the legacy application, the reconstruction is never independently scaled.
    When ``intensity_range`` is omitted, the range is derived from the
    reference alone, so a bad reconstruction cannot make its own normalized
    errors look smaller by widening the evaluation range.  Pass an explicit
    task-level range when comparing multiple references.
    """

    reference_array = _real_metric_array(reference, "reference")
    reconstruction_array = _real_metric_array(reconstruction, "reconstruction")
    if reference_array.shape != reconstruction_array.shape:
        raise ValidationError(
            "reference and reconstruction must be 2-D arrays with identical shapes."
        )
    if intensity_range is None:
        low = float(reference_array.min())
        high = float(reference_array.max())
        range_source = "reference"
        if high <= low:
            raise ValidationError(
                "A constant reference requires an explicit finite increasing intensity_range."
            )
    else:
        low, high = _evaluation_range(intensity_range)
        range_source = "explicit"
    low, high = _evaluation_range((low, high))
    normalized_unit = str(unit).strip() if unit is not None else None
    if normalized_unit == "":
        normalized_unit = None
    scale = high - low
    with np.errstate(over="ignore", invalid="ignore"):
        normalized_reference = (reference_array - low) / scale
        normalized_reconstruction = (reconstruction_array - low) / scale
        difference = normalized_reconstruction - normalized_reference
    if not all(
        np.all(np.isfinite(array))
        for array in (normalized_reference, normalized_reconstruction, difference)
    ):
        raise ValidationError("Normalized metric values exceed the finite numeric range.")
    heatmap = np.abs(difference)
    roi_values: dict[str, ScalarMetrics] = {}
    raw_roi_values: dict[str, RawDomainMetrics] = {}
    for name, rectangle in (rois or {}).items():
        roi_name = str(name)
        if roi_name in roi_values:
            raise ValidationError(f"Duplicate ROI name after normalization: {roi_name!r}.")
        x, y, width, height = _roi_rectangle(name, rectangle)
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValidationError(f"ROI {name!r} has invalid coordinates {rectangle!r}.")
        if x + width > reference_array.shape[1] or y + height > reference_array.shape[0]:
            raise ValidationError(f"ROI {name!r} extends outside the image.")
        normalized_reference_roi = normalized_reference[y : y + height, x : x + width]
        normalized_reconstruction_roi = normalized_reconstruction[y : y + height, x : x + width]
        raw_reference_roi = reference_array[y : y + height, x : x + width]
        raw_reconstruction_roi = reconstruction_array[y : y + height, x : x + width]
        roi_values[roi_name] = _scalar_metrics(
            normalized_reference_roi,
            normalized_reconstruction_roi,
        )
        raw_roi_values[roi_name] = _raw_domain_metrics(
            raw_reference_roi,
            raw_reconstruction_roi,
            normalized_unit,
        )
    return MetricReport(
        values=_scalar_metrics(normalized_reference, normalized_reconstruction),
        intensity_range=(low, high),
        normalized_reference=normalized_reference,
        normalized_reconstruction=normalized_reconstruction,
        difference=difference,
        error_heatmap=heatmap,
        roi=roi_values,
        raw_values=_raw_domain_metrics(
            reference_array,
            reconstruction_array,
            normalized_unit,
        ),
        raw_roi=raw_roi_values,
        evaluation_range_source=range_source,
    )
