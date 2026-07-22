"""Task-level metrics with explicit thresholds, calibration, and geometry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from types import MappingProxyType

import numpy as np

from ..errors import MissingDependencyError, ValidationError


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    @property
    def total(self) -> int:
        return self.true_negative + self.false_positive + self.false_negative + self.true_positive


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    method: str

    def __post_init__(self) -> None:
        values = (self.estimate, self.lower, self.upper, self.confidence_level)
        if not all(np.isfinite(value) for value in values):
            raise ValidationError("Confidence interval values must be finite.")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValidationError("confidence_level must be between zero and one.")
        if self.lower > self.estimate or self.estimate > self.upper:
            raise ValidationError("Confidence interval bounds must contain the estimate.")


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_score: float | None
    positive_fraction: float | None


@dataclass(frozen=True, slots=True)
class BinaryClassificationMetrics:
    threshold: float
    confusion: ConfusionMatrix
    accuracy: float
    sensitivity: float
    specificity: float
    positive_predictive_value: float | None
    negative_predictive_value: float | None
    f1: float | None
    auroc: float
    average_precision: float
    brier_score: float
    expected_calibration_error: float
    calibration_bins: tuple[CalibrationBin, ...]
    confidence_intervals: Mapping[str, ConfidenceInterval] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "confidence_intervals",
            MappingProxyType(dict(self.confidence_intervals)),
        )


@dataclass(frozen=True, slots=True)
class ThresholdPoint:
    threshold: float
    confusion: ConfusionMatrix
    sensitivity: float
    specificity: float
    positive_predictive_value: float | None
    negative_predictive_value: float | None
    f1: float | None


def _as_binary_inputs(
    truth: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw_truth = np.asarray(truth)
    raw_scores = np.asarray(scores)
    if raw_truth.ndim != 1 or raw_scores.ndim != 1 or raw_truth.shape != raw_scores.shape:
        raise ValidationError("truth and scores must be equal-length one-dimensional arrays.")
    if raw_truth.size < 2:
        raise ValidationError("Classification evaluation requires at least two samples.")
    if np.iscomplexobj(raw_truth) or np.iscomplexobj(raw_scores):
        raise ValidationError("Classification inputs must be real-valued.")
    try:
        truth_array = raw_truth.astype(np.int8, copy=False)
        score_array = raw_scores.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("Classification inputs must be numeric.") from exc
    try:
        truth_numeric = raw_truth.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("truth must contain binary numeric labels.") from exc
    if not np.all(np.isfinite(truth_numeric)) or not np.all(np.isin(truth_numeric, (0.0, 1.0))):
        raise ValidationError("truth must contain binary labels 0 and 1 only.")
    if not np.all(np.isfinite(score_array)) or np.any((score_array < 0.0) | (score_array > 1.0)):
        raise ValidationError("scores must be finite probabilities in the range [0, 1].")
    if np.unique(truth_array).size != 2:
        raise ValidationError("AUROC and average precision require both outcome classes.")
    return truth_array, score_array


def _validated_threshold(value: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValidationError("threshold must be a probability in the range [0, 1].")
    try:
        threshold = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("threshold must be numeric.") from exc
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValidationError("threshold must be a probability in the range [0, 1].")
    return threshold


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _confusion(truth: np.ndarray, scores: np.ndarray, threshold: float) -> ConfusionMatrix:
    predicted = scores >= threshold
    positive = truth == 1
    return ConfusionMatrix(
        true_negative=int(np.count_nonzero(~positive & ~predicted)),
        false_positive=int(np.count_nonzero(~positive & predicted)),
        false_negative=int(np.count_nonzero(positive & ~predicted)),
        true_positive=int(np.count_nonzero(positive & predicted)),
    )


def _auroc(truth: np.ndarray, scores: np.ndarray) -> float:
    """Compute tie-aware AUROC using the Mann-Whitney rank identity."""

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.arange(1, scores.size + 1, dtype=np.float64)
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[start:stop] = float(np.mean(ranks[start:stop]))
        start = stop
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    positives = truth == 1
    positive_count = int(np.count_nonzero(positives))
    negative_count = truth.size - positive_count
    rank_sum = float(np.sum(original_ranks[positives]))
    return float(
        (rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)
    )


def _average_precision(truth: np.ndarray, scores: np.ndarray) -> float:
    """Compute non-interpolated, tie-grouped average precision."""

    order = np.argsort(-scores, kind="mergesort")
    sorted_truth = truth[order]
    sorted_scores = scores[order]
    true_positives = np.cumsum(sorted_truth == 1)
    false_positives = np.cumsum(sorted_truth == 0)
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), scores.size - 1]
    tp = true_positives[group_ends].astype(np.float64)
    fp = false_positives[group_ends].astype(np.float64)
    recall = tp / float(np.count_nonzero(truth == 1))
    precision = tp / (tp + fp)
    recall_increments = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increments * precision))


def _calibration(
    truth: np.ndarray,
    scores: np.ndarray,
    n_bins: int,
) -> tuple[tuple[CalibrationBin, ...], float]:
    if (
        isinstance(n_bins, bool)
        or not isinstance(n_bins, (int, np.integer))
        or not 2 <= n_bins <= 100
    ):
        raise ValidationError("n_bins must be an integer from 2 through 100.")
    indices = np.minimum((scores * n_bins).astype(np.int64), n_bins - 1)
    bins: list[CalibrationBin] = []
    weighted_gap = 0.0
    for index in range(n_bins):
        selected = indices == index
        count = int(np.count_nonzero(selected))
        lower = index / n_bins
        upper = (index + 1) / n_bins
        if count == 0:
            bins.append(CalibrationBin(lower, upper, 0, None, None))
            continue
        mean_score = float(np.mean(scores[selected]))
        positive_fraction = float(np.mean(truth[selected]))
        weighted_gap += count * abs(mean_score - positive_fraction)
        bins.append(CalibrationBin(lower, upper, count, mean_score, positive_fraction))
    return tuple(bins), float(weighted_gap / truth.size)


def _wilson_interval(
    successes: int,
    total: int,
    *,
    confidence_level: float,
    method_name: str = "Wilson score",
) -> ConfidenceInterval | None:
    if total == 0:
        return None
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return ConfidenceInterval(
        estimate=float(proportion),
        lower=float(max(0.0, centre - half_width)),
        upper=float(min(1.0, centre + half_width)),
        confidence_level=confidence_level,
        method=method_name,
    )


def _auroc_interval(
    estimate: float,
    positive_count: int,
    negative_count: int,
    confidence_level: float,
) -> ConfidenceInterval:
    """Hanley-McNeil large-sample interval, clipped to the metric domain."""

    q1 = estimate / (2.0 - estimate)
    q2 = 2.0 * estimate * estimate / (1.0 + estimate)
    variance = (
        estimate * (1.0 - estimate)
        + (positive_count - 1) * (q1 - estimate * estimate)
        + (negative_count - 1) * (q2 - estimate * estimate)
    ) / (positive_count * negative_count)
    standard_error = float(np.sqrt(max(0.0, variance)))
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    return ConfidenceInterval(
        estimate=estimate,
        lower=max(0.0, estimate - z * standard_error),
        upper=min(1.0, estimate + z * standard_error),
        confidence_level=confidence_level,
        method="Hanley-McNeil",
    )


def compute_binary_classification_metrics(
    truth: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
    n_bins: int = 10,
    confidence_level: float = 0.95,
) -> BinaryClassificationMetrics:
    """Evaluate binary probabilities without silently selecting a threshold."""

    truth_array, score_array = _as_binary_inputs(truth, scores)
    selected_threshold = _validated_threshold(threshold)
    if not 0.0 < confidence_level < 1.0:
        raise ValidationError("confidence_level must be between zero and one.")
    confusion = _confusion(truth_array, score_array, selected_threshold)
    sensitivity = confusion.true_positive / (confusion.true_positive + confusion.false_negative)
    specificity = confusion.true_negative / (confusion.true_negative + confusion.false_positive)
    ppv = _safe_ratio(confusion.true_positive, confusion.true_positive + confusion.false_positive)
    npv = _safe_ratio(confusion.true_negative, confusion.true_negative + confusion.false_negative)
    f1 = _safe_ratio(
        2 * confusion.true_positive,
        2 * confusion.true_positive + confusion.false_positive + confusion.false_negative,
    )
    accuracy = (confusion.true_positive + confusion.true_negative) / confusion.total
    auroc = _auroc(truth_array, score_array)
    calibration_bins, ece = _calibration(truth_array, score_array, n_bins)
    intervals: dict[str, ConfidenceInterval] = {}
    interval_inputs = {
        "accuracy": (confusion.true_positive + confusion.true_negative, confusion.total),
        "sensitivity": (
            confusion.true_positive,
            confusion.true_positive + confusion.false_negative,
        ),
        "specificity": (
            confusion.true_negative,
            confusion.true_negative + confusion.false_positive,
        ),
        "positive_predictive_value": (
            confusion.true_positive,
            confusion.true_positive + confusion.false_positive,
        ),
        "negative_predictive_value": (
            confusion.true_negative,
            confusion.true_negative + confusion.false_negative,
        ),
    }
    for name, (successes, total) in interval_inputs.items():
        interval = _wilson_interval(successes, total, confidence_level=confidence_level)
        if interval is not None:
            intervals[name] = interval
    intervals["auroc"] = _auroc_interval(
        auroc,
        confusion.true_positive + confusion.false_negative,
        confusion.true_negative + confusion.false_positive,
        confidence_level,
    )
    return BinaryClassificationMetrics(
        threshold=selected_threshold,
        confusion=confusion,
        accuracy=float(accuracy),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        positive_predictive_value=ppv,
        negative_predictive_value=npv,
        f1=f1,
        auroc=auroc,
        average_precision=_average_precision(truth_array, score_array),
        brier_score=float(np.mean(np.square(score_array - truth_array))),
        expected_calibration_error=ece,
        calibration_bins=calibration_bins,
        confidence_intervals=intervals,
    )


def threshold_sweep(
    truth: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    thresholds: Sequence[float],
) -> tuple[ThresholdPoint, ...]:
    """Evaluate caller-selected operating points in the supplied order."""

    truth_array, score_array = _as_binary_inputs(truth, scores)
    if len(thresholds) == 0:
        raise ValidationError("thresholds must contain at least one operating point.")
    points: list[ThresholdPoint] = []
    seen: set[float] = set()
    for raw_threshold in thresholds:
        threshold = _validated_threshold(raw_threshold)
        if threshold in seen:
            raise ValidationError("thresholds must not contain duplicates.")
        seen.add(threshold)
        matrix = _confusion(truth_array, score_array, threshold)
        points.append(
            ThresholdPoint(
                threshold=threshold,
                confusion=matrix,
                sensitivity=matrix.true_positive / (matrix.true_positive + matrix.false_negative),
                specificity=matrix.true_negative / (matrix.true_negative + matrix.false_positive),
                positive_predictive_value=_safe_ratio(
                    matrix.true_positive,
                    matrix.true_positive + matrix.false_positive,
                ),
                negative_predictive_value=_safe_ratio(
                    matrix.true_negative,
                    matrix.true_negative + matrix.false_negative,
                ),
                f1=_safe_ratio(
                    2 * matrix.true_positive,
                    2 * matrix.true_positive + matrix.false_positive + matrix.false_negative,
                ),
            )
        )
    return tuple(points)


def bootstrap_classification_interval(
    truth: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    metric: str,
    threshold: float = 0.5,
    n_resamples: int = 1_000,
    confidence_level: float = 0.95,
    random_seed: int = 0,
) -> ConfidenceInterval:
    """Return a deterministic stratified percentile bootstrap interval."""

    truth_array, score_array = _as_binary_inputs(truth, scores)
    selected_threshold = _validated_threshold(threshold)
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, (int, np.integer)):
        raise ValidationError("n_resamples must be an integer.")
    if n_resamples < 100 or n_resamples > 100_000:
        raise ValidationError("n_resamples must be from 100 through 100,000.")
    if not 0.0 < confidence_level < 1.0:
        raise ValidationError("confidence_level must be between zero and one.")
    supported = {
        "accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "auroc",
        "average_precision",
        "brier_score",
    }
    if metric not in supported:
        raise ValidationError("Unsupported bootstrap metric: " + str(metric))

    def metric_value(labels: np.ndarray, probabilities: np.ndarray) -> float:
        matrix = _confusion(labels, probabilities, selected_threshold)
        if metric == "accuracy":
            return (matrix.true_positive + matrix.true_negative) / matrix.total
        if metric == "sensitivity":
            return matrix.true_positive / (matrix.true_positive + matrix.false_negative)
        if metric == "specificity":
            return matrix.true_negative / (matrix.true_negative + matrix.false_positive)
        if metric == "f1":
            value = _safe_ratio(
                2 * matrix.true_positive,
                2 * matrix.true_positive + matrix.false_positive + matrix.false_negative,
            )
            return 0.0 if value is None else value
        if metric == "auroc":
            return _auroc(labels, probabilities)
        if metric == "average_precision":
            return _average_precision(labels, probabilities)
        return float(np.mean(np.square(probabilities - labels)))

    generator = np.random.default_rng(random_seed)
    positive_indices = np.flatnonzero(truth_array == 1)
    negative_indices = np.flatnonzero(truth_array == 0)
    values = np.empty(int(n_resamples), dtype=np.float64)
    for index in range(int(n_resamples)):
        sampled = np.concatenate(
            (
                generator.choice(positive_indices, positive_indices.size, replace=True),
                generator.choice(negative_indices, negative_indices.size, replace=True),
            )
        )
        values[index] = metric_value(truth_array[sampled], score_array[sampled])
    alpha = (1.0 - confidence_level) / 2.0
    estimate = metric_value(truth_array, score_array)
    lower, upper = np.quantile(values, (alpha, 1.0 - alpha))
    # A percentile interval need not contain the point estimate in tiny samples.
    # Widening to include it keeps the contract explicit and useful for display.
    return ConfidenceInterval(
        estimate=float(estimate),
        lower=float(min(lower, estimate)),
        upper=float(max(upper, estimate)),
        confidence_level=confidence_level,
        method=f"stratified percentile bootstrap ({int(n_resamples)} resamples)",
    )


@dataclass(frozen=True, slots=True)
class SegmentationMetrics:
    dice: float
    intersection_over_union: float
    hd95: float | None
    average_symmetric_surface_distance: float | None
    surface_dice: float
    surface_tolerance: float
    distance_unit: str
    reference_voxels: int
    prediction_voxels: int


def _binary_mask(value: np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim not in {2, 3} or raw.size == 0:
        raise ValidationError(f"{name} must be a non-empty 2-D or 3-D binary array.")
    if np.iscomplexobj(raw):
        raise ValidationError(f"{name} must contain binary values 0 and 1 only.")
    try:
        numeric = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} must contain binary numeric values.") from exc
    if not np.all(np.isfinite(numeric)) or not np.all(np.isin(numeric, (0.0, 1.0))):
        raise ValidationError(f"{name} must contain binary values 0 and 1 only.")
    return np.asarray(numeric == 1.0, dtype=np.bool_)


def _validated_spacing(
    spacing: Sequence[float] | None,
    ndim: int,
) -> tuple[tuple[float, ...], str]:
    if spacing is None:
        return (1.0,) * ndim, "voxel"
    try:
        values = tuple(float(item) for item in spacing)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("spacing must contain positive finite values.") from exc
    if len(values) != ndim or not all(np.isfinite(item) and item > 0.0 for item in values):
        raise ValidationError(f"spacing must contain {ndim} positive finite values.")
    return values, "mm"


def compute_segmentation_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    spacing: Sequence[float] | None = None,
    surface_tolerance: float = 1.0,
) -> SegmentationMetrics:
    """Compute overlap and symmetric surface metrics for a binary mask pair.

    If exactly one mask is empty, overlap and surface Dice are zero while
    distance metrics are ``None`` because no finite boundary distance exists.
    If both masks are empty, all metrics represent perfect agreement.
    """

    reference_mask = _binary_mask(reference, name="reference")
    prediction_mask = _binary_mask(prediction, name="prediction")
    if reference_mask.shape != prediction_mask.shape:
        raise ValidationError("reference and prediction masks must have identical shapes.")
    spacing_values, distance_unit = _validated_spacing(spacing, reference_mask.ndim)
    try:
        tolerance = float(surface_tolerance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("surface_tolerance must be a non-negative finite value.") from exc
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValidationError("surface_tolerance must be a non-negative finite value.")
    reference_count = int(np.count_nonzero(reference_mask))
    prediction_count = int(np.count_nonzero(prediction_mask))
    intersection = int(np.count_nonzero(reference_mask & prediction_mask))
    union = int(np.count_nonzero(reference_mask | prediction_mask))
    if reference_count == 0 and prediction_count == 0:
        return SegmentationMetrics(
            dice=1.0,
            intersection_over_union=1.0,
            hd95=0.0,
            average_symmetric_surface_distance=0.0,
            surface_dice=1.0,
            surface_tolerance=tolerance,
            distance_unit=distance_unit,
            reference_voxels=0,
            prediction_voxels=0,
        )
    dice = 2.0 * intersection / (reference_count + prediction_count)
    iou = intersection / union
    if reference_count == 0 or prediction_count == 0:
        return SegmentationMetrics(
            dice=float(dice),
            intersection_over_union=float(iou),
            hd95=None,
            average_symmetric_surface_distance=None,
            surface_dice=0.0,
            surface_tolerance=tolerance,
            distance_unit=distance_unit,
            reference_voxels=reference_count,
            prediction_voxels=prediction_count,
        )
    try:
        from scipy.ndimage import (  # type: ignore[import-untyped]
            binary_erosion,
            distance_transform_edt,
            generate_binary_structure,
        )
    except ImportError as exc:  # pragma: no cover - base dependency
        raise MissingDependencyError("Surface metrics require SciPy.") from exc
    structure = generate_binary_structure(reference_mask.ndim, 1)
    reference_surface = reference_mask & ~binary_erosion(
        reference_mask,
        structure=structure,
        border_value=0,
    )
    prediction_surface = prediction_mask & ~binary_erosion(
        prediction_mask,
        structure=structure,
        border_value=0,
    )
    reference_to_prediction = distance_transform_edt(
        ~prediction_surface,
        sampling=spacing_values,
    )[reference_surface]
    prediction_to_reference = distance_transform_edt(
        ~reference_surface,
        sampling=spacing_values,
    )[prediction_surface]
    distances = np.concatenate((reference_to_prediction, prediction_to_reference))
    surface_matches = int(np.count_nonzero(reference_to_prediction <= tolerance)) + int(
        np.count_nonzero(prediction_to_reference <= tolerance)
    )
    surface_points = reference_to_prediction.size + prediction_to_reference.size
    return SegmentationMetrics(
        dice=float(dice),
        intersection_over_union=float(iou),
        hd95=float(np.percentile(distances, 95.0)),
        average_symmetric_surface_distance=float(np.mean(distances)),
        surface_dice=float(surface_matches / surface_points),
        surface_tolerance=tolerance,
        distance_unit=distance_unit,
        reference_voxels=reference_count,
        prediction_voxels=prediction_count,
    )
