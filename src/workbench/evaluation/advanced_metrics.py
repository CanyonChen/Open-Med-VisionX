"""Detection and registration metrics with explicit evaluation units."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from ..errors import ValidationError
from .metrics import compute_segmentation_metrics


@dataclass(frozen=True, slots=True)
class FrocPoint:
    score_threshold: float
    sensitivity: float
    false_positives_per_case: float


@dataclass(frozen=True, slots=True)
class DetectionMetrics:
    case_count: int
    reference_count: int
    prediction_count: int
    score_threshold: float
    matching_iou_threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float
    average_precision_by_iou: Mapping[float, float] = field(default_factory=dict)
    mean_average_precision: float = 0.0
    froc_curve: tuple[FrocPoint, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "average_precision_by_iou",
            MappingProxyType(dict(self.average_precision_by_iou)),
        )


def _boxes(value: np.ndarray | Sequence[Sequence[float]], *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] not in {4, 6}:
        raise ValidationError(f"{name} must have shape N×4 (2-D) or N×6 (3-D).")
    if not np.all(np.isfinite(array)):
        raise ValidationError(f"{name} must contain finite coordinates.")
    dimensions = array.shape[1] // 2
    if np.any(array[:, dimensions:] <= array[:, :dimensions]):
        raise ValidationError(f"{name} maximum coordinates must exceed minimum coordinates.")
    return array


def box_iou_matrix(
    left: np.ndarray | Sequence[Sequence[float]],
    right: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Return pairwise IoU for 2-D boxes or axis-aligned 3-D boxes."""

    left_array = _boxes(left, name="left boxes")
    right_array = _boxes(right, name="right boxes")
    if left_array.shape[1] != right_array.shape[1]:
        raise ValidationError("Both box sets must use the same dimensionality.")
    dimensions = left_array.shape[1] // 2
    intersection_min = np.maximum(
        left_array[:, None, :dimensions],
        right_array[None, :, :dimensions],
    )
    intersection_max = np.minimum(
        left_array[:, None, dimensions:],
        right_array[None, :, dimensions:],
    )
    intersection_size = np.maximum(0.0, intersection_max - intersection_min)
    intersection = np.prod(intersection_size, axis=2)
    left_volume = np.prod(left_array[:, dimensions:] - left_array[:, :dimensions], axis=1)
    right_volume = np.prod(right_array[:, dimensions:] - right_array[:, :dimensions], axis=1)
    union = left_volume[:, None] + right_volume[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0.0)


def _probability(value: float, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValidationError(f"{name} must be a number in [0, 1].")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} must be a number in [0, 1].") from exc
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValidationError(f"{name} must be a number in [0, 1].")
    return result


def _match_predictions(
    references: tuple[np.ndarray, ...],
    predictions: tuple[np.ndarray, ...],
    scores: tuple[np.ndarray, ...],
    *,
    iou_threshold: float,
    score_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    ranked: list[tuple[float, int, int]] = []
    for case_index, case_scores in enumerate(scores):
        ranked.extend(
            (float(score), case_index, prediction_index)
            for prediction_index, score in enumerate(case_scores)
            if score >= score_threshold
        )
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    matched = [np.zeros(reference.shape[0], dtype=np.bool_) for reference in references]
    true_positive = np.zeros(len(ranked), dtype=np.int64)
    false_positive = np.zeros(len(ranked), dtype=np.int64)
    for rank, (_score, case_index, prediction_index) in enumerate(ranked):
        reference = references[case_index]
        if reference.shape[0] == 0:
            false_positive[rank] = 1
            continue
        ious = box_iou_matrix(
            predictions[case_index][prediction_index : prediction_index + 1],
            reference,
        )[0]
        available = np.flatnonzero(~matched[case_index])
        if available.size == 0:
            false_positive[rank] = 1
            continue
        best_available = available[int(np.argmax(ious[available]))]
        if ious[best_available] >= iou_threshold:
            matched[case_index][best_available] = True
            true_positive[rank] = 1
        else:
            false_positive[rank] = 1
    return true_positive, false_positive


def _average_precision(
    true_positive: np.ndarray,
    false_positive: np.ndarray,
    reference_count: int,
) -> float:
    if true_positive.size == 0:
        return 0.0
    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(false_positive)
    recall = cumulative_tp / reference_count
    precision = cumulative_tp / np.maximum(1, cumulative_tp + cumulative_fp)
    extended_recall = np.r_[0.0, recall, 1.0]
    extended_precision = np.r_[0.0, precision, 0.0]
    for index in range(extended_precision.size - 2, -1, -1):
        extended_precision[index] = max(extended_precision[index], extended_precision[index + 1])
    changes = np.flatnonzero(extended_recall[1:] != extended_recall[:-1])
    return float(
        np.sum(
            (extended_recall[changes + 1] - extended_recall[changes])
            * extended_precision[changes + 1]
        )
    )


def compute_detection_metrics(
    reference_boxes: Sequence[np.ndarray | Sequence[Sequence[float]]],
    predicted_boxes: Sequence[np.ndarray | Sequence[Sequence[float]]],
    predicted_scores: Sequence[np.ndarray | Sequence[float]],
    *,
    score_threshold: float = 0.5,
    iou_thresholds: Sequence[float] = (0.5, 0.75),
) -> DetectionMetrics:
    """Evaluate one class across image/series cases using one-to-one matching."""

    if (
        not reference_boxes
        or len(reference_boxes) != len(predicted_boxes)
        or len(reference_boxes) != len(predicted_scores)
    ):
        raise ValidationError(
            "Detection evaluation needs equal, non-empty reference/prediction case lists."
        )
    references = tuple(_boxes(value, name="reference boxes") for value in reference_boxes)
    predictions = tuple(_boxes(value, name="predicted boxes") for value in predicted_boxes)
    dimensions = {item.shape[1] for item in (*references, *predictions)}
    if len(dimensions) != 1:
        raise ValidationError("Every detection case must use the same box dimensionality.")
    scores: list[np.ndarray] = []
    paired_predictions = zip(predictions, predicted_scores, strict=True)
    for case_index, (boxes, raw_scores) in enumerate(paired_predictions):
        score_array = np.asarray(raw_scores, dtype=np.float64)
        if score_array.ndim != 1 or score_array.shape[0] != boxes.shape[0]:
            raise ValidationError(
                f"predicted_scores[{case_index}] must match its prediction count."
            )
        if (
            not np.all(np.isfinite(score_array))
            or np.any(score_array < 0.0)
            or np.any(score_array > 1.0)
        ):
            raise ValidationError("Prediction scores must be finite probabilities in [0, 1].")
        scores.append(score_array)
    reference_count = sum(item.shape[0] for item in references)
    if reference_count == 0:
        raise ValidationError("Detection evaluation requires at least one reference box.")
    selected_score = _probability(score_threshold, name="score_threshold")
    thresholds = tuple(_probability(value, name="IoU threshold") for value in iou_thresholds)
    if (
        not thresholds
        or len(set(thresholds)) != len(thresholds)
        or any(value <= 0.0 for value in thresholds)
    ):
        raise ValidationError("iou_thresholds must contain unique values in (0, 1].")
    score_tuple = tuple(scores)
    ap_by_iou: dict[float, float] = {}
    for threshold in thresholds:
        tp, fp = _match_predictions(
            references,
            predictions,
            score_tuple,
            iou_threshold=threshold,
            score_threshold=0.0,
        )
        ap_by_iou[threshold] = _average_precision(tp, fp, reference_count)

    matching_threshold = thresholds[0]
    operating_tp, operating_fp = _match_predictions(
        references,
        predictions,
        score_tuple,
        iou_threshold=matching_threshold,
        score_threshold=selected_score,
    )
    true_positives = int(np.sum(operating_tp))
    false_positives = int(np.sum(operating_fp))
    false_negatives = reference_count - true_positives
    denominator = true_positives + false_positives
    precision = None if denominator == 0 else true_positives / denominator
    recall = true_positives / reference_count

    unique_scores = sorted(
        {float(value) for array in score_tuple for value in array},
        reverse=True,
    )
    froc: list[FrocPoint] = []
    for threshold in (1.0 + np.finfo(np.float64).eps, *unique_scores, 0.0):
        tp, fp = _match_predictions(
            references,
            predictions,
            score_tuple,
            iou_threshold=matching_threshold,
            score_threshold=threshold,
        )
        froc.append(
            FrocPoint(
                score_threshold=float(min(1.0, threshold)),
                sensitivity=float(np.sum(tp) / reference_count),
                false_positives_per_case=float(np.sum(fp) / len(references)),
            )
        )
    return DetectionMetrics(
        case_count=len(references),
        reference_count=reference_count,
        prediction_count=sum(item.shape[0] for item in predictions),
        score_threshold=selected_score,
        matching_iou_threshold=matching_threshold,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=None if precision is None else float(precision),
        recall=float(recall),
        average_precision_by_iou=ap_by_iou,
        mean_average_precision=float(np.mean(tuple(ap_by_iou.values()))),
        froc_curve=tuple(froc),
    )


@dataclass(frozen=True, slots=True)
class RegistrationMetrics:
    landmark_unit: str
    tre_mean: float | None
    tre_median: float | None
    tre_95th_percentile: float | None
    tre_maximum: float | None
    overlap_dice: float | None
    jacobian_mean: float | None
    jacobian_minimum: float | None
    jacobian_maximum: float | None
    folding_fraction: float | None
    inverse_consistency_mean: float | None
    inverse_consistency_95th_percentile: float | None


def _finite_numeric_array(value: np.ndarray, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} must contain finite numeric values.") from exc
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValidationError(f"{name} must contain finite numeric values.")
    return array


def compute_registration_metrics(
    *,
    fixed_landmarks: np.ndarray | None = None,
    transformed_landmarks: np.ndarray | None = None,
    fixed_mask: np.ndarray | None = None,
    warped_mask: np.ndarray | None = None,
    jacobian_determinant: np.ndarray | None = None,
    inverse_consistency_error: np.ndarray | None = None,
    landmark_unit: str = "mm",
) -> RegistrationMetrics:
    """Summarize geometry-aware registration evidence without image similarity claims."""

    tre: np.ndarray | None = None
    if (fixed_landmarks is None) != (transformed_landmarks is None):
        raise ValidationError("Both fixed and transformed landmarks are required for TRE.")
    if fixed_landmarks is not None and transformed_landmarks is not None:
        fixed = _finite_numeric_array(fixed_landmarks, name="fixed_landmarks")
        transformed = _finite_numeric_array(
            transformed_landmarks,
            name="transformed_landmarks",
        )
        if fixed.ndim != 2 or fixed.shape != transformed.shape or fixed.shape[1] not in {2, 3}:
            raise ValidationError("Landmarks must have equal N×2 or N×3 shapes.")
        tre = np.linalg.norm(fixed - transformed, axis=1)

    overlap_dice: float | None = None
    if (fixed_mask is None) != (warped_mask is None):
        raise ValidationError("Both fixed and warped masks are required for overlap Dice.")
    if fixed_mask is not None and warped_mask is not None:
        overlap_dice = compute_segmentation_metrics(fixed_mask, warped_mask).dice

    jacobian: np.ndarray | None = None
    if jacobian_determinant is not None:
        jacobian = _finite_numeric_array(
            jacobian_determinant,
            name="jacobian_determinant",
        )
        if jacobian.ndim not in {2, 3}:
            raise ValidationError("jacobian_determinant must be a 2-D or 3-D field.")

    inverse_error: np.ndarray | None = None
    if inverse_consistency_error is not None:
        inverse_error = _finite_numeric_array(
            inverse_consistency_error,
            name="inverse_consistency_error",
        )
        if np.any(inverse_error < 0.0):
            raise ValidationError("inverse_consistency_error cannot be negative.")

    if tre is None and overlap_dice is None and jacobian is None and inverse_error is None:
        raise ValidationError("Registration evaluation requires at least one evidence source.")
    unit = str(landmark_unit).strip()
    if not unit or len(unit) > 16:
        raise ValidationError("landmark_unit must be short non-empty text.")
    return RegistrationMetrics(
        landmark_unit=unit,
        tre_mean=None if tre is None else float(np.mean(tre)),
        tre_median=None if tre is None else float(np.median(tre)),
        tre_95th_percentile=None if tre is None else float(np.percentile(tre, 95.0)),
        tre_maximum=None if tre is None else float(np.max(tre)),
        overlap_dice=overlap_dice,
        jacobian_mean=None if jacobian is None else float(np.mean(jacobian)),
        jacobian_minimum=None if jacobian is None else float(np.min(jacobian)),
        jacobian_maximum=None if jacobian is None else float(np.max(jacobian)),
        folding_fraction=None
        if jacobian is None
        else float(np.count_nonzero(jacobian <= 0.0) / jacobian.size),
        inverse_consistency_mean=None if inverse_error is None else float(np.mean(inverse_error)),
        inverse_consistency_95th_percentile=None
        if inverse_error is None
        else float(np.percentile(inverse_error, 95.0)),
    )
