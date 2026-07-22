from __future__ import annotations

import numpy as np
import pytest

from workbench.errors import ValidationError
from workbench.evaluation.advanced_metrics import (
    box_iou_matrix,
    compute_detection_metrics,
    compute_registration_metrics,
)


def test_box_iou_supports_2d_and_3d_without_pixel_inclusive_offsets() -> None:
    two_dimensional = box_iou_matrix([[0, 0, 2, 2]], [[1, 1, 3, 3]])
    three_dimensional = box_iou_matrix([[0, 0, 0, 2, 2, 2]], [[1, 1, 1, 3, 3, 3]])

    assert two_dimensional[0, 0] == pytest.approx(1 / 7)
    assert three_dimensional[0, 0] == pytest.approx(1 / 15)


def test_detection_metrics_use_case_safe_one_to_one_matching_and_froc() -> None:
    references = (
        np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float),
        np.array([[5, 5, 15, 15]], dtype=float),
    )
    predictions = (
        np.array(
            [
                [0, 0, 10, 10],
                [0, 0, 10, 10],
                [20, 20, 29, 29],
            ],
            dtype=float,
        ),
        np.array([[40, 40, 50, 50]], dtype=float),
    )
    scores = (np.array([0.95, 0.9, 0.8]), np.array([0.7]))

    metrics = compute_detection_metrics(
        references,
        predictions,
        scores,
        score_threshold=0.75,
        iou_thresholds=(0.5, 0.75),
    )

    assert metrics.case_count == 2
    assert metrics.true_positives == 2
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert set(metrics.average_precision_by_iou) == {0.5, 0.75}
    assert 0.0 <= metrics.mean_average_precision <= 1.0
    assert metrics.froc_curve[-1].false_positives_per_case == pytest.approx(1.0)


def test_detection_rejects_mismatched_scores_and_empty_ground_truth() -> None:
    with pytest.raises(ValidationError, match="prediction count"):
        compute_detection_metrics(
            (np.array([[0, 0, 1, 1]]),),
            (np.array([[0, 0, 1, 1]]),),
            (np.array([]),),
        )
    with pytest.raises(ValidationError, match="reference box"):
        compute_detection_metrics(
            (np.empty((0, 4)),),
            (np.empty((0, 4)),),
            (np.array([]),),
        )


def test_registration_metrics_cover_tre_overlap_folding_and_inverse_consistency() -> None:
    fixed_landmarks = np.array([[0, 0, 0], [2, 2, 2]], dtype=float)
    transformed = np.array([[0, 0, 0], [3, 2, 2]], dtype=float)
    fixed_mask = np.zeros((3, 3, 3), dtype=np.uint8)
    fixed_mask[1, 1, 1] = 1
    warped_mask = fixed_mask.copy()
    jacobian = np.array([[[1.0, 0.8], [0.0, -0.2]]])
    inverse_error = np.array([0.1, 0.3, 0.2])

    metrics = compute_registration_metrics(
        fixed_landmarks=fixed_landmarks,
        transformed_landmarks=transformed,
        fixed_mask=fixed_mask,
        warped_mask=warped_mask,
        jacobian_determinant=jacobian,
        inverse_consistency_error=inverse_error,
    )

    assert metrics.tre_mean == pytest.approx(0.5)
    assert metrics.tre_maximum == pytest.approx(1.0)
    assert metrics.overlap_dice == 1.0
    assert metrics.folding_fraction == pytest.approx(0.5)
    assert metrics.inverse_consistency_mean == pytest.approx(0.2)


def test_registration_requires_paired_evidence_and_rejects_negative_inverse_error() -> None:
    with pytest.raises(ValidationError, match="Both fixed and transformed"):
        compute_registration_metrics(fixed_landmarks=np.zeros((1, 3)))
    with pytest.raises(ValidationError, match="cannot be negative"):
        compute_registration_metrics(inverse_consistency_error=np.array([0.1, -0.1]))
