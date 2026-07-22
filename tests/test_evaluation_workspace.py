from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from workbench.errors import ValidationError
from workbench.evaluation.contracts import (
    DatasetManifest,
    DatasetSample,
    dataset_manifest_from_mapping,
    load_dataset_manifest,
    validate_group_splits,
)
from workbench.evaluation.metrics import (
    bootstrap_classification_interval,
    compute_binary_classification_metrics,
    compute_segmentation_metrics,
    threshold_sweep,
)
from workbench.evaluation.records import (
    ArtifactReference,
    ExperimentRecord,
    TransformStep,
    export_experiment_record,
)


def _sample(
    sample_id: str,
    group_id: str,
    split: str,
    *,
    digest_character: str,
) -> DatasetSample:
    return DatasetSample(
        sample_id=sample_id,
        group_id=group_id,
        split=split,
        artifact_sha256=digest_character * 64,
        modality="MR",
        labels={"target": 1},
        site="site-a",
        scanner="scanner-3t",
    )


def _record() -> ExperimentRecord:
    return ExperimentRecord(
        record_id="experiment-001",
        created_at=datetime(2026, 7, 21, 10, 30, tzinfo=timezone.utc),
        application_version="0.1.0",
        code_revision="abc123def",
        task="segmentation",
        model_id="baseline-unet",
        model_version="1.2",
        dataset_manifest_id="dataset-demo-v1",
        inputs=(
            ArtifactReference(
                artifact_id="input-volume-01",
                sha256="a" * 64,
                kind="volume",
                media_type="application/x-nifti",
                shape=(16, 32, 32),
                coordinate_system="RAS+",
            ),
        ),
        transforms=(
            TransformStep(
                name="normalize-zscore",
                version="1",
                parameters={"mean": 0.0, "standard_deviation": 1.0},
            ),
        ),
        outputs=(
            ArtifactReference(
                artifact_id="output-mask-01",
                sha256="b" * 64,
                kind="segmentation",
                media_type="application/x-nifti",
                shape=(16, 32, 32),
                coordinate_system="RAS+",
                label_schema={"foreground": 1},
            ),
        ),
        parameters={"threshold": 0.5, "device": "cuda"},
        metrics={"dice": 0.875, "confidence_interval": {"lower": 0.81, "upper": 0.91}},
        warnings=("CPU/GPU results may differ within the stated tolerance.",),
        duration_ms=125.5,
    )


def test_dataset_manifest_round_trip_loads_without_retaining_source_path(
    tmp_path: Path,
) -> None:
    manifest = DatasetManifest(
        dataset_id="demo-set",
        dataset_version="v1",
        task="classification",
        license_id="CC-BY-4.0",
        samples=(
            _sample("sample-1", "group-a", "train", digest_character="a"),
            _sample("sample-2", "group-b", "test", digest_character="b"),
        ),
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

    loaded = load_dataset_manifest(path)

    assert loaded == manifest
    assert str(path) not in repr(loaded)


def test_dataset_manifest_parser_rejects_unknown_fields_and_missing_consent() -> None:
    payload = {
        "dataset_id": "demo-set",
        "dataset_version": "v1",
        "task": "classification",
        "license_id": "CC-BY-4.0",
        "deidentified": True,
        "samples": [],
        "unexpected": "value",
    }
    with pytest.raises(ValidationError, match="unknown fields"):
        dataset_manifest_from_mapping(payload)

    payload.pop("unexpected")
    payload["samples"] = [
        {
            "sample_id": "sample-1",
            "group_id": "group-a",
            "split": "train",
            "artifact_sha256": "a" * 64,
            "modality": "MR",
        }
    ]
    with pytest.raises(ValidationError, match="explicitly marked deidentified"):
        dataset_manifest_from_mapping(payload)


def test_anonymous_manifest_is_immutable_and_reports_group_splits() -> None:
    train_a = _sample("sample-a1", "group-a", "train", digest_character="a")
    train_b = _sample("sample-a2", "group-a", "train", digest_character="b")
    test = _sample("sample-b1", "group-b", "test", digest_character="c")
    manifest = DatasetManifest(
        dataset_id="demo-cohort",
        dataset_version="v1",
        task="binary-classification",
        license_id="CC-BY-4.0",
        samples=(train_a, train_b, test),
        label_schema={"target": {"negative": 0, "positive": 1}},
        provenance={"source_url": "https://example.org/dataset", "revision": "2026-07"},
    )

    report = manifest.split_report
    assert report.sample_counts == {"test": 1, "train": 2, "validation": 0}
    assert report.group_counts == {"test": 1, "train": 1, "validation": 0}
    assert report.total_samples == 3
    assert report.total_groups == 2
    assert "path" not in json.dumps(manifest.to_dict()).lower()
    with pytest.raises(TypeError):
        train_a.labels["target"] = 0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        manifest.dataset_id = "changed"  # type: ignore[misc]


def test_group_level_split_leakage_and_duplicate_samples_are_rejected() -> None:
    train = _sample("sample-a", "group-a", "train", digest_character="a")
    leaked = _sample("sample-b", "group-a", "test", digest_character="b")
    with pytest.raises(ValidationError, match="leaks across"):
        validate_group_splits((train, leaked))
    with pytest.raises(ValidationError, match="Duplicate sample_id"):
        validate_group_splits((train, train))
    with pytest.raises(ValidationError, match="leaks across"):
        DatasetManifest(
            dataset_id="leaky",
            dataset_version="v1",
            task="classification",
            license_id="MIT",
            samples=(train, leaked),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("labels", {"patient_id": "123"}),
        ("labels", {"source_path": "safe-looking"}),
        ("labels", {"note": "C:\\private\\scan.nii.gz"}),
    ],
)
def test_dataset_sample_rejects_identity_fields_and_paths(
    field: str,
    value: object,
) -> None:
    arguments = {
        "sample_id": "sample-safe",
        "group_id": "group-safe",
        "split": "train",
        "artifact_sha256": "a" * 64,
        "modality": "CT",
        "labels": {},
    }
    arguments[field] = value
    with pytest.raises(ValidationError):
        DatasetSample(**arguments)  # type: ignore[arg-type]


def test_binary_classification_metrics_include_calibration_and_intervals() -> None:
    truth = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.05, 0.20, 0.45, 0.60, 0.80, 0.95])

    report = compute_binary_classification_metrics(
        truth,
        scores,
        threshold=0.5,
        n_bins=5,
    )

    assert report.confusion.true_positive == 3
    assert report.confusion.true_negative == 3
    assert report.accuracy == pytest.approx(1.0)
    assert report.sensitivity == pytest.approx(1.0)
    assert report.specificity == pytest.approx(1.0)
    assert report.auroc == pytest.approx(1.0)
    assert report.average_precision == pytest.approx(1.0)
    assert report.brier_score == pytest.approx(np.mean((scores - truth) ** 2))
    assert 0.0 <= report.expected_calibration_error <= 1.0
    assert sum(calibration_bin.count for calibration_bin in report.calibration_bins) == len(truth)
    assert report.confidence_intervals["accuracy"].lower < 1.0
    assert report.confidence_intervals["accuracy"].upper == pytest.approx(1.0)
    with pytest.raises(TypeError):
        report.confidence_intervals["new"] = report.confidence_intervals["accuracy"]  # type: ignore[index]


def test_ties_threshold_sweep_and_bootstrap_interval_are_deterministic() -> None:
    truth = [0, 1, 0, 1, 0, 1, 0, 1]
    scores = [0.1, 0.9, 0.4, 0.6, 0.4, 0.6, 0.2, 0.8]
    report = compute_binary_classification_metrics(truth, scores)
    assert report.auroc == pytest.approx(1.0)

    points = threshold_sweep(truth, scores, (0.25, 0.5, 0.75))
    assert [point.threshold for point in points] == [0.25, 0.5, 0.75]
    assert points[0].sensitivity >= points[1].sensitivity >= points[2].sensitivity
    assert points[0].specificity <= points[1].specificity <= points[2].specificity

    first = bootstrap_classification_interval(
        truth,
        scores,
        metric="brier_score",
        n_resamples=200,
        random_seed=17,
    )
    second = bootstrap_classification_interval(
        truth,
        scores,
        metric="brier_score",
        n_resamples=200,
        random_seed=17,
    )
    assert first == second
    assert first.lower <= first.estimate <= first.upper
    assert first.method == "stratified percentile bootstrap (200 resamples)"


@pytest.mark.parametrize(
    ("truth", "scores", "message"),
    [
        ([0, 0], [0.1, 0.2], "both outcome classes"),
        ([0, 2], [0.1, 0.2], "binary labels"),
        ([0, 1], [0.1, 1.2], "range"),
    ],
)
def test_classification_metrics_reject_invalid_inputs(
    truth: list[int],
    scores: list[float],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        compute_binary_classification_metrics(truth, scores)


def test_segmentation_overlap_hd95_and_surface_metrics_use_spacing() -> None:
    reference = np.zeros((8, 8), dtype=np.uint8)
    prediction = np.zeros_like(reference)
    reference[2:6, 2:6] = 1
    prediction[2:6, 3:7] = 1

    report = compute_segmentation_metrics(
        reference,
        prediction,
        spacing=(2.0, 1.0),
        surface_tolerance=1.0,
    )

    assert report.dice == pytest.approx(0.75)
    assert report.intersection_over_union == pytest.approx(0.6)
    assert report.hd95 == pytest.approx(1.0)
    assert report.average_symmetric_surface_distance is not None
    assert 0.0 < report.average_symmetric_surface_distance <= report.hd95
    assert 0.0 < report.surface_dice <= 1.0
    assert report.distance_unit == "mm"


def test_segmentation_empty_mask_policy_is_explicit_and_stable() -> None:
    empty = np.zeros((4, 4, 4), dtype=np.uint8)
    foreground = empty.copy()
    foreground[1, 1, 1] = 1

    both_empty = compute_segmentation_metrics(empty, empty)
    assert both_empty.dice == 1.0
    assert both_empty.hd95 == 0.0
    assert both_empty.surface_dice == 1.0

    one_empty = compute_segmentation_metrics(empty, foreground)
    assert one_empty.dice == 0.0
    assert one_empty.intersection_over_union == 0.0
    assert one_empty.hd95 is None
    assert one_empty.average_symmetric_surface_distance is None
    assert one_empty.surface_dice == 0.0


def test_experiment_record_is_deeply_immutable_and_path_free() -> None:
    record = _record()
    detached = record.to_dict()

    assert detached["schema"] == "openmedvisionx-experiment-record/v1"
    assert detached["created_at"].endswith("Z")
    assert detached["contains_phi"] is False
    assert "path" not in json.dumps(detached).lower()
    detached["metrics"]["dice"] = 0.0
    assert record.metrics["dice"] == pytest.approx(0.875)
    with pytest.raises(TypeError):
        record.metrics["dice"] = 0.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        record.task = "changed"  # type: ignore[misc]


def test_experiment_record_rejects_phi_fields_and_local_paths() -> None:
    base = _record()
    common = {
        "record_id": base.record_id,
        "created_at": base.created_at,
        "application_version": base.application_version,
        "code_revision": base.code_revision,
        "task": base.task,
        "model_id": base.model_id,
        "model_version": base.model_version,
        "inputs": base.inputs,
        "transforms": base.transforms,
        "outputs": base.outputs,
        "metrics": base.metrics,
    }
    with pytest.raises(ValidationError, match="identity or a path"):
        ExperimentRecord(**common, parameters={"patient_name": "Example"})
    with pytest.raises(ValidationError, match="path"):
        ExperimentRecord(**common, parameters={"config": "D:\\secret\\config.yaml"})
    with pytest.raises(ValidationError, match="PHI"):
        ExperimentRecord(**common, contains_phi=True)


def test_json_and_yaml_exports_are_atomic_safe_and_never_overwrite_by_default(
    tmp_path: Path,
) -> None:
    record = _record()
    json_path = export_experiment_record(tmp_path / "record.json", record)
    yaml_path = export_experiment_record(tmp_path / "record.yaml", record)

    json_payload = json.loads(json_path.read_text(encoding="utf-8"))
    yaml_payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert json_payload == yaml_payload
    assert json_payload["metrics"]["dice"] == pytest.approx(0.875)
    assert not list(tmp_path.glob(".*.tmp"))

    json_path.write_text("original", encoding="utf-8")
    with pytest.raises(ValidationError, match="already exists"):
        export_experiment_record(json_path, record)
    assert json_path.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob(".*.tmp"))


def test_explicit_atomic_overwrite_replaces_the_complete_record(tmp_path: Path) -> None:
    target = tmp_path / "record.yml"
    target.write_text("old", encoding="utf-8")

    export_experiment_record(target, _record(), overwrite=True)

    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert payload["record_id"] == "experiment-001"
    assert not list(tmp_path.glob(".*.tmp"))
