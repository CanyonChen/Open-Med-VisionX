from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from workbench.errors import ValidationError
from workbench.evaluation import (
    create_brats_experiment_record,
    export_experiment_record,
)
from workbench.services import create_brats_experiment_record as service_record_builder
from workbench.services.brats_segmentation import (
    BraTSRegionEvaluation,
    BraTSSegmentationResult,
    ChannelNormalization,
)


def _result() -> BraTSSegmentationResult:
    probabilities = np.zeros((3, 4, 5, 6), dtype=np.float32)
    masks = np.zeros((3, 4, 5, 6), dtype=np.bool_)
    masks[0, 1:3, 1:4, 2:5] = True
    masks[1, 1:2, 2:4, 3:5] = True
    masks[2, 1, 2, 3] = True
    normalizations = tuple(
        ChannelNormalization(
            modality=modality,
            nonzero_voxels=100 + index,
            nonzero_mean=2.0 + index,
            nonzero_standard_deviation=0.5 + index,
        )
        for index, modality in enumerate(("T1ce", "T1", "T2", "FLAIR"))
    )
    evaluations = MappingProxyType(
        {
            region: BraTSRegionEvaluation(
                region=region,
                ground_truth_labels=labels,
                dice=0.8 + index * 0.01,
                hd95_mm=1.5 + index,
                prediction_volume_ml=1.2 + index,
                ground_truth_volume_ml=1.0 + index,
                signed_volume_error_ml=0.2,
                absolute_volume_error_ml=0.2,
            )
            for index, (region, labels) in enumerate(
                (("WT", (1, 2, 4)), ("TC", (1, 4)), ("ET", (4,)))
            )
        }
    )
    return BraTSSegmentationResult(
        case_alias="brats-2021-abcdef0123456789",
        probabilities=probabilities,
        masks=masks,
        source_affine=np.diag((1.0, 1.0, 1.0, 1.0)),
        canonical_affine=np.diag((1.0, 1.0, 1.0, 1.0)),
        source_orientation=("R", "A", "S"),
        input_channels=("T1ce", "T1", "T2", "FLAIR"),
        native_output_regions=("TC", "WT", "ET"),
        output_regions=("WT", "TC", "ET"),
        normalizations=normalizations,
        evaluations=evaluations,
        threshold=0.5,
        patch_size=(96, 96, 96),
        overlap=0.5,
        patch_count=8,
        device="cuda",
        model_elapsed_seconds=0.25,
        elapsed_seconds=0.5,
        model_sha256="f" * 64,
        fallback_reason=None,
        warnings=(
            "Domain shift: model training domain is BraTS 2018; the selected case and "
            "evaluation domain are BraTS 2021 Task 1.",
            "Education and research only; not for diagnosis.",
        ),
        input_artifact_sha256=tuple(
            (modality, character * 64)
            for modality, character in zip(
                ("T1ce", "T1", "T2", "FLAIR"),
                "abcd",
                strict=True,
            )
        ),
        ground_truth_artifact_sha256="e" * 64,
        requested_device="auto",
    )


def _walk(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def test_builder_records_bra_ts_provenance_metrics_and_only_references() -> None:
    created = datetime(2026, 7, 21, 12, 30, tzinfo=timezone.utc)

    record = create_brats_experiment_record(
        _result(),
        record_id="brats-run-001",
        created_at=created,
        code_revision="abc123def",
        dataset_manifest_id="brats-demo-v1",
    )
    payload = record.to_dict()

    assert service_record_builder is create_brats_experiment_record
    assert record.task == "brats-2021-task-1-segmentation"
    assert [item.artifact_id for item in record.inputs[:4]] == [
        "input-t1ce",
        "input-t1",
        "input-t2",
        "input-flair",
    ]
    assert [item.sha256 for item in record.inputs[:4]] == [value * 64 for value in "abcd"]
    assert record.inputs[4].artifact_id == "evaluation-reference-seg"
    assert record.parameters["model_sha256"] == "f" * 64
    assert record.parameters["requested_device"] == "auto"
    assert record.parameters["execution_device"] == "cuda"
    assert record.parameters["fallback_applied"] is False
    assert record.duration_ms == pytest.approx(500.0)
    assert tuple(record.metrics) == ("WT", "TC", "ET")
    assert record.metrics["WT"]["dice"] == pytest.approx(0.8)
    assert record.metrics["ET"]["hd95_mm"] == pytest.approx(3.5)
    assert len(record.outputs) == 3
    assert {item.shape for item in record.outputs} == {(4, 5, 6)}
    assert {item.label_schema["region"] for item in record.outputs} == {"WT", "TC", "ET"}
    assert all(len(item.sha256) == 64 for item in record.outputs)
    assert sum(step.name.startswith("nonzero-zscore-") for step in record.transforms) == 4
    assert "BraTS 2018" in " ".join(record.warnings)
    assert "BraTS 2021" in " ".join(record.warnings)

    flattened = tuple(_walk(payload))
    assert not any(isinstance(value, np.ndarray) for value in flattened)
    assert "probabilities" not in flattened
    assert "masks" not in flattened
    assert "pixels" not in flattened


def test_output_hashes_are_deterministic_and_cover_mask_content() -> None:
    first_result = _result()
    first = create_brats_experiment_record(
        first_result,
        record_id="first",
        created_at=datetime(2026, 7, 21, tzinfo=timezone.utc),
    )
    second = create_brats_experiment_record(
        first_result,
        record_id="second",
        created_at=datetime(2026, 7, 22, tzinfo=timezone.utc),
    )
    changed_masks = np.array(first_result.masks, copy=True)
    changed_masks[0, 0, 0, 0] = True
    changed = create_brats_experiment_record(
        replace(first_result, masks=changed_masks),
        record_id="changed",
        created_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
    )

    assert [item.sha256 for item in first.outputs] == [item.sha256 for item in second.outputs]
    assert first.outputs[0].sha256 != changed.outputs[0].sha256
    assert first.outputs[1].sha256 == changed.outputs[1].sha256


def test_json_export_is_atomic_path_free_and_contains_no_forbidden_role_term(
    tmp_path: Path,
) -> None:
    record = create_brats_experiment_record(
        _result(),
        record_id="brats-run-export",
        created_at=datetime(2026, 7, 21, tzinfo=timezone.utc),
    )
    destination = tmp_path / "brats-run.json"

    exported = export_experiment_record(destination, record)
    decoded = json.loads(exported.read_text(encoding="utf-8"))

    assert decoded["record_id"] == "brats-run-export"
    assert decoded["contains_phi"] is False
    encoded = exported.read_text(encoding="utf-8")
    assert str(tmp_path) not in encoded
    assert "case_t1ce.nii.gz" not in encoded
    assert "\u5b66\u751f" not in encoded
    assert not tuple(tmp_path.glob(".brats-run.json.*.tmp"))
    with pytest.raises(ValidationError, match="already exists"):
        export_experiment_record(destination, record)
    assert not tuple(tmp_path.glob(".brats-run.json.*.tmp"))


def test_builder_rejects_missing_run_bound_hashes() -> None:
    result = _result()
    with pytest.raises(ValidationError, match="T1ce, T1, T2, and FLAIR"):
        create_brats_experiment_record(replace(result, input_artifact_sha256=()))
    with pytest.raises(ValidationError, match="SEG artifact hash"):
        create_brats_experiment_record(replace(result, ground_truth_artifact_sha256=None))


@pytest.mark.parametrize(
    "unsafe_alias",
    (
        r"D:\private\case",
        "".join(chr(codepoint) for codepoint in (0x5B66, 0x751F)),
    ),
)
def test_builder_rejects_paths_and_nonanonymous_aliases(unsafe_alias: str) -> None:
    with pytest.raises(ValidationError, match="anonymous content-derived alias"):
        create_brats_experiment_record(replace(_result(), case_alias=unsafe_alias))
