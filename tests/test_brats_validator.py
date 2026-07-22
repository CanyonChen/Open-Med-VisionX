from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from workbench.cases.brats import (
    BRATS_2021_MODALITIES,
    BRATS_2021_OFFICIAL_ACQUISITION_URL,
    build_brats2021_manifest,
    validate_brats2021_case,
    write_brats2021_manifest,
)

nib = pytest.importorskip("nibabel")


def _write_volume(
    path: Path,
    values: np.ndarray,
    affine: np.ndarray,
    *,
    qform: np.ndarray | None = None,
    sform: np.ndarray | None = None,
    qform_code: int = 1,
    sform_code: int = 1,
) -> None:
    image = nib.Nifti1Image(values, affine)
    image.set_qform(affine if qform is None else qform, code=qform_code)
    image.set_sform(affine if sform is None else sform, code=sform_code)
    nib.save(image, path)


def _make_case(root: Path, *, seg_values: np.ndarray | None = None) -> Path:
    root.mkdir()
    affine = np.diag([1.0, 1.5, 2.0, 1.0])
    intensities = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    segmentation = np.zeros((4, 5, 6), dtype=np.int16)
    segmentation[0, 0, 0] = 1
    segmentation[1, 1, 1] = 2
    segmentation[2, 2, 2] = 4
    if seg_values is not None:
        segmentation = seg_values
    for token in ("t1", "t1ce", "t2", "flair"):
        _write_volume(root / f"Private_Name_MRN123_{token}.nii.gz", intensities, affine)
    _write_volume(root / "Private_Name_MRN123_seg.nii.gz", segmentation, affine)
    return root


def test_valid_case_is_complete_deterministic_and_path_free(tmp_path: Path) -> None:
    case = _make_case(tmp_path / "Sensitive-Case-Name")

    first = validate_brats2021_case(case)
    second = validate_brats2021_case(case)

    assert first == second
    assert first.is_valid
    assert tuple(record.modality for record in first.files) == BRATS_2021_MODALITIES
    assert first.case_alias.startswith("brats-2021-")
    assert first.case_alias != "brats-2021-unresolved"
    assert first.label_voxel_counts == {0: 117, 1: 1, 2: 1, 4: 1}
    assert first.file_for("t1ce") is not None
    assert all(record.relative_id.startswith("modalities/") for record in first.files)
    assert all(len(record.sha256) == 64 for record in first.files)

    report_text = repr(first)
    assert str(case) not in report_text
    assert "Sensitive-Case-Name" not in report_text
    assert "Private_Name" not in report_text
    assert "MRN123" not in report_text


def test_missing_duplicate_ambiguous_and_unrecognized_names_are_reported(tmp_path: Path) -> None:
    case = _make_case(tmp_path / "case")
    (case / "Private_Name_MRN123_seg.nii.gz").unlink()
    affine = np.eye(4)
    data = np.zeros((4, 5, 6), dtype=np.float32)
    _write_volume(case / "copy_t1.nii.gz", data, affine)
    _write_volume(case / "ambiguous_t2_flair.nii.gz", data, affine)
    _write_volume(case / "unknown.nii.gz", data, affine)

    report = validate_brats2021_case(case)

    assert not report.is_valid
    assert "missing_modality" in report.error_codes
    assert "duplicate_modality" in report.error_codes
    assert "ambiguous_modality_name" in report.error_codes
    assert any(issue.code == "unrecognized_nifti_name" for issue in report.issues)
    assert all("Private_Name" not in issue.message for issue in report.issues)


def test_corrupt_recognized_candidate_is_reported_without_exception(tmp_path: Path) -> None:
    case = _make_case(tmp_path / "case")
    t2 = case / "Private_Name_MRN123_t2.nii.gz"
    t2.write_bytes(b"not a nifti")

    report = validate_brats2021_case(case)

    assert not report.is_valid
    assert any(
        issue.code == "nifti_decode_failed" and issue.modality == "T2" for issue in report.issues
    )
    assert str(t2) not in repr(report)


def test_geometry_and_spatial_form_differences_block_installation(tmp_path: Path) -> None:
    case = _make_case(tmp_path / "case")
    t2 = case / "Private_Name_MRN123_t2.nii.gz"
    affine = np.diag([1.0, 1.5, 2.0, 1.0])
    translated = affine.copy()
    translated[0, 3] = 8.0
    conflicting_qform = translated.copy()
    conflicting_qform[1, 3] = 3.0
    values = np.zeros((4, 5, 6), dtype=np.float32)
    _write_volume(t2, values, translated, qform=conflicting_qform, sform=translated)

    report = validate_brats2021_case(case)

    assert not report.is_valid
    assert "qform_sform_conflict" in report.error_codes
    assert "affine_mismatch" in report.error_codes
    assert "world_coverage_mismatch" in report.error_codes
    assert "qform_mismatch" in report.error_codes
    assert "sform_mismatch" in report.error_codes


@pytest.mark.parametrize(
    ("segmentation", "expected_code"),
    [
        (np.zeros((4, 5, 6), dtype=np.float32), "seg_not_integer"),
        (np.full((4, 5, 6), 3, dtype=np.int16), "seg_invalid_labels"),
        (np.full((4, 5, 6), np.nan, dtype=np.float32), "seg_nonfinite"),
    ],
)
def test_invalid_segmentation_values_block_installation(
    tmp_path: Path,
    segmentation: np.ndarray,
    expected_code: str,
) -> None:
    case = _make_case(tmp_path / "case", seg_values=segmentation)

    report = validate_brats2021_case(case)

    assert not report.is_valid
    assert expected_code in report.error_codes


def test_manifest_is_anonymous_local_only_and_reproducible(tmp_path: Path) -> None:
    case = _make_case(tmp_path / "Jane-Doe-Case")
    report = validate_brats2021_case(case)
    installed_at = datetime(2026, 7, 21, 8, 9, 10, tzinfo=timezone.utc)

    first = build_brats2021_manifest(
        report,
        terms_confirmed_by_user=True,
        installed_at=installed_at,
    )
    second = build_brats2021_manifest(
        report,
        terms_confirmed_by_user=True,
        installed_at=installed_at,
    )

    assert first == second
    assert first["dataset"]["official_acquisition_url"] == BRATS_2021_OFFICIAL_ACQUISITION_URL
    assert first["source_data"] == {
        "copied": False,
        "uploaded": False,
        "storage": "user-managed-local-files",
    }
    assert first["installed_at_utc"] == "2026-07-21T08:09:10Z"
    assert first["segmentation"]["voxel_counts"] == {"0": 117, "1": 1, "2": 1, "4": 1}
    encoded = json.dumps(first, sort_keys=True)
    assert str(case) not in encoded
    assert "Jane-Doe-Case" not in encoded
    assert "Private_Name" not in encoded
    assert "MRN123" not in encoded
    assert ":\\" not in encoded
    assert all(not Path(item["relative_id"]).is_absolute() for item in first["files"])


def test_manifest_requires_valid_case_confirmation_and_aware_time(tmp_path: Path) -> None:
    valid = validate_brats2021_case(_make_case(tmp_path / "valid"))
    invalid = validate_brats2021_case(tmp_path / "missing")

    with pytest.raises(PermissionError):
        build_brats2021_manifest(valid, terms_confirmed_by_user=False)
    with pytest.raises(ValueError):
        build_brats2021_manifest(invalid, terms_confirmed_by_user=True)
    with pytest.raises(ValueError, match="timezone-aware"):
        build_brats2021_manifest(
            valid,
            terms_confirmed_by_user=True,
            installed_at=datetime(2026, 7, 21),
        )


def test_manifest_write_is_atomic_json_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    report = validate_brats2021_case(_make_case(tmp_path / "case"))
    destination = tmp_path / "install" / "brats-case.json"

    returned = write_brats2021_manifest(
        report,
        destination,
        terms_confirmed_by_user=True,
        installed_at=datetime(2026, 7, 21, tzinfo=timezone.utc),
    )

    assert returned == destination
    assert json.loads(destination.read_text(encoding="utf-8"))["case_alias"] == report.case_alias
    assert not list(destination.parent.glob(".*.tmp"))
