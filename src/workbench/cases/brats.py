"""Privacy-minimized validation for a user-provided BraTS 2021 Task 1 case.

This module deliberately has no download capability.  It validates files that a
user obtained through the official BraTS process and can write an anonymous
local manifest without copying the source volumes or retaining source paths.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from ..errors import ResourceLimitError

BRATS_2021_OFFICIAL_ACQUISITION_URL = "https://www.med.upenn.edu/cbica/brats2021/"
BRATS_2021_MODALITIES = ("T1", "T1CE", "T2", "FLAIR", "SEG")
BRATS_2021_SEGMENTATION_LABELS = (0, 1, 2, 4)

_MODALITY_TOKENS = {
    "t1": "T1",
    "t1ce": "T1CE",
    "t2": "T2",
    "flair": "FLAIR",
    "seg": "SEG",
}
_MODALITY_PATTERN = re.compile(r"(?<![a-z0-9])(t1ce|t1|t2|flair|seg)(?![a-z0-9])")
_MATRIX_ATOL = 1e-4
_MATRIX_RTOL = 1e-5
_MAX_FILE_BYTES = 2 * 1024**3
_MAX_VOXELS = 512 * 1024**2


@dataclass(frozen=True, slots=True)
class BraTS2021Issue:
    """A stable, path-free validation finding."""

    code: str
    severity: Literal["error", "warning"]
    message: str
    modality: str | None = None
    candidate_id: str | None = None


@dataclass(frozen=True, slots=True)
class BraTS2021Geometry:
    """Serializable spatial metadata used for cross-modality checks."""

    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    affine: tuple[tuple[float, float, float, float], ...]
    orientation: tuple[str, str, str]
    world_bounds_mm: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    qform_code: int
    qform: tuple[tuple[float, float, float, float], ...] | None
    sform_code: int
    sform: tuple[tuple[float, float, float, float], ...] | None

    def to_manifest(self) -> dict[str, Any]:
        """Return a JSON-compatible geometry summary."""

        return {
            "shape": list(self.shape),
            "spacing_mm": list(self.spacing_mm),
            "affine": [list(row) for row in self.affine],
            "orientation": list(self.orientation),
            "world_bounds_mm": [list(bounds) for bounds in self.world_bounds_mm],
            "qform": {
                "code": self.qform_code,
                "matrix": None if self.qform is None else [list(row) for row in self.qform],
            },
            "sform": {
                "code": self.sform_code,
                "matrix": None if self.sform is None else [list(row) for row in self.sform],
            },
        }


@dataclass(frozen=True, slots=True)
class BraTS2021FileRecord:
    """An anonymous content record for one required modality."""

    modality: str
    relative_id: str
    sha256: str
    size_bytes: int
    geometry: BraTS2021Geometry


@dataclass(frozen=True, slots=True)
class BraTS2021ValidationReport:
    """Complete deterministic result of validating one selected directory."""

    case_alias: str
    files: tuple[BraTS2021FileRecord, ...]
    issues: tuple[BraTS2021Issue, ...]
    segmentation_counts: tuple[tuple[int, int], ...]

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def error_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues if issue.severity == "error")

    @property
    def label_voxel_counts(self) -> dict[int, int]:
        return dict(self.segmentation_counts)

    def file_for(self, modality: str) -> BraTS2021FileRecord | None:
        normalized = str(modality).upper()
        return next((record for record in self.files if record.modality == normalized), None)


@dataclass(frozen=True, slots=True, eq=False)
class BraTS2021LocalCase:
    """Path-free, in-memory view of a freshly revalidated local case.

    Image arrays are reoriented to canonical RAS+ for model preprocessing.  The
    original affine and the inverse orientation transform are retained only so
    derived arrays can be mapped back to the source voxel grid.  No source path
    or filename is retained by this object.
    """

    case_alias: str
    modalities: Mapping[str, np.ndarray] = dataclass_field(repr=False)
    segmentation: np.ndarray | None = dataclass_field(repr=False)
    source_affine: np.ndarray = dataclass_field(repr=False)
    canonical_affine: np.ndarray = dataclass_field(repr=False)
    canonical_to_source_orientation: np.ndarray = dataclass_field(repr=False)
    source_orientation: tuple[str, str, str]
    artifact_sha256: Mapping[str, str] = dataclass_field(repr=False)

    def modality(self, name: str) -> np.ndarray:
        """Return one canonical RAS+ modality by its case-insensitive name."""

        normalized = str(name).upper()
        try:
            return self.modalities[normalized]
        except KeyError as exc:
            raise KeyError(f"Unknown BraTS modality: {name!r}") from exc


class BraTS2021CaseLoadError(ValueError):
    """A freshly selected local case could not cross the safe load boundary."""


def _issue(
    code: str,
    message: str,
    *,
    severity: Literal["error", "warning"] = "error",
    modality: str | None = None,
    candidate_id: str | None = None,
) -> BraTS2021Issue:
    return BraTS2021Issue(code, severity, message, modality, candidate_id)


def _is_nifti(path: Path) -> bool:
    name = path.name.casefold()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _modality_tokens(path: Path) -> tuple[str, ...]:
    name = path.name.casefold()
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    return tuple(_MODALITY_TOKENS[token] for token in _MODALITY_PATTERN.findall(name))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix_tuple(value: Any) -> tuple[tuple[float, float, float, float], ...]:
    matrix = np.asarray(value, dtype=np.float64)
    return tuple(tuple(float(item) for item in row) for row in matrix)


def _world_bounds(
    shape: tuple[int, int, int], affine: np.ndarray
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    corners = np.asarray(
        [(*corner, 1.0) for corner in itertools.product(*((0, size - 1) for size in shape))],
        dtype=np.float64,
    )
    world = (affine @ corners.T).T[:, :3]
    return tuple(
        (float(np.min(world[:, axis])), float(np.max(world[:, axis]))) for axis in range(3)
    )  # type: ignore[return-value]


def _geometry_from_image(image: Any, nib: Any) -> BraTS2021Geometry:
    shape = tuple(int(size) for size in image.shape)
    if len(shape) != 3 or any(size <= 0 for size in shape):
        raise ValueError("empty_or_non_3d_geometry")
    voxel_count = int(np.prod(shape, dtype=np.int64))
    if voxel_count > _MAX_VOXELS:
        raise OverflowError("voxel_limit")

    affine = np.asarray(image.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError("invalid_affine")
    if abs(float(np.linalg.det(affine[:3, :3]))) <= np.finfo(np.float64).eps:
        raise ValueError("degenerate_affine")

    spacing = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("invalid_spacing")
    affine_spacing = np.asarray(nib.affines.voxel_sizes(affine), dtype=np.float64)
    if not np.allclose(spacing, affine_spacing, rtol=_MATRIX_RTOL, atol=_MATRIX_ATOL):
        raise ValueError("spacing_affine_mismatch")

    orientation_values = nib.orientations.aff2axcodes(affine)
    if len(orientation_values) != 3 or any(value is None for value in orientation_values):
        raise ValueError("invalid_orientation")
    orientation = tuple(str(value) for value in orientation_values)

    qform_value, qform_code_value = image.get_qform(coded=True)
    sform_value, sform_code_value = image.get_sform(coded=True)
    qform_code = int(qform_code_value)
    sform_code = int(sform_code_value)
    qform = None if qform_value is None else np.asarray(qform_value, dtype=np.float64)
    sform = None if sform_value is None else np.asarray(sform_value, dtype=np.float64)
    for form in (qform, sform):
        if form is not None and (form.shape != (4, 4) or not np.all(np.isfinite(form))):
            raise ValueError("invalid_spatial_form")

    return BraTS2021Geometry(
        shape=shape,  # type: ignore[arg-type]
        spacing_mm=tuple(float(value) for value in spacing),  # type: ignore[arg-type]
        affine=_matrix_tuple(affine),
        orientation=orientation,  # type: ignore[arg-type]
        world_bounds_mm=_world_bounds(shape, affine),  # type: ignore[arg-type]
        qform_code=qform_code,
        qform=None if qform is None else _matrix_tuple(qform),
        sform_code=sform_code,
        sform=None if sform is None else _matrix_tuple(sform),
    )


def _form_matches(
    left: tuple[tuple[float, float, float, float], ...] | None,
    right: tuple[tuple[float, float, float, float], ...] | None,
) -> bool:
    if left is None or right is None:
        return left is right
    return bool(np.allclose(left, right, rtol=_MATRIX_RTOL, atol=_MATRIX_ATOL))


def _compare_geometry(
    reference: BraTS2021Geometry,
    current: BraTS2021Geometry,
    modality: str,
) -> list[BraTS2021Issue]:
    issues: list[BraTS2021Issue] = []
    checks = (
        ("shape_mismatch", reference.shape == current.shape, "shape"),
        (
            "spacing_mismatch",
            np.allclose(
                reference.spacing_mm,
                current.spacing_mm,
                rtol=_MATRIX_RTOL,
                atol=_MATRIX_ATOL,
            ),
            "voxel spacing",
        ),
        (
            "affine_mismatch",
            np.allclose(
                reference.affine,
                current.affine,
                rtol=_MATRIX_RTOL,
                atol=_MATRIX_ATOL,
            ),
            "affine",
        ),
        ("orientation_mismatch", reference.orientation == current.orientation, "orientation"),
        (
            "world_coverage_mismatch",
            np.allclose(
                reference.world_bounds_mm,
                current.world_bounds_mm,
                rtol=_MATRIX_RTOL,
                atol=_MATRIX_ATOL,
            ),
            "world-space coverage",
        ),
        ("qform_code_mismatch", reference.qform_code == current.qform_code, "qform code"),
        ("qform_mismatch", _form_matches(reference.qform, current.qform), "qform"),
        ("sform_code_mismatch", reference.sform_code == current.sform_code, "sform code"),
        ("sform_mismatch", _form_matches(reference.sform, current.sform), "sform"),
    )
    for code, matches, label in checks:
        if not bool(matches):
            issues.append(
                _issue(
                    code,
                    f"{modality} {label} does not match the T1 reference geometry.",
                    modality=modality,
                )
            )
    return issues


def _segmentation_counts(image: Any) -> tuple[tuple[tuple[int, int], ...], list[BraTS2021Issue]]:
    issues: list[BraTS2021Issue] = []
    disk_dtype = np.dtype(image.get_data_dtype())
    try:
        values = np.asanyarray(image.dataobj)
    except (OSError, ValueError, TypeError, MemoryError):
        return (), [
            _issue("seg_decode_failed", "SEG voxel data could not be decoded.", modality="SEG")
        ]
    try:
        finite = bool(np.all(np.isfinite(values)))
    except TypeError:
        finite = False
    if not finite:
        issues.append(
            _issue("seg_nonfinite", "SEG contains NaN or infinite values.", modality="SEG")
        )
        return (), issues
    if not np.issubdtype(disk_dtype, np.integer) or not np.issubdtype(values.dtype, np.integer):
        issues.append(
            _issue(
                "seg_not_integer",
                "SEG must use an integer NIfTI data type with unscaled integer voxels.",
                modality="SEG",
            )
        )
        return (), issues

    labels, counts = np.unique(values, return_counts=True)
    actual = {int(label): int(count) for label, count in zip(labels, counts, strict=True)}
    invalid = sorted(set(actual) - set(BRATS_2021_SEGMENTATION_LABELS))
    if invalid:
        issues.append(
            _issue(
                "seg_invalid_labels",
                f"SEG contains labels outside {{0, 1, 2, 4}}: {invalid}.",
                modality="SEG",
            )
        )
    result = tuple((label, actual.get(label, 0)) for label in BRATS_2021_SEGMENTATION_LABELS)
    return result, issues


def _anonymous_alias(records: list[BraTS2021FileRecord]) -> str:
    if not records:
        return "brats-2021-unresolved"
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: BRATS_2021_MODALITIES.index(item.modality)):
        digest.update(record.modality.encode("ascii"))
        digest.update(bytes.fromhex(record.sha256))
    return f"brats-2021-{digest.hexdigest()[:16]}"


def _discover_modality_paths(
    root: Path,
    *,
    require_segmentation: bool,
) -> tuple[dict[str, tuple[str, Path]], list[BraTS2021Issue]]:
    nifti_paths = sorted(
        (path for path in root.rglob("*") if path.is_file() and _is_nifti(path)),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )
    candidates: dict[str, list[tuple[str, Path]]] = {name: [] for name in BRATS_2021_MODALITIES}
    issues: list[BraTS2021Issue] = []
    for index, path in enumerate(nifti_paths, start=1):
        candidate_id = f"candidate-{index:03d}"
        try:
            path.resolve().relative_to(root)
        except (OSError, ValueError):
            issues.append(
                _issue(
                    "candidate_outside_directory",
                    "A NIfTI candidate resolves outside the selected directory.",
                    candidate_id=candidate_id,
                )
            )
            continue
        tokens = _modality_tokens(path)
        if not tokens:
            issues.append(
                _issue(
                    "unrecognized_nifti_name",
                    "A NIfTI file does not identify a required BraTS modality.",
                    severity="warning",
                    candidate_id=candidate_id,
                )
            )
            continue
        if len(tokens) != 1:
            issues.append(
                _issue(
                    "ambiguous_modality_name",
                    "A NIfTI filename identifies more than one modality token.",
                    candidate_id=candidate_id,
                )
            )
            continue
        candidates[tokens[0]].append((candidate_id, path))

    selected: dict[str, tuple[str, Path]] = {}
    for modality in BRATS_2021_MODALITIES:
        matches = candidates[modality]
        if not matches:
            if modality == "SEG" and not require_segmentation:
                continue
            issues.append(
                _issue(
                    "missing_modality",
                    f"Required modality {modality} is missing.",
                    modality=modality,
                )
            )
        elif len(matches) > 1:
            issues.append(
                _issue(
                    "duplicate_modality",
                    f"Required modality {modality} has {len(matches)} candidates.",
                    modality=modality,
                )
            )
        else:
            selected[modality] = matches[0]
    return selected, issues


def validate_brats2021_case(
    directory: str | Path,
    *,
    require_segmentation: bool = True,
) -> BraTS2021ValidationReport:
    """Validate a local Task 1 case without downloading, copying, or uploading data.

    The returned report never includes the selected directory or source filenames.
    Candidate identifiers are deterministic ordinals, and a valid case alias is
    derived from content hashes rather than any person- or site-provided text.
    """

    supplied_root = Path(directory)
    if not supplied_root.is_dir():
        return BraTS2021ValidationReport(
            case_alias="brats-2021-unresolved",
            files=(),
            issues=(
                _issue(
                    "directory_unavailable", "The selected BraTS case directory is unavailable."
                ),
            ),
            segmentation_counts=(),
        )

    root = supplied_root.resolve()
    selected, issues = _discover_modality_paths(
        root,
        require_segmentation=require_segmentation,
    )

    try:
        import nibabel as nib
    except ImportError:
        issues.append(
            _issue(
                "nibabel_unavailable",
                "BraTS validation requires nibabel from the NIfTI dependency group.",
            )
        )
        return BraTS2021ValidationReport(
            case_alias="brats-2021-unresolved",
            files=(),
            issues=tuple(issues),
            segmentation_counts=(),
        )

    records: list[BraTS2021FileRecord] = []
    segmentation_counts: tuple[tuple[int, int], ...] = ()
    for modality in BRATS_2021_MODALITIES:
        candidate = selected.get(modality)
        if candidate is None:
            continue
        candidate_id, path = candidate
        try:
            before = path.stat()
        except OSError:
            issues.append(
                _issue(
                    "candidate_unreadable",
                    "A required NIfTI candidate cannot be read.",
                    modality=modality,
                    candidate_id=candidate_id,
                )
            )
            continue
        if before.st_size <= 0:
            issues.append(
                _issue(
                    "candidate_empty",
                    "A required NIfTI candidate is empty.",
                    modality=modality,
                    candidate_id=candidate_id,
                )
            )
            continue
        if before.st_size > _MAX_FILE_BYTES:
            issues.append(
                _issue(
                    "candidate_too_large",
                    "A required NIfTI candidate exceeds the 2 GiB safety limit.",
                    modality=modality,
                    candidate_id=candidate_id,
                )
            )
            continue
        try:
            digest = _sha256(path)
            image = nib.load(str(path), mmap=True, keep_file_open=False)
            if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
                raise TypeError("not_nifti")
            geometry = _geometry_from_image(image, nib)
            if (
                geometry.qform is not None
                and geometry.sform is not None
                and not _form_matches(geometry.qform, geometry.sform)
            ):
                issues.append(
                    _issue(
                        "qform_sform_conflict",
                        f"{modality} qform and sform describe different geometry.",
                        modality=modality,
                        candidate_id=candidate_id,
                    )
                )
            if geometry.qform is None:
                issues.append(
                    _issue(
                        "qform_missing",
                        f"{modality} has no coded qform; affine and sform checks remain active.",
                        severity="warning",
                        modality=modality,
                        candidate_id=candidate_id,
                    )
                )
            if geometry.sform is None:
                issues.append(
                    _issue(
                        "sform_missing",
                        f"{modality} has no coded sform; affine and qform checks remain active.",
                        severity="warning",
                        modality=modality,
                        candidate_id=candidate_id,
                    )
                )
            if modality == "SEG":
                segmentation_counts, segmentation_issues = _segmentation_counts(image)
                issues.extend(segmentation_issues)
            after = path.stat()
        except OverflowError:
            issues.append(
                _issue(
                    "voxel_limit_exceeded",
                    "A required NIfTI candidate exceeds the voxel safety limit.",
                    modality=modality,
                    candidate_id=candidate_id,
                )
            )
            continue
        except (OSError, ValueError, TypeError, MemoryError, nib.filebasedimages.ImageFileError):
            issues.append(
                _issue(
                    "nifti_decode_failed",
                    "A required candidate is not a supported, valid 3D NIfTI volume.",
                    modality=modality,
                    candidate_id=candidate_id,
                )
            )
            continue
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            issues.append(
                _issue(
                    "file_changed_during_validation",
                    "A required NIfTI candidate changed during validation.",
                    modality=modality,
                    candidate_id=candidate_id,
                )
            )
            continue
        records.append(
            BraTS2021FileRecord(
                modality=modality,
                relative_id=f"modalities/{modality.casefold()}",
                sha256=digest,
                size_bytes=before.st_size,
                geometry=geometry,
            )
        )

    by_modality = {record.modality: record for record in records}
    reference = by_modality.get("T1")
    if reference is not None:
        for modality in BRATS_2021_MODALITIES[1:]:
            record = by_modality.get(modality)
            if record is not None:
                issues.extend(_compare_geometry(reference.geometry, record.geometry, modality))

    return BraTS2021ValidationReport(
        case_alias=_anonymous_alias(records),
        files=tuple(records),
        issues=tuple(issues),
        segmentation_counts=segmentation_counts,
    )


def _readonly_owned(value: Any, *, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def load_brats2021_local_case(
    directory: str | Path,
    *,
    require_segmentation: bool = False,
    max_voxels: int = _MAX_VOXELS,
) -> BraTS2021LocalCase:
    """Revalidate and decode a local BraTS case without retaining its paths.

    Four MRI modalities are mandatory.  Task 1 ``SEG`` is optional by default
    because it is evaluation ground truth rather than a model input.  Every
    decoded array is an owned, read-only array in canonical RAS+ orientation.
    Source files are hashed again immediately before decoding so a changed case
    never inherits an earlier validation result.
    """

    if isinstance(max_voxels, bool) or not isinstance(max_voxels, int) or max_voxels <= 0:
        raise ValueError("max_voxels must be a positive integer")
    report = validate_brats2021_case(
        directory,
        require_segmentation=require_segmentation,
    )
    if not report.is_valid:
        codes = ", ".join(report.error_codes) or "unknown_validation_error"
        raise BraTS2021CaseLoadError(f"BraTS case validation failed: {codes}.")

    root = Path(directory).resolve()
    selected, discovery_issues = _discover_modality_paths(
        root,
        require_segmentation=require_segmentation,
    )
    discovery_errors = tuple(issue.code for issue in discovery_issues if issue.severity == "error")
    if discovery_errors:
        raise BraTS2021CaseLoadError(
            f"BraTS case changed after validation: {', '.join(discovery_errors)}."
        )

    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - validator reports this first
        raise BraTS2021CaseLoadError("BraTS case loading requires nibabel.") from exc

    records = {record.modality: record for record in report.files}
    decoded: dict[str, np.ndarray] = {}
    segmentation: np.ndarray | None = None
    source_affine: np.ndarray | None = None
    canonical_affine: np.ndarray | None = None
    source_orientation: tuple[str, str, str] | None = None
    canonical_to_source: np.ndarray | None = None
    input_modalities = ("T1", "T1CE", "T2", "FLAIR")
    decode_order = (*input_modalities, *(("SEG",) if "SEG" in selected else ()))
    for modality in decode_order:
        candidate = selected.get(modality)
        expected = records.get(modality)
        if candidate is None or expected is None:
            raise BraTS2021CaseLoadError(
                f"BraTS case changed after validation: {modality} is unavailable."
            )
        _, path = candidate
        if int(np.prod(expected.geometry.shape, dtype=np.int64)) > max_voxels:
            raise ResourceLimitError(f"BraTS {modality} exceeds the configured voxel safety limit.")
        try:
            before = path.stat()
            if _sha256(path) != expected.sha256:
                raise BraTS2021CaseLoadError(
                    f"BraTS case changed after validation: {modality} hash mismatch."
                )
            image = nib.load(str(path), mmap=True, keep_file_open=False)
            original_affine = np.asarray(image.affine, dtype=np.float64)
            if not np.allclose(
                original_affine,
                expected.geometry.affine,
                rtol=_MATRIX_RTOL,
                atol=_MATRIX_ATOL,
            ):
                raise BraTS2021CaseLoadError(
                    f"BraTS case changed after validation: {modality} affine mismatch."
                )
            canonical = nib.as_closest_canonical(image)
            current_canonical_affine = np.asarray(canonical.affine, dtype=np.float64)
            orientation_values = nib.orientations.aff2axcodes(current_canonical_affine)
            if tuple(orientation_values) != ("R", "A", "S"):
                raise BraTS2021CaseLoadError(
                    f"BraTS {modality} could not be represented as canonical RAS+."
                )
            values = np.asanyarray(canonical.dataobj)
            after = path.stat()
        except BraTS2021CaseLoadError:
            raise
        except (
            OSError,
            ValueError,
            TypeError,
            MemoryError,
            nib.filebasedimages.ImageFileError,
        ) as exc:
            raise BraTS2021CaseLoadError(
                f"BraTS {modality} could not be decoded after validation."
            ) from exc
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise BraTS2021CaseLoadError(f"BraTS case changed while {modality} was being decoded.")
        if values.ndim != 3 or tuple(values.shape) != tuple(canonical.shape):
            raise BraTS2021CaseLoadError(f"BraTS {modality} is not a valid 3-D volume.")
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise BraTS2021CaseLoadError(f"BraTS {modality} contains invalid voxel values.")

        if canonical_affine is None:
            canonical_affine = current_canonical_affine
        elif tuple(values.shape) != next(iter(decoded.values())).shape or not np.allclose(
            canonical_affine,
            current_canonical_affine,
            rtol=_MATRIX_RTOL,
            atol=_MATRIX_ATOL,
        ):
            raise BraTS2021CaseLoadError(
                f"BraTS {modality} geometry is inconsistent after RAS+ canonicalization."
            )

        if modality == "T1":
            source_affine = original_affine
            source_codes = nib.orientations.aff2axcodes(source_affine)
            source_orientation = tuple(str(code) for code in source_codes)
            source_ornt = nib.orientations.io_orientation(source_affine)
            ras_ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))
            canonical_to_source = nib.orientations.ornt_transform(ras_ornt, source_ornt)

        if modality == "SEG":
            if not np.issubdtype(values.dtype, np.integer):
                raise BraTS2021CaseLoadError("BraTS SEG must contain integer labels.")
            if not np.isin(values, BRATS_2021_SEGMENTATION_LABELS).all():
                raise BraTS2021CaseLoadError("BraTS SEG contains labels outside {0, 1, 2, 4}.")
            segmentation = _readonly_owned(values, dtype=np.int16)
        else:
            decoded[modality] = _readonly_owned(values, dtype=np.float32)

    if (
        source_affine is None
        or canonical_affine is None
        or source_orientation is None
        or canonical_to_source is None
    ):
        raise BraTS2021CaseLoadError("BraTS case did not produce complete spatial metadata.")
    return BraTS2021LocalCase(
        case_alias=report.case_alias,
        modalities=MappingProxyType(decoded),
        segmentation=segmentation,
        source_affine=_readonly_owned(source_affine, dtype=np.float64),
        canonical_affine=_readonly_owned(canonical_affine, dtype=np.float64),
        canonical_to_source_orientation=_readonly_owned(canonical_to_source, dtype=np.float64),
        source_orientation=source_orientation,
        artifact_sha256=MappingProxyType(
            {modality: records[modality].sha256 for modality in decode_order if modality in records}
        ),
    )


def build_brats2021_manifest(
    report: BraTS2021ValidationReport,
    *,
    terms_confirmed_by_user: bool,
    installed_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a path-free manifest for a successfully validated local case."""

    if not report.is_valid:
        raise ValueError("Cannot install a BraTS case until all validation errors are resolved.")
    if report.file_for("SEG") is None:
        raise ValueError("A BraTS installation manifest requires the Task 1 SEG ground truth.")
    if terms_confirmed_by_user is not True:
        raise PermissionError("The user must confirm the official BraTS data terms first.")
    timestamp = datetime.now(timezone.utc) if installed_at is None else installed_at
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("installed_at must be timezone-aware")
    installed_at_utc = timestamp.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    installed_at_utc = installed_at_utc.replace("+00:00", "Z")

    ordered_files = sorted(
        report.files,
        key=lambda record: BRATS_2021_MODALITIES.index(record.modality),
    )
    return {
        "schema_version": 1,
        "dataset": {
            "id": "BraTS-2021-Task-1",
            "official_acquisition_url": BRATS_2021_OFFICIAL_ACQUISITION_URL,
        },
        "case_alias": report.case_alias,
        "installed_at_utc": installed_at_utc,
        "terms_confirmed_by_user": True,
        "source_data": {
            "copied": False,
            "uploaded": False,
            "storage": "user-managed-local-files",
        },
        "files": [
            {
                "modality": record.modality,
                "relative_id": record.relative_id,
                "sha256": record.sha256,
                "size_bytes": record.size_bytes,
                "geometry": record.geometry.to_manifest(),
            }
            for record in ordered_files
        ],
        "segmentation": {
            "allowed_labels": list(BRATS_2021_SEGMENTATION_LABELS),
            "voxel_counts": {str(label): count for label, count in report.segmentation_counts},
        },
        "privacy": {
            "contains_absolute_paths": False,
            "contains_source_filenames": False,
            "contains_person_identifiers": False,
        },
    }


def write_brats2021_manifest(
    report: BraTS2021ValidationReport,
    destination: str | Path,
    *,
    terms_confirmed_by_user: bool,
    installed_at: datetime | None = None,
) -> Path:
    """Atomically write a validated manifest and return its destination."""

    manifest = build_brats2021_manifest(
        report,
        terms_confirmed_by_user=terms_confirmed_by_user,
        installed_at=installed_at,
    )
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target
