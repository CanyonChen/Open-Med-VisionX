"""Bounded, header-only discovery for DICOM studies and series.

The public summaries intentionally contain no patient, accession, institution,
path, or raw UID fields.  UIDs are represented by short SHA-256 fingerprints so
the UI can distinguish candidates without echoing source identifiers.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from ..errors import DecodeError, MissingDependencyError, ResourceLimitError
from .base import CancelCheck, LoadLimits, PathLike, raise_if_cancelled
from .dicom_frames import dicom_frame_attribute

_SEGMENTATION_STORAGE_UID = "1.2.840.10008.5.1.4.1.1.66.4"
_RT_STRUCTURE_SET_STORAGE_UID = "1.2.840.10008.5.1.4.1.1.481.3"
_UNSUPPORTED_STRUCTURE_UIDS = {
    _SEGMENTATION_STORAGE_UID: "DICOM SEG is not supported by the stable image loader.",
    _RT_STRUCTURE_SET_STORAGE_UID: "DICOM RTSTRUCT is not supported by the stable image loader.",
}

# SeriesDescription is an unconstrained DICOM free-text field and can contain
# identifiers.  Only a deliberately small protocol vocabulary is shown.  All
# other values are withheld; no best-effort regex can promise de-identification.
_SAFE_DESCRIPTION_WORDS = frozenset(
    {
        "2D",
        "3D",
        "ABDOMEN",
        "ADC",
        "ANGIO",
        "ARTERIAL",
        "AX",
        "AXIAL",
        "BRAIN",
        "CHEST",
        "CONTRAST",
        "COR",
        "CORONAL",
        "CT",
        "DIFFUSION",
        "DWI",
        "ENHANCED",
        "FLAIR",
        "HEAD",
        "LOCALIZER",
        "LUNG",
        "MIP",
        "MR",
        "MRI",
        "NONCONTRAST",
        "PELVIS",
        "POST",
        "PRE",
        "SAG",
        "SAGITTAL",
        "SCOUT",
        "T1",
        "T2",
        "TSE",
        "VENOUS",
        "WITHOUT",
        "WITH",
    }
)
_DESCRIPTION_TOKEN = re.compile(r"[A-Z0-9]+")

_DISCOVERY_TAGS = (
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SeriesDescription",
    "SeriesNumber",
    "SOPClassUID",
    "Modality",
    "FrameOfReferenceUID",
    "Rows",
    "Columns",
    "SamplesPerPixel",
    "PhotometricInterpretation",
    "NumberOfFrames",
    "PixelSpacing",
    "SpacingBetweenSlices",
    "SliceThickness",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "InstanceNumber",
    "SharedFunctionalGroupsSequence",
    "PerFrameFunctionalGroupsSequence",
)


@dataclass(frozen=True, slots=True)
class SeriesSummary:
    """Immutable, PHI-minimized metadata for one Study/Series UID pair."""

    selector: str
    study_identifier: str
    series_identifier: str
    modality: str
    series_description: str
    series_number: int | None
    instance_count: int
    slice_count: int
    frame_count: int
    rows: int | None
    columns: int | None
    pixel_spacing_mm: tuple[float, float] | None
    slice_thickness_mm: float | None
    estimated_slice_spacing_mm: float | None
    geometry_consistent: bool
    supported_by_stable_loader: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DicomSeriesDiscovery:
    """Result of a bounded source scan; it never owns decoded pixel data."""

    source_kind: str
    series: tuple[SeriesSummary, ...]
    inspected_member_count: int
    dicom_object_count: int
    skipped_member_count: int
    warnings: tuple[str, ...] = ()


class SeriesSelectionRequiredError(DecodeError):
    """A multi-series source was rejected and safe candidates are available."""

    def __init__(self, candidates: tuple[SeriesSummary, ...]) -> None:
        self.candidates = tuple(candidates)
        super().__init__(
            "DICOM source contains "
            f"{len(self.candidates):,} series and no series was selected. "
            "Inspect the provided candidates and choose one explicitly; "
            "OpenMedVisionX will not guess which series to load."
        )


def _uid_fingerprint(kind: str, value: str) -> str:
    digest = hashlib.sha256(f"openmedvisionx:{kind}:{value}".encode()).hexdigest()
    return f"sha256:{digest[:12]}"


def _clean_uid(value: object) -> str:
    return str(value or "").strip()


def dicom_series_selector_for_dataset(dataset: object, *, ordinal: int = 0) -> str:
    """Return the same PHI-minimized selector used by discovery summaries."""

    study_uid = _clean_uid(getattr(dataset, "StudyInstanceUID", ""))
    series_uid = _clean_uid(getattr(dataset, "SeriesInstanceUID", ""))
    selector_source = f"{study_uid or '__missing_study'}\0{series_uid or ordinal}"
    return _uid_fingerprint("selector", selector_source)


def _safe_text(value: object, *, default: str) -> str:
    text = " ".join(unicodedata.normalize("NFKC", str(value or "")).split())
    if not text:
        return default
    if len(text) > 64 or any(unicodedata.category(char).startswith("C") for char in text):
        return "Withheld (untrusted DICOM free text)"
    upper = text.upper()
    tokens = _DESCRIPTION_TOKEN.findall(upper)
    if not tokens or any(token not in _SAFE_DESCRIPTION_WORDS for token in tokens):
        return "Withheld (untrusted DICOM free text)"
    if re.search(r"[@\\^]|https?://|\b(?:NAME|PATIENT|DOB|ID)\b", upper):
        return "Withheld (untrusted DICOM free text)"
    return text


def _safe_modality(value: object) -> str:
    modality = str(value or "UNKNOWN").strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{1,16}", modality):
        return "UNKNOWN"
    return modality


def _optional_int(value: object) -> int | None:
    try:
        result = int(str(value))
    except (TypeError, ValueError, OverflowError):
        return None
    return result


def _optional_positive_float(value: object) -> float | None:
    try:
        result = float(str(value))
    except (TypeError, ValueError, OverflowError):
        return None
    if result <= 0.0 or not np.isfinite(result):
        return None
    return result


def _vector(value: object, length: int, *, positive: bool = False) -> np.ndarray | None:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        return None
    if positive and np.any(result <= 0.0):
        return None
    return result


def _dataset_group_key(dataset: object, ordinal: int) -> tuple[str, str]:
    study_uid = _clean_uid(getattr(dataset, "StudyInstanceUID", ""))
    series_uid = _clean_uid(getattr(dataset, "SeriesInstanceUID", ""))
    # A missing series UID must not silently combine unrelated objects.  The
    # scan ordinal is local-only and never exposed in the summary.
    series_key = series_uid or f"__missing_series_{ordinal}"
    return study_uid or "__missing_study", series_key


def group_dicom_datasets(datasets: list[object]) -> tuple[tuple[object, ...], ...]:
    """Group by StudyInstanceUID and SeriesInstanceUID without exposing either."""

    groups: dict[tuple[str, str], list[object]] = defaultdict(list)
    for ordinal, dataset in enumerate(datasets):
        groups[_dataset_group_key(dataset, ordinal)].append(dataset)
    return tuple(tuple(group) for group in groups.values())


def _series_summary(
    series: tuple[object, ...],
    *,
    ordinal: int,
    limits: LoadLimits,
) -> SeriesSummary:
    first = series[0]
    study_uid = _clean_uid(getattr(first, "StudyInstanceUID", ""))
    series_uid = _clean_uid(getattr(first, "SeriesInstanceUID", ""))
    selector = dicom_series_selector_for_dataset(first, ordinal=ordinal)
    study_identifier = _uid_fingerprint("study", study_uid or f"missing:{ordinal}")
    series_identifier = _uid_fingerprint("series", series_uid or f"missing:{ordinal}")

    warning_set: set[str] = set()
    modalities = {_safe_modality(getattr(item, "Modality", "UNKNOWN")) for item in series}
    modality = next(iter(modalities)) if len(modalities) == 1 else "MIXED"
    if len(modalities) != 1:
        warning_set.add("Instances declare inconsistent modalities.")
    if not study_uid:
        warning_set.add("StudyInstanceUID is missing; study grouping is not reliable.")
    if not series_uid:
        warning_set.add("SeriesInstanceUID is missing; this candidate cannot be selected safely.")

    descriptions = {
        _safe_text(
            getattr(item, "SeriesDescription", ""),
            default="Not provided",
        )
        for item in series
    }
    description = next(iter(descriptions)) if len(descriptions) == 1 else "Mixed descriptions"
    if len(descriptions) != 1:
        warning_set.add("Instances declare inconsistent SeriesDescription values.")
    if "Withheld" in description:
        warning_set.add("SeriesDescription was withheld because DICOM free text may contain PHI.")

    frame_counts: list[int] = []
    for item in series:
        count = _optional_int(getattr(item, "NumberOfFrames", 1))
        if count is None or count <= 0:
            count = 1
            warning_set.add("NumberOfFrames is invalid; one frame was assumed for discovery.")
        frame_counts.append(count)
    frame_count = sum(frame_counts)
    enhanced_single_instance = len(series) == 1 and frame_count > 1
    supported_frame_layout = enhanced_single_instance or all(count == 1 for count in frame_counts)
    slice_count = (
        frame_count if enhanced_single_instance else sum(1 for count in frame_counts if count == 1)
    )
    if any(count != 1 for count in frame_counts) and not enhanced_single_instance:
        warning_set.add("A series cannot mix multi-frame and separate pixel-bearing instances.")
    if frame_count > limits.max_frames:
        warning_set.add("Series frame count exceeds the configured safety limit.")

    row_values = [_optional_int(getattr(item, "Rows", None)) for item in series]
    column_values = [_optional_int(getattr(item, "Columns", None)) for item in series]
    valid_rows = {value for value in row_values if value is not None and value > 0}
    valid_columns = {value for value in column_values if value is not None and value > 0}
    dimensions_consistent = (
        len(valid_rows) == 1
        and len(valid_columns) == 1
        and len(valid_rows) == len(set(row_values))
        and len(valid_columns) == len(set(column_values))
    )
    rows = next(iter(valid_rows)) if len(valid_rows) == 1 else None
    columns = next(iter(valid_columns)) if len(valid_columns) == 1 else None
    if not dimensions_consistent:
        warning_set.add("Rows/Columns are missing, invalid, or inconsistent.")
    elif rows is not None and columns is not None and rows * columns > limits.max_pixels:
        warning_set.add("Per-frame pixel count exceeds the configured safety limit.")

    if enhanced_single_instance:
        spacings = [
            _vector(
                dicom_frame_attribute(
                    first,
                    index,
                    sequence_name="PixelMeasuresSequence",
                    attribute_name="PixelSpacing",
                ),
                2,
                positive=True,
            )
            for index in range(frame_count)
        ]
        orientations = [
            _vector(
                dicom_frame_attribute(
                    first,
                    index,
                    sequence_name="PlaneOrientationSequence",
                    attribute_name="ImageOrientationPatient",
                ),
                6,
            )
            for index in range(frame_count)
        ]
        positions = [
            _vector(
                dicom_frame_attribute(
                    first,
                    index,
                    sequence_name="PlanePositionSequence",
                    attribute_name="ImagePositionPatient",
                ),
                3,
            )
            for index in range(frame_count)
        ]
        thickness_values = [
            _optional_positive_float(
                dicom_frame_attribute(
                    first,
                    index,
                    sequence_name="PixelMeasuresSequence",
                    attribute_name="SliceThickness",
                )
            )
            for index in range(frame_count)
        ]
        between_slice_values = [
            _optional_positive_float(
                dicom_frame_attribute(
                    first,
                    index,
                    sequence_name="PixelMeasuresSequence",
                    attribute_name="SpacingBetweenSlices",
                )
            )
            for index in range(frame_count)
        ]
    else:
        spacings = [
            _vector(getattr(item, "PixelSpacing", None), 2, positive=True) for item in series
        ]
        orientations = [
            _vector(getattr(item, "ImageOrientationPatient", None), 6) for item in series
        ]
        positions = [_vector(getattr(item, "ImagePositionPatient", None), 3) for item in series]
        thickness_values = [
            _optional_positive_float(getattr(item, "SliceThickness", None)) for item in series
        ]
        between_slice_values = [
            _optional_positive_float(getattr(item, "SpacingBetweenSlices", None)) for item in series
        ]

    reference_spacing = spacings[0] if spacings else None
    pixel_spacing_consistent = reference_spacing is not None and all(
        value is not None and np.allclose(value, reference_spacing, rtol=1e-4, atol=1e-5)
        for value in spacings[1:]
    )
    pixel_spacing: tuple[float, float] | None = None
    if pixel_spacing_consistent and spacings[0] is not None:
        pixel_spacing = (float(spacings[0][0]), float(spacings[0][1]))
    else:
        warning_set.add("PixelSpacing is missing, invalid, or inconsistent.")

    reference_orientation = orientations[0] if orientations else None
    orientation_consistent = reference_orientation is not None and all(
        value is not None and np.allclose(value, reference_orientation, rtol=1e-5, atol=1e-5)
        for value in orientations[1:]
    )
    normal: np.ndarray | None = None
    if orientation_consistent and orientations[0] is not None:
        first_orientation = orientations[0]
        row_direction = first_orientation[:3]
        column_direction = first_orientation[3:]
        normal_value = np.cross(row_direction, column_direction)
        if (
            np.linalg.norm(row_direction) <= 0.0
            or np.linalg.norm(column_direction) <= 0.0
            or not np.isclose(np.dot(row_direction, column_direction), 0.0, atol=1e-4)
            or np.linalg.norm(normal_value) <= 0.0
        ):
            orientation_consistent = False
        else:
            normal = normal_value / np.linalg.norm(normal_value)
    if not orientation_consistent:
        warning_set.add("ImageOrientationPatient is missing, invalid, or inconsistent.")

    positions_valid = bool(positions and all(value is not None for value in positions))
    estimated_spacing: float | None = None
    if positions_valid and normal is not None and len(positions) > 1:
        projections = np.sort(
            np.asarray([np.dot(value, normal) for value in positions if value is not None])
        )
        differences = np.abs(np.diff(projections))
        estimated_spacing = float(np.median(differences)) if differences.size else None
        if (
            estimated_spacing is None
            or estimated_spacing <= 0.0
            or not np.allclose(differences, estimated_spacing, rtol=0.05, atol=1e-3)
        ):
            positions_valid = False
            estimated_spacing = None
            warning_set.add("Slice positions are duplicated or non-uniform.")
    elif not positions_valid:
        warning_set.add("ImagePositionPatient is missing or invalid.")

    valid_thickness = {value for value in thickness_values if value is not None}
    thickness_consistent = len(valid_thickness) == 1 and len(valid_thickness) == len(
        set(thickness_values)
    )
    slice_thickness = next(iter(valid_thickness)) if thickness_consistent else None
    if slice_thickness is None:
        warning_set.add("SliceThickness is missing, invalid, or inconsistent.")
    if estimated_spacing is None and len(series) == 1:
        valid_between_slice = {value for value in between_slice_values if value is not None}
        consistent_between_slice = len(valid_between_slice) == 1 and len(
            valid_between_slice
        ) == len(set(between_slice_values))
        estimated_spacing = (
            next(iter(valid_between_slice)) if consistent_between_slice else None
        ) or slice_thickness

    frame_uids = [_clean_uid(getattr(item, "FrameOfReferenceUID", "")) for item in series]
    present_frame_uids = {value for value in frame_uids if value}
    frame_of_reference_consistent = not present_frame_uids or (
        len(present_frame_uids) == 1 and all(frame_uids)
    )
    if not frame_of_reference_consistent:
        warning_set.add("FrameOfReferenceUID is missing on some instances or inconsistent.")

    sop_classes = {_clean_uid(getattr(item, "SOPClassUID", "")) for item in series}
    for sop_class in sop_classes:
        unsupported_message = _UNSUPPORTED_STRUCTURE_UIDS.get(sop_class)
        if unsupported_message:
            warning_set.add(unsupported_message)
    if any(
        rows_value is None or columns_value is None
        for rows_value, columns_value in zip(row_values, column_values, strict=True)
    ):
        warning_set.add("This DICOM object does not contain a conventional image frame.")

    samples = {_optional_int(getattr(item, "SamplesPerPixel", 1)) for item in series}
    photometric = {
        str(getattr(item, "PhotometricInterpretation", "")).strip().upper() for item in series
    }
    monochrome_consistent = (
        samples == {1}
        and len(photometric) == 1
        and next(iter(photometric), "") in {"MONOCHROME1", "MONOCHROME2"}
    )
    if not monochrome_consistent:
        warning_set.add("The stable loader requires one consistent monochrome interpretation.")

    geometry_consistent = bool(
        dimensions_consistent
        and pixel_spacing_consistent
        and orientation_consistent
        and positions_valid
        and frame_of_reference_consistent
        and estimated_spacing is not None
    )
    supported = bool(
        study_uid
        and series_uid
        and geometry_consistent
        and monochrome_consistent
        and len(modalities) == 1
        and supported_frame_layout
        and frame_count <= limits.max_frames
        and rows is not None
        and columns is not None
        and rows * columns <= limits.max_pixels
        and not sop_classes.intersection(_UNSUPPORTED_STRUCTURE_UIDS)
    )

    return SeriesSummary(
        selector=selector,
        study_identifier=study_identifier,
        series_identifier=series_identifier,
        modality=modality,
        series_description=description,
        series_number=_optional_int(getattr(first, "SeriesNumber", None)),
        instance_count=len(series),
        slice_count=slice_count,
        frame_count=frame_count,
        rows=rows,
        columns=columns,
        pixel_spacing_mm=pixel_spacing,
        slice_thickness_mm=slice_thickness,
        estimated_slice_spacing_mm=estimated_spacing,
        geometry_consistent=geometry_consistent,
        supported_by_stable_loader=supported,
        warnings=tuple(sorted(warning_set)),
    )


def summarize_dicom_datasets(
    datasets: list[object],
    *,
    limits: LoadLimits | None = None,
) -> tuple[SeriesSummary, ...]:
    """Build safe summaries from already parsed datasets without retaining them."""

    active_limits = limits or LoadLimits()
    summaries = tuple(
        _series_summary(group, ordinal=ordinal, limits=active_limits)
        for ordinal, group in enumerate(group_dicom_datasets(datasets))
    )
    return tuple(
        sorted(
            summaries,
            key=lambda item: (
                item.series_number is None,
                item.series_number if item.series_number is not None else 0,
                item.modality,
                item.series_identifier,
            ),
        )
    )


def select_dicom_series_datasets(
    datasets: list[object],
    selector: str,
    *,
    limits: LoadLimits | None = None,
) -> list[object]:
    """Resolve one PHI-minimized discovery selector against decoded datasets.

    The selector is derived from Study/Series UIDs but never reveals them.  A
    missing or ambiguous selector is rejected rather than falling back to a
    largest-series heuristic.
    """

    if not isinstance(selector, str) or not selector.startswith("sha256:"):
        raise DecodeError("A valid DICOM series selector is required.")
    active_limits = limits or LoadLimits()
    matches: list[tuple[object, ...]] = []
    for ordinal, group in enumerate(group_dicom_datasets(datasets)):
        summary = _series_summary(group, ordinal=ordinal, limits=active_limits)
        if summary.selector == selector:
            matches.append(group)
    if len(matches) != 1:
        raise DecodeError(
            "The selected DICOM series is missing or ambiguous; scan the source again."
        )
    return list(matches[0])


def _validate_zip_member(member: zipfile.ZipInfo, limits: LoadLimits) -> None:
    normalized = member.filename.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if not normalized or normalized.startswith(("/", "\\")) or ".." in parts:
        raise ResourceLimitError("ZIP contains an unsafe path; extraction was refused.")
    if parts and ":" in parts[0]:
        raise ResourceLimitError("ZIP contains an absolute drive path; extraction was refused.")
    if member.flag_bits & 0x1:
        raise ResourceLimitError("Encrypted ZIP members are not supported.")
    if member.file_size > limits.max_zip_member_bytes:
        raise ResourceLimitError("A ZIP member exceeds the configured expanded-size limit.")
    if member.file_size > 0:
        ratio = member.file_size / max(member.compress_size, 1)
        if ratio > limits.max_zip_compression_ratio:
            raise ResourceLimitError(
                f"ZIP member compression ratio {ratio:.1f} exceeds the safe limit."
            )


def read_dicom_header(pydicom: Any, source: object) -> object:
    """Read only the bounded discovery tags without consuming pixel data."""

    return pydicom.dcmread(
        source,
        force=False,
        stop_before_pixels=True,
        specific_tags=_DISCOVERY_TAGS,
    )


def _scan_directory(
    path: Path,
    pydicom: Any,
    limits: LoadLimits,
    cancel: CancelCheck,
) -> tuple[list[object], int, int]:
    try:
        root = path.resolve(strict=True)
    except OSError as exc:
        raise DecodeError("Could not inspect the selected DICOM directory.") from exc
    datasets: list[object] = []
    inspected = 0
    skipped = 0
    total_bytes = 0
    for candidate in path.rglob("*"):
        raise_if_cancelled(cancel)
        if not candidate.is_file():
            continue
        inspected += 1
        if inspected > limits.max_zip_members:
            raise ResourceLimitError(
                "DICOM directory contains more files than the configured limit."
            )
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as exc:
            raise ResourceLimitError(
                "DICOM directory contains a link outside the selected folder."
            ) from exc
        try:
            file_size = resolved.stat().st_size
        except OSError:
            skipped += 1
            continue
        total_bytes += file_size
        if file_size > limits.max_zip_member_bytes or total_bytes > limits.max_zip_total_bytes:
            raise ResourceLimitError("DICOM directory exceeds configured byte limits.")
        try:
            datasets.append(read_dicom_header(pydicom, str(resolved)))
        except Exception:
            skipped += 1
        raise_if_cancelled(cancel)
    return datasets, inspected, skipped


def _scan_zip(
    path: Path,
    pydicom: Any,
    limits: LoadLimits,
    cancel: CancelCheck,
) -> tuple[list[object], int, int]:
    datasets: list[object] = []
    skipped = 0
    try:
        with zipfile.ZipFile(path) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            if len(members) > limits.max_zip_members:
                raise ResourceLimitError(
                    f"ZIP contains {len(members):,} files, above the configured limit."
                )
            total_bytes = 0
            for member in members:
                raise_if_cancelled(cancel)
                _validate_zip_member(member, limits)
                total_bytes += member.file_size
                if total_bytes > limits.max_zip_total_bytes:
                    raise ResourceLimitError(
                        "ZIP expanded size exceeds the configured total limit."
                    )
                try:
                    with archive.open(member) as stream:
                        datasets.append(read_dicom_header(pydicom, stream))
                except Exception:
                    skipped += 1
                raise_if_cancelled(cancel)
            return datasets, len(members), skipped
    except ResourceLimitError:
        raise
    except zipfile.BadZipFile as exc:
        raise DecodeError("The selected DICOM ZIP is corrupt or truncated.") from exc
    except (OSError, RuntimeError) as exc:
        raise DecodeError("Could not inspect the selected DICOM ZIP.") from exc


def _scan_file(
    path: Path,
    pydicom: Any,
    limits: LoadLimits,
    cancel: CancelCheck,
) -> tuple[list[object], int, int]:
    raise_if_cancelled(cancel)
    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise DecodeError("Could not inspect the selected DICOM file.") from exc
    if file_size > limits.max_zip_member_bytes or file_size > limits.max_zip_total_bytes:
        raise ResourceLimitError("DICOM file exceeds the configured byte limit.")
    try:
        dataset = read_dicom_header(pydicom, str(path))
    except Exception as exc:
        raise DecodeError("The selected file does not contain a readable DICOM header.") from exc
    raise_if_cancelled(cancel)
    return [dataset], 1, 0


def discover_dicom_series(
    source: PathLike,
    *,
    limits: LoadLimits | None = None,
    cancel: CancelCheck = None,
) -> DicomSeriesDiscovery:
    """Scan a directory, safe ZIP, or file without decoding image pixels."""

    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - base dependency
        raise MissingDependencyError(
            "DICOM discovery requires pydicom. Install the base project dependencies."
        ) from exc

    path = Path(source)
    active_limits = limits or LoadLimits()
    if path.is_dir():
        source_kind = "directory"
        datasets, inspected, skipped = _scan_directory(path, pydicom, active_limits, cancel)
    elif path.is_file():
        try:
            with path.open("rb") as stream:
                signature = stream.read(4)
        except OSError as exc:
            raise DecodeError("Could not inspect the selected DICOM source.") from exc
        if signature == b"PK\x03\x04":
            if path.suffix.lower() != ".zip":
                raise DecodeError("ZIP signature detected, but the filename must end in .zip.")
            source_kind = "zip"
            datasets, inspected, skipped = _scan_zip(path, pydicom, active_limits, cancel)
        else:
            source_kind = "file"
            datasets, inspected, skipped = _scan_file(path, pydicom, active_limits, cancel)
    else:
        raise DecodeError("The selected DICOM source does not exist.")

    if not datasets:
        raise DecodeError(
            "No readable DICOM headers were found after inspecting "
            f"{inspected:,} source member(s); {skipped:,} member(s) could not be parsed."
        )
    series_datasets = [
        dataset
        for dataset in datasets
        if _clean_uid(getattr(dataset, "SeriesInstanceUID", ""))
        or (
            _optional_int(getattr(dataset, "Rows", None)) is not None
            and _optional_int(getattr(dataset, "Columns", None)) is not None
        )
    ]
    unlisted_dicom_objects = len(datasets) - len(series_datasets)
    if not series_datasets:
        raise DecodeError(
            "Readable DICOM objects were found, but none identified an image or series."
        )
    summaries = summarize_dicom_datasets(series_datasets, limits=active_limits)
    warnings: list[str] = []
    if skipped:
        warnings.append(
            f"{skipped:,} of {inspected:,} source member(s) were not readable DICOM objects."
        )
    if unlisted_dicom_objects:
        warnings.append(
            f"{unlisted_dicom_objects:,} readable DICOM object(s) did not identify a series "
            "and were not listed."
        )
    if len(summaries) > 1:
        warnings.append(
            "Multiple DICOM series were found; select a candidate explicitly before loading."
        )
    return DicomSeriesDiscovery(
        source_kind=source_kind,
        series=summaries,
        inspected_member_count=inspected,
        dicom_object_count=len(datasets),
        skipped_member_count=skipped,
        warnings=tuple(warnings),
    )


__all__ = [
    "DicomSeriesDiscovery",
    "SeriesSelectionRequiredError",
    "SeriesSummary",
    "dicom_series_selector_for_dataset",
    "discover_dicom_series",
    "read_dicom_header",
    "select_dicom_series_datasets",
]
