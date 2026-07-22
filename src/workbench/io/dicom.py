"""DICOM folder/ZIP loader with HU conversion and RAS+ geometry."""

from __future__ import annotations

import re
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import numpy as np

from ..domain.images import ImageVolume, IntensitySemantics, SourceType
from ..errors import DecodeError, FormatMismatchError, MissingDependencyError, ResourceLimitError
from .base import CancelCheck, ImageLoader, LoadLimits, PathLike, ProbeResult, raise_if_cancelled
from .dicom_frames import dicom_frame_attribute, dicom_frame_count
from .dicom_series import (
    DicomSeriesDiscovery,
    SeriesSelectionRequiredError,
    SeriesSummary,
    dicom_series_selector_for_dataset,
    discover_dicom_series,
    group_dicom_datasets,
    read_dicom_header,
    select_dicom_series_datasets,
    summarize_dicom_datasets,
)

_DICOM_UID_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)*$")
_OPAQUE_DICOM_ID_RE = re.compile(r"^sha256:[0-9a-f]{12}$")


def _reference_uid(value: object, field_name: str, *, required: bool) -> str | None:
    normalized = str(value or "").strip()
    if not normalized:
        if required:
            raise DecodeError(
                f"The selected DICOM image series is missing {field_name}; "
                "annotation references cannot be matched safely."
            )
        return None
    if len(normalized) > 64 or not _DICOM_UID_RE.fullmatch(normalized):
        raise DecodeError(f"The selected DICOM image series has an invalid {field_name}.")
    return normalized


@dataclass(frozen=True, slots=True)
class DicomReferenceIdentity:
    """Exact in-memory DICOM reference identity plus anonymous public selectors.

    Exact UIDs are deliberately excluded from ``repr`` so incidental logging
    does not echo source identifiers.  They must never be copied into image
    ``runtime_metadata`` or UI-facing series summaries.
    """

    selector: str
    study_identifier: str
    series_identifier: str
    study_instance_uid: str = field(repr=False)
    series_instance_uid: str = field(repr=False)
    frame_of_reference_uid: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        for value, name in (
            (self.selector, "selector"),
            (self.study_identifier, "study_identifier"),
            (self.series_identifier, "series_identifier"),
        ):
            if not _OPAQUE_DICOM_ID_RE.fullmatch(str(value)):
                raise DecodeError(f"DICOM {name} must be an anonymous SHA-256 identifier.")
        object.__setattr__(
            self,
            "study_instance_uid",
            _reference_uid(self.study_instance_uid, "StudyInstanceUID", required=True),
        )
        object.__setattr__(
            self,
            "series_instance_uid",
            _reference_uid(self.series_instance_uid, "SeriesInstanceUID", required=True),
        )
        object.__setattr__(
            self,
            "frame_of_reference_uid",
            _reference_uid(
                self.frame_of_reference_uid,
                "FrameOfReferenceUID",
                required=False,
            ),
        )


@dataclass(frozen=True, slots=True)
class DicomLoadResult:
    """One decoded volume and its exact, non-serialized reference identity."""

    volume: ImageVolume
    reference: DicomReferenceIdentity


def _has_dicom_preamble(header: bytes) -> bool:
    return len(header) >= 132 and header[128:132] == b"DICM"


def _normalized_dicom_text(value: object) -> str:
    return " ".join(str(value or "").upper().replace("_", " ").split())


def _dicom_code_tokens(value: object) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        values = value.replace("\\", " ").split()
    elif isinstance(value, Iterable):
        values = [str(item) for item in value]
    else:
        values = [str(value)]
    return frozenset(item.strip().upper() for item in values if item.strip())


_SUV_TYPES = frozenset({"BSA", "BW", "LBM", "LBMJANMA", "IBW"})
_DECAY_CORRECTIONS = frozenset({"ADMIN", "START"})


def _rescale_parameters(dataset: object) -> tuple[float, float]:
    try:
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise DecodeError("DICOM rescale slope/intercept is missing or invalid.") from exc
    if not np.isfinite(slope) or not np.isfinite(intercept):
        raise DecodeError("DICOM rescale slope/intercept must contain finite values.")
    return slope, intercept


def _infer_dicom_intensity_semantics(
    dataset: object,
    modality: str,
) -> tuple[IntensitySemantics, str]:
    """Infer only semantics supported by explicit DICOM evidence.

    Rescale slope/intercept describe a transform, not by themselves the unit
    of the transformed values.  In particular, MR and projection radiography
    signals remain arbitrary, and PET/SPECT values are not called SUV unless
    their units, valid SUV normalization contract, and correction evidence
    are present.  DICOM permits an absent SUV Type to mean body-weight SUV
    when Units is GML.
    """

    normalized_modality = modality.upper().strip()
    if normalized_modality == "CT":
        if not hasattr(dataset, "RescaleSlope") or not hasattr(dataset, "RescaleIntercept"):
            return IntensitySemantics.UNKNOWN, "dicom_ct_modality_lut_missing"
        rescale_type = _normalized_dicom_text(getattr(dataset, "RescaleType", ""))
        if rescale_type in {"", "HU", "HOUNSFIELD UNIT", "HOUNSFIELD UNITS"}:
            source = "dicom_modality_ct" if not rescale_type else "dicom_rescale_type_hu"
            return IntensitySemantics.HOUNSFIELD_UNIT, source
        return IntensitySemantics.UNKNOWN, "dicom_ct_non_hu_rescale_type"

    if normalized_modality in {"MR", "CR", "DX", "MG", "XA", "RF", "US", "OCT"}:
        return IntensitySemantics.ARBITRARY_SIGNAL, "dicom_modality_signal"

    if normalized_modality in {"PT", "PET", "NM"}:
        if not hasattr(dataset, "RescaleSlope") or not hasattr(dataset, "RescaleIntercept"):
            return IntensitySemantics.UNKNOWN, "dicom_nuclear_modality_lut_missing"
        units = _normalized_dicom_text(getattr(dataset, "Units", ""))
        suv_type = _normalized_dicom_text(getattr(dataset, "SUVType", ""))
        decay_correction = _normalized_dicom_text(getattr(dataset, "DecayCorrection", ""))
        corrected_image = _dicom_code_tokens(getattr(dataset, "CorrectedImage", None))
        _slope, intercept = _rescale_parameters(dataset)
        valid_suv_type = not suv_type or suv_type in _SUV_TYPES
        corrections_declared = decay_correction in _DECAY_CORRECTIONS and {"ATTN", "DECY"}.issubset(
            corrected_image
        )
        if (
            units in {"GML", "G/ML"}
            and valid_suv_type
            and corrections_declared
            and intercept == 0.0
        ):
            return IntensitySemantics.SUV, "dicom_pet_suv_tags"
        # Counts and activity concentration can be useful for relative
        # comparisons, but without a complete calibration/correction contract
        # the stable loader must not advertise them as quantitative values.
        return IntensitySemantics.UNKNOWN, "dicom_nuclear_units_or_corrections_incomplete"

    return IntensitySemantics.UNKNOWN, "dicom_semantics_not_declared"


class DicomLoader(ImageLoader):
    name = "dicom"

    def discover_series(
        self,
        source: PathLike,
        *,
        limits: LoadLimits | None = None,
        cancel: CancelCheck = None,
    ) -> DicomSeriesDiscovery:
        """Return bounded, header-only, PHI-minimized series candidates."""

        return discover_dicom_series(source, limits=limits, cancel=cancel)

    def probe(self, source: PathLike) -> ProbeResult:
        path = Path(source)
        if path.is_dir():
            return ProbeResult(True, "DICOM_DIRECTORY", 70, {"directory": True})
        if not path.is_file():
            return ProbeResult(False)
        try:
            with path.open("rb") as stream:
                header = stream.read(132)
        except OSError:
            return ProbeResult(False)
        if header.startswith(b"PK\x03\x04"):
            return ProbeResult(
                True,
                "DICOM_ZIP",
                80 if path.suffix.lower() == ".zip" else 75,
                {"zip_signature": True, "extension_matches": path.suffix.lower() == ".zip"},
            )
        signature = _has_dicom_preamble(header)
        expected = path.suffix.lower() in {".dcm", ".dicom"}
        return ProbeResult(
            signature or expected,
            "DICOM",
            100 if signature and expected else 90 if signature else 40,
            {"dicom_preamble": signature, "extension_matches": signature and expected},
        )

    def load(
        self,
        source: PathLike,
        *,
        limits: LoadLimits | None = None,
        cancel: CancelCheck = None,
        series_selector: str | None = None,
    ) -> ImageVolume:
        volume, _datasets, _summary = self._load_selected_series(
            source,
            limits=limits,
            cancel=cancel,
            series_selector=series_selector,
        )
        return volume

    def load_with_reference(
        self,
        source: PathLike,
        *,
        limits: LoadLimits | None = None,
        cancel: CancelCheck = None,
        series_selector: str | None = None,
    ) -> DicomLoadResult:
        """Decode once and retain exact DICOM identities only in memory.

        This is the service-facing import path used when later DICOM SEG or
        RTSTRUCT objects must be matched to the base image series.  It shares
        the same complete dataset reads and volume construction as ``load``;
        no second pixel-bearing read is performed to recover the UIDs.
        """

        volume, datasets, summary = self._load_selected_series(
            source,
            limits=limits,
            cancel=cancel,
            series_selector=series_selector,
        )
        reference = self._reference_identity(datasets, summary)
        return DicomLoadResult(volume=volume, reference=reference)

    def _load_selected_series(
        self,
        source: PathLike,
        *,
        limits: LoadLimits | None,
        cancel: CancelCheck,
        series_selector: str | None,
    ) -> tuple[ImageVolume, list[object], SeriesSummary]:
        try:
            import pydicom
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise MissingDependencyError(
                "DICOM loading requires pydicom. Install the base project dependencies."
            ) from exc

        path = Path(source)
        active_limits = limits or LoadLimits()
        probe = self.probe(path)
        if not probe.accepted:
            raise DecodeError(f"{path.name!r} is not a DICOM source.")
        raise_if_cancelled(cancel)
        discovery = discover_dicom_series(
            path,
            limits=active_limits,
            cancel=cancel,
        )
        decode_selector = self._resolve_series_selector(discovery, series_selector)
        summary = next(item for item in discovery.series if item.selector == decode_selector)
        if path.is_dir():
            datasets = self._read_directory(
                path,
                pydicom,
                active_limits,
                cancel,
                series_selector=decode_selector,
            )
            container = "directory"
        elif probe.format_name == "DICOM_ZIP":
            if path.suffix.lower() != ".zip":
                raise FormatMismatchError(
                    "ZIP signature detected, but the filename must end in .zip."
                )
            datasets = self._read_zip(
                path,
                pydicom,
                active_limits,
                cancel,
                series_selector=decode_selector,
            )
            container = "zip"
        else:
            datasets = self._read_single(path, pydicom, active_limits)
            container = "file"
        raise_if_cancelled(cancel)
        volume = self._build_volume(
            datasets,
            container=container,
            limits=active_limits,
            cancel=cancel,
            series_selector=series_selector,
        )
        return volume, datasets, summary

    @staticmethod
    def _reference_identity(
        datasets: list[object],
        summary: SeriesSummary,
    ) -> DicomReferenceIdentity:
        image_datasets = [item for item in datasets if hasattr(item, "PixelData")]
        groups = group_dicom_datasets(image_datasets)
        if len(groups) != 1:
            raise DecodeError(
                "The selected DICOM image series became ambiguous while retaining references."
            )
        series = groups[0]
        study_uids = {
            _reference_uid(item_uid, "StudyInstanceUID", required=True)
            for item_uid in (getattr(item, "StudyInstanceUID", "") for item in series)
        }
        series_uids = {
            _reference_uid(item_uid, "SeriesInstanceUID", required=True)
            for item_uid in (getattr(item, "SeriesInstanceUID", "") for item in series)
        }
        frame_uids = {
            _reference_uid(item_uid, "FrameOfReferenceUID", required=False)
            for item_uid in (getattr(item, "FrameOfReferenceUID", "") for item in series)
        }
        if len(study_uids) != 1:
            raise DecodeError("The selected DICOM instances disagree on StudyInstanceUID.")
        if len(series_uids) != 1:
            raise DecodeError("The selected DICOM instances disagree on SeriesInstanceUID.")
        if len(frame_uids) != 1:
            raise DecodeError(
                "The selected DICOM instances have missing or inconsistent "
                "FrameOfReferenceUID values."
            )
        study_uid = next(iter(study_uids))
        series_uid = next(iter(series_uids))
        assert study_uid is not None
        assert series_uid is not None
        return DicomReferenceIdentity(
            selector=summary.selector,
            study_identifier=summary.study_identifier,
            series_identifier=summary.series_identifier,
            study_instance_uid=study_uid,
            series_instance_uid=series_uid,
            frame_of_reference_uid=next(iter(frame_uids)),
        )

    @staticmethod
    def _resolve_series_selector(
        discovery: DicomSeriesDiscovery,
        requested_selector: str | None,
    ) -> str:
        candidates = discovery.series
        if requested_selector is None and len(candidates) > 1:
            raise SeriesSelectionRequiredError(candidates)
        matches = (
            candidates
            if requested_selector is None
            else tuple(item for item in candidates if item.selector == requested_selector)
        )
        if len(matches) != 1:
            raise DecodeError(
                "The selected DICOM series is missing or ambiguous; scan the source again."
            )
        return matches[0].selector

    @staticmethod
    def _read_single(path: Path, pydicom: object, limits: LoadLimits) -> list[object]:
        try:
            file_size = path.stat().st_size
        except OSError as exc:
            raise DecodeError(f"Could not inspect DICOM file {path.name!r}: {exc}") from exc
        if file_size > limits.max_zip_member_bytes or file_size > limits.max_zip_total_bytes:
            raise ResourceLimitError("DICOM file exceeds the configured byte limit.")
        try:
            return [pydicom.dcmread(str(path), force=False)]  # type: ignore[attr-defined]
        except Exception as exc:
            raise DecodeError(f"Could not decode DICOM file {path.name!r}: {exc}") from exc

    def _read_directory(
        self,
        path: Path,
        pydicom: object,
        limits: LoadLimits,
        cancel: CancelCheck,
        *,
        series_selector: str | None = None,
    ) -> list[object]:
        root = path.resolve(strict=True)
        datasets: list[object] = []
        member_count = 0
        total_bytes = 0
        for candidate in path.rglob("*"):
            raise_if_cancelled(cancel)
            if not candidate.is_file():
                continue
            member_count += 1
            if member_count > limits.max_zip_members:
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
                continue
            total_bytes += file_size
            if file_size > limits.max_zip_member_bytes or total_bytes > limits.max_zip_total_bytes:
                raise ResourceLimitError("DICOM directory exceeds configured byte limits.")
            try:
                header = read_dicom_header(pydicom, str(resolved))
            except Exception:
                # Directories often contain DICOMDIR or unrelated notes.  A
                # supported-looking .dcm that fails is reported if no valid
                # image remains, without exposing local paths in logs.
                continue
            if (
                series_selector is not None
                and dicom_series_selector_for_dataset(header) != series_selector
            ):
                continue
            try:
                datasets.append(pydicom.dcmread(str(resolved), force=False))  # type: ignore[attr-defined]
            except Exception:
                continue
        if not datasets:
            raise DecodeError("No decodable DICOM instances were found for the selected series.")
        return datasets

    def _read_zip(
        self,
        path: Path,
        pydicom: object,
        limits: LoadLimits,
        cancel: CancelCheck,
        *,
        series_selector: str | None = None,
    ) -> list[object]:
        datasets: list[object] = []
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
                    self._validate_zip_member(member, limits)
                    total_bytes += member.file_size
                    if total_bytes > limits.max_zip_total_bytes:
                        raise ResourceLimitError(
                            "ZIP expanded size exceeds the configured total limit."
                        )
                    try:
                        with archive.open(member) as stream:
                            header = read_dicom_header(pydicom, stream)
                    except Exception:
                        continue
                    if (
                        series_selector is not None
                        and dicom_series_selector_for_dataset(header) != series_selector
                    ):
                        continue
                    with archive.open(member) as stream:
                        payload = stream.read(limits.max_zip_member_bytes + 1)
                    if len(payload) > limits.max_zip_member_bytes:
                        raise ResourceLimitError("A ZIP member exceeds the configured byte limit.")
                    try:
                        datasets.append(pydicom.dcmread(BytesIO(payload), force=False))  # type: ignore[attr-defined]
                    except Exception:
                        continue
        except ResourceLimitError:
            raise
        except zipfile.BadZipFile as exc:
            raise DecodeError(f"{path.name!r} is a corrupt or truncated ZIP archive.") from exc
        except (OSError, RuntimeError) as exc:
            raise DecodeError(f"Could not read DICOM ZIP {path.name!r}: {exc}") from exc
        if not datasets:
            raise DecodeError("No decodable DICOM instances were found for the selected series.")
        return datasets

    @staticmethod
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

    def _build_volume(
        self,
        datasets: list[object],
        *,
        container: str,
        limits: LoadLimits,
        cancel: CancelCheck,
        series_selector: str | None = None,
    ) -> ImageVolume:
        image_datasets = [item for item in datasets if hasattr(item, "PixelData")]
        if not image_datasets:
            raise DecodeError("DICOM source contains no pixel-bearing image instances.")

        if series_selector is not None:
            image_datasets = select_dicom_series_datasets(
                image_datasets,
                series_selector,
                limits=limits,
            )

        series_groups = group_dicom_datasets(image_datasets)
        if len(series_groups) != 1:
            raise SeriesSelectionRequiredError(
                summarize_dicom_datasets(image_datasets, limits=limits)
            )
        series = list(series_groups[0])
        frame_counts = [dicom_frame_count(item) for item in series]
        total_frames = sum(frame_counts)
        if total_frames > limits.max_frames:
            raise ResourceLimitError(
                f"DICOM series contains {total_frames:,} frames, above the configured limit."
            )
        if any(count > 1 for count in frame_counts):
            if len(series) != 1 or frame_counts[0] <= 1:
                raise DecodeError(
                    "A DICOM series cannot mix a multi-frame object with separate image instances."
                )
            return self._build_multiframe_volume(
                series[0],
                container=container,
                limits=limits,
                cancel=cancel,
                series_selector=series_selector,
            )

        first = series[0]
        if any(int(getattr(item, "SamplesPerPixel", 1)) != 1 for item in series):
            raise DecodeError(
                "The stable DICOM volume loader currently requires monochrome slices."
            )
        try:
            rows = int(first.Rows)
            columns = int(first.Columns)
        except Exception as exc:
            raise DecodeError("DICOM rows/columns metadata is missing or invalid.") from exc
        if rows * columns > limits.max_pixels:
            raise ResourceLimitError("DICOM slice pixel count exceeds the configured limit.")
        if any(int(item.Rows) != rows or int(item.Columns) != columns for item in series):
            raise DecodeError("DICOM series contains inconsistent slice dimensions.")

        self._validate_frame_of_reference(series)
        orientation = self._orientation(series)
        row_direction_lps, column_direction_lps, normal_lps = orientation
        sorted_series, projections = self._sort_series(series, normal_lps)
        spacing_x, spacing_y = self._in_plane_spacing(sorted_series)
        spacing_z = self._slice_spacing(sorted_series, projections)

        decoded: list[np.ndarray] = []
        slopes: list[float] = []
        intercepts: list[float] = []
        photometric = {
            str(getattr(dataset, "PhotometricInterpretation", "UNKNOWN")).upper().strip()
            for dataset in sorted_series
        }
        if len(photometric) != 1:
            raise DecodeError(
                "DICOM series mixes photometric interpretations; a single display inversion "
                "cannot represent it safely."
            )
        photometric_mode = next(iter(photometric))
        if photometric_mode not in {"MONOCHROME1", "MONOCHROME2"}:
            raise DecodeError(
                f"Unsupported monochrome photometric interpretation {photometric_mode!r}."
            )
        modalities = {
            str(getattr(dataset, "Modality", "UNKNOWN")).upper().strip()
            for dataset in sorted_series
        }
        if len(modalities) != 1:
            raise DecodeError("DICOM series contains inconsistent Modality values.")
        modality = next(iter(modalities)) or "UNKNOWN"
        decoded_bytes = 0
        for dataset in sorted_series:
            raise_if_cancelled(cancel)
            try:
                pixels = np.asarray(dataset.pixel_array)
            except Exception as exc:
                raise DecodeError(
                    "DICOM pixel data could not be decoded; "
                    "a transfer-syntax plugin may be required."
                ) from exc
            if pixels.ndim != 2 or pixels.shape != (rows, columns):
                raise DecodeError(f"Unexpected DICOM pixel array shape {pixels.shape}.")
            slope, intercept = _rescale_parameters(dataset)
            with np.errstate(over="ignore", invalid="ignore"):
                converted = pixels.astype(np.float32) * slope + intercept
            if not np.all(np.isfinite(converted)):
                raise DecodeError("DICOM rescale produced non-finite image values.")
            decoded_bytes += int(converted.nbytes)
            if decoded_bytes > limits.max_decoded_bytes:
                raise ResourceLimitError(
                    "Decoded DICOM volume exceeds the configured memory limit."
                )
            decoded.append(converted)
            slopes.append(slope)
            intercepts.append(intercept)

        volume = np.stack(decoded, axis=0)
        first_position_lps = self._position(sorted_series[0])
        affine_lps = np.eye(4, dtype=np.float64)
        # DICOM IOP's first vector advances image columns (x); its second
        # advances rows (y). PixelSpacing is ordered (row, column).
        affine_lps[:3, 0] = row_direction_lps * spacing_x
        affine_lps[:3, 1] = column_direction_lps * spacing_y
        affine_lps[:3, 2] = normal_lps * spacing_z
        affine_lps[:3, 3] = first_position_lps
        lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
        affine_ras = lps_to_ras @ affine_lps
        basis = affine_ras[:3, :3]
        spacing = tuple(float(value) for value in np.linalg.norm(basis, axis=0))
        direction = basis @ np.diag([1.0 / value for value in spacing])
        origin = tuple(float(value) for value in affine_ras[:3, 3])
        semantics_evidence = [
            _infer_dicom_intensity_semantics(dataset, modality) for dataset in sorted_series
        ]
        semantics_set = {item[0] for item in semantics_evidence}
        if len(semantics_set) == 1:
            semantics = next(iter(semantics_set))
            evidence_sources = {item[1] for item in semantics_evidence}
            semantics_source = (
                next(iter(evidence_sources))
                if len(evidence_sources) == 1
                else "dicom_series_consistent_semantics"
            )
        else:
            semantics = IntensitySemantics.UNKNOWN
            semantics_source = "dicom_series_inconsistent_intensity_semantics"
        if semantics is IntensitySemantics.SUV and np.any(volume < 0.0):
            semantics = IntensitySemantics.UNKNOWN
            semantics_source = "dicom_pet_suv_tags_with_invalid_negative_values"
        return ImageVolume(
            array=volume,
            source_type=SourceType.DICOM,
            intensity_semantics=semantics,
            runtime_metadata={
                "loader": self.name,
                "format": "DICOM",
                "container": container,
                "slice_count": len(sorted_series),
                "ignored_series_count": 0,
                "series_selection": (
                    "explicit-user-selection" if series_selector is not None else "single-series"
                ),
                "series_selector_fingerprint": series_selector,
                "rescale_applied": True,
                "rescale_slope_range": (min(slopes), max(slopes)),
                "rescale_intercept_range": (min(intercepts), max(intercepts)),
                "rescale_type": _normalized_dicom_text(
                    getattr(sorted_series[0], "RescaleType", "")
                ),
                "intensity_semantics_source": semantics_source,
                "units": _normalized_dicom_text(getattr(sorted_series[0], "Units", "")),
                "photometric_interpretations": (photometric_mode,),
                "display_inverted": photometric_mode == "MONOCHROME1",
                "frame_of_reference_uid_present": bool(
                    str(getattr(sorted_series[0], "FrameOfReferenceUID", "")).strip()
                ),
                "canonical_orientation": "RAS+",
                "decoded_bytes": decoded_bytes,
            },
            affine=affine_ras,
            spacing=spacing,
            origin=origin,
            direction=direction,
            modality=modality,
        )

    def _build_multiframe_volume(
        self,
        dataset: object,
        *,
        container: str,
        limits: LoadLimits,
        cancel: CancelCheck,
        series_selector: str | None,
    ) -> ImageVolume:
        """Decode one monochrome Enhanced CT/MR object using frame geometry."""

        frame_count = dicom_frame_count(dataset)
        if frame_count <= 1:
            raise DecodeError("The selected DICOM object is not multi-frame.")
        if int(getattr(dataset, "SamplesPerPixel", 1)) != 1:
            raise DecodeError("Enhanced DICOM loading currently requires monochrome frames.")
        try:
            rows = int(dataset.Rows)  # type: ignore[attr-defined]
            columns = int(dataset.Columns)  # type: ignore[attr-defined]
        except Exception as exc:
            raise DecodeError("DICOM rows/columns metadata is missing or invalid.") from exc
        if rows <= 0 or columns <= 0:
            raise DecodeError("DICOM rows/columns must be positive.")
        if rows * columns > limits.max_pixels:
            raise ResourceLimitError("DICOM frame pixel count exceeds the configured limit.")

        photometric_mode = (
            str(getattr(dataset, "PhotometricInterpretation", "UNKNOWN")).upper().strip()
        )
        if photometric_mode not in {"MONOCHROME1", "MONOCHROME2"}:
            raise DecodeError(
                f"Unsupported monochrome photometric interpretation {photometric_mode!r}."
            )
        modality = str(getattr(dataset, "Modality", "UNKNOWN")).upper().strip() or "UNKNOWN"

        frames: list[SimpleNamespace] = []
        for index in range(frame_count):
            orientation = dicom_frame_attribute(
                dataset,
                index,
                sequence_name="PlaneOrientationSequence",
                attribute_name="ImageOrientationPatient",
            )
            position = dicom_frame_attribute(
                dataset,
                index,
                sequence_name="PlanePositionSequence",
                attribute_name="ImagePositionPatient",
            )
            pixel_spacing = dicom_frame_attribute(
                dataset,
                index,
                sequence_name="PixelMeasuresSequence",
                attribute_name="PixelSpacing",
            )
            slice_thickness = dicom_frame_attribute(
                dataset,
                index,
                sequence_name="PixelMeasuresSequence",
                attribute_name="SliceThickness",
            )
            between_slices = dicom_frame_attribute(
                dataset,
                index,
                sequence_name="PixelMeasuresSequence",
                attribute_name="SpacingBetweenSlices",
            )
            frames.append(
                SimpleNamespace(
                    frame_index=index,
                    ImageOrientationPatient=orientation,
                    ImagePositionPatient=position,
                    PixelSpacing=pixel_spacing,
                    SliceThickness=slice_thickness,
                    SpacingBetweenSlices=between_slices,
                )
            )

        row_direction_lps, column_direction_lps, normal_lps = self._orientation(frames)
        sorted_frames, projections = self._sort_series(frames, normal_lps)
        spacing_x, spacing_y = self._in_plane_spacing(sorted_frames)
        spacing_z = self._slice_spacing(sorted_frames, projections)

        raise_if_cancelled(cancel)
        try:
            stored_pixels = np.asarray(dataset.pixel_array)  # type: ignore[attr-defined]
        except Exception as exc:
            raise DecodeError(
                "Enhanced DICOM pixel data could not be decoded; a transfer-syntax plugin "
                "may be required."
            ) from exc
        if stored_pixels.shape != (frame_count, rows, columns):
            raise DecodeError(f"Unexpected Enhanced DICOM pixel array shape {stored_pixels.shape}.")

        decoded: list[np.ndarray] = []
        slopes: list[float] = []
        intercepts: list[float] = []
        semantics_evidence: list[tuple[IntensitySemantics, str]] = []
        rescale_types: list[str] = []
        decoded_bytes = 0
        for frame in sorted_frames:
            raise_if_cancelled(cancel)
            index = int(frame.frame_index)
            slope_value = dicom_frame_attribute(
                dataset,
                index,
                sequence_name="PixelValueTransformationSequence",
                attribute_name="RescaleSlope",
            )
            intercept_value = dicom_frame_attribute(
                dataset,
                index,
                sequence_name="PixelValueTransformationSequence",
                attribute_name="RescaleIntercept",
            )
            transform = SimpleNamespace(
                Units=getattr(dataset, "Units", ""),
                SUVType=getattr(dataset, "SUVType", ""),
                DecayCorrection=getattr(dataset, "DecayCorrection", ""),
                CorrectedImage=getattr(dataset, "CorrectedImage", None),
            )
            if slope_value is not None:
                transform.RescaleSlope = slope_value
            if intercept_value is not None:
                transform.RescaleIntercept = intercept_value
            transform.RescaleType = (
                dicom_frame_attribute(
                    dataset,
                    index,
                    sequence_name="PixelValueTransformationSequence",
                    attribute_name="RescaleType",
                )
                or ""
            )
            slope, intercept = _rescale_parameters(transform)
            with np.errstate(over="ignore", invalid="ignore"):
                converted = stored_pixels[index].astype(np.float32) * slope + intercept
            if not np.all(np.isfinite(converted)):
                raise DecodeError("DICOM rescale produced non-finite image values.")
            decoded_bytes += int(converted.nbytes)
            if decoded_bytes > limits.max_decoded_bytes:
                raise ResourceLimitError(
                    "Decoded Enhanced DICOM volume exceeds the configured memory limit."
                )
            decoded.append(converted)
            slopes.append(slope)
            intercepts.append(intercept)
            rescale_types.append(_normalized_dicom_text(transform.RescaleType))
            semantics_evidence.append(_infer_dicom_intensity_semantics(transform, modality))

        volume = np.stack(decoded, axis=0)
        first_position_lps = self._position(sorted_frames[0])
        affine_lps = np.eye(4, dtype=np.float64)
        affine_lps[:3, 0] = row_direction_lps * spacing_x
        affine_lps[:3, 1] = column_direction_lps * spacing_y
        affine_lps[:3, 2] = normal_lps * spacing_z
        affine_lps[:3, 3] = first_position_lps
        affine_ras = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine_lps
        basis = affine_ras[:3, :3]
        spacing = tuple(float(value) for value in np.linalg.norm(basis, axis=0))
        direction = basis @ np.diag([1.0 / value for value in spacing])
        origin = tuple(float(value) for value in affine_ras[:3, 3])

        semantics_set = {item[0] for item in semantics_evidence}
        evidence_sources = {item[1] for item in semantics_evidence}
        semantics = (
            next(iter(semantics_set)) if len(semantics_set) == 1 else IntensitySemantics.UNKNOWN
        )
        semantics_source = (
            next(iter(evidence_sources))
            if len(semantics_set) == 1 and len(evidence_sources) == 1
            else "dicom_multiframe_inconsistent_intensity_semantics"
        )
        if semantics is IntensitySemantics.SUV and np.any(volume < 0.0):
            semantics = IntensitySemantics.UNKNOWN
            semantics_source = "dicom_pet_suv_tags_with_invalid_negative_values"

        return ImageVolume(
            array=volume,
            source_type=SourceType.DICOM,
            intensity_semantics=semantics,
            runtime_metadata={
                "loader": self.name,
                "format": "DICOM_ENHANCED_MULTIFRAME",
                "container": container,
                "instance_count": 1,
                "slice_count": frame_count,
                "frame_count": frame_count,
                "enhanced_multiframe": True,
                "ignored_series_count": 0,
                "series_selection": (
                    "explicit-user-selection" if series_selector is not None else "single-series"
                ),
                "series_selector_fingerprint": series_selector,
                "rescale_applied": any(
                    not np.isclose(slope, 1.0) or not np.isclose(intercept, 0.0)
                    for slope, intercept in zip(slopes, intercepts, strict=True)
                ),
                "rescale_slope_range": (min(slopes), max(slopes)),
                "rescale_intercept_range": (min(intercepts), max(intercepts)),
                "rescale_types": tuple(sorted(set(rescale_types))),
                "intensity_semantics_source": semantics_source,
                "photometric_interpretations": (photometric_mode,),
                "display_inverted": photometric_mode == "MONOCHROME1",
                "frame_of_reference_uid_present": bool(
                    str(getattr(dataset, "FrameOfReferenceUID", "")).strip()
                ),
                "canonical_orientation": "RAS+",
                "decoded_bytes": decoded_bytes,
            },
            affine=affine_ras,
            spacing=spacing,
            origin=origin,
            direction=direction,
            modality=modality,
        )

    @staticmethod
    def _validate_frame_of_reference(series: list[object]) -> None:
        """Reject slices that cannot safely share one physical coordinate frame.

        ``FrameOfReferenceUID`` is optional in some teaching fixtures and older
        exports, so absence alone is not fatal.  If any instance declares it,
        however, every instance must declare the same value.  Orientation is
        checked here as a series property rather than silently trusting the
        first slice; otherwise a mixed localizer/axial series can produce a
        plausible-looking but physically false volume.
        """

        frame_uids = [
            str(getattr(dataset, "FrameOfReferenceUID", "")).strip() for dataset in series
        ]
        present_uids = {value for value in frame_uids if value}
        if present_uids and (len(present_uids) != 1 or any(not value for value in frame_uids)):
            raise DecodeError(
                "DICOM series contains missing or inconsistent FrameOfReferenceUID values."
            )

        reference: np.ndarray | None = None
        for dataset in series:
            try:
                orientation = np.asarray(dataset.ImageOrientationPatient, dtype=np.float64)
            except Exception as exc:
                raise DecodeError(
                    "DICOM ImageOrientationPatient is required on every slice."
                ) from exc
            if orientation.shape != (6,) or not np.all(np.isfinite(orientation)):
                raise DecodeError("DICOM ImageOrientationPatient must contain six finite values.")
            if reference is None:
                reference = orientation
            elif not np.allclose(orientation, reference, rtol=1e-5, atol=1e-5):
                raise DecodeError(
                    "DICOM series contains inconsistent ImageOrientationPatient values."
                )

    @staticmethod
    def _orientation(series: list[object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        def normalized(dataset: object) -> tuple[np.ndarray, np.ndarray]:
            try:
                values = np.asarray(dataset.ImageOrientationPatient, dtype=np.float64)
            except Exception as exc:
                raise DecodeError(
                    "DICOM ImageOrientationPatient is required for every slice."
                ) from exc
            if values.shape != (6,) or not np.all(np.isfinite(values)):
                raise DecodeError("DICOM ImageOrientationPatient must contain six finite values.")
            row_direction = values[:3].copy()
            column_direction = values[3:].copy()
            row_norm = float(np.linalg.norm(row_direction))
            column_norm = float(np.linalg.norm(column_direction))
            if row_norm <= 0.0 or column_norm <= 0.0:
                raise DecodeError("DICOM row and column directions must be non-zero.")
            row_direction /= row_norm
            column_direction /= column_norm
            if not np.isclose(np.dot(row_direction, column_direction), 0.0, atol=1e-4):
                raise DecodeError("DICOM row and column directions are not orthogonal.")
            return row_direction, column_direction

        row_direction, column_direction = normalized(series[0])
        for dataset in series[1:]:
            other_row, other_column = normalized(dataset)
            if not np.allclose(
                other_row,
                row_direction,
                rtol=1e-4,
                atol=1e-5,
            ) or not np.allclose(
                other_column,
                column_direction,
                rtol=1e-4,
                atol=1e-5,
            ):
                raise DecodeError(
                    "DICOM series contains inconsistent ImageOrientationPatient values."
                )
        normal = np.cross(row_direction, column_direction)
        normal /= np.linalg.norm(normal)
        return row_direction, column_direction, normal

    @staticmethod
    def _position(dataset: object) -> np.ndarray:
        try:
            value = np.asarray(dataset.ImagePositionPatient, dtype=np.float64)
        except Exception as exc:
            raise DecodeError(
                "DICOM ImagePositionPatient is required for reliable volume geometry."
            ) from exc
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise DecodeError("DICOM ImagePositionPatient must contain three finite values.")
        return value

    def _sort_series(
        self,
        series: list[object],
        normal: np.ndarray,
    ) -> tuple[list[object], np.ndarray]:
        try:
            projections = np.asarray([np.dot(self._position(item), normal) for item in series])
            order = np.argsort(projections)
            return [series[int(index)] for index in order], projections[order]
        except DecodeError:
            if all(hasattr(item, "InstanceNumber") for item in series):
                ordered = sorted(series, key=lambda item: int(item.InstanceNumber))
                return ordered, np.arange(len(ordered), dtype=np.float64)
            raise

    @staticmethod
    def _in_plane_spacing(series: list[object]) -> tuple[float, float]:
        try:
            values = np.asarray(series[0].PixelSpacing, dtype=np.float64)
        except Exception as exc:
            raise DecodeError("DICOM PixelSpacing is required for physical measurements.") from exc
        if values.shape != (2,) or np.any(values <= 0) or not np.all(np.isfinite(values)):
            raise DecodeError("DICOM PixelSpacing must contain two positive finite values.")
        for dataset in series[1:]:
            other = np.asarray(getattr(dataset, "PixelSpacing", values), dtype=np.float64)
            if not np.allclose(other, values, rtol=1e-4, atol=1e-5):
                raise DecodeError("DICOM series contains inconsistent in-plane pixel spacing.")
        return float(values[1]), float(values[0])

    @staticmethod
    def _slice_spacing(series: list[object], projections: np.ndarray) -> float:
        if len(series) > 1 and len(np.unique(projections)) > 1:
            differences = np.diff(projections)
            spacing = float(np.median(np.abs(differences)))
            if spacing <= 0 or not np.allclose(np.abs(differences), spacing, rtol=0.05, atol=1e-3):
                raise DecodeError(
                    "DICOM slice positions are non-uniform; resample the series before loading."
                )
            return spacing
        for attribute in ("SpacingBetweenSlices", "SliceThickness"):
            value = getattr(series[0], attribute, None)
            if value is not None and float(value) > 0:
                return float(value)
        raise DecodeError("DICOM slice spacing is unavailable; physical volume geometry is unsafe.")
