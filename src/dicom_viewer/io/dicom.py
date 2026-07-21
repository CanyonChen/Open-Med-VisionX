"""DICOM folder/ZIP loader with HU conversion and RAS+ geometry."""

from __future__ import annotations

import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path, PurePosixPath

import numpy as np

from ..domain.images import ImageVolume, IntensitySemantics, SourceType
from ..errors import DecodeError, FormatMismatchError, MissingDependencyError, ResourceLimitError
from .base import CancelCheck, ImageLoader, LoadLimits, PathLike, ProbeResult, raise_if_cancelled


def _has_dicom_preamble(header: bytes) -> bool:
    return len(header) >= 132 and header[128:132] == b"DICM"


class DicomLoader(ImageLoader):
    name = "dicom"

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
    ) -> ImageVolume:
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
        if path.is_dir():
            datasets = self._read_directory(path, pydicom, active_limits, cancel)
            container = "directory"
        elif probe.format_name == "DICOM_ZIP":
            if path.suffix.lower() != ".zip":
                raise FormatMismatchError(
                    "ZIP signature detected, but the filename must end in .zip."
                )
            datasets = self._read_zip(path, pydicom, active_limits, cancel)
            container = "zip"
        else:
            datasets = self._read_single(path, pydicom, active_limits)
            container = "file"
        raise_if_cancelled(cancel)
        return self._build_volume(
            datasets, container=container, limits=active_limits, cancel=cancel
        )

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
                datasets.append(pydicom.dcmread(str(resolved), force=False))  # type: ignore[attr-defined]
            except Exception:
                # Directories often contain DICOMDIR or unrelated notes.  A
                # supported-looking .dcm that fails is reported if no valid
                # image remains, without exposing local paths in logs.
                continue
        if not datasets:
            raise DecodeError("No decodable DICOM instances were found in the selected directory.")
        return datasets

    def _read_zip(
        self,
        path: Path,
        pydicom: object,
        limits: LoadLimits,
        cancel: CancelCheck,
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
            raise DecodeError("No decodable DICOM instances were found in the selected ZIP.")
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
    ) -> ImageVolume:
        image_datasets = [item for item in datasets if hasattr(item, "PixelData")]
        if not image_datasets:
            raise DecodeError("DICOM source contains no pixel-bearing image instances.")

        by_series: dict[str, list[object]] = defaultdict(list)
        for dataset in image_datasets:
            by_series[str(getattr(dataset, "SeriesInstanceUID", "__missing__"))].append(dataset)
        series = max(by_series.values(), key=len)
        ignored_series = len(by_series) - 1
        if len(series) > limits.max_frames:
            raise ResourceLimitError(
                f"DICOM series contains {len(series):,} slices, above the configured limit."
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
        modality = str(getattr(sorted_series[0], "Modality", "UNKNOWN")).upper()
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
            slope = float(getattr(dataset, "RescaleSlope", 1.0))
            intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
            converted = pixels.astype(np.float32) * slope + intercept
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
        semantics = (
            IntensitySemantics.HOUNSFIELD_UNIT
            if modality == "CT"
            else IntensitySemantics.QUANTITATIVE
        )
        return ImageVolume(
            array=volume,
            source_type=SourceType.DICOM,
            intensity_semantics=semantics,
            runtime_metadata={
                "loader": self.name,
                "format": "DICOM",
                "container": container,
                "slice_count": len(sorted_series),
                "ignored_series_count": ignored_series,
                "rescale_applied": True,
                "rescale_slope_range": (min(slopes), max(slopes)),
                "rescale_intercept_range": (min(intercepts), max(intercepts)),
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
