"""Conservative import of user-selected clinical label maps.

PNG, explicitly lossless TIFF, and 3-D NIfTI are accepted.  The importer
never guesses a target series and never resamples: callers must supply an
``ImageSeries`` and geometry mismatches are returned as hidden, pending
layers whose ``reference_report`` explains why overlay is disabled.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from ..domain.images import SourceType
from ..domain.studies import (
    CompressionKind,
    ImageSeries,
    LabelDefinition,
    LayerCreator,
    LayerPresentation,
    LayerReference,
    LayerValidationState,
    SegmentationLayer,
    SegmentationValueType,
    SourceFormat,
    SourceReference,
    SpatialGeometry,
)
from ..errors import (
    DecodeError,
    FormatMismatchError,
    MissingDependencyError,
    OperationCancelled,
    ResourceLimitError,
    UnsupportedFormatError,
    ValidationError,
)
from .base import CancelCheck, LoadLimits, raise_if_cancelled
from .nifti import NiftiLoader

_RASTER_EXTENSIONS = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}
_LOSSLESS_TIFF_COMPRESSION_CODES = {
    1,  # none
    2,  # CCITT 1-D
    3,  # Group 3 fax
    4,  # Group 4 fax
    5,  # LZW
    8,  # Adobe Deflate
    32773,  # PackBits
    32946,  # Deflate
    34925,  # LZMA
    50000,  # Zstandard
}
_INTEGER_RASTER_MODES = {
    "1",
    "L",
    "P",
    "I",
    "I;16",
    "I;16B",
    "I;16L",
    "I;16N",
    "I;16S",
    "I;16BS",
    "I;16LS",
    "I;32",
    "I;32B",
    "I;32L",
    "I;32S",
    "I;32BS",
    "I;32LS",
}
_LABEL_COLORS = (
    "#0A84FF",
    "#30D158",
    "#FF9F0A",
    "#BF5AF2",
    "#FF375F",
    "#64D2FF",
    "#FFD60A",
    "#5E5CE6",
)


@dataclass(frozen=True, slots=True)
class LabelMapLimits:
    """Hard limits applied to untrusted local label-map inputs."""

    max_file_bytes: int = 512 * 1024 * 1024
    max_voxels: int = 100_000_000
    max_frames: int = 2_048
    max_decoded_bytes: int = 512 * 1024 * 1024
    max_unique_labels: int = 4_096

    def __post_init__(self) -> None:
        for name in (
            "max_file_bytes",
            "max_voxels",
            "max_frames",
            "max_decoded_bytes",
            "max_unique_labels",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValidationError(f"{name} must be positive.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class _DecodedLabelMap:
    array: np.ndarray
    geometry: SpatialGeometry
    source_type: SourceType
    source_format: SourceFormat
    compression: CompressionKind
    digest: str
    provenance: dict[str, Any]


def import_label_map(
    source: str | Path,
    *,
    reference_series: ImageSeries,
    label_schema: Sequence[LabelDefinition] | None = None,
    volume_index: int | None = None,
    limits: LabelMapLimits | None = None,
    cancel: CancelCheck = None,
) -> SegmentationLayer:
    """Import a discrete label map without automatically changing its grid.

    ``reference_series`` is mandatory and represents an explicit user choice.
    A raster has no patient-space affine, so its grid axes are associated with
    the chosen reference while its own dimensions are preserved.  Only an
    exact shape/affine match is made visible.  A NIfTI keeps its canonical
    RAS+ affine and is likewise left hidden when it does not match.

    A 4-D NIfTI always requires ``volume_index``.  ``label_schema`` may retain
    reviewed names and colors; otherwise a deterministic schema is generated
    from the stored non-zero integer values.
    """

    if not isinstance(reference_series, ImageSeries):
        raise ValidationError("reference_series must be an ImageSeries selected by the user.")
    active_limits = limits or LabelMapLimits()
    if not isinstance(active_limits, LabelMapLimits):
        raise ValidationError("limits must be LabelMapLimits.")
    path = Path(source)
    encoded_size = _inspect_path(path, active_limits)
    detected_format = _detect_format(path)
    raise_if_cancelled(cancel)

    if detected_format in {"PNG", "TIFF"}:
        if volume_index is not None:
            raise ValidationError("volume_index can only be used with a 4-D NIfTI label map.")
        payload = _read_snapshot(path, encoded_size, active_limits, cancel)
        array, raster_provenance = _decode_raster(
            payload,
            detected_format,
            limits=active_limits,
            cancel=cancel,
        )
        digest = hashlib.sha256(payload).hexdigest()
        shape_zyx = _spatial_shape(array)
        geometry = SpatialGeometry(shape_zyx, reference_series.geometry.affine_ras)
        decoded = _DecodedLabelMap(
            array=array,
            geometry=geometry,
            source_type=SourceType.RASTER,
            source_format=(SourceFormat.PNG if detected_format == "PNG" else SourceFormat.TIFF),
            compression=CompressionKind.LOSSLESS,
            digest=digest,
            provenance={
                **raster_provenance,
                "geometry_basis": "explicit-reference-grid",
            },
        )
    else:
        decoded = _decode_nifti(
            path,
            volume_index=volume_index,
            limits=active_limits,
            cancel=cancel,
        )

    raise_if_cancelled(cancel)
    array = _validate_discrete_array(decoded.array, active_limits)
    labels, schema_origin = _prepare_label_schema(array, label_schema, active_limits)
    geometry_matches = decoded.geometry.matches(reference_series.geometry)
    reference = LayerReference(
        local_series_id=reference_series.series_id,
        dicom_series_uid=reference_series.series_instance_uid,
        frame_of_reference_uid=reference_series.frame_of_reference_uid,
    )
    source_reference = SourceReference(
        source_id=f"label-map-{decoded.digest[:16]}",
        source_type=decoded.source_type,
        source_format=decoded.source_format,
        content_sha256=decoded.digest,
        compression=decoded.compression,
        provenance={
            "importer": "clinical-label-map",
            "selection": "explicit-local-selection",
            "content_identity": "sha256",
            **decoded.provenance,
        },
    )
    return SegmentationLayer(
        layer_id=f"label-map-{decoded.digest[:16]}",
        series_id=reference_series.series_id,
        name="Imported label map",
        source=source_reference,
        created_by=LayerCreator.IMPORT,
        validation_state=(
            LayerValidationState.VALIDATED if geometry_matches else LayerValidationState.PENDING
        ),
        presentation=LayerPresentation(
            visible=geometry_matches,
            opacity=0.45,
            color=labels[0].color,
        ),
        provenance={
            "geometry_match": "matched" if geometry_matches else "requires-resampling",
            "overlay_enabled": geometry_matches,
            "automatic_resampling": False,
            "label_schema_origin": schema_origin,
            **decoded.provenance,
        },
        array=array,
        geometry=decoded.geometry,
        value_type=SegmentationValueType.DISCRETE,
        reference=reference,
        labels=labels,
        reference_geometry=reference_series.geometry,
    )


def _inspect_path(path: Path, limits: LabelMapLimits) -> int:
    try:
        if not path.is_file():
            raise DecodeError("The label-map source must be a regular file.")
        size = int(path.stat().st_size)
    except DecodeError:
        raise
    except OSError as exc:
        raise DecodeError("The label-map source could not be inspected.") from exc
    if size <= 0:
        raise DecodeError("The selected label-map file is empty.")
    if size > limits.max_file_bytes:
        raise ResourceLimitError("The label-map file exceeds the encoded-size limit.")
    return size


def _raster_signature(prefix: bytes) -> str | None:
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if prefix.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if prefix.startswith((b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")):
        return "TIFF"
    return None


def _detect_format(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            prefix = stream.read(16)
    except OSError as exc:
        raise DecodeError("The label-map source could not be read.") from exc

    detected_raster = _raster_signature(prefix)
    expected_raster = _RASTER_EXTENSIONS.get(path.suffix.lower())
    if detected_raster == "JPEG":
        raise UnsupportedFormatError(
            "JPEG label maps are always rejected because JPEG compression is lossy."
        )
    if detected_raster is not None:
        if expected_raster != detected_raster:
            raise FormatMismatchError(
                "The selected label map's extension does not match its raster signature."
            )
        return detected_raster
    if expected_raster == "JPEG":
        raise UnsupportedFormatError(
            "JPEG label maps are always rejected because JPEG compression is lossy."
        )
    if expected_raster is not None:
        raise DecodeError("The selected raster label map has an invalid or truncated signature.")

    probe = NiftiLoader().probe(path)
    signature_valid = bool(probe.details.get("signature_valid"))
    nifti_suffix = path.name.lower().endswith((".nii", ".nii.gz"))
    if signature_valid and not nifti_suffix:
        raise FormatMismatchError(
            "A NIfTI label map must use the .nii or .nii.gz filename extension."
        )
    if nifti_suffix and not signature_valid:
        raise DecodeError("The selected file does not contain a valid NIfTI signature.")
    if signature_valid:
        return "NIFTI"
    raise UnsupportedFormatError(
        "Label-map import supports lossless PNG/TIFF and single-volume NIfTI files."
    )


def _read_snapshot(
    path: Path,
    inspected_size: int,
    limits: LabelMapLimits,
    cancel: CancelCheck,
) -> bytes:
    raise_if_cancelled(cancel)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise DecodeError("The raster label map could not be read.") from exc
    if not payload:
        raise DecodeError("The selected label-map file is empty.")
    if len(payload) > limits.max_file_bytes:
        raise ResourceLimitError("The label-map file exceeds the encoded-size limit.")
    if len(payload) != inspected_size:
        raise DecodeError("The selected raster label map changed while it was being read.")
    return payload


def _decode_raster(
    payload: bytes,
    format_name: str,
    *,
    limits: LabelMapLimits,
    cancel: CancelCheck,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - base dependency
        raise MissingDependencyError(
            "PNG/TIFF label-map import requires Pillow from the base dependencies."
        ) from exc

    frames: list[np.ndarray] = []
    orientations: list[int] = []
    compression_codes: list[int] = []
    voxel_count = 0
    decoded_bytes = 0
    try:
        image = Image.open(BytesIO(payload))
        try:
            actual_format = str(image.format or "").upper()
            if actual_format != format_name:
                raise DecodeError("The raster decoder disagrees with the selected file signature.")
            frame_count = int(getattr(image, "n_frames", 1))
            if frame_count <= 0 or frame_count > limits.max_frames:
                raise ResourceLimitError(
                    "The raster label-map frame count exceeds the configured limit."
                )
            for frame_index in range(frame_count):
                raise_if_cancelled(cancel)
                if frame_index:
                    image.seek(frame_index)
                width, height = (int(item) for item in image.size)
                if width <= 0 or height <= 0:
                    raise DecodeError("The raster label map reports invalid dimensions.")
                voxel_count += width * height
                if voxel_count > limits.max_voxels:
                    raise ResourceLimitError(
                        "The raster label map exceeds the configured voxel limit."
                    )
                if image.mode not in _INTEGER_RASTER_MODES:
                    raise ValidationError(
                        "A label map must be a single-channel integer raster; "
                        f"color/continuous mode {image.mode!r} is not accepted."
                    )
                if format_name == "TIFF":
                    compression_codes.append(_verified_tiff_compression(image))
                frame = image.copy()
                orientation = int(frame.getexif().get(274, 1) or 1)
                if not 1 <= orientation <= 8:
                    raise DecodeError("The raster label map has an invalid EXIF orientation.")
                oriented = ImageOps.exif_transpose(frame)
                array = np.asarray(oriented)
                if array.ndim != 2 or (
                    not np.issubdtype(array.dtype, np.integer) and array.dtype != np.bool_
                ):
                    raise ValidationError(
                        "A label map must decode to one plane of integer label values."
                    )
                array = np.array(array, copy=True)
                decoded_bytes += int(array.nbytes)
                if decoded_bytes > limits.max_decoded_bytes:
                    raise ResourceLimitError(
                        "The decoded raster label map exceeds the configured memory limit."
                    )
                frames.append(array)
                orientations.append(orientation)
        finally:
            image.close()
    except (OperationCancelled, ResourceLimitError, ValidationError, DecodeError):
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise DecodeError("The raster label map is corrupt or could not be decoded.") from exc

    shapes = {item.shape for item in frames}
    dtypes = {item.dtype.str for item in frames}
    if len(shapes) != 1 or len(dtypes) != 1:
        raise DecodeError("TIFF label-map pages must use identical dimensions and dtypes.")
    array = frames[0] if len(frames) == 1 else np.stack(frames, axis=0)
    return array, {
        "decoder": "pillow",
        "format": format_name,
        "lossless_verified": True,
        "frame_count": len(frames),
        "orientation_values": tuple(orientations),
        "compression_codes": tuple(compression_codes),
    }


def _verified_tiff_compression(image: Any) -> int:
    tags = getattr(image, "tag_v2", None)
    raw_value = None if tags is None else tags.get(259)
    try:
        compression = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "TIFF label maps require an explicitly identifiable lossless compression."
        ) from exc
    if compression not in _LOSSLESS_TIFF_COMPRESSION_CODES:
        raise ValidationError(
            "TIFF label-map compression is lossy or cannot be verified as lossless."
        )
    return compression


def _decode_nifti(
    path: Path,
    *,
    volume_index: int | None,
    limits: LabelMapLimits,
    cancel: CancelCheck,
) -> _DecodedLabelMap:
    digest_before = _sha256_path(path, limits, cancel)
    loader_limits = LoadLimits(
        max_pixels=limits.max_voxels,
        max_frames=limits.max_frames,
        max_decoded_bytes=limits.max_decoded_bytes,
    )
    volume = NiftiLoader().load(
        path,
        limits=loader_limits,
        cancel=cancel,
        intensity_semantics="discrete_label",
        volume_index=volume_index,
    )
    raise_if_cancelled(cancel)
    if volume.array.size > limits.max_voxels:
        raise ResourceLimitError("The NIfTI label map exceeds the configured voxel limit.")
    if int(volume.array.nbytes) > limits.max_decoded_bytes:
        raise ResourceLimitError("The decoded NIfTI label map exceeds the configured memory limit.")
    if not np.issubdtype(volume.array.dtype, np.integer) and volume.array.dtype != np.bool_:
        raise ValidationError("NIfTI label maps must store discrete labels with an integer dtype.")
    digest_after = _sha256_path(path, limits, cancel)
    if digest_before != digest_after:
        raise DecodeError("The selected NIfTI label map changed while it was being decoded.")
    selected_index = volume.runtime_metadata.get("selected_volume_index")
    return _DecodedLabelMap(
        array=volume.array,
        geometry=SpatialGeometry.from_volume(volume),
        source_type=SourceType.NIFTI,
        source_format=SourceFormat.NIFTI,
        compression=(
            CompressionKind.LOSSLESS
            if volume.runtime_metadata.get("storage_compression") == "gzip"
            else CompressionKind.NONE
        ),
        digest=digest_before,
        provenance={
            "decoder": "nibabel",
            "format": "NIFTI",
            "canonical_orientation": "RAS+",
            "selected_volume_index": selected_index,
            "volume_selection": volume.runtime_metadata.get("volume_selection"),
        },
    )


def _sha256_path(path: Path, limits: LabelMapLimits, cancel: CancelCheck) -> str:
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                raise_if_cancelled(cancel)
                total += len(chunk)
                if total > limits.max_file_bytes:
                    raise ResourceLimitError("The label-map file exceeds the encoded-size limit.")
                digest.update(chunk)
    except (OperationCancelled, ResourceLimitError):
        raise
    except OSError as exc:
        raise DecodeError("The label-map source could not be read.") from exc
    if total <= 0:
        raise DecodeError("The selected label-map file is empty.")
    return digest.hexdigest()


def _spatial_shape(array: np.ndarray) -> tuple[int, int, int]:
    if array.ndim == 2:
        return 1, int(array.shape[0]), int(array.shape[1])
    if array.ndim == 3:
        return tuple(int(item) for item in array.shape)
    raise ValidationError("Label-map arrays must be two- or three-dimensional.")


def _validate_discrete_array(array: np.ndarray, limits: LabelMapLimits) -> np.ndarray:
    candidate = np.asarray(array)
    if candidate.ndim not in {2, 3}:
        raise ValidationError("Label-map arrays must be two- or three-dimensional.")
    if candidate.size > limits.max_voxels:
        raise ResourceLimitError("The label map exceeds the configured voxel limit.")
    if int(candidate.nbytes) > limits.max_decoded_bytes:
        raise ResourceLimitError("The label map exceeds the configured memory limit.")
    if not np.issubdtype(candidate.dtype, np.integer) and candidate.dtype != np.bool_:
        raise ValidationError("Label maps must contain discrete integer values.")
    if np.any(candidate < 0):
        raise ValidationError("Label-map values cannot be negative.")
    return candidate


def _prepare_label_schema(
    array: np.ndarray,
    label_schema: Sequence[LabelDefinition] | None,
    limits: LabelMapLimits,
) -> tuple[tuple[LabelDefinition, ...], str]:
    unique_values = tuple(int(item) for item in np.unique(array))
    if len(unique_values) > limits.max_unique_labels:
        raise ResourceLimitError("The label map exceeds the configured unique-label limit.")
    if label_schema is None:
        foreground = tuple(value for value in unique_values if value != 0)
        values = foreground or (0,)
        labels = tuple(
            LabelDefinition(
                value=value,
                name="Background" if value == 0 else f"Label {value}",
                color=_LABEL_COLORS[index % len(_LABEL_COLORS)],
                visible=value != 0,
            )
            for index, value in enumerate(values)
        )
        return labels, "derived-from-stored-values"

    labels = tuple(label_schema)
    if not labels or not all(isinstance(item, LabelDefinition) for item in labels):
        raise ValidationError("label_schema must contain one or more LabelDefinition values.")
    if len(labels) > limits.max_unique_labels:
        raise ResourceLimitError("The label schema exceeds the configured label limit.")
    schema_values = {item.value for item in labels}
    missing = sorted(set(unique_values) - {0} - schema_values)
    if missing:
        raise ValidationError(f"The label schema is missing stored values: {missing}.")
    return labels, "caller-supplied"


__all__ = ["LabelMapLimits", "import_label_map"]
