"""Safe, geometry-preserving import of DICOM SEG and RT Structure Set objects.

This module is intentionally separate from the ordinary DICOM image loader.
An encoded object is read once, inspected without pixels, and dispatched only
after its SOP Class UID identifies it as DICOM SEG or RTSTRUCT.  Imported
objects never inherit a target series reference from caller context: the
reference must be present in the DICOM object and match the supplied series.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from ..domain.images import SourceType
from ..domain.studies import (
    CompressionKind,
    ContourLayer,
    ContourPath,
    FractionalType,
    ImageSeries,
    LabelDefinition,
    LayerCreator,
    LayerPresentation,
    LayerReference,
    LayerValidationState,
    RegionOfInterest,
    SegmentationLayer,
    SegmentationValueType,
    SourceFormat,
    SourceReference,
    SpatialGeometry,
)
from ..errors import (
    DecodeError,
    MissingDependencyError,
    ResourceLimitError,
    UnsupportedFormatError,
    ValidationError,
)

SEGMENTATION_STORAGE_UID = "1.2.840.10008.5.1.4.1.1.66.4"
RT_STRUCTURE_SET_STORAGE_UID = "1.2.840.10008.5.1.4.1.1.481.3"

_DICOM_UID_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)*$")
_ABSOLUTE_PATH_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|[/\\]{2}|/)")
_LPS_TO_RAS = np.diag((-1.0, -1.0, 1.0, 1.0))
_GEOMETRY_ATOL_MM = 1e-3
_DIRECTION_ATOL = 1e-4

_UNCOMPRESSED_TRANSFER_SYNTAXES = {
    "1.2.840.10008.1.2",
    "1.2.840.10008.1.2.1",
    "1.2.840.10008.1.2.1.99",
    "1.2.840.10008.1.2.2",
}
_LOSSLESS_TRANSFER_SYNTAXES = {
    "1.2.840.10008.1.2.4.57",  # JPEG Lossless, Non-Hierarchical
    "1.2.840.10008.1.2.4.70",  # JPEG Lossless, First-Order Prediction
    "1.2.840.10008.1.2.4.80",  # JPEG-LS Lossless
    "1.2.840.10008.1.2.4.90",  # JPEG 2000 Lossless
    "1.2.840.10008.1.2.4.92",  # JPEG 2000 Multi-component Lossless
    "1.2.840.10008.1.2.4.201",  # HTJ2K Lossless
    "1.2.840.10008.1.2.5",  # RLE Lossless
}
_LOSSY_TRANSFER_SYNTAXES = {
    "1.2.840.10008.1.2.4.50",  # JPEG Baseline
    "1.2.840.10008.1.2.4.51",  # JPEG Extended
    "1.2.840.10008.1.2.4.81",  # JPEG-LS Near-lossless
    "1.2.840.10008.1.2.4.91",  # JPEG 2000
    "1.2.840.10008.1.2.4.93",  # JPEG 2000 Multi-component
    "1.2.840.10008.1.2.4.202",  # HTJ2K
    "1.2.840.10008.1.2.4.203",  # HTJ2K RPCL
}
_FALLBACK_COLORS = (
    "#0A84FF",
    "#30D158",
    "#FF9F0A",
    "#BF5AF2",
    "#FF375F",
    "#64D2FF",
    "#FFD60A",
    "#5E5CE6",
)


class DicomAnnotationKind(str, Enum):
    """Supported annotation object kinds."""

    SEGMENTATION = "dicom-seg"
    RTSTRUCT = "rtstruct"


@dataclass(frozen=True, slots=True)
class DicomAnnotationLimits:
    """Hard limits applied before large DICOM payloads are materialized."""

    max_file_bytes: int = 256 * 1024 * 1024
    max_frames: int = 50_000
    max_pixels_per_frame: int = 16_777_216
    max_decoded_bytes: int = 1024 * 1024 * 1024
    max_segments: int = 1024
    max_rois: int = 4096
    max_contours: int = 200_000
    max_contour_points: int = 10_000_000

    def __post_init__(self) -> None:
        for name in (
            "max_file_bytes",
            "max_frames",
            "max_pixels_per_frame",
            "max_decoded_bytes",
            "max_segments",
            "max_rois",
            "max_contours",
            "max_contour_points",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValidationError(f"{name} must be positive.")
            object.__setattr__(self, name, int(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class DicomAnnotationImport:
    """One validated annotation import, with no path or patient metadata."""

    kind: DicomAnnotationKind
    layers: tuple[SegmentationLayer | ContourLayer, ...]

    def __post_init__(self) -> None:
        kind = DicomAnnotationKind(self.kind)
        layers = tuple(self.layers)
        if not layers:
            raise ValidationError("A DICOM annotation import must contain at least one layer.")
        if kind is DicomAnnotationKind.SEGMENTATION and not all(
            isinstance(layer, SegmentationLayer) for layer in layers
        ):
            raise ValidationError("DICOM SEG imports may only contain segmentation layers.")
        if kind is DicomAnnotationKind.RTSTRUCT and not all(
            isinstance(layer, ContourLayer) for layer in layers
        ):
            raise ValidationError("RTSTRUCT imports may only contain contour layers.")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "layers", layers)


@dataclass(frozen=True, slots=True)
class _EncodedDicom:
    payload: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class _SegmentDescription:
    number: int
    name: str
    color: str


@dataclass(frozen=True, slots=True)
class _FrameGeometry:
    frame_index: int
    segment_number: int
    position_lps: np.ndarray
    column_direction_lps: np.ndarray
    row_direction_lps: np.ndarray
    pixel_spacing_row: float
    pixel_spacing_column: float
    slice_spacing: float | None
    slice_thickness: float | None
    source_sop_instance_uids: tuple[str, ...]


def import_dicom_annotation(
    source: str | Path | bytes | bytearray | memoryview,
    *,
    reference_series: ImageSeries,
    limits: DicomAnnotationLimits | None = None,
) -> DicomAnnotationImport:
    """Import DICOM SEG or RTSTRUCT after explicit SOP-class dispatch.

    ``reference_series`` is mandatory because annotation references are
    validated rather than inferred.  The returned objects retain only opaque
    DICOM identifiers required for reference matching; absolute paths and
    descriptive patient/study fields are never copied into provenance.
    """

    if not isinstance(reference_series, ImageSeries):
        raise ValidationError("reference_series must be an ImageSeries.")
    if reference_series.source.source_type is not SourceType.DICOM:
        raise ValidationError("DICOM annotations require a DICOM reference series.")
    active_limits = limits or DicomAnnotationLimits()
    encoded = _read_encoded_source(source, active_limits)
    pydicom = _require_pydicom()
    header = _read_dataset(pydicom, encoded.payload, stop_before_pixels=True)
    sop_class_uid = _sop_class_uid(header)

    if sop_class_uid == SEGMENTATION_STORAGE_UID:
        kind = DicomAnnotationKind.SEGMENTATION
    elif sop_class_uid == RT_STRUCTURE_SET_STORAGE_UID:
        kind = DicomAnnotationKind.RTSTRUCT
    else:
        raise UnsupportedFormatError(
            "The selected DICOM object is not DICOM SEG or RTSTRUCT; "
            "annotation import does not decode ordinary image-storage objects."
        )

    dataset = _read_dataset(pydicom, encoded.payload, stop_before_pixels=False)
    if _sop_class_uid(dataset) != sop_class_uid:
        raise DecodeError("The DICOM SOP Class UID changed between header and full decoding.")
    _required_uid(dataset, "SOPInstanceUID")

    if kind is DicomAnnotationKind.SEGMENTATION:
        layers = _import_segmentation(
            dataset,
            digest=encoded.sha256,
            reference_series=reference_series,
            limits=active_limits,
        )
    else:
        layers = (
            _import_rtstruct(
                dataset,
                digest=encoded.sha256,
                reference_series=reference_series,
                limits=active_limits,
            ),
        )
    return DicomAnnotationImport(kind=kind, layers=layers)


def _require_pydicom() -> Any:
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise MissingDependencyError(
            "DICOM annotation import requires pydicom. Install the base dependencies."
        ) from exc
    return pydicom


def _read_encoded_source(
    source: str | Path | bytes | bytearray | memoryview,
    limits: DicomAnnotationLimits,
) -> _EncodedDicom:
    if isinstance(source, (bytes, bytearray, memoryview)):
        payload = bytes(source)
    else:
        path = Path(source)
        try:
            if not path.is_file():
                raise DecodeError("The DICOM annotation source must be a regular file.")
            encoded_size = path.stat().st_size
        except OSError as exc:
            raise DecodeError("The DICOM annotation source could not be inspected.") from exc
        if encoded_size > limits.max_file_bytes:
            raise ResourceLimitError("The DICOM annotation file exceeds the encoded-size limit.")
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise DecodeError("The DICOM annotation source could not be read.") from exc
    if not payload:
        raise DecodeError("The DICOM annotation source is empty.")
    if len(payload) > limits.max_file_bytes:
        raise ResourceLimitError("The DICOM annotation file exceeds the encoded-size limit.")
    return _EncodedDicom(payload=payload, sha256=hashlib.sha256(payload).hexdigest())


def _read_dataset(pydicom: Any, payload: bytes, *, stop_before_pixels: bool) -> Any:
    try:
        return pydicom.dcmread(
            BytesIO(payload),
            force=False,
            stop_before_pixels=stop_before_pixels,
        )
    except Exception as exc:
        stage = "header" if stop_before_pixels else "dataset"
        raise DecodeError(f"The DICOM annotation {stage} could not be decoded.") from exc


def _sop_class_uid(dataset: Any) -> str:
    dataset_uid = _required_uid(dataset, "SOPClassUID")
    file_meta = getattr(dataset, "file_meta", None)
    meta_value = None if file_meta is None else getattr(file_meta, "MediaStorageSOPClassUID", None)
    if meta_value is not None:
        meta_uid = _validated_uid(meta_value, "MediaStorageSOPClassUID")
        if meta_uid != dataset_uid:
            raise DecodeError("DICOM file-meta and dataset SOP Class UIDs do not match.")
    return dataset_uid


def _import_segmentation(
    dataset: Any,
    *,
    digest: str,
    reference_series: ImageSeries,
    limits: DicomAnnotationLimits,
) -> tuple[SegmentationLayer, ...]:
    if _safe_code(getattr(dataset, "Modality", "")) != "SEG":
        raise DecodeError("DICOM Segmentation Storage requires Modality 'SEG'.")
    if int(getattr(dataset, "SamplesPerPixel", 0)) != 1:
        raise DecodeError("DICOM SEG import requires one monochrome sample per pixel.")
    if _safe_code(getattr(dataset, "PhotometricInterpretation", "")) != "MONOCHROME2":
        raise DecodeError("DICOM SEG import requires MONOCHROME2 pixel data.")

    rows = _positive_integer(dataset, "Rows")
    columns = _positive_integer(dataset, "Columns")
    frame_count = _positive_integer(dataset, "NumberOfFrames")
    if frame_count > limits.max_frames:
        raise ResourceLimitError("DICOM SEG frame count exceeds the configured limit.")
    if rows * columns > limits.max_pixels_per_frame:
        raise ResourceLimitError("DICOM SEG frame dimensions exceed the configured limit.")
    if rows * columns * frame_count > limits.max_decoded_bytes:
        raise ResourceLimitError("DICOM SEG decoded samples exceed the memory limit.")

    segmentation_type = _safe_code(getattr(dataset, "SegmentationType", ""))
    if segmentation_type == "BINARY":
        value_type = SegmentationValueType.BINARY
        maximum_fractional_value = None
        fractional_type = None
        bits_allocated = _positive_integer(dataset, "BitsAllocated")
        if bits_allocated != 1:
            raise DecodeError("Binary DICOM SEG requires BitsAllocated equal to 1.")
        _validate_pixel_encoding(dataset, bits_allocated=bits_allocated, bits_stored=1)
    elif segmentation_type == "FRACTIONAL":
        value_type = SegmentationValueType.FRACTIONAL
        bits_allocated = _positive_integer(dataset, "BitsAllocated")
        if bits_allocated not in {8, 16}:
            raise DecodeError("Fractional DICOM SEG requires 8-bit or 16-bit stored values.")
        _validate_pixel_encoding(
            dataset,
            bits_allocated=bits_allocated,
            bits_stored=bits_allocated,
        )
        maximum_fractional_value = _positive_integer(dataset, "MaximumFractionalValue")
        if maximum_fractional_value >= 2**bits_allocated:
            raise DecodeError("MaximumFractionalValue does not fit the declared DICOM pixel depth.")
        raw_fractional_type = _safe_code(getattr(dataset, "SegmentationFractionalType", ""))
        try:
            fractional_type = FractionalType(raw_fractional_type.lower())
        except ValueError as exc:
            raise DecodeError(
                "SegmentationFractionalType must be PROBABILITY or OCCUPANCY."
            ) from exc
    else:
        raise DecodeError("DICOM SEG SegmentationType must be BINARY or FRACTIONAL.")
    bytes_per_decoded_sample = 1 if bits_allocated <= 8 else bits_allocated // 8
    if rows * columns * frame_count * bytes_per_decoded_sample > limits.max_decoded_bytes:
        raise ResourceLimitError("DICOM SEG decoded pixels exceed the memory limit.")

    compression = _segmentation_compression(dataset)
    if compression is CompressionKind.LOSSY:
        raise DecodeError("Lossy transfer syntaxes cannot be used for segmentation masks.")
    if compression is CompressionKind.UNKNOWN:
        raise DecodeError("The DICOM SEG transfer syntax is not verified as lossless.")

    segment_descriptions = _segment_descriptions(dataset, limits)
    frame_geometries = _segmentation_frame_geometries(
        dataset,
        frame_count=frame_count,
        segment_numbers=set(segment_descriptions),
    )
    referenced_series_uid, top_level_source_uids = _seg_references(dataset)
    frame_source_uids = {
        uid for frame in frame_geometries for uid in frame.source_sop_instance_uids
    }
    if not top_level_source_uids and any(
        not frame.source_sop_instance_uids for frame in frame_geometries
    ):
        raise DecodeError(
            "DICOM SEG frames without source-image references require an explicit "
            "ReferencedInstanceSequence."
        )
    if top_level_source_uids and not frame_source_uids.issubset(top_level_source_uids):
        raise DecodeError(
            "DICOM SEG per-frame source references conflict with ReferencedSeriesSequence."
        )
    referenced_instance_uids = tuple(sorted(top_level_source_uids | frame_source_uids))
    if not referenced_instance_uids:
        raise DecodeError("DICOM SEG does not contain referenced source instances.")
    frame_of_reference_uid = _required_uid(dataset, "FrameOfReferenceUID")
    _match_reference_series(
        reference_series,
        series_instance_uid=referenced_series_uid,
        frame_of_reference_uid=frame_of_reference_uid,
    )

    if not hasattr(dataset, "PixelData"):
        raise DecodeError("DICOM SEG pixel data is missing.")
    try:
        decoded = np.asarray(dataset.pixel_array)
    except Exception as exc:
        raise DecodeError(
            "DICOM SEG pixels could not be decoded with the installed transfer-syntax support."
        ) from exc
    if frame_count == 1 and decoded.shape == (rows, columns):
        decoded = decoded[np.newaxis, ...]
    if decoded.shape != (frame_count, rows, columns):
        raise DecodeError(f"Unexpected DICOM SEG pixel array shape {decoded.shape}.")
    if decoded.dtype == object or not np.issubdtype(decoded.dtype, np.unsignedinteger):
        raise DecodeError("DICOM SEG pixels must decode to native integer values.")
    if int(decoded.nbytes) > limits.max_decoded_bytes:
        raise ResourceLimitError("Decoded DICOM SEG pixels exceed the memory limit.")
    if value_type is SegmentationValueType.BINARY:
        if not set(int(item) for item in np.unique(decoded)).issubset({0, 1}):
            raise DecodeError("Binary DICOM SEG pixels may only contain 0 and 1.")
    else:
        assert maximum_fractional_value is not None
        if np.any(decoded < 0) or np.any(decoded > maximum_fractional_value):
            raise DecodeError("Fractional DICOM SEG pixels exceed MaximumFractionalValue.")

    source = SourceReference(
        source_id=f"dicom-seg-{digest[:20]}",
        source_type=SourceType.DICOM,
        source_format=SourceFormat.DICOM_SEG,
        content_sha256=digest,
        compression=compression,
        provenance={
            "importer": "dicom_annotations",
            "annotation_kind": "dicom-seg",
            "segment_count": len(segment_descriptions),
            "frame_count": frame_count,
            "canonical_orientation": "RAS+",
        },
    )
    reference = LayerReference(
        local_series_id=reference_series.series_id,
        dicom_series_uid=referenced_series_uid,
        frame_of_reference_uid=frame_of_reference_uid,
        referenced_sop_instance_uids=referenced_instance_uids,
    )

    layers: list[SegmentationLayer] = []
    for segment_number, description in sorted(segment_descriptions.items()):
        selected_frames = tuple(
            frame for frame in frame_geometries if frame.segment_number == segment_number
        )
        if not selected_frames:
            raise DecodeError(
                f"DICOM SEG SegmentSequence item {segment_number} has no pixel frames."
            )
        geometry, ordered_indices = _geometry_from_frames(
            selected_frames,
            rows=rows,
            columns=columns,
        )
        segment_array = np.stack([decoded[index] for index in ordered_indices], axis=0)
        if len(ordered_indices) == 1:
            # SegmentationLayer accepts either 2-D or a one-slice 3-D array; keep
            # the explicit frame axis to preserve DICOM frame dimensionality.
            segment_array = segment_array.reshape((1, rows, columns))
        layers.append(
            SegmentationLayer(
                layer_id=f"seg-{digest[:12]}-{segment_number}",
                series_id=reference_series.series_id,
                name=description.name,
                source=source,
                created_by=LayerCreator.IMPORT,
                validation_state=LayerValidationState.VALIDATED,
                presentation=LayerPresentation(color=description.color),
                provenance={
                    "importer": "dicom_annotations",
                    "annotation_kind": "dicom-seg",
                    "segment_number": segment_number,
                    "native_values_preserved": True,
                    "frame_count": len(ordered_indices),
                },
                array=segment_array,
                geometry=geometry,
                value_type=value_type,
                reference=reference,
                labels=(
                    LabelDefinition(
                        value=segment_number,
                        name=description.name,
                        color=description.color,
                    ),
                ),
                reference_geometry=reference_series.geometry,
                maximum_fractional_value=maximum_fractional_value,
                fractional_type=fractional_type,
            )
        )
    return tuple(layers)


def _segment_descriptions(
    dataset: Any,
    limits: DicomAnnotationLimits,
) -> dict[int, _SegmentDescription]:
    sequence = _required_sequence(dataset, "SegmentSequence")
    if len(sequence) > limits.max_segments:
        raise ResourceLimitError("DICOM SEG segment count exceeds the configured limit.")
    descriptions: dict[int, _SegmentDescription] = {}
    for item in sequence:
        number = _positive_integer(item, "SegmentNumber")
        if number in descriptions:
            raise DecodeError("DICOM SEG SegmentNumber values must be unique.")
        name = _safe_name(
            getattr(item, "SegmentLabel", ""),
            fallback=f"Segment {number}",
            maximum=128,
        )
        color = _segmentation_color(item, number)
        descriptions[number] = _SegmentDescription(number=number, name=name, color=color)
    return descriptions


def _segmentation_frame_geometries(
    dataset: Any,
    *,
    frame_count: int,
    segment_numbers: set[int],
) -> tuple[_FrameGeometry, ...]:
    per_frame = _required_sequence(dataset, "PerFrameFunctionalGroupsSequence")
    if len(per_frame) != frame_count:
        raise DecodeError("PerFrameFunctionalGroupsSequence length must equal NumberOfFrames.")
    shared = _optional_sequence(dataset, "SharedFunctionalGroupsSequence")
    if len(shared) > 1:
        raise DecodeError("SharedFunctionalGroupsSequence may contain at most one item.")

    frames: list[_FrameGeometry] = []
    for index in range(frame_count):
        segment_value = _functional_group_attribute(
            per_frame[index],
            shared,
            sequence_name="SegmentIdentificationSequence",
            attribute_name="ReferencedSegmentNumber",
            frame_number=index + 1,
        )
        try:
            segment_number = int(segment_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DecodeError("ReferencedSegmentNumber must be a positive integer.") from exc
        if segment_number not in segment_numbers:
            raise DecodeError("A DICOM SEG frame references an undefined SegmentSequence item.")
        orientation = _numeric_vector(
            _functional_group_attribute(
                per_frame[index],
                shared,
                sequence_name="PlaneOrientationSequence",
                attribute_name="ImageOrientationPatient",
                frame_number=index + 1,
            ),
            length=6,
            field_name="ImageOrientationPatient",
        )
        column_direction, row_direction = _validate_orientation(orientation)
        position = _numeric_vector(
            _functional_group_attribute(
                per_frame[index],
                shared,
                sequence_name="PlanePositionSequence",
                attribute_name="ImagePositionPatient",
                frame_number=index + 1,
            ),
            length=3,
            field_name="ImagePositionPatient",
        )
        pixel_spacing = _numeric_vector(
            _functional_group_attribute(
                per_frame[index],
                shared,
                sequence_name="PixelMeasuresSequence",
                attribute_name="PixelSpacing",
                frame_number=index + 1,
            ),
            length=2,
            field_name="PixelSpacing",
        )
        if np.any(pixel_spacing <= 0.0):
            raise DecodeError("DICOM SEG PixelSpacing values must be positive.")
        slice_spacing_value = _optional_functional_group_attribute(
            per_frame[index],
            shared,
            sequence_name="PixelMeasuresSequence",
            attribute_name="SpacingBetweenSlices",
        )
        slice_spacing = _optional_positive_float(
            slice_spacing_value,
            "SpacingBetweenSlices",
        )
        slice_thickness = _optional_positive_float(
            _optional_functional_group_attribute(
                per_frame[index],
                shared,
                sequence_name="PixelMeasuresSequence",
                attribute_name="SliceThickness",
            ),
            "SliceThickness",
        )
        source_uids = _frame_source_uids(per_frame[index], shared)
        frames.append(
            _FrameGeometry(
                frame_index=index,
                segment_number=segment_number,
                position_lps=position,
                column_direction_lps=column_direction,
                row_direction_lps=row_direction,
                pixel_spacing_row=float(pixel_spacing[0]),
                pixel_spacing_column=float(pixel_spacing[1]),
                slice_spacing=slice_spacing,
                slice_thickness=slice_thickness,
                source_sop_instance_uids=source_uids,
            )
        )
    return tuple(frames)


def _geometry_from_frames(
    frames: Sequence[_FrameGeometry],
    *,
    rows: int,
    columns: int,
) -> tuple[SpatialGeometry, tuple[int, ...]]:
    if not frames:
        raise DecodeError("Cannot construct segmentation geometry without frames.")
    first = frames[0]
    for frame in frames[1:]:
        if not np.allclose(
            frame.column_direction_lps,
            first.column_direction_lps,
            atol=_DIRECTION_ATOL,
            rtol=0.0,
        ) or not np.allclose(
            frame.row_direction_lps,
            first.row_direction_lps,
            atol=_DIRECTION_ATOL,
            rtol=0.0,
        ):
            raise DecodeError("DICOM SEG frames have inconsistent image orientation.")
        if not math.isclose(
            frame.pixel_spacing_row,
            first.pixel_spacing_row,
            rel_tol=1e-5,
            abs_tol=1e-6,
        ) or not math.isclose(
            frame.pixel_spacing_column,
            first.pixel_spacing_column,
            rel_tol=1e-5,
            abs_tol=1e-6,
        ):
            raise DecodeError("DICOM SEG frames have inconsistent in-plane spacing.")

    normal_lps = np.cross(first.column_direction_lps, first.row_direction_lps)
    normal_lps = normal_lps / np.linalg.norm(normal_lps)
    projections = np.asarray(
        [float(np.dot(frame.position_lps, normal_lps)) for frame in frames],
        dtype=np.float64,
    )
    order = np.argsort(projections, kind="stable")
    sorted_frames = [frames[int(index)] for index in order]
    sorted_projections = projections[order]
    origin_lps = sorted_frames[0].position_lps
    for frame, projection in zip(sorted_frames, sorted_projections, strict=True):
        displacement = frame.position_lps - origin_lps
        in_plane = displacement - normal_lps * (projection - sorted_projections[0])
        if np.linalg.norm(in_plane) > _GEOMETRY_ATOL_MM:
            raise DecodeError("DICOM SEG frame positions do not form one regular slice axis.")

    if len(sorted_frames) == 1:
        spacing_values = {
            round(float(value), 9)
            for frame in sorted_frames
            for value in (
                frame.slice_spacing if frame.slice_spacing is not None else frame.slice_thickness,
            )
            if value is not None
        }
        if len(spacing_values) != 1:
            raise DecodeError(
                "Single-frame DICOM SEG geometry requires explicit positive slice spacing."
            )
        spacing_z = next(iter(spacing_values))
    else:
        steps = np.diff(sorted_projections)
        if np.any(steps <= _GEOMETRY_ATOL_MM):
            raise DecodeError("DICOM SEG frame positions are duplicated or unordered.")
        spacing_z = float(np.median(steps))
        if not np.allclose(steps, spacing_z, rtol=1e-4, atol=_GEOMETRY_ATOL_MM):
            raise DecodeError("DICOM SEG frame positions have non-uniform slice spacing.")
        declared = [
            float(frame.slice_spacing) for frame in sorted_frames if frame.slice_spacing is not None
        ]
        if declared and (
            len(declared) != len(sorted_frames)
            or not np.allclose(declared, spacing_z, rtol=1e-4, atol=_GEOMETRY_ATOL_MM)
        ):
            raise DecodeError("DICOM SEG declared slice spacing conflicts with frame positions.")

    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, 0] = first.column_direction_lps * first.pixel_spacing_column
    affine_lps[:3, 1] = first.row_direction_lps * first.pixel_spacing_row
    affine_lps[:3, 2] = normal_lps * spacing_z
    affine_lps[:3, 3] = origin_lps
    geometry = SpatialGeometry(
        shape_zyx=(len(sorted_frames), rows, columns),
        affine_ras=_LPS_TO_RAS @ affine_lps,
    )
    return geometry, tuple(frame.frame_index for frame in sorted_frames)


def _seg_references(dataset: Any) -> tuple[str, set[str]]:
    referenced_series = _required_sequence(dataset, "ReferencedSeriesSequence")
    series_uids: set[str] = set()
    instance_uids: set[str] = set()
    for item in referenced_series:
        series_uids.add(_required_uid(item, "SeriesInstanceUID"))
        for instance in _optional_sequence(item, "ReferencedInstanceSequence"):
            instance_uids.add(_referenced_sop_instance_uid(instance))
    if len(series_uids) != 1:
        raise DecodeError("DICOM SEG must reference exactly one source image series.")
    return next(iter(series_uids)), instance_uids


def _frame_source_uids(per_frame: Any, shared: Sequence[Any]) -> tuple[str, ...]:
    groups = [per_frame, *shared]
    result: set[str] = set()
    for group in groups:
        for derivation in _optional_sequence(group, "DerivationImageSequence"):
            for source in _required_sequence(derivation, "SourceImageSequence"):
                result.add(_referenced_sop_instance_uid(source))
    return tuple(sorted(result))


def _import_rtstruct(
    dataset: Any,
    *,
    digest: str,
    reference_series: ImageSeries,
    limits: DicomAnnotationLimits,
) -> ContourLayer:
    if _safe_code(getattr(dataset, "Modality", "")) != "RTSTRUCT":
        raise DecodeError("RT Structure Set Storage requires Modality 'RTSTRUCT'.")
    if hasattr(dataset, "PixelData"):
        raise DecodeError("RTSTRUCT objects must not contain a pixel payload.")

    referenced_series_uid, frame_of_reference_uid, instance_uids = _rt_references(dataset)
    _match_reference_series(
        reference_series,
        series_instance_uid=referenced_series_uid,
        frame_of_reference_uid=frame_of_reference_uid,
    )

    structure_sequence = _required_sequence(dataset, "StructureSetROISequence")
    contour_sequence = _required_sequence(dataset, "ROIContourSequence")
    if len(structure_sequence) > limits.max_rois or len(contour_sequence) > limits.max_rois:
        raise ResourceLimitError("RTSTRUCT ROI count exceeds the configured limit.")

    structures: dict[int, tuple[str, str]] = {}
    for item in structure_sequence:
        roi_number = _positive_integer(item, "ROINumber")
        if roi_number in structures:
            raise DecodeError("RTSTRUCT ROINumber values must be unique.")
        roi_frame_uid = _required_uid(item, "ReferencedFrameOfReferenceUID")
        if roi_frame_uid != frame_of_reference_uid:
            raise DecodeError("An RTSTRUCT ROI references a different Frame of Reference UID.")
        name = _safe_name(
            getattr(item, "ROIName", ""),
            fallback=f"ROI {roi_number}",
            maximum=128,
        )
        structures[roi_number] = (name, roi_frame_uid)

    contours_by_roi: dict[int, Any] = {}
    for item in contour_sequence:
        roi_number = _positive_integer(item, "ReferencedROINumber")
        if roi_number not in structures:
            raise DecodeError("ROIContourSequence references an undefined ROINumber.")
        if roi_number in contours_by_roi:
            raise DecodeError("RTSTRUCT contains duplicate ROIContourSequence entries.")
        contours_by_roi[roi_number] = item
    if set(contours_by_roi) != set(structures):
        raise DecodeError("Every imported RTSTRUCT ROI must have exactly one contour entry.")

    total_contours = 0
    total_points = 0
    top_level_instance_uids = set(instance_uids)
    all_instance_uids = set(instance_uids)
    rois: list[RegionOfInterest] = []
    for roi_number, (name, _) in sorted(structures.items()):
        roi_item = contours_by_roi[roi_number]
        color = _rt_color(getattr(roi_item, "ROIDisplayColor", None), roi_number)
        paths: list[ContourPath] = []
        for contour in _required_sequence(roi_item, "ContourSequence"):
            total_contours += 1
            if total_contours > limits.max_contours:
                raise ResourceLimitError("RTSTRUCT contour count exceeds the configured limit.")
            geometric_type = _safe_code(getattr(contour, "ContourGeometricType", ""))
            if geometric_type not in {"CLOSED_PLANAR", "CLOSEDPLANAR_XOR"}:
                raise DecodeError("Only closed planar RTSTRUCT contours can be represented safely.")
            number_of_points = _positive_integer(contour, "NumberOfContourPoints")
            total_points += number_of_points
            if total_points > limits.max_contour_points:
                raise ResourceLimitError(
                    "RTSTRUCT contour point count exceeds the configured limit."
                )
            raw_points = _numeric_vector(
                getattr(contour, "ContourData", None),
                length=number_of_points * 3,
                field_name="ContourData",
            )
            points_lps = raw_points.reshape((-1, 3))
            _validate_planar_contour(points_lps)
            points_ras = np.array(points_lps, copy=True)
            points_ras[:, :2] *= -1.0
            contour_refs = {
                _referenced_sop_instance_uid(item)
                for item in _optional_sequence(contour, "ContourImageSequence")
            }
            if len(contour_refs) > 1:
                raise DecodeError(
                    "One RTSTRUCT contour cannot be assigned to multiple source instances."
                )
            if top_level_instance_uids and not contour_refs.issubset(top_level_instance_uids):
                raise DecodeError(
                    "RTSTRUCT contour references conflict with the referenced image series."
                )
            all_instance_uids.update(contour_refs)
            paths.append(
                ContourPath(
                    points_ras=points_ras,
                    referenced_sop_instance_uid=(
                        next(iter(contour_refs)) if contour_refs else None
                    ),
                )
            )
        rois.append(
            RegionOfInterest(
                roi_number=roi_number,
                name=name,
                color=color,
                contours=tuple(paths),
            )
        )
    if not all_instance_uids:
        raise DecodeError("RTSTRUCT does not contain referenced source instances.")

    source = SourceReference(
        source_id=f"rtstruct-{digest[:20]}",
        source_type=SourceType.DICOM,
        source_format=SourceFormat.RTSTRUCT,
        content_sha256=digest,
        compression=CompressionKind.NONE,
        provenance={
            "importer": "dicom_annotations",
            "annotation_kind": "rtstruct",
            "roi_count": len(rois),
            "contour_count": total_contours,
            "canonical_orientation": "RAS+",
        },
    )
    return ContourLayer(
        layer_id=f"rtstruct-{digest[:12]}",
        series_id=reference_series.series_id,
        name="RT Structure Set contours",
        source=source,
        created_by=LayerCreator.IMPORT,
        validation_state=LayerValidationState.VALIDATED,
        presentation=LayerPresentation(color=rois[0].color),
        provenance={
            "importer": "dicom_annotations",
            "annotation_kind": "rtstruct",
            "coordinate_conversion": "DICOM LPS to RAS+",
            "roi_count": len(rois),
            "contour_count": total_contours,
        },
        reference=LayerReference(
            local_series_id=reference_series.series_id,
            dicom_series_uid=referenced_series_uid,
            frame_of_reference_uid=frame_of_reference_uid,
            referenced_sop_instance_uids=tuple(sorted(all_instance_uids)),
        ),
        rois=tuple(rois),
        reference_geometry=reference_series.geometry,
    )


def _rt_references(dataset: Any) -> tuple[str, str, set[str]]:
    frame_sequence = _required_sequence(dataset, "ReferencedFrameOfReferenceSequence")
    frame_uids: set[str] = set()
    series_uids: set[str] = set()
    instance_uids: set[str] = set()
    for frame_item in frame_sequence:
        frame_uids.add(_required_uid(frame_item, "FrameOfReferenceUID"))
        for study_item in _required_sequence(frame_item, "RTReferencedStudySequence"):
            for series_item in _required_sequence(study_item, "RTReferencedSeriesSequence"):
                series_uids.add(_required_uid(series_item, "SeriesInstanceUID"))
                for image_item in _optional_sequence(series_item, "ContourImageSequence"):
                    instance_uids.add(_referenced_sop_instance_uid(image_item))
    if len(frame_uids) != 1:
        raise DecodeError("RTSTRUCT must reference exactly one Frame of Reference UID.")
    if len(series_uids) != 1:
        raise DecodeError("RTSTRUCT must reference exactly one source image series.")
    return next(iter(series_uids)), next(iter(frame_uids)), instance_uids


def _match_reference_series(
    reference_series: ImageSeries,
    *,
    series_instance_uid: str,
    frame_of_reference_uid: str,
) -> None:
    if reference_series.series_instance_uid != series_instance_uid:
        raise DecodeError(
            "The annotation's referenced Series Instance UID does not match the selected series."
        )
    if reference_series.frame_of_reference_uid != frame_of_reference_uid:
        raise DecodeError(
            "The annotation's Frame of Reference UID does not match the selected series."
        )


def _functional_group_attribute(
    per_frame: Any,
    shared: Sequence[Any],
    *,
    sequence_name: str,
    attribute_name: str,
    frame_number: int,
) -> Any:
    value = _optional_functional_group_attribute(
        per_frame,
        shared,
        sequence_name=sequence_name,
        attribute_name=attribute_name,
    )
    if value is None:
        raise DecodeError(
            f"DICOM SEG frame {frame_number} is missing {attribute_name} in its functional groups."
        )
    return value


def _optional_functional_group_attribute(
    per_frame: Any,
    shared: Sequence[Any],
    *,
    sequence_name: str,
    attribute_name: str,
) -> Any | None:
    for group in (per_frame, *shared):
        sequence = _optional_sequence(group, sequence_name)
        if len(sequence) > 1:
            raise DecodeError(f"{sequence_name} must contain at most one item.")
        if sequence:
            value = getattr(sequence[0], attribute_name, None)
            if value is not None:
                return value
    return None


def _validate_orientation(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    column_direction = np.asarray(values[:3], dtype=np.float64)
    row_direction = np.asarray(values[3:], dtype=np.float64)
    column_norm = float(np.linalg.norm(column_direction))
    row_norm = float(np.linalg.norm(row_direction))
    if not math.isclose(column_norm, 1.0, abs_tol=1e-3, rel_tol=0.0) or not math.isclose(
        row_norm, 1.0, abs_tol=1e-3, rel_tol=0.0
    ):
        raise DecodeError("DICOM SEG orientation direction cosines must have unit length.")
    column_direction = column_direction / column_norm
    row_direction = row_direction / row_norm
    if abs(float(np.dot(column_direction, row_direction))) > 1e-3:
        raise DecodeError("DICOM SEG orientation axes must be orthogonal.")
    return column_direction, row_direction


def _validate_planar_contour(points_lps: np.ndarray) -> None:
    if points_lps.shape[0] < 3:
        raise DecodeError("RTSTRUCT closed contours require at least three points.")
    centered = points_lps - points_lps[0]
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size < 2 or singular_values[1] <= 1e-8:
        raise DecodeError("RTSTRUCT contour points must define a non-degenerate plane.")
    normal = right_vectors[-1]
    distances = np.abs(centered @ normal)
    if np.max(distances) > _GEOMETRY_ATOL_MM:
        raise DecodeError("RTSTRUCT CLOSED_PLANAR contour points are not coplanar.")


def _segmentation_compression(dataset: Any) -> CompressionKind:
    file_meta = getattr(dataset, "file_meta", None)
    if file_meta is None:
        raise DecodeError("DICOM SEG file-meta transfer syntax is missing.")
    uid = _required_uid(file_meta, "TransferSyntaxUID")
    if uid in _UNCOMPRESSED_TRANSFER_SYNTAXES:
        return CompressionKind.NONE
    if uid in _LOSSLESS_TRANSFER_SYNTAXES:
        return CompressionKind.LOSSLESS
    if uid in _LOSSY_TRANSFER_SYNTAXES:
        return CompressionKind.LOSSY
    name = _safe_text(getattr(file_meta.TransferSyntaxUID, "name", ""))
    if "lossless" in name.lower() and "near-lossless" not in name.lower():
        return CompressionKind.LOSSLESS
    return CompressionKind.UNKNOWN


def _validate_pixel_encoding(
    dataset: Any,
    *,
    bits_allocated: int,
    bits_stored: int,
) -> None:
    declared_stored = _positive_integer(dataset, "BitsStored")
    high_bit = _nonnegative_integer(dataset, "HighBit")
    pixel_representation = _nonnegative_integer(dataset, "PixelRepresentation")
    if declared_stored != bits_stored or declared_stored > bits_allocated:
        raise DecodeError("DICOM SEG BitsStored is inconsistent with BitsAllocated.")
    if high_bit != declared_stored - 1:
        raise DecodeError("DICOM SEG HighBit is inconsistent with BitsStored.")
    if pixel_representation != 0:
        raise DecodeError("DICOM SEG stored values must use unsigned pixel representation.")


def _segmentation_color(item: Any, number: int) -> str:
    rgb = getattr(item, "RecommendedDisplayRGBValue", None)
    parsed = _parse_rgb(rgb)
    if parsed is not None:
        return parsed
    cielab = getattr(item, "RecommendedDisplayCIELabValue", None)
    converted = _dicom_cielab_to_rgb(cielab)
    return converted or _fallback_color(number)


def _rt_color(value: Any, number: int) -> str:
    return _parse_rgb(value) or _fallback_color(number)


def _parse_rgb(value: Any) -> str | None:
    try:
        values = tuple(int(item) for item in value)
    except (TypeError, ValueError, OverflowError):
        return None
    if len(values) != 3 or any(item < 0 or item > 255 for item in values):
        return None
    return f"#{values[0]:02X}{values[1]:02X}{values[2]:02X}"


def _dicom_cielab_to_rgb(value: Any) -> str | None:
    try:
        encoded = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        len(encoded) != 3
        or not np.all(np.isfinite(encoded))
        or any(item < 0.0 or item > 65535.0 for item in encoded)
    ):
        return None
    lightness = encoded[0] * 100.0 / 65535.0
    a_star = encoded[1] * 255.0 / 65535.0 - 128.0
    b_star = encoded[2] * 255.0 / 65535.0 - 128.0
    fy = (lightness + 16.0) / 116.0
    fx = fy + a_star / 500.0
    fz = fy - b_star / 200.0

    def inverse_lab(value_: float) -> float:
        cube = value_**3
        return cube if cube > 216.0 / 24389.0 else (116.0 * value_ - 16.0) / 903.3

    # DICOM CIELab uses the D50 reference white.  A Bradford-adapted D50 to
    # sRGB matrix keeps this conversion deterministic without external color
    # management dependencies.
    xyz = np.asarray(
        [
            0.96422 * inverse_lab(fx),
            1.00000 * inverse_lab(fy),
            0.82521 * inverse_lab(fz),
        ]
    )
    linear = (
        np.asarray(
            [
                [3.1338561, -1.6168667, -0.4906146],
                [-0.9787684, 1.9161415, 0.0334540],
                [0.0719453, -0.2289914, 1.4052427],
            ]
        )
        @ xyz
    )
    linear = np.clip(linear, 0.0, 1.0)
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    values = np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(int)
    return f"#{values[0]:02X}{values[1]:02X}{values[2]:02X}"


def _fallback_color(number: int) -> str:
    return _FALLBACK_COLORS[(number - 1) % len(_FALLBACK_COLORS)]


def _required_sequence(dataset: Any, name: str) -> tuple[Any, ...]:
    sequence = _optional_sequence(dataset, name)
    if not sequence:
        raise DecodeError(f"DICOM annotation sequence {name} is missing or empty.")
    return sequence


def _optional_sequence(dataset: Any, name: str) -> tuple[Any, ...]:
    value = getattr(dataset, name, None)
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        raise DecodeError(f"DICOM annotation field {name} is not a valid sequence.")
    try:
        return tuple(value)
    except TypeError as exc:
        raise DecodeError(f"DICOM annotation field {name} is not a valid sequence.") from exc


def _required_uid(dataset: Any, name: str) -> str:
    value = getattr(dataset, name, None)
    if value is None:
        raise DecodeError(f"DICOM annotation field {name} is missing.")
    return _validated_uid(value, name)


def _referenced_sop_instance_uid(item: Any) -> str:
    sop_class_uid = _required_uid(item, "ReferencedSOPClassUID")
    if sop_class_uid in {SEGMENTATION_STORAGE_UID, RT_STRUCTURE_SET_STORAGE_UID}:
        raise DecodeError(
            "An imported annotation cannot use another annotation object as its source image."
        )
    return _required_uid(item, "ReferencedSOPInstanceUID")


def _validated_uid(value: Any, name: str) -> str:
    normalized = str(value).strip()
    if not normalized or len(normalized) > 64 or not _DICOM_UID_RE.fullmatch(normalized):
        raise DecodeError(f"DICOM annotation field {name} is not a valid UID.")
    return normalized


def _positive_integer(dataset: Any, name: str) -> int:
    value = getattr(dataset, name, None)
    if value is None:
        raise DecodeError(f"DICOM annotation field {name} must be a positive integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DecodeError(f"DICOM annotation field {name} must be a positive integer.") from exc
    if result <= 0:
        raise DecodeError(f"DICOM annotation field {name} must be a positive integer.")
    return result


def _nonnegative_integer(dataset: Any, name: str) -> int:
    value = getattr(dataset, name, None)
    if value is None:
        raise DecodeError(f"DICOM annotation field {name} must be a nonnegative integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DecodeError(f"DICOM annotation field {name} must be a nonnegative integer.") from exc
    if result < 0:
        raise DecodeError(f"DICOM annotation field {name} must be a nonnegative integer.")
    return result


def _numeric_vector(value: Any, *, length: int, field_name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DecodeError(f"DICOM annotation field {field_name} is not numeric.") from exc
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise DecodeError(
            f"DICOM annotation field {field_name} must contain {length} finite values."
        )
    return result


def _optional_positive_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DecodeError(f"DICOM annotation field {field_name} must be numeric.") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise DecodeError(f"DICOM annotation field {field_name} must be positive.")
    return result


def _safe_code(value: Any) -> str:
    return _safe_text(value).upper()


def _safe_name(value: Any, *, fallback: str, maximum: int) -> str:
    normalized = _safe_text(value, fallback=fallback, maximum=maximum)
    return fallback if _ABSOLUTE_PATH_RE.match(normalized) else normalized


def _safe_text(value: Any, *, fallback: str = "", maximum: int = 128) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    normalized = "".join(
        " " if unicodedata.category(character).startswith("C") else character
        for character in normalized
    )
    normalized = " ".join(normalized.split())[:maximum].strip()
    return normalized or fallback


def annotation_layers(
    imports: Iterable[DicomAnnotationImport],
) -> tuple[SegmentationLayer | ContourLayer, ...]:
    """Flatten validated results without changing layer identity or geometry."""

    return tuple(layer for result in imports for layer in result.layers)


__all__ = [
    "RT_STRUCTURE_SET_STORAGE_UID",
    "SEGMENTATION_STORAGE_UID",
    "DicomAnnotationImport",
    "DicomAnnotationKind",
    "DicomAnnotationLimits",
    "annotation_layers",
    "import_dicom_annotation",
]
