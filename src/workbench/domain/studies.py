"""Versioned study, series, and layer contracts for medical imaging workflows.

The objects in this module deliberately separate source identity, spatial
geometry, presentation state, and pixel/voxel payloads.  They are immutable:
operations that change geometry or layer collections return new objects so an
imported layer is never silently overwritten.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from copy import copy as shallow_copy
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Self

import numpy as np

from ..errors import ValidationError
from .images import (
    ImageVolume,
    IntensitySemantics,
    SourceType,
    sanitize_runtime_metadata,
)

STUDY_LAYER_SCHEMA_VERSION = 1


class WorldCoordinateConvention(str, Enum):
    """World-coordinate convention used by every spatial public contract."""

    RAS_PLUS = "RAS+"


class SourceFormat(str, Enum):
    PNG = "png"
    JPEG = "jpeg"
    TIFF = "tiff"
    DICOM = "dicom"
    DICOM_SEG = "dicom-seg"
    RTSTRUCT = "rtstruct"
    NIFTI = "nifti"
    GENERATED = "generated"
    PLUGIN = "plugin"


class CompressionKind(str, Enum):
    NONE = "none"
    LOSSLESS = "lossless"
    LOSSY = "lossy"
    UNKNOWN = "unknown"


class LayerCreator(str, Enum):
    IMPORT = "import"
    USER = "user"
    ALGORITHM = "algorithm"
    MODEL = "model"
    LLM = "llm"


class LayerValidationState(str, Enum):
    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"


class DeidentificationStatus(str, Enum):
    NOT_ASSESSED = "not-assessed"
    USER_CONFIRMED = "user-confirmed"
    VERIFIED = "verified"
    FAILED = "failed"


class SegmentationValueType(str, Enum):
    DISCRETE = "discrete"
    BINARY = "binary"
    FRACTIONAL = "fractional"
    PROBABILITY = "probability"


class FractionalType(str, Enum):
    PROBABILITY = "probability"
    OCCUPANCY = "occupancy"


class GeometryMatchStatus(str, Enum):
    UNRESOLVED_REFERENCE = "unresolved-reference"
    REFERENCE_MISMATCH = "reference-mismatch"
    REQUIRES_RESAMPLING = "requires-resampling"
    MATCHED = "matched"
    RESAMPLED = "resampled"


class TransformKind(str, Enum):
    RESAMPLE = "resample"
    RASTERIZE = "rasterize"
    THRESHOLD = "threshold"
    MERGE = "merge"
    OTHER = "other"


class InterpolationMode(str, Enum):
    NEAREST = "nearest"
    LINEAR = "linear"
    BSPLINE = "bspline"


_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DICOM_UID_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)*$")


def _nonempty(value: str, field_name: str, *, maximum: int = 256) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValidationError(f"{field_name} cannot be empty.")
    if len(normalized) > maximum:
        raise ValidationError(f"{field_name} cannot exceed {maximum} characters.")
    return normalized


def _opaque_id(value: str, field_name: str) -> str:
    normalized = _nonempty(value, field_name, maximum=128)
    if any(character in normalized for character in ("/", "\\", "\0", "\n", "\r")):
        raise ValidationError(f"{field_name} must be an opaque identifier, not a path.")
    return normalized


def _dicom_uid(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = _nonempty(value, field_name, maximum=64)
    if not _DICOM_UID_RE.fullmatch(normalized):
        raise ValidationError(f"{field_name} is not a valid DICOM UID.")
    return normalized


def _color(value: str, field_name: str = "color") -> str:
    normalized = str(value).upper()
    if not _HEX_COLOR_RE.fullmatch(normalized):
        raise ValidationError(f"{field_name} must use #RRGGBB notation.")
    return normalized


def _readonly_array(
    value: np.ndarray | Sequence[Any],
    *,
    field_name: str,
    numeric_only: bool = True,
) -> np.ndarray:
    array = np.asanyarray(value)
    if array.dtype == object:
        raise ValidationError(f"{field_name} cannot use object dtype.")
    if numeric_only and not (
        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValidationError(f"{field_name} must contain numeric values.")
    if any(dimension <= 0 for dimension in array.shape):
        raise ValidationError(f"{field_name} dimensions must be positive.")
    copied = np.array(array, copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True, slots=True)
class SourceReference:
    """Opaque, immutable source identity without file paths or patient fields."""

    source_id: str
    source_type: SourceType
    source_format: SourceFormat
    version: str = "1"
    content_sha256: str | None = None
    compression: CompressionKind = CompressionKind.UNKNOWN
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source_id = _opaque_id(self.source_id, "source_id")
        source_type = SourceType(self.source_type)
        source_format = SourceFormat(self.source_format)
        version = _nonempty(self.version, "source version", maximum=64)
        digest = self.content_sha256
        if digest is not None:
            digest = str(digest).lower()
            if not _SHA256_RE.fullmatch(digest):
                raise ValidationError(
                    "content_sha256 must contain 64 lowercase hexadecimal digits."
                )

        compression = CompressionKind(self.compression)
        if source_format is SourceFormat.JPEG:
            if compression is CompressionKind.UNKNOWN:
                compression = CompressionKind.LOSSY
            elif compression is not CompressionKind.LOSSY:
                raise ValidationError("JPEG sources must be represented as lossy.")
        elif source_format is SourceFormat.PNG:
            if compression is CompressionKind.UNKNOWN:
                compression = CompressionKind.LOSSLESS
            elif compression is CompressionKind.LOSSY:
                raise ValidationError("PNG sources cannot be marked as lossy.")
        elif source_format is SourceFormat.GENERATED and compression is CompressionKind.UNKNOWN:
            compression = CompressionKind.NONE

        expected_source_types = {
            SourceFormat.PNG: {SourceType.RASTER},
            SourceFormat.JPEG: {SourceType.RASTER},
            SourceFormat.TIFF: {SourceType.RASTER},
            SourceFormat.DICOM: {SourceType.DICOM},
            SourceFormat.DICOM_SEG: {SourceType.DICOM},
            SourceFormat.RTSTRUCT: {SourceType.DICOM},
            SourceFormat.NIFTI: {SourceType.NIFTI},
            SourceFormat.GENERATED: {SourceType.GENERATED},
            SourceFormat.PLUGIN: {SourceType.PLUGIN},
        }[source_format]
        if source_type not in expected_source_types:
            raise ValidationError(
                f"Source type {source_type.value!r} is incompatible with "
                f"format {source_format.value!r}."
            )

        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "source_format", source_format)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "content_sha256", digest)
        object.__setattr__(self, "compression", compression)
        object.__setattr__(self, "provenance", sanitize_runtime_metadata(self.provenance))

    @property
    def is_lossy(self) -> bool:
        return self.compression is CompressionKind.LOSSY


@dataclass(frozen=True, slots=True, eq=False)
class SpatialGeometry:
    """A three-dimensional voxel grid whose affine maps XYZ indices to RAS+ mm."""

    shape_zyx: tuple[int, int, int]
    affine_ras: np.ndarray
    convention: WorldCoordinateConvention = WorldCoordinateConvention.RAS_PLUS

    def __post_init__(self) -> None:
        shape = tuple(int(item) for item in self.shape_zyx)
        if len(shape) != 3 or any(item <= 0 for item in shape):
            raise ValidationError("shape_zyx must contain three positive dimensions.")
        affine = np.asarray(self.affine_ras, dtype=np.float64)
        if affine.shape != (4, 4):
            raise ValidationError("affine_ras must have shape 4x4.")
        if not np.all(np.isfinite(affine)):
            raise ValidationError("affine_ras cannot contain NaN or infinity.")
        if not np.allclose(affine[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
            raise ValidationError("affine_ras must use a homogeneous [0, 0, 0, 1] final row.")
        spatial = affine[:3, :3]
        spacing = np.linalg.norm(spatial, axis=0)
        if np.any(spacing <= 0.0) or np.isclose(np.linalg.det(spatial), 0.0):
            raise ValidationError("affine_ras must describe three independent spatial axes.")
        affine = affine.copy()
        affine.setflags(write=False)
        object.__setattr__(self, "shape_zyx", shape)
        object.__setattr__(self, "affine_ras", affine)
        object.__setattr__(self, "convention", WorldCoordinateConvention(self.convention))

    @classmethod
    def from_volume(cls, volume: ImageVolume) -> SpatialGeometry:
        return cls(tuple(int(item) for item in volume.shape), volume.affine)

    @property
    def spacing_xyz(self) -> tuple[float, float, float]:
        values = np.linalg.norm(self.affine_ras[:3, :3], axis=0)
        return tuple(float(item) for item in values)

    @property
    def direction_ras(self) -> np.ndarray:
        direction = self.affine_ras[:3, :3] / np.asarray(self.spacing_xyz)[None, :]
        direction = np.array(direction, copy=True)
        direction.setflags(write=False)
        return direction

    @property
    def origin_ras(self) -> tuple[float, float, float]:
        return tuple(float(item) for item in self.affine_ras[:3, 3])

    @property
    def fingerprint(self) -> str:
        payload = {
            "convention": self.convention.value,
            "shape_zyx": self.shape_zyx,
            "affine_ras": self.affine_ras.tolist(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def compare_to(
        self,
        other: SpatialGeometry,
        *,
        spacing_atol: float = 1e-5,
        orientation_atol: float = 1e-5,
        affine_atol: float = 1e-4,
    ) -> GeometryDifference:
        if not isinstance(other, SpatialGeometry):
            raise ValidationError("Spatial geometry can only be compared with SpatialGeometry.")
        shape_matches = self.shape_zyx == other.shape_zyx
        spacing_matches = bool(
            np.allclose(self.spacing_xyz, other.spacing_xyz, atol=spacing_atol, rtol=0.0)
        )
        orientation_matches = bool(
            np.allclose(
                self.direction_ras,
                other.direction_ras,
                atol=orientation_atol,
                rtol=0.0,
            )
        )
        affine_matches = bool(
            np.allclose(self.affine_ras, other.affine_ras, atol=affine_atol, rtol=0.0)
        )
        return GeometryDifference(
            shape_matches=shape_matches,
            spacing_matches=spacing_matches,
            orientation_matches=orientation_matches,
            affine_matches=affine_matches,
        )

    def matches(self, other: SpatialGeometry, *, affine_atol: float = 1e-4) -> bool:
        return self.compare_to(other, affine_atol=affine_atol).matches


@dataclass(frozen=True, slots=True)
class GeometryDifference:
    shape_matches: bool
    spacing_matches: bool
    orientation_matches: bool
    affine_matches: bool

    @property
    def matches(self) -> bool:
        return all(
            (
                self.shape_matches,
                self.spacing_matches,
                self.orientation_matches,
                self.affine_matches,
            )
        )

    @property
    def mismatched_components(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, matched in (
                ("shape", self.shape_matches),
                ("spacing", self.spacing_matches),
                ("orientation", self.orientation_matches),
                ("affine", self.affine_matches),
            )
            if not matched
        )


@dataclass(frozen=True, slots=True)
class LayerPresentation:
    visible: bool = True
    locked: bool = False
    opacity: float = 1.0
    color: str = "#58A6FF"

    def __post_init__(self) -> None:
        opacity = float(self.opacity)
        if not np.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
            raise ValidationError("Layer opacity must be a finite value in [0, 1].")
        object.__setattr__(self, "visible", bool(self.visible))
        object.__setattr__(self, "locked", bool(self.locked))
        object.__setattr__(self, "opacity", opacity)
        object.__setattr__(self, "color", _color(self.color))


@dataclass(frozen=True, slots=True)
class GeometryTransformRecord:
    kind: TransformKind
    source_geometry_fingerprint: str
    target_geometry_fingerprint: str
    interpolation: InterpolationMode | None = None
    user_confirmed: bool = False
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = TransformKind(self.kind)
        source = str(self.source_geometry_fingerprint).lower()
        target = str(self.target_geometry_fingerprint).lower()
        if not _SHA256_RE.fullmatch(source) or not _SHA256_RE.fullmatch(target):
            raise ValidationError("Transform geometry fingerprints must be SHA-256 values.")
        interpolation = (
            None if self.interpolation is None else InterpolationMode(self.interpolation)
        )
        if kind is TransformKind.RESAMPLE:
            if interpolation is None:
                raise ValidationError("A resampling record must declare its interpolation.")
            if not self.user_confirmed:
                raise ValidationError("A resampling record requires explicit user confirmation.")
        elif interpolation is not None:
            raise ValidationError("Interpolation is only valid for resampling records.")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source_geometry_fingerprint", source)
        object.__setattr__(self, "target_geometry_fingerprint", target)
        object.__setattr__(self, "interpolation", interpolation)
        object.__setattr__(self, "user_confirmed", bool(self.user_confirmed))
        object.__setattr__(self, "parameters", sanitize_runtime_metadata(self.parameters))

    @classmethod
    def resampling(
        cls,
        source: SpatialGeometry,
        target: SpatialGeometry,
        *,
        interpolation: InterpolationMode,
        user_confirmed: bool,
        parameters: Mapping[str, Any] | None = None,
    ) -> GeometryTransformRecord:
        return cls(
            kind=TransformKind.RESAMPLE,
            source_geometry_fingerprint=source.fingerprint,
            target_geometry_fingerprint=target.fingerprint,
            interpolation=interpolation,
            user_confirmed=user_confirmed,
            parameters={} if parameters is None else parameters,
        )


@dataclass(frozen=True, slots=True)
class LabelDefinition:
    value: int
    name: str
    color: str
    visible: bool = True

    def __post_init__(self) -> None:
        value = int(self.value)
        if value < 0:
            raise ValidationError("Label values must be nonnegative integers.")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "name", _nonempty(self.name, "label name", maximum=128))
        object.__setattr__(self, "color", _color(self.color, "label color"))
        object.__setattr__(self, "visible", bool(self.visible))


@dataclass(frozen=True, slots=True)
class LabelStatistics:
    label_value: int
    voxel_count: int
    nonzero_voxel_count: int
    minimum: float
    maximum: float
    mean: float


@dataclass(frozen=True, slots=True)
class LayerReference:
    """Series and instance references carried by SEG or RTSTRUCT objects."""

    local_series_id: str | None = None
    dicom_series_uid: str | None = None
    frame_of_reference_uid: str | None = None
    referenced_layer_id: str | None = None
    referenced_sop_instance_uids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        local_series_id = (
            None
            if self.local_series_id is None
            else _opaque_id(self.local_series_id, "local_series_id")
        )
        series_uid = _dicom_uid(self.dicom_series_uid, "dicom_series_uid")
        frame_uid = _dicom_uid(self.frame_of_reference_uid, "frame_of_reference_uid")
        layer_id = (
            None
            if self.referenced_layer_id is None
            else _opaque_id(self.referenced_layer_id, "referenced_layer_id")
        )
        sop_uids = tuple(
            _dicom_uid(item, "referenced_sop_instance_uid")
            for item in self.referenced_sop_instance_uids
        )
        if len(set(sop_uids)) != len(sop_uids):
            raise ValidationError("Referenced SOP Instance UIDs must be unique.")
        if local_series_id is None and series_uid is None:
            raise ValidationError(
                "A layer reference needs a local series ID or DICOM Series Instance UID."
            )
        object.__setattr__(self, "local_series_id", local_series_id)
        object.__setattr__(self, "dicom_series_uid", series_uid)
        object.__setattr__(self, "frame_of_reference_uid", frame_uid)
        object.__setattr__(self, "referenced_layer_id", layer_id)
        object.__setattr__(self, "referenced_sop_instance_uids", sop_uids)


@dataclass(frozen=True, slots=True)
class ReferenceMatchReport:
    status: GeometryMatchStatus
    local_series_matches: bool | None
    dicom_series_matches: bool | None
    frame_of_reference_matches: bool | None
    geometry_difference: GeometryDifference | None
    was_resampled: bool = False

    @property
    def overlay_allowed(self) -> bool:
        return self.status in {GeometryMatchStatus.MATCHED, GeometryMatchStatus.RESAMPLED}


@dataclass(frozen=True, slots=True, kw_only=True)
class LayerBase:
    layer_id: str
    series_id: str
    name: str
    source: SourceReference
    created_by: LayerCreator = LayerCreator.IMPORT
    validation_state: LayerValidationState = LayerValidationState.PENDING
    presentation: LayerPresentation = field(default_factory=LayerPresentation)
    original_geometry: SpatialGeometry | None = None
    transform_chain: tuple[GeometryTransformRecord, ...] = ()
    derived_from_layer_ids: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = STUDY_LAYER_SCHEMA_VERSION
    revision: int = 1

    def __post_init__(self) -> None:
        if int(self.schema_version) != STUDY_LAYER_SCHEMA_VERSION:
            raise ValidationError(
                f"Unsupported layer schema version {self.schema_version}; "
                f"expected {STUDY_LAYER_SCHEMA_VERSION}."
            )
        if int(self.revision) < 1:
            raise ValidationError("Layer revision must be at least 1.")
        derived = tuple(
            _opaque_id(item, "derived_from_layer_id") for item in self.derived_from_layer_ids
        )
        if len(set(derived)) != len(derived):
            raise ValidationError("derived_from_layer_ids must be unique.")
        if self.layer_id in derived:
            raise ValidationError("A layer cannot derive from itself.")
        if not isinstance(self.source, SourceReference):
            raise ValidationError("source must be a SourceReference.")
        if not isinstance(self.presentation, LayerPresentation):
            raise ValidationError("presentation must be a LayerPresentation.")
        if self.original_geometry is not None and not isinstance(
            self.original_geometry, SpatialGeometry
        ):
            raise ValidationError("original_geometry must be SpatialGeometry or None.")
        if not all(isinstance(item, GeometryTransformRecord) for item in self.transform_chain):
            raise ValidationError("transform_chain contains an unsupported record.")
        object.__setattr__(self, "layer_id", _opaque_id(self.layer_id, "layer_id"))
        object.__setattr__(self, "series_id", _opaque_id(self.series_id, "series_id"))
        object.__setattr__(self, "name", _nonempty(self.name, "layer name", maximum=160))
        object.__setattr__(self, "created_by", LayerCreator(self.created_by))
        object.__setattr__(self, "validation_state", LayerValidationState(self.validation_state))
        object.__setattr__(self, "presentation", self.presentation)
        object.__setattr__(self, "transform_chain", tuple(self.transform_chain))
        object.__setattr__(self, "derived_from_layer_ids", derived)
        object.__setattr__(self, "provenance", sanitize_runtime_metadata(self.provenance))
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "revision", int(self.revision))

    def with_presentation(
        self,
        *,
        visible: bool | None = None,
        locked: bool | None = None,
        opacity: float | None = None,
        color: str | None = None,
    ) -> Self:
        current = self.presentation
        presentation = LayerPresentation(
            visible=current.visible if visible is None else visible,
            locked=current.locked if locked is None else locked,
            opacity=current.opacity if opacity is None else opacity,
            color=current.color if color is None else color,
        )
        # Presentation edits must stay responsive for large 3-D segmentation
        # arrays.  A regular dataclasses.replace() would reconstruct the
        # subclass and defensively copy its entire voxel payload.  Every other
        # field on this already-validated immutable instance is safe to share.
        updated = shallow_copy(self)
        object.__setattr__(updated, "presentation", presentation)
        object.__setattr__(updated, "revision", self.revision + 1)
        return updated

    def _validate_transform_chain(self, current_geometry: SpatialGeometry) -> None:
        original = self.original_geometry
        if original is None:
            raise ValidationError("Spatial layers must preserve their original geometry.")
        chain = self.transform_chain
        if not chain:
            if not original.matches(current_geometry):
                raise ValidationError(
                    "Changed layer geometry requires an explicit geometry transform chain."
                )
            return
        if chain[0].source_geometry_fingerprint != original.fingerprint:
            raise ValidationError("The transform chain does not start at original_geometry.")
        for previous, following in zip(chain, chain[1:], strict=False):
            if previous.target_geometry_fingerprint != following.source_geometry_fingerprint:
                raise ValidationError("The geometry transform chain is disconnected.")
        if chain[-1].target_geometry_fingerprint != current_geometry.fingerprint:
            raise ValidationError("The transform chain does not end at current geometry.")


@dataclass(frozen=True, slots=True, kw_only=True)
class VolumeLayer(LayerBase):
    volume: ImageVolume
    is_base_image: bool = False
    display_mapping: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super(VolumeLayer, self).__post_init__()
        geometry = SpatialGeometry.from_volume(self.volume)
        if self.original_geometry is None:
            object.__setattr__(self, "original_geometry", geometry)
        self._validate_transform_chain(geometry)
        object.__setattr__(self, "is_base_image", bool(self.is_base_image))
        object.__setattr__(self, "display_mapping", sanitize_runtime_metadata(self.display_mapping))

    @property
    def current_geometry(self) -> SpatialGeometry:
        return SpatialGeometry.from_volume(self.volume)


@dataclass(frozen=True, slots=True, kw_only=True)
class SegmentationLayer(LayerBase):
    array: np.ndarray
    geometry: SpatialGeometry
    value_type: SegmentationValueType
    reference: LayerReference
    labels: tuple[LabelDefinition, ...]
    reference_geometry: SpatialGeometry | None = None
    maximum_fractional_value: int | None = None
    fractional_type: FractionalType | None = None
    threshold: float | None = None
    statistics: tuple[LabelStatistics, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super(SegmentationLayer, self).__post_init__()
        if self.source.source_format is SourceFormat.JPEG or self.source.is_lossy:
            raise ValidationError("Lossy sources cannot be imported as segmentation layers.")
        if (
            self.source.source_format is SourceFormat.TIFF
            and self.source.compression is CompressionKind.UNKNOWN
        ):
            raise ValidationError(
                "TIFF segmentation sources must be verified as lossless before import."
            )
        supported_formats = {
            SourceFormat.PNG,
            SourceFormat.TIFF,
            SourceFormat.NIFTI,
            SourceFormat.DICOM_SEG,
            SourceFormat.GENERATED,
            SourceFormat.PLUGIN,
        }
        if self.source.source_format not in supported_formats:
            raise ValidationError(
                f"{self.source.source_format.value!r} is not a supported segmentation format."
            )

        array = _readonly_array(self.array, field_name="segmentation array")
        expected_shape = self.geometry.shape_zyx
        if array.ndim == 2:
            if expected_shape[0] != 1 or array.shape != expected_shape[1:]:
                raise ValidationError(
                    "A two-dimensional segmentation requires a single-slice matching geometry."
                )
        elif array.ndim == 3:
            if array.shape != expected_shape:
                raise ValidationError(
                    "Segmentation array shape must match its current spatial geometry."
                )
        else:
            raise ValidationError("Segmentation arrays must be two- or three-dimensional.")
        if np.iscomplexobj(array) or not np.all(np.isfinite(array)):
            raise ValidationError("Segmentation arrays must contain finite real values.")

        value_type = SegmentationValueType(self.value_type)
        labels = tuple(self.labels)
        if not all(isinstance(item, LabelDefinition) for item in labels):
            raise ValidationError("labels must contain LabelDefinition values.")
        label_values = tuple(item.value for item in labels)
        if not labels or len(set(label_values)) != len(labels):
            raise ValidationError("Segmentation labels must be nonempty and have unique values.")

        maximum = self.maximum_fractional_value
        fractional_type = self.fractional_type
        threshold = self.threshold
        if value_type in {SegmentationValueType.DISCRETE, SegmentationValueType.BINARY}:
            if not np.issubdtype(array.dtype, np.integer) and array.dtype != np.bool_:
                raise ValidationError("Discrete and binary segmentations require integer values.")
            if np.any(array < 0):
                raise ValidationError("Segmentation label values cannot be negative.")
            unique = set(int(item) for item in np.unique(array))
            if value_type is SegmentationValueType.BINARY:
                if not unique.issubset({0, 1}):
                    raise ValidationError("Binary segmentations may only contain 0 and 1.")
                if len(labels) != 1:
                    raise ValidationError("A binary segmentation represents one foreground label.")
            elif not (unique - {0}).issubset(set(label_values)):
                missing = sorted((unique - {0}) - set(label_values))
                raise ValidationError(f"Label schema is missing values present in data: {missing}.")
            if maximum is not None or fractional_type is not None or threshold is not None:
                raise ValidationError(
                    "Discrete and binary segmentations cannot declare fractional settings."
                )
        elif value_type is SegmentationValueType.FRACTIONAL:
            if not np.issubdtype(array.dtype, np.integer):
                raise ValidationError(
                    "Fractional DICOM SEG values must remain in their stored integer scale."
                )
            if maximum is None or int(maximum) <= 0:
                raise ValidationError(
                    "Fractional segmentation requires a positive maximum_fractional_value."
                )
            maximum = int(maximum)
            if np.any(array < 0) or np.any(array > maximum):
                raise ValidationError(
                    "Fractional segmentation values exceed their declared native scale."
                )
            if fractional_type is None:
                raise ValidationError("Fractional segmentation requires a fractional_type.")
            fractional_type = FractionalType(fractional_type)
            if len(labels) != 1:
                raise ValidationError("A fractional layer represents one segment definition.")
        else:
            if np.any(array < 0.0) or np.any(array > 1.0):
                raise ValidationError("Probability segmentations must contain values in [0, 1].")
            if maximum is not None or fractional_type is not None:
                raise ValidationError(
                    "Probability arrays use unit scale and cannot declare DICOM fractional fields."
                )
            if len(labels) != 1:
                raise ValidationError("A probability layer represents one segment definition.")

        if threshold is not None:
            threshold = float(threshold)
            if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValidationError("A non-destructive threshold must be in [0, 1].")

        if self.source.source_format is SourceFormat.DICOM_SEG:
            if value_type not in {
                SegmentationValueType.BINARY,
                SegmentationValueType.FRACTIONAL,
            }:
                raise ValidationError("DICOM SEG layers must be binary or fractional.")
            if (
                self.reference.dicom_series_uid is None
                or not self.reference.referenced_sop_instance_uids
            ):
                raise ValidationError("DICOM SEG requires referenced series and SOP Instance UIDs.")

        if self.original_geometry is None:
            object.__setattr__(self, "original_geometry", self.geometry)
        self._validate_transform_chain(self.geometry)
        self._validate_interpolation(value_type)
        statistics = _segmentation_statistics(array, value_type, labels, maximum)

        object.__setattr__(self, "array", array)
        object.__setattr__(self, "value_type", value_type)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "maximum_fractional_value", maximum)
        object.__setattr__(self, "fractional_type", fractional_type)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "statistics", statistics)

    @property
    def current_geometry(self) -> SpatialGeometry:
        return self.geometry

    @property
    def geometry_match_status(self) -> GeometryMatchStatus:
        if self.reference_geometry is None:
            return GeometryMatchStatus.UNRESOLVED_REFERENCE
        if not self.geometry.matches(self.reference_geometry):
            return GeometryMatchStatus.REQUIRES_RESAMPLING
        if any(item.kind is TransformKind.RESAMPLE for item in self.transform_chain):
            return GeometryMatchStatus.RESAMPLED
        return GeometryMatchStatus.MATCHED

    def reference_report(self, series: ImageSeries) -> ReferenceMatchReport:
        return _reference_report(
            self.reference,
            series,
            layer_geometry=self.geometry,
            transform_chain=self.transform_chain,
        )

    def with_label_visibility(self, label_value: int, visible: bool) -> SegmentationLayer:
        """Return a lightweight revision with one label's display state changed."""

        value = int(label_value)
        matches = [index for index, label in enumerate(self.labels) if label.value == value]
        if not matches:
            raise ValidationError(f"Label value {value} is not present in this segmentation.")
        index = matches[0]
        if self.labels[index].visible is bool(visible):
            return self
        labels = list(self.labels)
        labels[index] = replace(labels[index], visible=bool(visible))
        updated = shallow_copy(self)
        object.__setattr__(updated, "labels", tuple(labels))
        object.__setattr__(updated, "revision", self.revision + 1)
        return updated

    def derive_resampled(
        self,
        *,
        layer_id: str,
        name: str,
        array: np.ndarray,
        target_geometry: SpatialGeometry,
        interpolation: InterpolationMode,
        user_confirmed: bool,
        parameters: Mapping[str, Any] | None = None,
    ) -> SegmentationLayer:
        """Create a new layer on a target grid; the imported layer remains unchanged."""

        if self.presentation.locked:
            raise ValidationError("A locked layer cannot be used to create a resampled derivative.")
        new_id = _opaque_id(layer_id, "layer_id")
        if new_id == self.layer_id:
            raise ValidationError("A resampled derivative requires a new layer ID.")
        record = GeometryTransformRecord.resampling(
            self.geometry,
            target_geometry,
            interpolation=interpolation,
            user_confirmed=user_confirmed,
            parameters=parameters,
        )
        return SegmentationLayer(
            layer_id=new_id,
            series_id=self.series_id,
            name=name,
            source=self.source,
            created_by=LayerCreator.ALGORITHM,
            validation_state=LayerValidationState.PENDING,
            presentation=self.presentation,
            original_geometry=self.original_geometry,
            transform_chain=(*self.transform_chain, record),
            derived_from_layer_ids=(*self.derived_from_layer_ids, self.layer_id),
            provenance=self.provenance,
            array=array,
            geometry=target_geometry,
            value_type=self.value_type,
            reference=self.reference,
            labels=self.labels,
            reference_geometry=self.reference_geometry,
            maximum_fractional_value=self.maximum_fractional_value,
            fractional_type=self.fractional_type,
            threshold=self.threshold,
        )

    def _validate_interpolation(self, value_type: SegmentationValueType) -> None:
        for record in self.transform_chain:
            if record.kind is not TransformKind.RESAMPLE:
                continue
            if value_type in {
                SegmentationValueType.DISCRETE,
                SegmentationValueType.BINARY,
            }:
                if record.interpolation is not InterpolationMode.NEAREST:
                    raise ValidationError(
                        "Discrete labels can only use nearest-neighbour resampling."
                    )
            elif record.interpolation is InterpolationMode.NEAREST:
                raise ValidationError(
                    "Fractional and probability layers require an explicit "
                    "continuous interpolation."
                )


def _segmentation_statistics(
    array: np.ndarray,
    value_type: SegmentationValueType,
    labels: tuple[LabelDefinition, ...],
    maximum_fractional_value: int | None,
) -> tuple[LabelStatistics, ...]:
    if value_type is SegmentationValueType.DISCRETE:
        result = []
        for label in labels:
            selected = array == label.value
            count = int(np.count_nonzero(selected))
            result.append(
                LabelStatistics(
                    label.value,
                    count,
                    count,
                    float(label.value),
                    float(label.value),
                    float(label.value),
                )
            )
        return tuple(result)

    foreground = array > 0
    selected_values = np.asarray(array[foreground], dtype=np.float64)
    if value_type is SegmentationValueType.FRACTIONAL and maximum_fractional_value:
        selected_values = selected_values / maximum_fractional_value
    if selected_values.size:
        minimum = float(np.min(selected_values))
        maximum = float(np.max(selected_values))
        mean = float(np.mean(selected_values))
    else:
        minimum = maximum = mean = 0.0
    label = labels[0]
    return (
        LabelStatistics(
            label_value=label.value,
            voxel_count=int(array.size),
            nonzero_voxel_count=int(np.count_nonzero(foreground)),
            minimum=minimum,
            maximum=maximum,
            mean=mean,
        ),
    )


@dataclass(frozen=True, slots=True)
class ContourPath:
    points_ras: np.ndarray
    referenced_sop_instance_uid: str | None = None

    def __post_init__(self) -> None:
        points = _readonly_array(self.points_ras, field_name="contour points")
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
            raise ValidationError("A contour path requires at least three RAS+ points (N, 3).")
        if not np.all(np.isfinite(points)):
            raise ValidationError("Contour points cannot contain NaN or infinity.")
        points = np.array(points, dtype=np.float64, copy=True)
        points.setflags(write=False)
        object.__setattr__(self, "points_ras", points)
        object.__setattr__(
            self,
            "referenced_sop_instance_uid",
            _dicom_uid(self.referenced_sop_instance_uid, "referenced_sop_instance_uid"),
        )


@dataclass(frozen=True, slots=True)
class RegionOfInterest:
    roi_number: int
    name: str
    color: str
    contours: tuple[ContourPath, ...]
    visible: bool = True

    def __post_init__(self) -> None:
        number = int(self.roi_number)
        if number <= 0:
            raise ValidationError("RTSTRUCT ROI numbers must be positive.")
        contours = tuple(self.contours)
        if not contours:
            raise ValidationError("A region of interest must contain at least one contour.")
        object.__setattr__(self, "roi_number", number)
        object.__setattr__(self, "name", _nonempty(self.name, "ROI name", maximum=128))
        object.__setattr__(self, "color", _color(self.color, "ROI color"))
        object.__setattr__(self, "contours", contours)
        object.__setattr__(self, "visible", bool(self.visible))


@dataclass(frozen=True, slots=True)
class RasterizationSettings:
    target_geometry: SpatialGeometry
    fill_rule: str = "even-odd"
    supersampling: int = 1
    user_confirmed: bool = False

    def __post_init__(self) -> None:
        fill_rule = str(self.fill_rule).strip().lower()
        if fill_rule not in {"even-odd", "non-zero"}:
            raise ValidationError("Contour fill_rule must be 'even-odd' or 'non-zero'.")
        supersampling = int(self.supersampling)
        if not 1 <= supersampling <= 16:
            raise ValidationError("Contour supersampling must be between 1 and 16.")
        if not self.user_confirmed:
            raise ValidationError("Contour rasterization requires explicit user confirmation.")
        object.__setattr__(self, "fill_rule", fill_rule)
        object.__setattr__(self, "supersampling", supersampling)


@dataclass(frozen=True, slots=True, kw_only=True)
class ContourLayer(LayerBase):
    reference: LayerReference
    rois: tuple[RegionOfInterest, ...]
    reference_geometry: SpatialGeometry | None = None
    rasterization: RasterizationSettings | None = None

    def __post_init__(self) -> None:
        super(ContourLayer, self).__post_init__()
        if self.source.source_format not in {
            SourceFormat.RTSTRUCT,
            SourceFormat.GENERATED,
            SourceFormat.PLUGIN,
        }:
            raise ValidationError("Contour layers require RTSTRUCT or a derived contour source.")
        rois = tuple(self.rois)
        roi_numbers = tuple(item.roi_number for item in rois)
        if not rois or len(set(roi_numbers)) != len(rois):
            raise ValidationError("Contour ROIs must be nonempty and use unique ROI numbers.")
        if self.source.source_format is SourceFormat.RTSTRUCT and (
            self.reference.dicom_series_uid is None or self.reference.frame_of_reference_uid is None
        ):
            raise ValidationError(
                "RTSTRUCT requires referenced Series Instance and Frame of Reference UIDs."
            )
        if self.original_geometry is None and self.reference_geometry is not None:
            object.__setattr__(self, "original_geometry", self.reference_geometry)
        if self.rasterization is not None:
            if self.original_geometry is None:
                raise ValidationError("Rasterized contours require a preserved reference geometry.")
            if not self.transform_chain:
                raise ValidationError("Contour rasterization requires a transform record.")
            self._validate_transform_chain(self.rasterization.target_geometry)
        object.__setattr__(self, "rois", rois)

    @property
    def current_geometry(self) -> SpatialGeometry | None:
        if self.rasterization is not None:
            return self.rasterization.target_geometry
        return self.reference_geometry

    def reference_report(self, series: ImageSeries) -> ReferenceMatchReport:
        return _reference_report(
            self.reference,
            series,
            layer_geometry=self.reference_geometry,
            transform_chain=self.transform_chain,
            geometry_optional=True,
        )


Layer = VolumeLayer | SegmentationLayer | ContourLayer


@dataclass(frozen=True, slots=True)
class ImageSeries:
    series_id: str
    modality: str
    source: SourceReference
    geometry: SpatialGeometry
    intensity_semantics: IntensitySemantics
    layers: tuple[Layer, ...] = ()
    study_instance_uid: str | None = None
    series_instance_uid: str | None = None
    frame_of_reference_uid: str | None = None
    data_shape: tuple[int, ...] | None = None
    time_axis: int | None = None
    channel_axis: int | None = None
    schema_version: int = STUDY_LAYER_SCHEMA_VERSION
    revision: int = 1

    def __post_init__(self) -> None:
        if int(self.schema_version) != STUDY_LAYER_SCHEMA_VERSION:
            raise ValidationError(
                f"Unsupported series schema version {self.schema_version}; "
                f"expected {STUDY_LAYER_SCHEMA_VERSION}."
            )
        if int(self.revision) < 1:
            raise ValidationError("Series revision must be at least 1.")
        series_id = _opaque_id(self.series_id, "series_id")
        modality = _nonempty(self.modality, "modality", maximum=32).upper()
        study_uid = _dicom_uid(self.study_instance_uid, "study_instance_uid")
        series_uid = _dicom_uid(self.series_instance_uid, "series_instance_uid")
        frame_uid = _dicom_uid(self.frame_of_reference_uid, "frame_of_reference_uid")
        if self.source.source_type is SourceType.DICOM and (
            study_uid is None or series_uid is None
        ):
            raise ValidationError("DICOM image series require Study and Series Instance UIDs.")

        data_shape = (
            self.geometry.shape_zyx
            if self.data_shape is None
            else tuple(int(item) for item in self.data_shape)
        )
        if len(data_shape) not in {3, 4} or any(item <= 0 for item in data_shape):
            raise ValidationError("ImageSeries data_shape must contain three or four dimensions.")
        time_axis = self.time_axis
        channel_axis = self.channel_axis
        for value, name in ((time_axis, "time_axis"), (channel_axis, "channel_axis")):
            if value is not None and not 0 <= int(value) < len(data_shape):
                raise ValidationError(f"{name} is outside data_shape.")
        if time_axis is not None:
            time_axis = int(time_axis)
        if channel_axis is not None:
            channel_axis = int(channel_axis)
        if time_axis is not None and time_axis == channel_axis:
            raise ValidationError("time_axis and channel_axis must be different.")
        declared_nonspatial_axes = tuple(
            item for item in (time_axis, channel_axis) if item is not None
        )
        if len(data_shape) == 3 and declared_nonspatial_axes:
            raise ValidationError("A 3-D series cannot declare a time or channel axis.")
        if len(data_shape) == 4 and len(declared_nonspatial_axes) != 1:
            raise ValidationError("A 4-D series must identify exactly one time or channel axis.")
        spatial_shape = tuple(
            size for index, size in enumerate(data_shape) if index not in declared_nonspatial_axes
        )
        if spatial_shape != self.geometry.shape_zyx:
            raise ValidationError(
                "ImageSeries spatial data dimensions must match shape_zyx geometry."
            )

        layers = tuple(self.layers)
        layer_ids = tuple(layer.layer_id for layer in layers)
        if len(set(layer_ids)) != len(layer_ids):
            raise ValidationError("Layer IDs must be unique within a series.")
        for layer in layers:
            if layer.series_id != series_id:
                raise ValidationError("Every layer must identify its owning series.")
            if (
                isinstance(layer, VolumeLayer)
                and layer.is_base_image
                and not layer.current_geometry.matches(self.geometry)
            ):
                raise ValidationError("A base volume layer must match its series geometry.")

        object.__setattr__(self, "series_id", series_id)
        object.__setattr__(self, "modality", modality)
        object.__setattr__(
            self, "intensity_semantics", IntensitySemantics(self.intensity_semantics)
        )
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "study_instance_uid", study_uid)
        object.__setattr__(self, "series_instance_uid", series_uid)
        object.__setattr__(self, "frame_of_reference_uid", frame_uid)
        object.__setattr__(self, "data_shape", data_shape)
        object.__setattr__(self, "time_axis", time_axis)
        object.__setattr__(self, "channel_axis", channel_axis)
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "revision", int(self.revision))

    @property
    def container_format(self) -> SourceFormat:
        return self.source.source_format

    @property
    def shape(self) -> tuple[int, ...]:
        assert self.data_shape is not None
        return self.data_shape

    @property
    def spacing(self) -> tuple[float, float, float]:
        return self.geometry.spacing_xyz

    @property
    def orientation(self) -> np.ndarray:
        return self.geometry.direction_ras

    @property
    def affine(self) -> np.ndarray:
        return self.geometry.affine_ras

    @property
    def base_layers(self) -> tuple[VolumeLayer, ...]:
        return tuple(
            layer for layer in self.layers if isinstance(layer, VolumeLayer) and layer.is_base_image
        )

    def with_layer(self, layer: Layer) -> ImageSeries:
        if layer.series_id != self.series_id:
            raise ValidationError("The layer belongs to a different series.")
        if any(existing.layer_id == layer.layer_id for existing in self.layers):
            raise ValidationError(f"Layer ID {layer.layer_id!r} already exists in this series.")
        return replace(self, layers=(*self.layers, layer), revision=self.revision + 1)

    def replace_layer(self, layer: Layer) -> ImageSeries:
        if layer.series_id != self.series_id:
            raise ValidationError("The layer belongs to a different series.")
        matches = [
            index for index, item in enumerate(self.layers) if item.layer_id == layer.layer_id
        ]
        if not matches:
            raise ValidationError(f"Layer ID {layer.layer_id!r} is not present in this series.")
        if self.layers[matches[0]].presentation.locked:
            raise ValidationError("A locked layer cannot be replaced.")
        updated = list(self.layers)
        updated[matches[0]] = layer
        return replace(self, layers=tuple(updated), revision=self.revision + 1)

    def with_layer_presentation(
        self,
        layer_id: str,
        *,
        visible: bool | None = None,
        locked: bool | None = None,
        opacity: float | None = None,
        color: str | None = None,
    ) -> ImageSeries:
        """Update display-only state, including hiding or unlocking a locked layer."""

        normalized = _opaque_id(layer_id, "layer_id")
        matches = [index for index, item in enumerate(self.layers) if item.layer_id == normalized]
        if not matches:
            raise ValidationError(f"Layer ID {normalized!r} is not present in this series.")
        index = matches[0]
        current = self.layers[index]
        if current.presentation.locked and (opacity is not None or color is not None):
            raise ValidationError("A locked layer's opacity or color cannot be changed.")
        updated_layer = current.with_presentation(
            visible=visible,
            locked=locked,
            opacity=opacity,
            color=color,
        )
        updated = list(self.layers)
        updated[index] = updated_layer
        return replace(self, layers=tuple(updated), revision=self.revision + 1)

    def with_segmentation_label_visibility(
        self,
        layer_id: str,
        label_value: int,
        visible: bool,
    ) -> ImageSeries:
        """Update one segmentation label without copying its voxel array."""

        normalized = _opaque_id(layer_id, "layer_id")
        matches = [index for index, item in enumerate(self.layers) if item.layer_id == normalized]
        if not matches:
            raise ValidationError(f"Layer ID {normalized!r} is not present in this series.")
        index = matches[0]
        current = self.layers[index]
        if not isinstance(current, SegmentationLayer):
            raise ValidationError("Only segmentation layers contain label visibility state.")
        if current.presentation.locked:
            raise ValidationError("A locked layer's labels cannot be changed.")
        updated_layer = current.with_label_visibility(label_value, visible)
        if updated_layer is current:
            return self
        updated = list(self.layers)
        updated[index] = updated_layer
        return replace(self, layers=tuple(updated), revision=self.revision + 1)


def _reference_report(
    reference: LayerReference,
    series: ImageSeries,
    *,
    layer_geometry: SpatialGeometry | None,
    transform_chain: tuple[GeometryTransformRecord, ...],
    geometry_optional: bool = False,
) -> ReferenceMatchReport:
    local_matches = (
        None if reference.local_series_id is None else reference.local_series_id == series.series_id
    )
    dicom_matches = (
        None
        if reference.dicom_series_uid is None
        else reference.dicom_series_uid == series.series_instance_uid
    )
    frame_matches = (
        None
        if reference.frame_of_reference_uid is None
        else reference.frame_of_reference_uid == series.frame_of_reference_uid
    )
    identity_checks = tuple(
        item for item in (local_matches, dicom_matches, frame_matches) if item is not None
    )
    if any(item is False for item in identity_checks):
        status = GeometryMatchStatus.REFERENCE_MISMATCH
        difference = None if layer_geometry is None else layer_geometry.compare_to(series.geometry)
    elif not identity_checks:
        status = GeometryMatchStatus.UNRESOLVED_REFERENCE
        difference = None
    elif layer_geometry is None:
        difference = None
        status = (
            GeometryMatchStatus.MATCHED
            if geometry_optional
            else GeometryMatchStatus.UNRESOLVED_REFERENCE
        )
    else:
        difference = layer_geometry.compare_to(series.geometry)
        if not difference.matches:
            status = GeometryMatchStatus.REQUIRES_RESAMPLING
        elif any(item.kind is TransformKind.RESAMPLE for item in transform_chain):
            status = GeometryMatchStatus.RESAMPLED
        else:
            status = GeometryMatchStatus.MATCHED
    return ReferenceMatchReport(
        status=status,
        local_series_matches=local_matches,
        dicom_series_matches=dicom_matches,
        frame_of_reference_matches=frame_matches,
        geometry_difference=difference,
        was_resampled=any(item.kind is TransformKind.RESAMPLE for item in transform_chain),
    )


@dataclass(frozen=True, slots=True)
class ImageStudy:
    study_id: str
    source_type: SourceType
    series: tuple[ImageSeries, ...]
    deidentification_status: DeidentificationStatus = DeidentificationStatus.NOT_ASSESSED
    provenance: Mapping[str, Any] = field(default_factory=dict)
    world_coordinate_convention: WorldCoordinateConvention = WorldCoordinateConvention.RAS_PLUS
    schema_version: int = STUDY_LAYER_SCHEMA_VERSION
    revision: int = 1

    def __post_init__(self) -> None:
        if int(self.schema_version) != STUDY_LAYER_SCHEMA_VERSION:
            raise ValidationError(
                f"Unsupported study schema version {self.schema_version}; "
                f"expected {STUDY_LAYER_SCHEMA_VERSION}."
            )
        if int(self.revision) < 1:
            raise ValidationError("Study revision must be at least 1.")
        series = tuple(self.series)
        series_ids = tuple(item.series_id for item in series)
        if len(set(series_ids)) != len(series_ids):
            raise ValidationError("Series IDs must be unique within a study.")
        dicom_uids = tuple(
            item.series_instance_uid for item in series if item.series_instance_uid is not None
        )
        if len(set(dicom_uids)) != len(dicom_uids):
            raise ValidationError("DICOM Series Instance UIDs must be unique within a study.")
        object.__setattr__(self, "study_id", _opaque_id(self.study_id, "study_id"))
        object.__setattr__(self, "source_type", SourceType(self.source_type))
        object.__setattr__(self, "series", series)
        object.__setattr__(
            self,
            "deidentification_status",
            DeidentificationStatus(self.deidentification_status),
        )
        object.__setattr__(self, "provenance", sanitize_runtime_metadata(self.provenance))
        object.__setattr__(
            self,
            "world_coordinate_convention",
            WorldCoordinateConvention(self.world_coordinate_convention),
        )
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "revision", int(self.revision))

    def find_series(self, series_id: str) -> ImageSeries:
        normalized = _opaque_id(series_id, "series_id")
        for item in self.series:
            if item.series_id == normalized:
                return item
        raise ValidationError(f"Series ID {normalized!r} is not present in this study.")

    def with_series(self, series: ImageSeries) -> ImageStudy:
        if any(existing.series_id == series.series_id for existing in self.series):
            raise ValidationError(f"Series ID {series.series_id!r} already exists in this study.")
        return replace(self, series=(*self.series, series), revision=self.revision + 1)

    def replace_series(self, series: ImageSeries) -> ImageStudy:
        matches = [
            index for index, item in enumerate(self.series) if item.series_id == series.series_id
        ]
        if not matches:
            raise ValidationError(f"Series ID {series.series_id!r} is not present in this study.")
        updated = list(self.series)
        updated[matches[0]] = series
        return replace(self, series=tuple(updated), revision=self.revision + 1)

    def with_layer(self, series_id: str, layer: Layer) -> ImageStudy:
        series = self.find_series(series_id)
        return self.replace_series(series.with_layer(layer))


__all__ = [
    "STUDY_LAYER_SCHEMA_VERSION",
    "CompressionKind",
    "ContourLayer",
    "ContourPath",
    "DeidentificationStatus",
    "FractionalType",
    "GeometryDifference",
    "GeometryMatchStatus",
    "GeometryTransformRecord",
    "ImageSeries",
    "ImageStudy",
    "InterpolationMode",
    "LabelDefinition",
    "LabelStatistics",
    "Layer",
    "LayerBase",
    "LayerCreator",
    "LayerPresentation",
    "LayerReference",
    "LayerValidationState",
    "RasterizationSettings",
    "ReferenceMatchReport",
    "RegionOfInterest",
    "SegmentationLayer",
    "SegmentationValueType",
    "SourceFormat",
    "SourceReference",
    "SpatialGeometry",
    "TransformKind",
    "VolumeLayer",
    "WorldCoordinateConvention",
]
