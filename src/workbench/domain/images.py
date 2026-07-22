"""Unified image-domain types with explicit medical and raster semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any

import numpy as np

from ..errors import ValidationError
from .transforms import TransformRecord


class SourceType(str, Enum):
    RASTER = "raster"
    DICOM = "dicom"
    NIFTI = "nifti"
    PLUGIN = "plugin"
    GENERATED = "generated"


class IntensitySemantics(str, Enum):
    """Meaning of stored sample values, independent from display mapping.

    ``QUANTITATIVE`` and ``LABEL`` remain for compatibility with existing
    plugins.  Loaders should prefer the more specific members below and use
    ``UNKNOWN`` whenever the file does not contain enough evidence.
    """

    GRAYSCALE = "grayscale"
    COLOR = "color"
    ARBITRARY_SIGNAL = "arbitrary_signal"
    RELATIVE_QUANTITATIVE = "relative_quantitative"
    HOUNSFIELD_UNIT = "hounsfield_unit"
    HU = "hounsfield_unit"
    SUV = "suv"
    PROBABILITY = "probability"
    DISCRETE_LABEL = "discrete_label"
    QUANTITATIVE = "quantitative"
    LABEL = "label"
    UNKNOWN = "unknown"


class ColorSpace(str, Enum):
    GRAYSCALE = "GRAY"
    RGB = "RGB"
    RGBA = "RGBA"


class AlphaSemantics(str, Enum):
    NONE = "none"
    STRAIGHT = "straight"
    PREMULTIPLIED = "premultiplied"


class SpacingSource(str, Enum):
    FILE = "file"
    USER = "user_provided"


class Capability(str, Enum):
    VIEW_2D = "view_2d"
    ZOOM_PAN = "zoom_pan"
    PIXEL_VALUE = "pixel_value"
    HISTOGRAM = "histogram"
    PIXEL_MEASUREMENT = "pixel_measurement"
    PHYSICAL_MEASUREMENT = "physical_measurement"
    ROI = "roi"
    ANNOTATION = "annotation"
    OVERLAY = "overlay"
    FRAME_NAVIGATION = "frame_navigation"
    FRAME_PLAYBACK = "frame_playback"
    ORTHOGONAL_VIEWS = "orthogonal_views"
    VOLUME_RENDERING = "volume_rendering"
    HU_WINDOWING = "hu_windowing"
    RECONSTRUCTION = "reconstruction"


_SENSITIVE_METADATA_TOKENS = (
    "patient",
    "birth",
    "accession",
    "physician",
    "institution",
    "operator",
    "address",
    "telephone",
    "medical_record",
    "subject_id",
    "study_date",
    "series_date",
    "acquisition_date",
    "referring",
    "study_uid",
    "series_uid",
    "sop_instance_uid",
    "study_description",
    "series_description",
    "file_path",
    "source_path",
    "local_path",
    "weight_path",
    "filename",
)


def _metadata_is_sensitive(key: str) -> bool:
    normalized = key.lower().replace("-", "_").replace(" ", "_")
    compact = normalized.replace("_", "")
    return any(
        token in normalized or token.replace("_", "") in compact
        for token in _SENSITIVE_METADATA_TOKENS
    )


def sanitize_runtime_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """Drop known PHI-bearing keys and deeply freeze remaining metadata."""

    sanitized: dict[str, Any] = {}
    for raw_key, value in metadata.items():
        key = str(raw_key)
        if _metadata_is_sensitive(key):
            continue
        if isinstance(value, Mapping):
            value = sanitize_runtime_metadata(value)
        elif isinstance(value, np.generic):
            value = value.item()
        elif isinstance(value, np.ndarray):
            value = _freeze_metadata_sequence(value.tolist())
        elif isinstance(value, (list, tuple)):
            value = _freeze_metadata_sequence(value)
        elif isinstance(value, (set, frozenset)):
            value = _freeze_metadata_sequence(tuple(sorted(value, key=repr)))
        elif isinstance(value, bytes):
            # Binary payloads have no place in runtime metadata and could
            # accidentally retain encoded patient data.
            value = f"<{len(value)} bytes omitted>"
        sanitized[key] = value
    return MappingProxyType(sanitized)


def _freeze_metadata_sequence(values: Sequence[Any]) -> tuple[Any, ...]:
    frozen: list[Any] = []
    for value in values:
        if isinstance(value, Mapping):
            frozen.append(sanitize_runtime_metadata(value))
        elif isinstance(value, np.ndarray):
            frozen.append(_freeze_metadata_sequence(value.tolist()))
        elif isinstance(value, np.generic):
            frozen.append(value.item())
        elif isinstance(value, (list, tuple)):
            frozen.append(_freeze_metadata_sequence(value))
        elif isinstance(value, (set, frozenset)):
            frozen.append(_freeze_metadata_sequence(tuple(sorted(value, key=repr))))
        elif isinstance(value, bytes):
            frozen.append(f"<{len(value)} bytes omitted>")
        else:
            frozen.append(value)
    return tuple(frozen)


def _readonly_array(value: np.ndarray | Sequence[Any]) -> np.ndarray:
    array = np.asanyarray(value)
    if array.dtype == object:
        raise ValidationError("Image arrays cannot use object dtype.")
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise ValidationError(f"Unsupported image dtype {array.dtype}.")
    if any(dimension <= 0 for dimension in array.shape):
        raise ValidationError(f"Image dimensions must be positive, got {array.shape}.")
    # A read-only view is insufficient: callers may still retain a writable
    # alias to the same buffer and mutate supposedly immutable session state.
    view = np.array(array, copy=True)
    view.setflags(write=False)
    return view


@dataclass(frozen=True, slots=True)
class ImageData:
    """Top-level image contract shared by every supported source."""

    array: np.ndarray
    source_type: SourceType
    intensity_semantics: IntensitySemantics
    runtime_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array = _readonly_array(self.array)
        semantics = IntensitySemantics(self.intensity_semantics)
        if semantics in {
            IntensitySemantics.RELATIVE_QUANTITATIVE,
            IntensitySemantics.HOUNSFIELD_UNIT,
        } and (np.iscomplexobj(array) or not np.all(np.isfinite(array))):
            raise ValidationError(
                f"{semantics.value} images must contain finite real-valued samples."
            )
        if semantics is IntensitySemantics.PROBABILITY and (
            np.iscomplexobj(array)
            or not np.all(np.isfinite(array))
            or np.any(array < 0.0)
            or np.any(array > 1.0)
        ):
            raise ValidationError("Probability images must contain finite values in [0, 1].")
        if semantics is IntensitySemantics.DISCRETE_LABEL and (
            np.iscomplexobj(array)
            or not np.all(np.isfinite(array))
            or np.any(array < 0.0)
            or not np.all(array == np.floor(array))
        ):
            raise ValidationError(
                "Discrete-label images must contain finite nonnegative integer values."
            )
        if semantics is IntensitySemantics.SUV and (
            np.iscomplexobj(array) or not np.all(np.isfinite(array)) or np.any(array < 0.0)
        ):
            raise ValidationError("SUV images must contain finite nonnegative values.")
        object.__setattr__(self, "array", array)
        object.__setattr__(self, "source_type", SourceType(self.source_type))
        object.__setattr__(self, "intensity_semantics", semantics)
        object.__setattr__(
            self, "runtime_metadata", sanitize_runtime_metadata(self.runtime_metadata)
        )

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.array.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def has_explicit_quantitative_semantics(self) -> bool:
        """Whether values have an explicitly declared quantitative meaning."""

        return self.intensity_semantics in {
            IntensitySemantics.RELATIVE_QUANTITATIVE,
            IntensitySemantics.HOUNSFIELD_UNIT,
            IntensitySemantics.SUV,
            IntensitySemantics.PROBABILITY,
            # Compatibility for callers that deliberately construct the
            # legacy generic semantic.  Built-in medical loaders never infer
            # this member for otherwise unknown data.
            IntensitySemantics.QUANTITATIVE,
        }

    @property
    def capabilities(self) -> frozenset[Capability]:
        return frozenset()


@dataclass(frozen=True, slots=True)
class RasterImage2D(ImageData):
    """One grayscale or color image in top-left-origin pixel coordinates."""

    bit_depth: int = 8
    color_space: ColorSpace = ColorSpace.GRAYSCALE
    channel_order: tuple[str, ...] = ()
    alpha_semantics: AlphaSemantics = AlphaSemantics.NONE
    transform_record: TransformRecord | None = None
    pixel_spacing: tuple[float, float] | None = None  # (x, y)
    spacing_source: SpacingSource | None = None

    def __post_init__(self) -> None:
        super(RasterImage2D, self).__post_init__()
        if self.array.ndim not in {2, 3}:
            raise ValidationError(f"RasterImage2D must be HxW or HxWxC, got {self.array.shape}.")
        channels = 1 if self.array.ndim == 2 else self.array.shape[2]
        if channels not in {1, 3, 4}:
            raise ValidationError(f"RasterImage2D supports 1, 3, or 4 channels, got {channels}.")
        color_space = ColorSpace(self.color_space)
        expected_channels = {
            ColorSpace.GRAYSCALE: 1,
            ColorSpace.RGB: 3,
            ColorSpace.RGBA: 4,
        }[color_space]
        if channels != expected_channels:
            raise ValidationError(
                f"{color_space.value} requires {expected_channels} channel(s), got {channels}."
            )
        if self.bit_depth <= 0:
            raise ValidationError("bit_depth must be positive.")
        expected_order = {
            ColorSpace.GRAYSCALE: ("Y",),
            ColorSpace.RGB: ("R", "G", "B"),
            ColorSpace.RGBA: ("R", "G", "B", "A"),
        }[color_space]
        channel_order = tuple(self.channel_order) or expected_order
        if channel_order != expected_order:
            raise ValidationError(
                f"Canonical {color_space.value} channel order is {expected_order}, "
                f"got {channel_order}."
            )
        alpha = AlphaSemantics(self.alpha_semantics)
        if color_space is ColorSpace.RGBA and alpha is AlphaSemantics.NONE:
            raise ValidationError("RGBA data must declare its alpha semantics.")
        if color_space is not ColorSpace.RGBA and alpha is not AlphaSemantics.NONE:
            raise ValidationError("Only RGBA data can declare alpha semantics.")
        height, width = self.array.shape[:2]
        transform = self.transform_record or TransformRecord.identity((height, width))
        if transform.output_shape != (height, width):
            raise ValidationError(
                "transform_record output shape must match the canonical raster shape."
            )
        spacing = self.pixel_spacing
        source = self.spacing_source
        if spacing is None and source is not None:
            raise ValidationError("spacing_source cannot be set without pixel_spacing.")
        if spacing is not None:
            if len(spacing) != 2 or any(float(item) <= 0 for item in spacing):
                raise ValidationError(
                    "pixel_spacing must contain two positive values in millimetres."
                )
            if source is not SpacingSource.USER:
                raise ValidationError(
                    "Raster physical spacing must be explicitly user-provided; "
                    "DPI is not medical spacing."
                )
            spacing = (float(spacing[0]), float(spacing[1]))
        object.__setattr__(self, "color_space", color_space)
        object.__setattr__(self, "channel_order", channel_order)
        object.__setattr__(self, "alpha_semantics", alpha)
        object.__setattr__(self, "transform_record", transform)
        object.__setattr__(self, "pixel_spacing", spacing)

    @property
    def capabilities(self) -> frozenset[Capability]:
        capabilities = {
            Capability.VIEW_2D,
            Capability.ZOOM_PAN,
            Capability.PIXEL_VALUE,
            Capability.HISTOGRAM,
            Capability.PIXEL_MEASUREMENT,
            Capability.ROI,
            Capability.ANNOTATION,
            Capability.OVERLAY,
        }
        if self.pixel_spacing is not None:
            capabilities.add(Capability.PHYSICAL_MEASUREMENT)
        return frozenset(capabilities)

    def with_user_spacing(self, x_mm: float, y_mm: float) -> RasterImage2D:
        """Return a new raster with explicitly user-supplied physical spacing."""

        return replace(
            self,
            pixel_spacing=(float(x_mm), float(y_mm)),
            spacing_source=SpacingSource.USER,
        )


@dataclass(frozen=True, slots=True)
class ImageSequence2D(ImageData):
    """A page/frame sequence without implied 3-D physical geometry."""

    bit_depth: int = 8
    color_space: ColorSpace = ColorSpace.GRAYSCALE
    channel_order: tuple[str, ...] = ()
    alpha_semantics: AlphaSemantics = AlphaSemantics.NONE
    frame_transforms: tuple[TransformRecord, ...] = ()

    def __post_init__(self) -> None:
        super(ImageSequence2D, self).__post_init__()
        if self.array.ndim not in {3, 4}:
            raise ValidationError(
                f"ImageSequence2D must be NxHxW or NxHxWxC, got {self.array.shape}."
            )
        channels = 1 if self.array.ndim == 3 else self.array.shape[3]
        color_space = ColorSpace(self.color_space)
        expected_channels = {ColorSpace.GRAYSCALE: 1, ColorSpace.RGB: 3, ColorSpace.RGBA: 4}[
            color_space
        ]
        if channels != expected_channels:
            raise ValidationError(
                f"{color_space.value} requires {expected_channels} channel(s), got {channels}."
            )
        expected_order = {
            ColorSpace.GRAYSCALE: ("Y",),
            ColorSpace.RGB: ("R", "G", "B"),
            ColorSpace.RGBA: ("R", "G", "B", "A"),
        }[color_space]
        order = tuple(self.channel_order) or expected_order
        if order != expected_order:
            raise ValidationError(f"Canonical channel order must be {expected_order}, got {order}.")
        alpha = AlphaSemantics(self.alpha_semantics)
        if (color_space is ColorSpace.RGBA) != (alpha is not AlphaSemantics.NONE):
            raise ValidationError("Alpha semantics and sequence color space disagree.")
        transforms = tuple(self.frame_transforms)
        if transforms and len(transforms) != self.array.shape[0]:
            raise ValidationError("frame_transforms length must match frame count.")
        if not transforms:
            transforms = tuple(
                TransformRecord.identity(self.array.shape[1:3]) for _ in range(self.array.shape[0])
            )
        if any(item.output_shape != self.array.shape[1:3] for item in transforms):
            raise ValidationError("Every frame transform must match the canonical frame shape.")
        object.__setattr__(self, "color_space", color_space)
        object.__setattr__(self, "channel_order", order)
        object.__setattr__(self, "alpha_semantics", alpha)
        object.__setattr__(self, "frame_transforms", transforms)

    @property
    def frame_count(self) -> int:
        return self.array.shape[0]

    @property
    def capabilities(self) -> frozenset[Capability]:
        return frozenset(
            {
                Capability.VIEW_2D,
                Capability.ZOOM_PAN,
                Capability.PIXEL_VALUE,
                Capability.HISTOGRAM,
                Capability.PIXEL_MEASUREMENT,
                Capability.ROI,
                Capability.ANNOTATION,
                Capability.OVERLAY,
                Capability.FRAME_NAVIGATION,
                Capability.FRAME_PLAYBACK,
            }
        )


@dataclass(frozen=True, slots=True)
class ImageVolume(ImageData):
    """A RAS+ medical volume stored as ``(z, y, x)`` voxels."""

    affine: np.ndarray = field(default_factory=lambda: np.eye(4))
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)  # x, y, z in mm
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)  # RAS+ mm
    direction: np.ndarray = field(default_factory=lambda: np.eye(3))
    modality: str = "UNKNOWN"

    def __post_init__(self) -> None:
        super(ImageVolume, self).__post_init__()
        if self.array.ndim != 3:
            raise ValidationError(f"ImageVolume must be ZxYxX, got {self.array.shape}.")
        affine = np.asarray(self.affine, dtype=np.float64)
        direction = np.asarray(self.direction, dtype=np.float64)
        if affine.shape != (4, 4) or direction.shape != (3, 3):
            raise ValidationError("ImageVolume affine/direction must be 4x4/3x3.")
        if not np.all(np.isfinite(affine)) or not np.all(np.isfinite(direction)):
            raise ValidationError("ImageVolume geometry contains NaN or infinity.")
        spacing = tuple(float(item) for item in self.spacing)
        origin = tuple(float(item) for item in self.origin)
        if len(spacing) != 3 or any(item <= 0 for item in spacing):
            raise ValidationError("ImageVolume spacing must contain three positive values.")
        if len(origin) != 3 or not np.all(np.isfinite(origin)):
            raise ValidationError("ImageVolume origin must contain three finite values.")
        column_norms = np.linalg.norm(direction, axis=0)
        if not np.allclose(column_norms, np.ones(3), atol=1e-4):
            raise ValidationError("ImageVolume direction columns must be unit vectors.")
        if np.isclose(np.linalg.det(direction), 0.0):
            raise ValidationError("ImageVolume direction axes must be linearly independent.")
        expected_affine = np.eye(4)
        expected_affine[:3, :3] = direction @ np.diag(spacing)
        expected_affine[:3, 3] = origin
        if not np.allclose(affine, expected_affine, atol=1e-4):
            raise ValidationError(
                "affine, spacing, origin, and direction describe inconsistent RAS+ geometry."
            )
        affine = affine.copy()
        affine.setflags(write=False)
        direction = direction.copy()
        direction.setflags(write=False)
        object.__setattr__(self, "affine", affine)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "modality", str(self.modality).upper())

    @property
    def capabilities(self) -> frozenset[Capability]:
        capabilities = {
            Capability.VIEW_2D,
            Capability.ZOOM_PAN,
            Capability.PIXEL_VALUE,
            Capability.HISTOGRAM,
            Capability.PIXEL_MEASUREMENT,
            Capability.PHYSICAL_MEASUREMENT,
            Capability.ROI,
            Capability.ANNOTATION,
            Capability.OVERLAY,
            Capability.FRAME_NAVIGATION,
            Capability.ORTHOGONAL_VIEWS,
            Capability.RECONSTRUCTION,
        }
        if self.intensity_semantics is IntensitySemantics.HOUNSFIELD_UNIT:
            capabilities.add(Capability.HU_WINDOWING)
        return frozenset(capabilities)

    def voxel_xyz_to_world_ras(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.shape == (3,):
            values = values.reshape(1, 3)
            squeeze = True
        else:
            squeeze = False
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValidationError("Voxel points must have shape (3,) or (N, 3).")
        homogeneous = np.column_stack((values, np.ones(len(values))))
        result = (homogeneous @ self.affine.T)[:, :3]
        return result[0] if squeeze else result

    def world_ras_to_voxel_xyz(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.shape == (3,):
            values = values.reshape(1, 3)
            squeeze = True
        else:
            squeeze = False
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValidationError("World points must have shape (3,) or (N, 3).")
        homogeneous = np.column_stack((values, np.ones(len(values))))
        result = (homogeneous @ np.linalg.inv(self.affine).T)[:, :3]
        return result[0] if squeeze else result

    def axial(self, index: int) -> np.ndarray:
        return self.array[index, :, :]

    def coronal(self, index: int) -> np.ndarray:
        return self.array[:, index, :]

    def sagittal(self, index: int) -> np.ndarray:
        return self.array[:, :, index]
