"""Declarative, reversible preprocessing contracts for two-dimensional inputs.

This module describes preprocessing; it intentionally does not decode images or
perform tensor operations.  A runtime implementation consumes these immutable
specifications and returns the future domain-layer ``TransformRecord`` alongside
the prepared tensor so spatial outputs can be mapped back to source pixels.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ._schema import (
    check_keys,
    expect_mapping,
    expect_sequence,
    fail,
    get_required,
    parse_bool,
    parse_enum,
    parse_number,
    parse_number_tuple,
    parse_size_2d,
    parse_string_tuple,
)
from .enums import (
    AlphaHandling,
    ColorSpace,
    CropAnchor,
    InterpolationMode,
    OrientationHandling,
    SpatialOperationKind,
    TensorDType,
    TensorLayout,
)
from .errors import ManifestValidationError

if TYPE_CHECKING:
    from dicom_viewer.domain.transforms import TransformRecord


_TWO_D_LAYOUTS = frozenset(
    {
        TensorLayout.HW,
        TensorLayout.HWC,
        TensorLayout.CHW,
        TensorLayout.NHWC,
        TensorLayout.NCHW,
    }
)


@dataclass(frozen=True, slots=True)
class NumericRange:
    """Inclusive declared numeric range before model normalization."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if self.minimum != self.minimum or self.maximum != self.maximum:
            raise ValueError("numeric range values must be finite")
        if self.minimum >= self.maximum:
            raise ValueError("numeric range minimum must be less than maximum")

    @classmethod
    def from_value(cls, value: Any, path: str = "value_range") -> NumericRange:
        values = parse_number_tuple(value, path, exact_length=2)
        if values[0] >= values[1]:
            fail(path, "minimum must be less than maximum", values)
        return cls(values[0], values[1])

    def to_list(self) -> list[float]:
        return [self.minimum, self.maximum]


@dataclass(frozen=True, slots=True)
class NormalizationSpec:
    """Channel normalization applied as ``(x * scale + offset - mean) / std``."""

    mean: tuple[float, ...]
    std: tuple[float, ...]
    scale: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        if not self.mean or not self.std:
            raise ValueError("normalization mean and std must not be empty")
        if len(self.mean) != len(self.std):
            raise ValueError("normalization mean and std must have equal lengths")
        if any(value <= 0 for value in self.std):
            raise ValueError("normalization std values must be greater than zero")

    @classmethod
    def from_mapping(cls, value: Any, path: str = "normalization") -> NormalizationSpec:
        data = expect_mapping(value, path)
        check_keys(
            data,
            path=path,
            allowed={"mean", "std", "scale", "offset"},
            required={"mean", "std"},
        )
        mean = parse_number_tuple(
            get_required(data, "mean", path),
            f"{path}.mean",
            minimum_length=1,
        )
        std = parse_number_tuple(get_required(data, "std", path), f"{path}.std", minimum_length=1)
        if len(mean) != len(std):
            fail(path, "mean and std must have equal lengths")
        if any(item <= 0 for item in std):
            fail(f"{path}.std", "all standard deviations must be greater than zero", std)
        scale = parse_number(data.get("scale", 1.0), f"{path}.scale")
        offset = parse_number(data.get("offset", 0.0), f"{path}.offset")
        return cls(mean=mean, std=std, scale=scale, offset=offset)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "mean": list(self.mean),
            "std": list(self.std),
            "scale": self.scale,
            "offset": self.offset,
        }


@dataclass(frozen=True, slots=True)
class SpatialOperation2D:
    """One ordered spatial operation in a reversible preprocessing pipeline."""

    operation: SpatialOperationKind
    size: tuple[int, int] | None = None
    interpolation: InterpolationMode | None = None
    anchor: CropAnchor = CropAnchor.CENTER
    pad_value: tuple[float, ...] = (0.0,)
    allow_upscale: bool = True

    def __post_init__(self) -> None:
        needs_size = self.operation is not SpatialOperationKind.NONE
        if needs_size and self.size is None:
            raise ValueError(f"{self.operation.value} requires a target size")
        if not needs_size and self.size is not None:
            raise ValueError("the none operation cannot declare a target size")
        resampling = {
            SpatialOperationKind.RESIZE,
            SpatialOperationKind.LETTERBOX,
            SpatialOperationKind.FIT_SHORTER_SIDE,
            SpatialOperationKind.FIT_LONGER_SIDE,
        }
        if self.operation in resampling and self.interpolation is None:
            raise ValueError(f"{self.operation.value} requires an interpolation mode")
        if self.operation not in resampling and self.interpolation is not None:
            raise ValueError(f"{self.operation.value} does not use interpolation")
        if self.operation is SpatialOperationKind.LETTERBOX and not self.pad_value:
            raise ValueError("letterbox requires at least one pad value")

    @classmethod
    def from_mapping(cls, value: Any, path: str = "spatial[]") -> SpatialOperation2D:
        data = expect_mapping(value, path)
        check_keys(
            data,
            path=path,
            allowed={
                "operation",
                "size",
                "interpolation",
                "anchor",
                "pad_value",
                "allow_upscale",
            },
            required={"operation"},
        )
        operation = parse_enum(
            SpatialOperationKind,
            get_required(data, "operation", path),
            f"{path}.operation",
        )
        size_value = data.get("size")
        size = None if size_value is None else parse_size_2d(size_value, f"{path}.size")
        interpolation_value = data.get("interpolation")
        interpolation = (
            None
            if interpolation_value is None
            else parse_enum(InterpolationMode, interpolation_value, f"{path}.interpolation")
        )
        anchor = parse_enum(CropAnchor, data.get("anchor", "center"), f"{path}.anchor")
        pad_value = parse_number_tuple(
            data.get("pad_value", [0.0]),
            f"{path}.pad_value",
            minimum_length=1,
        )
        allow_upscale = parse_bool(data.get("allow_upscale", True), f"{path}.allow_upscale")
        try:
            return cls(
                operation=operation,
                size=size,
                interpolation=interpolation,
                anchor=anchor,
                pad_value=pad_value,
                allow_upscale=allow_upscale,
            )
        except ManifestValidationError:
            raise
        except ValueError as exc:
            fail(path, str(exc))

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {"operation": self.operation.value}
        if self.size is not None:
            result["size"] = list(self.size)
        if self.interpolation is not None:
            result["interpolation"] = self.interpolation.value
        if self.anchor is not CropAnchor.CENTER:
            result["anchor"] = self.anchor.value
        if self.operation is SpatialOperationKind.LETTERBOX or self.pad_value != (0.0,):
            result["pad_value"] = list(self.pad_value)
        if not self.allow_upscale:
            result["allow_upscale"] = False
        return result


@dataclass(frozen=True, slots=True)
class Preprocessing2DSpec:
    """Complete model-facing preprocessing declaration for a 2D image."""

    layout: TensorLayout
    color_space: ColorSpace
    channel_order: tuple[str, ...]
    alpha_handling: AlphaHandling
    dtype: TensorDType
    value_range: NumericRange
    normalization: NormalizationSpec
    spatial: tuple[SpatialOperation2D, ...]
    orientation: OrientationHandling

    def __post_init__(self) -> None:
        if self.layout not in _TWO_D_LAYOUTS:
            raise ValueError(f"layout {self.layout.value!r} is not a 2D tensor layout")
        if not self.channel_order:
            raise ValueError("channel_order must not be empty")
        normalized_order = tuple(item.upper() for item in self.channel_order)
        if len(set(normalized_order)) != len(normalized_order):
            raise ValueError("channel_order entries must be unique")
        object.__setattr__(self, "channel_order", normalized_order)
        if not self.spatial:
            raise ValueError("spatial preprocessing must be explicitly declared")
        contains_none = any(
            operation.operation is SpatialOperationKind.NONE for operation in self.spatial
        )
        if contains_none and len(self.spatial) != 1:
            raise ValueError("the none operation must be the only spatial operation")
        self._validate_channels()

    def _validate_channels(self) -> None:
        channels = set(self.channel_order)
        expected: set[str] | None = None
        if self.color_space is ColorSpace.GRAYSCALE:
            expected = {"Y"}
        elif self.color_space in {ColorSpace.RGB, ColorSpace.BGR}:
            expected = {"R", "G", "B"}
        elif self.color_space in {ColorSpace.RGBA, ColorSpace.BGRA}:
            expected = {"R", "G", "B", "A"}
        if expected is not None and channels != expected:
            raise ValueError(
                f"channel_order {self.channel_order!r} does not match "
                f"{self.color_space.value} channels {tuple(sorted(expected))!r}"
            )
        if self.color_space in {ColorSpace.RGBA, ColorSpace.BGRA}:
            if self.alpha_handling not in {AlphaHandling.PRESERVE, AlphaHandling.PREMULTIPLY}:
                raise ValueError(
                    "rgba/bgra output requires alpha_handling preserve or premultiply; "
                    "use rgb/bgr when alpha is dropped or composited"
                )
        elif self.alpha_handling in {AlphaHandling.PRESERVE, AlphaHandling.PREMULTIPLY}:
            raise ValueError("preserve/premultiply alpha requires rgba or bgra color_space")
        channel_count = len(self.channel_order)
        norm_count = len(self.normalization.mean)
        if norm_count not in {1, channel_count}:
            raise ValueError(
                "normalization mean/std length must be 1 or match channel_order length"
            )

    @classmethod
    def from_mapping(cls, value: Any, path: str = "preprocessing") -> Preprocessing2DSpec:
        data = expect_mapping(value, path)
        required = {
            "layout",
            "color_space",
            "channel_order",
            "alpha_handling",
            "dtype",
            "value_range",
            "normalization",
            "spatial",
            "orientation",
        }
        check_keys(data, path=path, allowed=required, required=required)
        spatial_items = expect_sequence(get_required(data, "spatial", path), f"{path}.spatial")
        spatial = tuple(
            SpatialOperation2D.from_mapping(item, f"{path}.spatial[{index}]")
            for index, item in enumerate(spatial_items)
        )
        try:
            return cls(
                layout=parse_enum(TensorLayout, data["layout"], f"{path}.layout"),
                color_space=parse_enum(ColorSpace, data["color_space"], f"{path}.color_space"),
                channel_order=parse_string_tuple(
                    data["channel_order"], f"{path}.channel_order", minimum_length=1, unique=True
                ),
                alpha_handling=parse_enum(
                    AlphaHandling, data["alpha_handling"], f"{path}.alpha_handling"
                ),
                dtype=parse_enum(TensorDType, data["dtype"], f"{path}.dtype"),
                value_range=NumericRange.from_value(data["value_range"], f"{path}.value_range"),
                normalization=NormalizationSpec.from_mapping(
                    data["normalization"], f"{path}.normalization"
                ),
                spatial=spatial,
                orientation=parse_enum(
                    OrientationHandling, data["orientation"], f"{path}.orientation"
                ),
            )
        except ManifestValidationError:
            raise
        except ValueError as exc:
            fail(path, str(exc))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "layout": self.layout.value,
            "color_space": self.color_space.value,
            "channel_order": list(self.channel_order),
            "alpha_handling": self.alpha_handling.value,
            "dtype": self.dtype.value,
            "value_range": self.value_range.to_list(),
            "normalization": self.normalization.to_mapping(),
            "spatial": [operation.to_mapping() for operation in self.spatial],
            "orientation": self.orientation.value,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedInput2D:
    """Runtime output pairing a model tensor with its reversible transform."""

    tensor: Any
    transform_record: TransformRecord
    source_shape: tuple[int, ...]
    model_shape: tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
