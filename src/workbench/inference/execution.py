"""Reference 2-D preprocessing and typed standard-runtime postprocessing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..domain.images import RasterImage2D
from ..domain.transforms import TransformOperation, TransformRecord
from .enums import (
    ActivationKind,
    AlphaHandling,
    BoxFormat,
    ColorSpace,
    CoordinateSystem,
    CropAnchor,
    InterpolationMode,
    OutputSemantic,
    RuntimeKind,
    SpatialOperationKind,
    Task,
    TensorDType,
    TensorLayout,
)
from .errors import PluginContractError
from .manifest import ModelManifest, OutputSpec
from .plugin import InferenceRequest
from .preprocessing import PreparedInput2D, Preprocessing2DSpec
from .results import (
    AnomalyDetectionResult,
    BoundingBox,
    ClassificationResult,
    DetectionResult,
    GenerationResult,
    InferenceProvenance,
    InferenceResult,
    Keypoint,
    MultimodalResult,
    ReconstructionResult,
    RegistrationResult,
    RepresentationResult,
    RestorationResult,
    SamplingState,
    SegmentationResult,
    Track,
    TrackingResult,
    TrackObservation,
    WSIMILResult,
)

_DTYPES: dict[TensorDType, np.dtype[Any]] = {
    TensorDType.UINT8: np.dtype("uint8"),
    TensorDType.UINT16: np.dtype("uint16"),
    TensorDType.INT8: np.dtype("int8"),
    TensorDType.INT16: np.dtype("int16"),
    TensorDType.INT32: np.dtype("int32"),
    TensorDType.INT64: np.dtype("int64"),
    TensorDType.FLOAT16: np.dtype("float16"),
    TensorDType.FLOAT32: np.dtype("float32"),
    TensorDType.FLOAT64: np.dtype("float64"),
    TensorDType.BOOL: np.dtype("bool"),
}


def _resize_transform(
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
) -> TransformRecord:
    """Return the pixel-centre transform used by :func:`_resize`.

    The half-pixel translation matters for reversible overlays.  A plain
    ``out/in`` scale maps pixel edges, not the centres sampled by common image
    resizing implementations.
    """

    in_h, in_w = input_shape
    out_h, out_w = output_shape
    scale_x = out_w / in_w
    scale_y = out_h / in_h
    matrix = np.array(
        [
            [scale_x, 0.0, (scale_x - 1.0) / 2.0],
            [0.0, scale_y, (scale_y - 1.0) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return TransformRecord(
        matrix,
        input_shape,
        output_shape,
        (
            TransformOperation(
                "resize",
                {
                    "input_shape": input_shape,
                    "output_shape": output_shape,
                    "coordinate_convention": "half-pixel",
                },
            ),
        ),
    )


def _resize(array: np.ndarray, target: tuple[int, int], order: int) -> np.ndarray:
    try:
        from scipy.ndimage import affine_transform
    except ImportError as exc:  # pragma: no cover - base dependency
        raise PluginContractError("2-D model preprocessing requires SciPy.") from exc
    height, width = array.shape[:2]
    inverse_y = height / target[0]
    inverse_x = width / target[1]
    matrix = np.diag((inverse_y, inverse_x, 1.0) if array.ndim == 3 else (inverse_y, inverse_x))
    offset = (
        ((inverse_y - 1.0) / 2.0, (inverse_x - 1.0) / 2.0, 0.0)
        if array.ndim == 3
        else ((inverse_y - 1.0) / 2.0, (inverse_x - 1.0) / 2.0)
    )
    output_shape = (target[0], target[1], *array.shape[2:])
    return affine_transform(
        array,
        matrix=matrix,
        offset=offset,
        output_shape=output_shape,
        order=order,
        mode="nearest",
        prefilter=order > 1,
    )


def _crop_origin(
    shape: tuple[int, int], target: tuple[int, int], anchor: CropAnchor
) -> tuple[int, int]:
    height, width = shape
    target_h, target_w = target
    if target_h > height or target_w > width:
        raise PluginContractError(f"Crop target {target} exceeds preprocessed image shape {shape}.")
    vertical = {
        CropAnchor.TOP_LEFT: 0,
        CropAnchor.TOP_RIGHT: 0,
        CropAnchor.CENTER: (height - target_h) // 2,
        CropAnchor.BOTTOM_LEFT: height - target_h,
        CropAnchor.BOTTOM_RIGHT: height - target_h,
    }[anchor]
    horizontal = {
        CropAnchor.TOP_LEFT: 0,
        CropAnchor.BOTTOM_LEFT: 0,
        CropAnchor.CENTER: (width - target_w) // 2,
        CropAnchor.TOP_RIGHT: width - target_w,
        CropAnchor.BOTTOM_RIGHT: width - target_w,
    }[anchor]
    return vertical, horizontal


def _alpha_scale(alpha: np.ndarray) -> np.ndarray:
    values = alpha.astype(np.float64)
    if np.issubdtype(alpha.dtype, np.integer):
        info = np.iinfo(alpha.dtype)
        return (values - info.min) / (info.max - info.min)
    maximum = float(np.max(values)) if values.size else 1.0
    return np.clip(values / (255.0 if maximum > 1.0 else 1.0), 0.0, 1.0)


def _convert_color(array: np.ndarray, spec: Preprocessing2DSpec) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 2:
        channels: dict[str, np.ndarray] = {"Y": values}
        rgb = np.repeat(values[:, :, None], 3, axis=2)
    elif values.ndim == 3 and values.shape[2] in {1, 3, 4}:
        if values.shape[2] == 1:
            channels = {"Y": values[:, :, 0]}
            rgb = np.repeat(values, 3, axis=2)
        else:
            rgb = values[:, :, :3]
            channels = {"R": rgb[:, :, 0], "G": rgb[:, :, 1], "B": rgb[:, :, 2]}
            if values.shape[2] == 4:
                channels["A"] = values[:, :, 3]
    else:
        raise PluginContractError(f"2-D preprocessing expected HxW/HxWxC, got {values.shape}.")

    if "A" in channels:
        handling = spec.alpha_handling
        alpha = _alpha_scale(channels["A"])
        if handling is AlphaHandling.REJECT:
            raise PluginContractError(
                "Model manifest rejects alpha but the input has an alpha channel."
            )
        if handling in {AlphaHandling.COMPOSITE_BLACK, AlphaHandling.COMPOSITE_WHITE}:
            background = 0.0 if handling is AlphaHandling.COMPOSITE_BLACK else 1.0
            working = rgb.astype(np.float64)
            if np.issubdtype(rgb.dtype, np.integer):
                info = np.iinfo(rgb.dtype)
                background *= info.max
            rgb = working * alpha[:, :, None] + background * (1.0 - alpha[:, :, None])
            channels = {"R": rgb[:, :, 0], "G": rgb[:, :, 1], "B": rgb[:, :, 2]}
        elif handling is AlphaHandling.PREMULTIPLY:
            rgb = rgb.astype(np.float64) * alpha[:, :, None]
            channels.update({"R": rgb[:, :, 0], "G": rgb[:, :, 1], "B": rgb[:, :, 2]})
        elif handling is AlphaHandling.DROP:
            channels.pop("A", None)

    if spec.color_space is ColorSpace.GRAYSCALE:
        if "Y" not in channels:
            channels["Y"] = 0.2126 * channels["R"] + 0.7152 * channels["G"] + 0.0722 * channels["B"]
    elif spec.color_space in {ColorSpace.RGB, ColorSpace.RGBA, ColorSpace.BGR, ColorSpace.BGRA}:
        if "R" not in channels:
            channels.update({"R": values.squeeze(), "G": values.squeeze(), "B": values.squeeze()})
        if spec.color_space in {ColorSpace.RGBA, ColorSpace.BGRA} and "A" not in channels:
            raise PluginContractError("Model requires alpha but the input has no alpha channel.")
    elif spec.color_space is not ColorSpace.NATIVE:
        raise PluginContractError(
            f"Reference preprocessing does not silently approximate {spec.color_space.value}; "
            "use a Python adapter with an explicit color conversion."
        )

    try:
        ordered = [channels[name.upper()] for name in spec.channel_order]
    except KeyError as exc:
        raise PluginContractError(f"Input cannot supply declared channel {exc.args[0]!r}.") from exc
    return ordered[0] if len(ordered) == 1 else np.stack(ordered, axis=2)


def _interpolation_order(mode: InterpolationMode | None) -> int:
    if mode is InterpolationMode.NEAREST:
        return 0
    if mode is InterpolationMode.BICUBIC:
        return 3
    if mode is InterpolationMode.LANCZOS:
        raise PluginContractError(
            "The reference SciPy preprocessor cannot implement Lanczos exactly; "
            "use nearest, bilinear, or bicubic interpolation."
        )
    if mode is InterpolationMode.AREA:
        raise PluginContractError(
            "The reference SciPy preprocessor cannot implement area resampling exactly; "
            "use nearest, bilinear, or bicubic interpolation."
        )
    return 1


def prepare_input_2d(
    source: RasterImage2D | np.ndarray,
    spec: Preprocessing2DSpec,
) -> PreparedInput2D:
    """Apply declared preprocessing and return its reversible coordinate map."""

    if isinstance(source, RasterImage2D):
        array = np.asarray(source.array)
        transform = source.transform_record
        source_shape = source.transform_record.original_shape
    else:
        array = np.asarray(source)
        source_shape = array.shape[:2]
        transform = TransformRecord.identity(source_shape)
    array = _convert_color(array, spec)

    for operation in spec.spatial:
        kind = operation.operation
        if kind is SpatialOperationKind.NONE:
            continue
        assert operation.size is not None
        current_shape = array.shape[:2]
        if kind is SpatialOperationKind.RESIZE:
            target = operation.size
            array = _resize(array, target, _interpolation_order(operation.interpolation))
            transform = transform.then(_resize_transform(current_shape, target))
        elif kind in {SpatialOperationKind.CENTER_CROP, SpatialOperationKind.CROP}:
            top, left = _crop_origin(current_shape, operation.size, operation.anchor)
            target_h, target_w = operation.size
            array = array[top : top + target_h, left : left + target_w, ...]
            transform = transform.then(
                TransformRecord.crop(
                    current_shape,
                    left=left,
                    top=top,
                    width=target_w,
                    height=target_h,
                )
            )
        elif kind is SpatialOperationKind.LETTERBOX:
            target_h, target_w = operation.size
            scale = min(target_h / current_shape[0], target_w / current_shape[1])
            if not operation.allow_upscale:
                scale = min(scale, 1.0)
            resized_shape = (
                max(1, int(round(current_shape[0] * scale))),
                max(1, int(round(current_shape[1] * scale))),
            )
            resized = _resize(array, resized_shape, _interpolation_order(operation.interpolation))
            top, left = _crop_origin(operation.size, resized_shape, operation.anchor)
            pad = np.asarray(operation.pad_value, dtype=resized.dtype)
            if resized.ndim == 2:
                fill = float(pad[0])
                output = np.full(operation.size, fill, dtype=resized.dtype)
            else:
                if pad.size not in {1, resized.shape[2]}:
                    raise PluginContractError("Letterbox pad_value must be scalar or per-channel.")
                fill_values = np.repeat(pad, resized.shape[2]) if pad.size == 1 else pad
                output = np.empty((*operation.size, resized.shape[2]), dtype=resized.dtype)
                output[...] = fill_values
            output[top : top + resized_shape[0], left : left + resized_shape[1], ...] = resized
            array = output
            resize_record = _resize_transform(current_shape, resized_shape)
            translate = np.array([[1, 0, left], [0, 1, top], [0, 0, 1]], dtype=float)
            letterbox_record = TransformRecord(
                translate,
                resized_shape,
                operation.size,
                (
                    TransformOperation(
                        "letterbox_pad",
                        {"top": top, "left": left, "target": operation.size},
                    ),
                ),
            )
            transform = transform.then(resize_record).then(letterbox_record)
        elif kind in {SpatialOperationKind.FIT_SHORTER_SIDE, SpatialOperationKind.FIT_LONGER_SIDE}:
            target_h, target_w = operation.size
            ratios = (target_h / current_shape[0], target_w / current_shape[1])
            scale = max(ratios) if kind is SpatialOperationKind.FIT_SHORTER_SIDE else min(ratios)
            if not operation.allow_upscale:
                scale = min(scale, 1.0)
            target = (
                max(1, int(round(current_shape[0] * scale))),
                max(1, int(round(current_shape[1] * scale))),
            )
            array = _resize(array, target, _interpolation_order(operation.interpolation))
            transform = transform.then(_resize_transform(current_shape, target))
        else:  # pragma: no cover - controlled enum exhaustiveness
            raise PluginContractError(f"Unsupported spatial operation {kind.value}.")

    values = array.astype(np.float64, copy=False)
    tolerance = max(1e-6, abs(spec.value_range.maximum - spec.value_range.minimum) * 1e-5)
    if values.size and (
        float(np.min(values)) < spec.value_range.minimum - tolerance
        or float(np.max(values)) > spec.value_range.maximum + tolerance
    ):
        raise PluginContractError(
            f"Input values [{np.min(values):.6g}, {np.max(values):.6g}] violate the manifest "
            f"range [{spec.value_range.minimum}, {spec.value_range.maximum}]."
        )
    values = values * spec.normalization.scale + spec.normalization.offset
    channel_count = 1 if values.ndim == 2 else values.shape[2]
    means = np.asarray(spec.normalization.mean, dtype=float)
    stds = np.asarray(spec.normalization.std, dtype=float)
    if means.size == 1:
        means = np.repeat(means, channel_count)
        stds = np.repeat(stds, channel_count)
    if values.ndim == 2:
        values = (values - means[0]) / stds[0]
    else:
        values = (values - means.reshape(1, 1, -1)) / stds.reshape(1, 1, -1)

    if values.ndim == 2 and spec.layout in {
        TensorLayout.HWC,
        TensorLayout.CHW,
        TensorLayout.NHWC,
        TensorLayout.NCHW,
    }:
        values = values[:, :, None]
    if spec.layout in {TensorLayout.CHW, TensorLayout.NCHW}:
        values = np.transpose(values, (2, 0, 1))
    if spec.layout in {TensorLayout.NHWC, TensorLayout.NCHW}:
        values = values[None, ...]
    tensor = np.ascontiguousarray(values.astype(_DTYPES[spec.dtype], copy=False))
    return PreparedInput2D(
        tensor=tensor,
        transform_record=transform,
        source_shape=tuple(source_shape),
        model_shape=tuple(tensor.shape),
        metadata={
            "layout": spec.layout.value,
            "color_space": spec.color_space.value,
            "channel_order": spec.channel_order,
            "dtype": spec.dtype.value,
        },
    )


def _spatial_axes(array: np.ndarray, layout: str | None) -> tuple[int, int]:
    normalized = None if layout is None else layout.strip().lower().replace("_", "")
    if normalized in {None, "hw", "chw", "nchw"}:
        axes = (array.ndim - 2, array.ndim - 1)
    elif normalized in {"hwc", "nhwc"}:
        axes = (array.ndim - 3, array.ndim - 2)
    else:
        raise PluginContractError("Spatial output layout must be one of hw/hwc/chw/nhwc/nchw.")
    if min(axes) < 0 or axes[0] == axes[1]:
        raise PluginContractError(
            f"Spatial output layout {layout or 'default'} is incompatible with shape {array.shape}."
        )
    return axes


def inverse_map_spatial_array(
    value: Any,
    transform: TransformRecord,
    *,
    interpolation: InterpolationMode | str | None = InterpolationMode.LINEAR,
    discrete: bool = False,
    layout: str | None = None,
) -> np.ndarray:
    """Resample a model-space 2-D field back to original source pixels.

    Non-spatial axes (batch and/or channels) are preserved.  ``discrete=True``
    always uses nearest-neighbour interpolation, even if a different mode is
    supplied, so class IDs can never be invented by interpolation.
    """

    array = np.asarray(value)
    if array.ndim < 2:
        raise PluginContractError(
            f"A spatial output needs at least two dimensions, got {array.shape}."
        )
    if not isinstance(transform, TransformRecord):
        raise PluginContractError("Spatial inverse mapping requires a TransformRecord.")
    mode = (
        interpolation
        if isinstance(interpolation, InterpolationMode) or interpolation is None
        else InterpolationMode.coerce(interpolation)
    )
    order = 0 if discrete else _interpolation_order(mode)
    height_axis, width_axis = _spatial_axes(array, layout)
    moved = np.moveaxis(array, (height_axis, width_axis), (-2, -1))
    grid_shape = (int(moved.shape[-2]), int(moved.shape[-1]))
    grid_record = _resize_transform(transform.output_shape, grid_shape)
    source_to_grid = grid_record.matrix @ transform.matrix
    # scipy.ndimage uses array-axis order (y, x), while TransformRecord uses
    # Cartesian pixel order (x, y).
    matrix_yx = np.array(
        [
            [source_to_grid[1, 1], source_to_grid[1, 0]],
            [source_to_grid[0, 1], source_to_grid[0, 0]],
        ],
        dtype=float,
    )
    offset_yx = np.array([source_to_grid[1, 2], source_to_grid[0, 2]], dtype=float)
    try:
        from scipy.ndimage import affine_transform
    except ImportError as exc:  # pragma: no cover - base dependency
        raise PluginContractError("Spatial inverse mapping requires SciPy.") from exc

    prefix_shape = moved.shape[:-2]
    working = moved.reshape((-1, *grid_shape))
    was_bool = working.dtype == np.dtype("bool")
    if was_bool:
        working = working.astype(np.uint8)
    mapped = np.empty(
        (working.shape[0], *transform.original_shape),
        dtype=working.dtype,
    )
    for index, plane in enumerate(working):
        mapped[index] = affine_transform(
            plane,
            matrix=matrix_yx,
            offset=offset_yx,
            output_shape=transform.original_shape,
            order=order,
            mode="constant",
            cval=0,
            prefilter=order > 1,
        )
    restored = mapped.reshape((*prefix_shape, *transform.original_shape))
    if was_bool:
        restored = restored.astype(bool)
    return np.moveaxis(restored, (-2, -1), (height_axis, width_axis))


def _box_to_xyxy(coordinates: np.ndarray, box_format: BoxFormat) -> np.ndarray:
    if box_format is BoxFormat.XYXY:
        return coordinates.copy()
    if box_format is BoxFormat.XYWH:
        x, y, width, height = coordinates
        return np.asarray([x, y, x + width, y + height], dtype=float)
    if box_format is BoxFormat.CXCYWH:
        center_x, center_y, width, height = coordinates
        return np.asarray(
            [
                center_x - width / 2.0,
                center_y - height / 2.0,
                center_x + width / 2.0,
                center_y + height / 2.0,
            ],
            dtype=float,
        )
    raise PluginContractError("3-D boxes cannot be mapped through a 2-D TransformRecord.")


def _xyxy_to_box(coordinates: np.ndarray, box_format: BoxFormat) -> np.ndarray:
    if box_format is BoxFormat.XYXY:
        return coordinates
    x1, y1, x2, y2 = coordinates
    if box_format is BoxFormat.XYWH:
        return np.asarray([x1, y1, x2 - x1, y2 - y1], dtype=float)
    if box_format is BoxFormat.CXCYWH:
        return np.asarray(
            [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1],
            dtype=float,
        )
    raise PluginContractError("3-D boxes cannot be mapped through a 2-D TransformRecord.")


def inverse_map_boxes(
    boxes: Any,
    transform: TransformRecord,
    *,
    box_format: BoxFormat | str = BoxFormat.XYXY,
    coordinate_system: CoordinateSystem | str = CoordinateSystem.MODEL_INPUT_PIXEL,
) -> np.ndarray:
    """Map model-input or normalized 2-D boxes into source-pixel coordinates."""

    format_value = box_format if isinstance(box_format, BoxFormat) else BoxFormat.coerce(box_format)
    coordinates = (
        coordinate_system
        if isinstance(coordinate_system, CoordinateSystem)
        else CoordinateSystem.coerce(coordinate_system)
    )
    values = np.asarray(boxes, dtype=float)
    squeeze = values.ndim == 1
    if squeeze:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 4:
        raise PluginContractError(f"2-D boxes must have shape (N, 4), got {values.shape}.")
    if coordinates is CoordinateSystem.SOURCE_PIXEL:
        result = values.copy()
        return result[0] if squeeze else result
    if coordinates not in {
        CoordinateSystem.MODEL_INPUT_PIXEL,
        CoordinateSystem.NORMALIZED,
    }:
        raise PluginContractError(
            f"Cannot map {coordinates.value!r} boxes through a 2-D pixel transform."
        )
    if not len(values):
        return values[0] if squeeze else values.copy()
    xyxy_values = np.stack([_box_to_xyxy(row, format_value) for row in values])
    model_h, model_w = transform.output_shape
    if coordinates is CoordinateSystem.NORMALIZED:
        xyxy_values *= np.asarray([model_w, model_h, model_w, model_h], dtype=float)
    source_values = transform.inverse_boxes(xyxy_values)
    converted = [_xyxy_to_box(row, format_value) for row in source_values]
    result = np.stack(converted)
    return result[0] if squeeze else result


def inverse_map_keypoints(
    points: Any,
    transform: TransformRecord,
    *,
    coordinate_system: CoordinateSystem | str = CoordinateSystem.MODEL_INPUT_PIXEL,
) -> np.ndarray:
    """Map model-input or normalized 2-D keypoints to source pixels."""

    coordinates = (
        coordinate_system
        if isinstance(coordinate_system, CoordinateSystem)
        else CoordinateSystem.coerce(coordinate_system)
    )
    values = np.asarray(points, dtype=float)
    if values.shape == (2,):
        squeeze = True
        values = values.reshape(1, 2)
    else:
        squeeze = False
    if values.ndim != 2 or values.shape[1] != 2:
        raise PluginContractError(f"2-D keypoints must have shape (N, 2), got {values.shape}.")
    if coordinates is CoordinateSystem.SOURCE_PIXEL:
        mapped = values.copy()
    elif coordinates in {
        CoordinateSystem.MODEL_INPUT_PIXEL,
        CoordinateSystem.NORMALIZED,
    }:
        if coordinates is CoordinateSystem.NORMALIZED:
            model_h, model_w = transform.output_shape
            values = values * np.asarray([max(model_w - 1, 1), max(model_h - 1, 1)])
        mapped = transform.inverse(values)
    else:
        raise PluginContractError(
            f"Cannot map {coordinates.value!r} keypoints through a 2-D pixel transform."
        )
    return mapped[0] if squeeze else mapped


def _inverse_output_array(
    spec: OutputSpec,
    value: Any,
    transform: TransformRecord | None,
    *,
    discrete: bool | None = None,
) -> Any:
    if transform is None or spec.coordinate_system is CoordinateSystem.SOURCE_PIXEL:
        return value
    supported = {
        CoordinateSystem.MODEL_INPUT_PIXEL,
        CoordinateSystem.NORMALIZED,
        CoordinateSystem.FEATURE_GRID,
    }
    if spec.coordinate_system not in supported:
        return value
    if spec.coordinate_system is CoordinateSystem.FEATURE_GRID and not bool(
        spec.postprocessing.parameters.get("map_to_source", True)
    ):
        return value
    return inverse_map_spatial_array(
        value,
        transform,
        interpolation=spec.postprocessing.interpolation,
        discrete=spec.postprocessing.discrete_labels if discrete is None else discrete,
        layout=spec.postprocessing.parameters.get("layout"),
    )


def _class_axis(array: np.ndarray, spec: OutputSpec) -> int:
    parameters = spec.postprocessing.parameters
    declared = parameters.get("class_axis", parameters.get("axis"))
    if declared is not None:
        if isinstance(declared, bool):
            raise PluginContractError(
                f"Output {spec.name!r} class_axis must be an integer, not bool."
            )
        try:
            axis = int(declared)
        except (TypeError, ValueError) as exc:
            raise PluginContractError(
                f"Output {spec.name!r} class_axis must be an integer."
            ) from exc
    else:
        layout = parameters.get("layout")
        normalized = None if layout is None else str(layout).strip().lower().replace("_", "")
        axes = {"chw": 0, "nchw": 1, "hwc": 2, "nhwc": 3}
        if normalized in axes:
            axis = axes[normalized]
        elif normalized is not None:
            raise PluginContractError(
                f"Output {spec.name!r} layout {layout!r} does not declare a class axis."
            )
        elif spec.semantic is OutputSemantic.MASKS and array.ndim in {3, 4}:
            axis = 0 if array.ndim == 3 else 1
        else:
            axis = array.ndim - 1
    if axis < 0:
        axis += array.ndim
    if not 0 <= axis < array.ndim:
        raise PluginContractError(
            f"Output {spec.name!r} class_axis {axis} is invalid for shape {array.shape}."
        )
    return axis


def apply_activation(array: np.ndarray, spec: OutputSpec) -> np.ndarray:
    values = np.asarray(array)
    activation = spec.postprocessing.activation
    if activation is ActivationKind.SOFTMAX:
        axis = _class_axis(values, spec)
        shifted = values - np.max(values, axis=axis, keepdims=True)
        exponent = np.exp(shifted)
        values = exponent / np.sum(exponent, axis=axis, keepdims=True)
    elif activation is ActivationKind.SIGMOID:
        values = 1.0 / (1.0 + np.exp(-values))
    return values


def _semantic(
    manifest: ModelManifest,
    outputs: Mapping[str, Any],
    semantic: OutputSemantic,
) -> tuple[OutputSpec, Any] | None:
    for spec in manifest.outputs:
        if spec.semantic is semantic and spec.name in outputs:
            return spec, outputs[spec.name]
    return None


def _scores(spec: OutputSpec, value: Any) -> dict[str, float]:
    array = np.asarray(apply_activation(np.asarray(value), spec)).squeeze()
    if array.ndim != 1:
        raise PluginContractError(f"Class-score output {spec.name!r} must reduce to one vector.")
    labels = [spec.labels.get(str(index), str(index)) for index in range(len(array))]
    return {label: float(array[index]) for index, label in enumerate(labels)}


def validate_tensor_shape(
    value: Any,
    declared_shape: tuple[int | str | None, ...],
    *,
    label: str,
) -> tuple[int, ...]:
    """Validate concrete dimensions while honoring symbolic manifest axes."""

    actual = tuple(int(item) for item in np.asarray(value).shape)
    if len(actual) != len(declared_shape):
        raise PluginContractError(
            f"{label} has shape {actual}, but the manifest declares {declared_shape}."
        )
    symbols: dict[str, int] = {}
    for axis, (expected, observed) in enumerate(zip(declared_shape, actual, strict=True)):
        if isinstance(expected, int) and observed != expected:
            raise PluginContractError(
                f"{label} axis {axis} has size {observed}, expected {expected}."
            )
        if isinstance(expected, str):
            previous = symbols.setdefault(expected, observed)
            if previous != observed:
                raise PluginContractError(
                    f"{label} symbolic axis {expected!r} is inconsistent "
                    f"({previous} versus {observed})."
                )
    return actual


def validate_tensor_dtype(
    value: Any,
    declared_dtype: TensorDType | str,
    *,
    label: str,
) -> np.dtype[Any]:
    """Require a runtime tensor to use the exact dtype declared by the manifest."""

    dtype = (
        declared_dtype
        if isinstance(declared_dtype, TensorDType)
        else TensorDType.coerce(declared_dtype)
    )
    expected = _DTYPES[dtype]
    actual = np.asarray(value).dtype
    if actual != expected:
        raise PluginContractError(f"{label} has dtype {actual.name!r}, expected {expected.name!r}.")
    return actual


def _mapped_boxes(
    spec: OutputSpec,
    raw: Any,
    transform: TransformRecord | None,
) -> tuple[BoundingBox, ...]:
    values = np.asarray(raw)
    if values.ndim == 0:
        raise PluginContractError(f"Box output {spec.name!r} must contain a coordinate axis.")
    array = values.reshape(-1, values.shape[-1])
    format_name = spec.postprocessing.parameters.get("box_format", "xyxy")
    box_format = BoxFormat.coerce(format_name)
    coordinate_count = 6 if box_format is BoxFormat.XYZXYZ else 4
    if array.shape[1] < coordinate_count:
        raise PluginContractError(
            f"Box output {spec.name!r} has {array.shape[1]} values per box; "
            f"{box_format.value} requires {coordinate_count}."
        )
    coordinates = array[:, :coordinate_count].astype(float, copy=False)
    result_coordinate_system = spec.coordinate_system
    if (
        transform is not None
        and box_format is not BoxFormat.XYZXYZ
        and spec.coordinate_system
        in {
            CoordinateSystem.MODEL_INPUT_PIXEL,
            CoordinateSystem.NORMALIZED,
            CoordinateSystem.SOURCE_PIXEL,
        }
    ):
        coordinates = inverse_map_boxes(
            coordinates,
            transform,
            box_format=box_format,
            coordinate_system=spec.coordinate_system,
        )
        result_coordinate_system = CoordinateSystem.SOURCE_PIXEL
    boxes: list[BoundingBox] = []
    for index, row in enumerate(array):
        score = float(row[coordinate_count]) if len(row) > coordinate_count else None
        class_id = int(row[coordinate_count + 1]) if len(row) > coordinate_count + 1 else None
        label = spec.labels.get(str(class_id)) if class_id is not None else None
        boxes.append(
            BoundingBox(
                coordinates=tuple(float(item) for item in coordinates[index]),
                format=box_format,
                coordinate_system=result_coordinate_system,
                score=score,
                class_id=class_id,
                label=label,
            )
        )
    return tuple(boxes)


def _mapped_keypoints(
    spec: OutputSpec,
    raw: Any,
    transform: TransformRecord | None,
) -> tuple[Keypoint, ...]:
    values = np.asarray(raw)
    if values.ndim == 0 or values.shape[-1] < 2:
        raise PluginContractError(
            f"Keypoint output {spec.name!r} must end with at least x and y coordinates."
        )
    rows = values.reshape(-1, values.shape[-1])
    coordinates = rows[:, :2].astype(float, copy=False)
    result_coordinate_system = spec.coordinate_system
    if transform is not None and spec.coordinate_system in {
        CoordinateSystem.MODEL_INPUT_PIXEL,
        CoordinateSystem.NORMALIZED,
        CoordinateSystem.SOURCE_PIXEL,
    }:
        coordinates = inverse_map_keypoints(
            coordinates,
            transform,
            coordinate_system=spec.coordinate_system,
        )
        result_coordinate_system = CoordinateSystem.SOURCE_PIXEL
    points: list[Keypoint] = []
    for index, row in enumerate(rows):
        score = float(row[2]) if len(row) > 2 else None
        class_id = int(row[3]) if len(row) > 3 else None
        points.append(
            Keypoint(
                coordinates=tuple(float(item) for item in coordinates[index]),
                coordinate_system=result_coordinate_system,
                score=score,
                label=spec.labels.get(str(class_id)) if class_id is not None else None,
            )
        )
    return tuple(points)


def _sampling_states(value: Any) -> tuple[SamplingState, ...]:
    if isinstance(value, Mapping):
        items: list[Any] = [value]
    elif isinstance(value, (np.ndarray, list, tuple)):
        items = list(value)
    else:
        items = [value]
    states: list[SamplingState] = []
    for index, item in enumerate(items):
        if isinstance(item, Mapping):
            if "sample" not in item:
                raise PluginContractError("Sampling trajectory items require a 'sample' value.")
            states.append(
                SamplingState(
                    step=int(item.get("step", index)),
                    sample=item["sample"],
                    time=None if item.get("time") is None else float(item["time"]),
                    metadata=dict(item.get("metadata", {})),
                )
            )
        else:
            states.append(SamplingState(step=index, sample=item))
    return tuple(states)


def _tracking_result(
    spec: OutputSpec,
    raw: Any,
    transform: TransformRecord | None,
) -> tuple[Track, ...]:
    values = raw.get("tracks", [raw]) if isinstance(raw, Mapping) else raw
    if not isinstance(values, (list, tuple)):
        raise PluginContractError("Track output must be a list of track mappings.")
    tracks: list[Track] = []
    for track_index, item in enumerate(values):
        if not isinstance(item, Mapping):
            raise PluginContractError("Each track output must be a mapping.")
        observation_values = item.get("observations")
        if not isinstance(observation_values, (list, tuple)):
            raise PluginContractError("Each track requires an observations list.")
        observations: list[TrackObservation] = []
        for observation in observation_values:
            if not isinstance(observation, Mapping):
                raise PluginContractError("Track observations must be mappings.")
            box: BoundingBox | None = None
            if observation.get("box") is not None:
                box = _mapped_boxes(spec, np.asarray([observation["box"]]), transform)[0]
            point_values = observation.get("keypoints", ())
            points = (
                ()
                if point_values is None or np.asarray(point_values).size == 0
                else _mapped_keypoints(spec, np.asarray(point_values), transform)
            )
            mask = observation.get("mask")
            if mask is not None:
                mask = _inverse_output_array(spec, mask, transform, discrete=True)
            observations.append(
                TrackObservation(
                    frame_index=int(observation["frame_index"]),
                    box=box,
                    mask=mask,
                    keypoints=points,
                    score=(
                        None if observation.get("score") is None else float(observation["score"])
                    ),
                )
            )
        tracks.append(
            Track(
                track_id=str(item.get("track_id", track_index)),
                observations=tuple(observations),
                label=None if item.get("label") is None else str(item["label"]),
            )
        )
    return tuple(tracks)


def build_typed_result(
    manifest: ModelManifest,
    outputs: Mapping[str, Any],
    request: InferenceRequest,
    *,
    runtime: RuntimeKind,
    duration_ms: float,
    device: str | None,
) -> InferenceResult:
    """Convert named standard-runtime tensors into the manifest's typed task result."""

    requested_task = request.parameters.get("task")
    if requested_task is None:
        if len(manifest.tasks) != 1:
            raise PluginContractError("A multi-task manifest requires request.parameters['task'].")
        task = manifest.tasks[0]
    else:
        task = Task.coerce(requested_task)
        if task not in manifest.tasks:
            raise PluginContractError(
                f"Requested task {task.value!r} is not declared by the manifest."
            )
    provenance = InferenceProvenance(
        model_name=manifest.name,
        model_version=manifest.version,
        runtime=runtime,
        request_id=request.request_id,
        duration_ms=duration_ms,
        device=device,
    )
    transform = next(iter(request.transform_records.values()), None)
    common = {"provenance": provenance, "transform_record": transform}

    scores_item = _semantic(manifest, outputs, OutputSemantic.CLASS_SCORES)
    masks_item = _semantic(manifest, outputs, OutputSemantic.MASKS)
    boxes_item = _semantic(manifest, outputs, OutputSemantic.BOXES)
    keypoints_item = _semantic(manifest, outputs, OutputSemantic.KEYPOINTS)
    embeddings_item = _semantic(manifest, outputs, OutputSemantic.EMBEDDINGS)
    feature_item = _semantic(manifest, outputs, OutputSemantic.FEATURE_MAPS)
    attention_item = _semantic(manifest, outputs, OutputSemantic.ATTENTION_MAPS)
    vector_item = _semantic(manifest, outputs, OutputSemantic.VECTOR_FIELDS)
    affine_item = _semantic(manifest, outputs, OutputSemantic.AFFINE_TRANSFORMS)
    reconstructed_item = _semantic(manifest, outputs, OutputSemantic.RECONSTRUCTED_IMAGES)
    restored_item = _semantic(manifest, outputs, OutputSemantic.RESTORED_IMAGES)
    generated_item = _semantic(manifest, outputs, OutputSemantic.GENERATED_SAMPLES)
    trajectory_item = _semantic(manifest, outputs, OutputSemantic.SAMPLING_TRAJECTORIES)
    tracks_item = _semantic(manifest, outputs, OutputSemantic.TRACKS)
    text_item = _semantic(manifest, outputs, OutputSemantic.TEXT)
    similarity_item = _semantic(manifest, outputs, OutputSemantic.SIMILARITY_SCORES)
    anomaly_score_item = _semantic(manifest, outputs, OutputSemantic.ANOMALY_SCORES)
    anomaly_map_item = _semantic(manifest, outputs, OutputSemantic.ANOMALY_MAPS)

    if task is Task.CLASSIFICATION and scores_item:
        scores = _scores(*scores_item)
        saliency_maps = {}
        if attention_item is not None:
            saliency_maps[attention_item[0].name] = _inverse_output_array(
                attention_item[0], attention_item[1], transform, discrete=False
            )
        return ClassificationResult(
            **common,
            scores=scores,
            predicted_label=max(scores, key=scores.get),
            logits=np.asarray(scores_item[1]),
            saliency_maps=saliency_maps,
        )
    if task is Task.SEGMENTATION and masks_item:
        spec, raw = masks_item
        probabilities = apply_activation(np.asarray(raw), spec)
        squeezed = np.squeeze(probabilities)
        class_axis = (
            None
            if spec.postprocessing.discrete_labels or probabilities.ndim < 3
            else _class_axis(probabilities, spec)
        )
        if class_axis is not None and probabilities.shape[class_axis] > 1:
            masks = np.squeeze(np.argmax(probabilities, axis=class_axis))
        elif spec.postprocessing.threshold is not None:
            masks = squeezed >= spec.postprocessing.threshold
        else:
            masks = squeezed
        labels = {int(key): value for key, value in spec.labels.items()}
        mapped_masks = _inverse_output_array(spec, masks, transform, discrete=True)
        mapped_probabilities = _inverse_output_array(
            spec,
            probabilities,
            transform,
            discrete=False,
        )
        mapped_logits = _inverse_output_array(
            spec,
            np.asarray(raw),
            transform,
            discrete=False,
        )
        return SegmentationResult(
            **common,
            masks=mapped_masks,
            labels=labels,
            probabilities=mapped_probabilities,
            logits=mapped_logits,
        )
    if task is Task.DETECTION and (boxes_item or keypoints_item or masks_item):
        boxes = () if boxes_item is None else _mapped_boxes(boxes_item[0], boxes_item[1], transform)
        keypoints = (
            ()
            if keypoints_item is None
            else _mapped_keypoints(keypoints_item[0], keypoints_item[1], transform)
        )
        detection_masks = (
            None
            if masks_item is None
            else _inverse_output_array(
                masks_item[0],
                masks_item[1],
                transform,
                discrete=masks_item[0].postprocessing.discrete_labels,
            )
        )
        return DetectionResult(
            **common,
            boxes=boxes,
            keypoints=keypoints,
            masks=detection_masks,
        )
    if task is Task.REPRESENTATION and embeddings_item:
        feature_maps = {feature_item[0].name: feature_item[1]} if feature_item else {}
        attention_maps = (
            {
                attention_item[0].name: _inverse_output_array(
                    attention_item[0], attention_item[1], transform, discrete=False
                )
            }
            if attention_item
            else {}
        )
        return RepresentationResult(
            **common,
            embeddings=embeddings_item[1],
            feature_maps=feature_maps,
            attention_maps=attention_maps,
        )
    if task is Task.REGISTRATION and (vector_item or affine_item):
        return RegistrationResult(
            **common,
            vector_field=None if vector_item is None else vector_item[1],
            affine_matrix=None if affine_item is None else affine_item[1],
        )
    if task is Task.RECONSTRUCTION and reconstructed_item:
        return ReconstructionResult(
            **common,
            reconstructed_image=_inverse_output_array(
                reconstructed_item[0], reconstructed_item[1], transform, discrete=False
            ),
            sampling_trajectory=(
                () if trajectory_item is None else _sampling_states(trajectory_item[1])
            ),
        )
    if task is Task.RESTORATION and restored_item:
        return RestorationResult(
            **common,
            restored_image=_inverse_output_array(
                restored_item[0], restored_item[1], transform, discrete=False
            ),
        )
    if task is Task.GENERATION and generated_item:
        value = np.asarray(generated_item[1])
        samples = tuple(value) if value.ndim > 2 else (value,)
        return GenerationResult(
            **common,
            samples=samples,
            sampling_trajectory=(
                () if trajectory_item is None else _sampling_states(trajectory_item[1])
            ),
        )
    if task in {Task.MULTIMODAL, Task.RETRIEVAL, Task.VQA, Task.REPORT_GENERATION}:
        similarity = _scores(*similarity_item) if similarity_item else {}
        text = None if text_item is None else str(np.asarray(text_item[1]).squeeze().item())
        grounding_boxes = (
            () if boxes_item is None else _mapped_boxes(boxes_item[0], boxes_item[1], transform)
        )
        return MultimodalResult(
            **common,
            task=task,
            text_output=text,
            similarity_scores=similarity,
            embeddings=None if embeddings_item is None else embeddings_item[1],
            grounding_boxes=grounding_boxes,
        )
    if task is Task.ANOMALY_DETECTION and anomaly_score_item:
        score = float(np.asarray(anomaly_score_item[1]).squeeze())
        threshold = anomaly_score_item[0].postprocessing.threshold
        return AnomalyDetectionResult(
            **common,
            anomaly_score=score,
            anomaly_map=(
                None
                if anomaly_map_item is None
                else _inverse_output_array(
                    anomaly_map_item[0], anomaly_map_item[1], transform, discrete=False
                )
            ),
            threshold=threshold,
            is_anomaly=None if threshold is None else score >= threshold,
        )
    if task is Task.WSI_MIL and scores_item:
        return WSIMILResult(
            **common,
            slide_scores=_scores(*scores_item),
            attention_map=None if attention_item is None else attention_item[1],
            embeddings=None if embeddings_item is None else embeddings_item[1],
        )
    if task is Task.TRACKING and tracks_item:
        return TrackingResult(
            **common,
            tracks=_tracking_result(tracks_item[0], tracks_item[1], transform),
            vector_fields=None if vector_item is None else vector_item[1],
        )
    raise PluginContractError(
        f"Standard-runtime outputs cannot be mapped to a typed {task.value!r} result. "
        "Correct the output semantics or use an explicitly reviewed Python adapter."
    )
