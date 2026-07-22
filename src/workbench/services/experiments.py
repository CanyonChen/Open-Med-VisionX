"""Local-only teaching experiment pairing and safe export helpers.

The functions in this module never discover neighbouring files, contact a
network service, or overwrite an existing path.  Callers must pass every
input and output path selected explicitly by the user.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from ..domain.display import ColorDisplayMapping, GrayscaleDisplayMapping
from ..domain.images import ImageData, ImageSequence2D, ImageVolume, RasterImage2D
from ..errors import DecodeError, OperationCancelled, ResourceLimitError, ValidationError
from ..io import LoadLimits, RasterImageLoader

CancelCheck: TypeAlias = Callable[[], bool] | None
ProgressCallback: TypeAlias = Callable[[float, str], None] | None

_MAX_ANNOTATION_BYTES = 4 * 1024 * 1024
_MAX_EXPERIMENT_BYTES = 1024 * 1024
_BLOCKED_RECORD_KEYS = {
    "array",
    "image",
    "image_data",
    "metadata",
    "patient",
    "patient_id",
    "path",
    "pixel_data",
    "pixels",
    "raw",
    "source_path",
}
_BLOCKED_RECORD_KEY_FRAGMENTS = {
    "patient",
    "sourcepath",
    "filepath",
    "pixeldata",
    "rawimage",
    "metadata",
    "medicalrecord",
    "dicomtag",
    "niftidata",
}
_EXPERIMENT_FIELDS = {
    "schema",
    "platform",
    "created_at",
    "experiment",
    "contains_raw_image_data",
    "image_summary",
    "parameters",
    "metrics",
}
_IMAGE_SUMMARY_FIELDS = {
    "type",
    "source_type",
    "shape",
    "dtype",
    "intensity_semantics",
    "capabilities",
    "bit_depth",
    "color_space",
    "pixel_spacing_mm",
    "spacing_source",
    "frame_count",
    "spatial_volume",
    "modality",
    "spacing_xyz_mm",
    "coordinate_system",
}


def _check_cancelled(cancel: CancelCheck) -> None:
    if cancel is not None and cancel():
        raise OperationCancelled("Local experiment operation was cancelled.")


def _report(progress: ProgressCallback, fraction: float, message: str) -> None:
    if progress is not None:
        progress(float(fraction), message)


@dataclass(frozen=True, slots=True)
class AnnotationOverlay:
    """Validated pixel-coordinate teaching annotations for one image plane."""

    boxes: tuple[tuple[float, float, float, float], ...] = ()
    box_labels: tuple[str, ...] = ()
    points: tuple[tuple[float, float], ...] = ()
    point_labels: tuple[str, ...] = ()


def load_local_mask(
    path: str | Path,
    expected_shape: tuple[int, int],
    *,
    cancel: CancelCheck = None,
    progress: ProgressCallback = None,
) -> np.ndarray:
    """Load one explicitly selected, lossless grayscale mask.

    The mask must match the current displayed plane exactly.  It is not
    resized because an implicit resize would make the teaching comparison
    spatially ambiguous.
    """

    source = Path(path)
    height, width = (int(expected_shape[0]), int(expected_shape[1]))
    if height <= 0 or width <= 0:
        raise ValidationError("Expected mask dimensions must be positive.")
    if source.suffix.lower() not in {".png", ".tif", ".tiff"}:
        raise ValidationError("Masks must be lossless PNG or single-page TIFF files.")
    _check_cancelled(cancel)
    _report(progress, 0.05, "Validating selected mask")
    limits = LoadLimits(
        max_pixels=height * width,
        max_frames=1,
        max_decoded_bytes=max(1024, height * width * 8),
    )
    loaded = RasterImageLoader().load(source, limits=limits, cancel=cancel)
    _check_cancelled(cancel)
    if isinstance(loaded, ImageSequence2D):
        raise ValidationError("A paired mask must contain exactly one page.")
    if loaded.array.ndim != 2:
        raise ValidationError(
            "A paired mask must be a grayscale label image; convert RGB/RGBA masks explicitly."
        )
    if loaded.array.shape != (height, width):
        raise ValidationError(
            f"Mask shape {loaded.array.shape} does not match the current plane {(height, width)}."
        )
    _report(progress, 0.85, "Converting non-zero labels to overlay")
    mask = np.asarray(loaded.array != 0, dtype=np.bool_)
    mask.setflags(write=False)
    _check_cancelled(cancel)
    _report(progress, 1.0, "Mask paired locally")
    return mask


def _read_limited_json(path: Path, *, cancel: CancelCheck) -> Any:
    if path.suffix.lower() != ".json":
        raise ValidationError("Annotations must be a UTF-8 JSON file.")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise DecodeError(f"Cannot inspect selected annotation file: {exc}") from exc
    if size > _MAX_ANNOTATION_BYTES:
        raise ResourceLimitError("Annotation JSON exceeds the 4 MiB safety limit.")
    chunks: list[bytes] = []
    total = 0
    try:
        with path.open("rb") as stream:
            while True:
                _check_cancelled(cancel)
                chunk = stream.read(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_ANNOTATION_BYTES:
                    raise ResourceLimitError("Annotation JSON exceeds the 4 MiB safety limit.")
                chunks.append(chunk)
        return json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DecodeError(f"Annotation file is not valid UTF-8 JSON: {exc}") from exc
    except OSError as exc:
        raise DecodeError(f"Cannot read selected annotation file: {exc}") from exc


def _label(value: Any, *, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str) or len(value) > 128:
        raise ValidationError(f"{field} must be a string of at most 128 characters.")
    return value


def _coordinate(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field} must be numeric.")
    number = float(value)
    if not np.isfinite(number):
        raise ValidationError(f"{field} must be finite.")
    return number


def load_local_annotations(
    path: str | Path,
    expected_shape: tuple[int, int],
    *,
    cancel: CancelCheck = None,
    progress: ProgressCallback = None,
) -> AnnotationOverlay:
    """Load a small, self-contained JSON annotation selected by the user.

    External references and unknown top-level keys are rejected, so loading an
    annotation can never trigger directory scans or follow another file path.
    Coordinates use the current canonical image's top-left pixel system.
    """

    height, width = (int(expected_shape[0]), int(expected_shape[1]))
    if height <= 0 or width <= 0:
        raise ValidationError("Expected annotation dimensions must be positive.")
    _report(progress, 0.05, "Reading selected annotation")
    payload = _read_limited_json(Path(path), cancel=cancel)
    _check_cancelled(cancel)
    if not isinstance(payload, dict):
        raise ValidationError("Annotation JSON root must be an object.")
    allowed = {"schema_version", "coordinate_system", "image_size", "boxes", "points"}
    unknown = set(payload) - allowed
    if unknown:
        raise ValidationError(
            "Unknown annotation fields are not followed: " + ", ".join(sorted(unknown))
        )
    if payload.get("schema_version", 1) != 1:
        raise ValidationError("Only annotation schema_version 1 is supported.")
    if payload.get("coordinate_system") != "pixel_xy_top_left":
        raise ValidationError("coordinate_system must be 'pixel_xy_top_left'.")
    image_size = payload.get("image_size")
    if image_size != [width, height]:
        raise ValidationError(
            f"Annotation image_size must be [{width}, {height}] for the current plane."
        )

    boxes: list[tuple[float, float, float, float]] = []
    box_labels: list[str] = []
    raw_boxes = payload.get("boxes", [])
    if not isinstance(raw_boxes, list) or len(raw_boxes) > 100_000:
        raise ValidationError("boxes must be a list with at most 100,000 entries.")
    for index, item in enumerate(raw_boxes):
        if not isinstance(item, dict) or set(item) - {"x1", "y1", "x2", "y2", "label"}:
            raise ValidationError(f"boxes[{index}] contains unsupported fields.")
        try:
            x1 = _coordinate(item["x1"], field=f"boxes[{index}].x1")
            y1 = _coordinate(item["y1"], field=f"boxes[{index}].y1")
            x2 = _coordinate(item["x2"], field=f"boxes[{index}].x2")
            y2 = _coordinate(item["y2"], field=f"boxes[{index}].y2")
        except KeyError as exc:
            raise ValidationError(f"boxes[{index}] is missing {exc.args[0]}.") from exc
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValidationError(f"boxes[{index}] lies outside the current image.")
        boxes.append((x1, y1, x2, y2))
        box_labels.append(_label(item.get("label"), field=f"boxes[{index}].label"))

    points: list[tuple[float, float]] = []
    point_labels: list[str] = []
    raw_points = payload.get("points", [])
    if not isinstance(raw_points, list) or len(raw_points) > 100_000:
        raise ValidationError("points must be a list with at most 100,000 entries.")
    for index, item in enumerate(raw_points):
        if not isinstance(item, dict) or set(item) - {"x", "y", "label"}:
            raise ValidationError(f"points[{index}] contains unsupported fields.")
        try:
            x = _coordinate(item["x"], field=f"points[{index}].x")
            y = _coordinate(item["y"], field=f"points[{index}].y")
        except KeyError as exc:
            raise ValidationError(f"points[{index}] is missing {exc.args[0]}.") from exc
        if not (0 <= x < width and 0 <= y < height):
            raise ValidationError(f"points[{index}] lies outside the current image.")
        points.append((x, y))
        point_labels.append(_label(item.get("label"), field=f"points[{index}].label"))

    if not boxes and not points:
        raise ValidationError("Annotation JSON contains no boxes or points.")
    _check_cancelled(cancel)
    _report(progress, 1.0, "Annotations paired locally")
    return AnnotationOverlay(
        boxes=tuple(boxes),
        box_labels=tuple(box_labels),
        points=tuple(points),
        point_labels=tuple(point_labels),
    )


def _json_parameter(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        if isinstance(value, str) and len(value) > 1_000:
            raise ValidationError(f"{field} is too long for an experiment record.")
        return value
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValidationError(f"{field} must be finite.")
        return value
    if isinstance(value, np.ndarray):
        raise ValidationError(f"{field} cannot contain image or tensor arrays.")
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        if len(value) > 16:
            raise ValidationError(f"{field} contains too many values for a parameter.")
        if any(isinstance(item, Sequence) and not isinstance(item, str) for item in value):
            raise ValidationError(f"{field} cannot contain nested arrays.")
        return [_json_parameter(item, field=field) for item in value]
    raise ValidationError(f"{field} uses unsupported type {type(value).__name__}.")


def _clean_mapping(values: Mapping[str, Any], *, numeric_only: bool) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    if len(values) > 128:
        raise ValidationError("Experiment mappings may contain at most 128 fields.")
    for raw_key, value in values.items():
        key = str(raw_key).strip()
        if not key or len(key) > 128:
            raise ValidationError("Experiment field names must contain 1–128 characters.")
        compact_key = "".join(character for character in key.lower() if character.isalnum())
        if key.lower() in _BLOCKED_RECORD_KEYS or any(
            token in compact_key for token in _BLOCKED_RECORD_KEY_FRAGMENTS
        ):
            raise ValidationError(f"Experiment field {key!r} may expose image data or identity.")
        converted = _json_parameter(value, field=key)
        if (
            numeric_only
            and converted is not None
            and (isinstance(converted, bool) or not isinstance(converted, (int, float)))
        ):
            raise ValidationError(f"Metric {key!r} must be a numeric scalar or null.")
        cleaned[key] = converted
    return cleaned


def create_experiment_record(
    experiment: str,
    *,
    image: ImageData | None,
    parameters: Mapping[str, Any],
    metrics: Mapping[str, Any] | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Create a deliberately pixel-free, path-free experiment record."""

    title = str(experiment).strip()
    if not title or len(title) > 200:
        raise ValidationError("Experiment name must contain 1–200 characters.")
    summary: dict[str, Any] | None = None
    if image is not None:
        summary = {
            "type": type(image).__name__,
            "source_type": image.source_type.value,
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "intensity_semantics": image.intensity_semantics.value,
            "capabilities": sorted(item.value for item in image.capabilities),
        }
        if isinstance(image, RasterImage2D):
            summary.update(
                {
                    "bit_depth": image.bit_depth,
                    "color_space": image.color_space.value,
                    "pixel_spacing_mm": (
                        list(image.pixel_spacing) if image.pixel_spacing is not None else None
                    ),
                    "spacing_source": (
                        image.spacing_source.value if image.spacing_source is not None else None
                    ),
                }
            )
        elif isinstance(image, ImageSequence2D):
            summary.update({"frame_count": image.frame_count, "spatial_volume": False})
        elif isinstance(image, ImageVolume):
            summary.update(
                {
                    "modality": image.modality,
                    "spacing_xyz_mm": list(image.spacing),
                    "coordinate_system": "RAS+",
                }
            )
    timestamp = created_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return {
        "schema": "openmedvisionx-experiment/v1",
        "platform": "OpenMedVisionX",
        "created_at": timestamp.astimezone(timezone.utc).isoformat(),
        "experiment": title,
        "contains_raw_image_data": False,
        "image_summary": summary,
        "parameters": _clean_mapping(parameters, numeric_only=False),
        "metrics": _clean_mapping(metrics or {}, numeric_only=True),
    }


def _validate_output_path(path: str | Path, *, suffix: str, source_path: str | Path | None) -> Path:
    target = Path(path)
    if target.suffix.lower() != suffix:
        raise ValidationError(f"Output filename must end in {suffix}.")
    if not target.parent.is_dir():
        raise ValidationError("Select an existing local output directory.")
    resolved = target.resolve(strict=False)
    if source_path is not None and resolved == Path(source_path).resolve(strict=False):
        raise ValidationError("The selected output is the original image; choose a new filename.")
    return target


def _write_new_file(
    path: Path,
    data: bytes,
    *,
    cancel: CancelCheck,
    progress: ProgressCallback,
) -> Path:
    _check_cancelled(cancel)
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            total = max(1, len(data))
            for offset in range(0, len(data), 1024 * 1024):
                _check_cancelled(cancel)
                stream.write(data[offset : offset + 1024 * 1024])
                _report(progress, min(0.99, (offset + 1024 * 1024) / total), "Writing new file")
            stream.flush()
        _check_cancelled(cancel)
    except FileExistsError as exc:
        raise ValidationError(
            "Output already exists; OpenMedVisionX does not overwrite it."
        ) from exc
    except BaseException:
        if created:
            with suppress(OSError):
                path.unlink(missing_ok=True)
        raise
    _report(progress, 1.0, "Saved without overwriting")
    return path


def save_experiment_record(
    path: str | Path,
    record: Mapping[str, Any],
    *,
    cancel: CancelCheck = None,
    progress: ProgressCallback = None,
) -> Path:
    """Save a validated experiment record to a newly created JSON file."""

    if set(record) != _EXPERIMENT_FIELDS:
        raise ValidationError("Experiment record contains missing or unsupported fields.")
    if (
        record.get("schema") != "openmedvisionx-experiment/v1"
        or record.get("platform") != "OpenMedVisionX"
        or record.get("contains_raw_image_data") is not False
    ):
        raise ValidationError("Experiment record does not satisfy the pixel-free schema.")
    experiment = record.get("experiment")
    created_at = record.get("created_at")
    if not isinstance(experiment, str) or not 1 <= len(experiment) <= 200:
        raise ValidationError("Experiment record has an invalid experiment name.")
    if not isinstance(created_at, str) or not 1 <= len(created_at) <= 80:
        raise ValidationError("Experiment record has an invalid UTC timestamp.")
    parameters = record.get("parameters")
    metrics = record.get("metrics")
    if not isinstance(parameters, Mapping) or not isinstance(metrics, Mapping):
        raise ValidationError("Experiment parameters and metrics must be objects.")
    summary = record.get("image_summary")
    if summary is not None and not isinstance(summary, Mapping):
        raise ValidationError("Experiment image_summary must be an object or null.")
    if isinstance(summary, Mapping) and not set(summary).issubset(_IMAGE_SUMMARY_FIELDS):
        raise ValidationError("Experiment image_summary contains unsupported fields.")
    safe_record = {
        "schema": "openmedvisionx-experiment/v1",
        "platform": "OpenMedVisionX",
        "created_at": created_at,
        "experiment": experiment,
        "contains_raw_image_data": False,
        "image_summary": None if summary is None else _clean_mapping(summary, numeric_only=False),
        "parameters": _clean_mapping(parameters, numeric_only=False),
        "metrics": _clean_mapping(metrics, numeric_only=True),
    }
    # Re-encode with strict JSON to reject NaN/Infinity and non-JSON payloads.
    try:
        encoded = (
            json.dumps(safe_record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Experiment record is not safe JSON: {exc}") from exc
    if len(encoded) > _MAX_EXPERIMENT_BYTES:
        raise ResourceLimitError("Experiment JSON exceeds the 1 MiB safety limit.")
    target = _validate_output_path(path, suffix=".json", source_path=None)
    return _write_new_file(target, encoded, cancel=cancel, progress=progress)


def export_rendered_png(
    path: str | Path,
    array: np.ndarray,
    *,
    source_path: str | Path | None = None,
    grayscale_mapping: GrayscaleDisplayMapping | None = None,
    color_mapping: ColorDisplayMapping | None = None,
    cancel: CancelCheck = None,
    progress: ProgressCallback = None,
) -> Path:
    """Render decoded or derived values to a user-selected, new PNG file."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - base dependency
        raise ValidationError("PNG export requires Pillow.") from exc
    values = np.asarray(array)
    _check_cancelled(cancel)
    _report(progress, 0.05, "Applying display mapping")
    if values.ndim == 2:
        mapping = grayscale_mapping or GrayscaleDisplayMapping.from_percentiles(values)
        rendered = mapping.map(values)
    elif values.ndim == 3 and values.shape[2] in {3, 4}:
        rendered = (color_mapping or ColorDisplayMapping()).map(values)
    else:
        raise ValidationError(
            f"PNG export requires one HxW, RGB, or RGBA plane; got {values.shape}."
        )
    _check_cancelled(cancel)
    _report(progress, 0.35, "Encoding rendered PNG")
    output = BytesIO()
    Image.fromarray(rendered).save(output, format="PNG")
    _check_cancelled(cancel)
    target = _validate_output_path(path, suffix=".png", source_path=source_path)
    _report(progress, 0.55, "Saving rendered PNG")
    return _write_new_file(target, output.getvalue(), cancel=cancel, progress=progress)
