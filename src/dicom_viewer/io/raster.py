"""Safe PNG/JPEG/TIFF loader with explicit color and EXIF semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..domain.images import (
    AlphaSemantics,
    ColorSpace,
    ImageSequence2D,
    IntensitySemantics,
    RasterImage2D,
    SourceType,
)
from ..domain.transforms import TransformRecord
from ..errors import DecodeError, FormatMismatchError, MissingDependencyError, ResourceLimitError
from .base import CancelCheck, ImageLoader, LoadLimits, PathLike, ProbeResult, raise_if_cancelled

_EXTENSIONS = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}


def _signature_format(header: bytes) -> str | None:
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if header.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if header.startswith((b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")):
        return "TIFF"
    return None


def _raw_dimensions(image: Any, format_name: str) -> tuple[int, int]:
    """Return storage-order width/height before EXIF orientation is applied.

    Pillow's TIFF plugin reports an orientation-adjusted ``image.size`` even
    before pixels are loaded. TIFF width/length tags remain the authoritative
    raw dimensions needed by :class:`TransformRecord`.
    """

    if format_name == "TIFF":
        tags = getattr(image, "tag_v2", None)
        if tags is not None:
            width = tags.get(256)
            height = tags.get(257)
            if isinstance(width, int) and isinstance(height, int):
                return width, height
    width, height = image.size
    return int(width), int(height)


def _mode_bit_depth(mode: str, dtype: np.dtype[Any]) -> int:
    if mode == "1":
        return 1
    if mode.startswith("I;16"):
        return 16
    if mode == "F":
        return 32
    if mode == "I":
        return int(dtype.itemsize * 8)
    return int(dtype.itemsize * 8)


def _mode_bit_depth_hint(mode: str) -> int:
    """Return a conservative bit-depth hint without decoding pixel data."""

    if mode == "1":
        return 1
    if mode.startswith("I;16"):
        return 16
    if mode in {"I", "F"}:
        return 32
    return 8


def _decoded_bytes_per_pixel(mode: str) -> int:
    """Return the in-memory byte width of one canonical Pillow pixel."""

    if mode in {"I", "F"}:
        return 4
    if mode.startswith("I;16"):
        return 2
    if mode == "RGB":
        return 3
    if mode == "RGBA":
        return 4
    return 1


def _canonicalization_plan(
    mode: str,
    *,
    has_transparency: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """Describe the canonical output mode without decoding the image."""

    conversions: list[dict[str, Any]] = []
    target = mode
    if mode == "P":
        target = "RGBA" if has_transparency else "RGB"
        conversions.append({"operation": "palette_expand", "from": "P", "to": target})
    elif mode == "CMYK":
        target = "RGB"
        conversions.append({"operation": "color_convert", "from": "CMYK", "to": target})
    elif mode in {"YCbCr", "HSV", "LAB"}:
        target = "RGB"
        conversions.append({"operation": "color_convert", "from": mode, "to": target})
    elif mode in {"LA", "La", "PA"}:
        target = "RGBA"
        conversions.append({"operation": "alpha_preserving_convert", "from": mode, "to": target})
    elif mode == "1":
        target = "L"
        conversions.append({"operation": "binary_expand", "from": "1", "to": target})
    elif mode not in {"L", "I", "F", "I;16", "I;16L", "I;16B", "RGB", "RGBA"}:
        raise DecodeError(f"Unsupported raster color mode {mode!r}.")
    return target, conversions


def _canonicalize_image(image: Any) -> tuple[Any, list[dict[str, Any]]]:
    target, conversions = _canonicalization_plan(
        image.mode,
        has_transparency="transparency" in image.info,
    )
    if target != image.mode:
        image = image.convert(target)
    return image, conversions


def _color_semantics(mode: str) -> tuple[ColorSpace, AlphaSemantics]:
    if mode == "RGB":
        return ColorSpace.RGB, AlphaSemantics.NONE
    if mode == "RGBA":
        return ColorSpace.RGBA, AlphaSemantics.STRAIGHT
    return ColorSpace.GRAYSCALE, AlphaSemantics.NONE


class RasterImageLoader(ImageLoader):
    name = "raster"

    def probe(self, source: PathLike) -> ProbeResult:
        path = Path(source)
        if not path.is_file():
            return ProbeResult(False)
        try:
            with path.open("rb") as stream:
                detected = _signature_format(stream.read(16))
        except OSError:
            return ProbeResult(False)
        expected = _EXTENSIONS.get(path.suffix.lower())
        accepted = detected is not None or expected is not None
        confidence = 100 if detected and expected == detected else 85 if detected else 45
        return ProbeResult(
            accepted,
            detected or expected,
            confidence,
            {
                "extension_format": expected,
                "signature_format": detected,
                "extension_matches": bool(detected and expected == detected),
            },
        )

    def load(
        self,
        source: PathLike,
        *,
        limits: LoadLimits | None = None,
        cancel: CancelCheck = None,
    ) -> RasterImage2D | ImageSequence2D:
        try:
            from PIL import Image, ImageOps, UnidentifiedImageError
        except ImportError as exc:  # pragma: no cover - depends on installation
            raise MissingDependencyError(
                "Raster loading requires Pillow. Install the base project dependencies."
            ) from exc

        path = Path(source)
        active_limits = limits or LoadLimits()
        result = self.probe(path)
        if not result.accepted:
            raise DecodeError(f"{path.name!r} is not a supported PNG, JPEG, or TIFF image.")
        expected = result.details.get("extension_format")
        detected = result.details.get("signature_format")
        if detected is None:
            raise DecodeError(
                f"{path.name!r} has a supported extension but its file signature "
                "is missing or truncated."
            )
        if expected != detected:
            raise FormatMismatchError(
                f"File extension identifies {expected or 'an unsupported format'}, "
                f"but the signature identifies {detected}. Rename or replace the file."
            )
        raise_if_cancelled(cancel)

        try:
            # Keep one verified file descriptor from signature check through
            # decoding. Apart from avoiding a replacement race, an externally
            # owned descriptor makes Pillow apply TIFF Orientation exactly
            # once instead of re-applying it during exclusive-file cleanup.
            with path.open("rb") as stream:
                current_detected = _signature_format(stream.read(16))
                if current_detected != detected:
                    raise DecodeError("The raster source changed before it could be decoded.")
                stream.seek(0)
                image = Image.open(stream)
                try:
                    actual_format = str(image.format or "").upper()
                    if actual_format != detected:
                        raise DecodeError(
                            f"Decoder reported {actual_format or 'unknown'} "
                            f"for a {detected} signature."
                        )
                    width, height = _raw_dimensions(image, detected)
                    frame_count = int(getattr(image, "n_frames", 1))
                    self._validate_dimensions(width, height, frame_count, active_limits)

                    # Pillow keeps Image.open() lazy, so reject an oversized
                    # homogeneous raster/sequence before image.copy() or
                    # np.asarray() can materialize its pixels.  Canonical mode
                    # matters here: palette transparency expands to RGBA,
                    # RGB has three channels, and I/F use four bytes per pixel.
                    initial_canonical_mode, _ = _canonicalization_plan(
                        image.mode,
                        has_transparency="transparency" in image.info,
                    )
                    estimated_total_bytes = (
                        width
                        * height
                        * frame_count
                        * _decoded_bytes_per_pixel(initial_canonical_mode)
                    )
                    self._validate_decoded_bytes(
                        estimated_total_bytes,
                        active_limits,
                        estimated=True,
                    )

                    frames: list[np.ndarray] = []
                    transforms: list[TransformRecord] = []
                    conversion_records: list[dict[str, Any]] = []
                    canonical_modes: list[str] = []
                    original_modes: list[str] = []
                    original_bit_depths: list[int] = []
                    total_bytes = 0
                    for frame_index in range(frame_count):
                        raise_if_cancelled(cancel)
                        # Re-seeking TIFF frame zero can change Pillow's lazy
                        # orientation state before the first decode. The decoder
                        # already starts on frame zero, so seek only when needed.
                        if frame_index:
                            image.seek(frame_index)
                        raw_width, raw_height = _raw_dimensions(image, detected)
                        self._validate_dimensions(
                            raw_width,
                            raw_height,
                            frame_count,
                            active_limits,
                        )
                        # Re-check each page's own planned canonical mode before
                        # decoding it.  This covers malformed/heterogeneous TIFF
                        # metadata while the initial check accounts for the
                        # complete frame count before any pixel materialization.
                        planned_mode, _ = _canonicalization_plan(
                            image.mode,
                            has_transparency="transparency" in image.info,
                        )
                        estimated_frame_bytes = (
                            raw_width * raw_height * _decoded_bytes_per_pixel(planned_mode)
                        )
                        self._validate_decoded_bytes(
                            total_bytes + estimated_frame_bytes,
                            active_limits,
                            estimated=True,
                        )
                        orientation = int(image.getexif().get(274, 1) or 1)
                        frame = image.copy()
                        original_mode = frame.mode
                        transform = TransformRecord.from_exif_orientation(
                            orientation,
                            (raw_height, raw_width),
                        )
                        corrected = ImageOps.exif_transpose(frame)
                        canonical, conversions = self._canonicalize(corrected)
                        array = np.asarray(canonical).copy()
                        if array.ndim == 3 and array.shape[2] == 1:
                            array = array[:, :, 0]
                        total_bytes += int(array.nbytes)
                        self._validate_decoded_bytes(total_bytes, active_limits)
                        frames.append(array)
                        transforms.append(transform)
                        canonical_modes.append(canonical.mode)
                        original_modes.append(original_mode)
                        original_bit_depths.append(_mode_bit_depth(original_mode, array.dtype))
                        record: dict[str, Any] = {"frame": frame_index, "steps": conversions}
                        if orientation != 1:
                            record["exif_orientation"] = orientation
                            record["steps"] = [
                                {"operation": "exif_orientation", "value": orientation},
                                *conversions,
                            ]
                        conversion_records.append(record)
                finally:
                    image.close()
        except ResourceLimitError:
            raise
        except FormatMismatchError:
            raise
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
            raise DecodeError(
                f"Could not decode {path.name!r}; the image may be corrupt or truncated: {exc}"
            ) from exc

        if len(set(canonical_modes)) != 1:
            raise DecodeError(
                "TIFF pages use incompatible color modes and cannot form one sequence."
            )
        shapes = {frame.shape for frame in frames}
        dtypes = {frame.dtype.str for frame in frames}
        if len(shapes) != 1 or len(dtypes) != 1:
            raise DecodeError("TIFF pages use incompatible dimensions or dtypes.")

        mode = canonical_modes[0]
        color_space, alpha = self._color_semantics(mode)
        metadata = {
            "loader": self.name,
            "format": detected,
            "original_modes": tuple(original_modes),
            "conversion_records": tuple(conversion_records),
            "lossy_compression": detected == "JPEG",
            "frame_count": frame_count,
            "decoded_bytes": total_bytes,
            "spatial_semantics": "pixel_coordinates_only",
        }
        semantics = (
            IntensitySemantics.GRAYSCALE
            if color_space is ColorSpace.GRAYSCALE
            else IntensitySemantics.COLOR
        )
        bit_depth = max(original_bit_depths)
        if frame_count == 1:
            return RasterImage2D(
                array=frames[0],
                source_type=SourceType.RASTER,
                intensity_semantics=semantics,
                runtime_metadata=metadata,
                bit_depth=bit_depth,
                color_space=color_space,
                alpha_semantics=alpha,
                transform_record=transforms[0],
            )
        return ImageSequence2D(
            array=np.stack(frames, axis=0),
            source_type=SourceType.RASTER,
            intensity_semantics=semantics,
            runtime_metadata=metadata,
            bit_depth=bit_depth,
            color_space=color_space,
            alpha_semantics=alpha,
            frame_transforms=tuple(transforms),
        )

    @staticmethod
    def _validate_dimensions(width: int, height: int, frames: int, limits: LoadLimits) -> None:
        if width <= 0 or height <= 0 or frames <= 0:
            raise DecodeError(f"Image reports invalid dimensions {width}x{height}x{frames}.")
        pixels = width * height
        if pixels > limits.max_pixels:
            raise ResourceLimitError(
                f"Image has {pixels:,} pixels, above the {limits.max_pixels:,} pixel limit."
            )
        if frames > limits.max_frames:
            raise ResourceLimitError(
                f"Image has {frames:,} frames, above the {limits.max_frames:,} frame limit."
            )

    @staticmethod
    def _validate_decoded_bytes(
        decoded_bytes: int,
        limits: LoadLimits,
        *,
        estimated: bool = False,
    ) -> None:
        if decoded_bytes > limits.max_decoded_bytes:
            qualifier = "Estimated decoded" if estimated else "Decoded"
            raise ResourceLimitError(
                f"{qualifier} raster data needs {decoded_bytes:,} bytes, above the "
                f"configured {limits.max_decoded_bytes:,}-byte memory limit."
            )

    @staticmethod
    def _canonicalize(image: Any) -> tuple[Any, list[dict[str, Any]]]:
        return _canonicalize_image(image)

    @staticmethod
    def _color_semantics(mode: str) -> tuple[ColorSpace, AlphaSemantics]:
        return _color_semantics(mode)
