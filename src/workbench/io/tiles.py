"""Bounded, cancellable page and tile access for ordinary raster images.

The built-in implementation deliberately exposes *flat* image pages.  It is
useful for large PNG/JPEG/TIFF previews, but it does not claim whole-slide
pathology (WSI) pyramid semantics.  A WSI implementation belongs behind the
separate :class:`WsiPyramidTileSource` plugin protocol below.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, BinaryIO, Literal, Protocol, runtime_checkable

import numpy as np

from ..domain.images import (
    AlphaSemantics,
    ColorSpace,
    IntensitySemantics,
    RasterImage2D,
    SourceType,
)
from ..domain.transforms import TransformRecord
from ..errors import (
    DecodeError,
    FormatMismatchError,
    MissingDependencyError,
    ResourceLimitError,
    ValidationError,
)
from .base import CancelCheck, LoadLimits, PathLike, raise_if_cancelled
from .raster import (
    _EXTENSIONS,
    _canonicalization_plan,
    _canonicalize_image,
    _color_semantics,
    _mode_bit_depth,
    _mode_bit_depth_hint,
    _raw_dimensions,
    _signature_format,
)


def _decoded_bytes_per_pixel(mode: str) -> int:
    if mode in {"I", "F"}:
        return 4
    if mode.startswith("I;16"):
        return 2
    if mode == "RGB":
        return 3
    if mode == "RGBA":
        return 4
    return 1


@dataclass(frozen=True, slots=True)
class RasterConversion:
    """One color/alpha conversion known before a page is decoded."""

    operation: str
    source_mode: str | None = None
    target_mode: str | None = None


@dataclass(frozen=True, slots=True)
class RasterPageInfo:
    """Non-pixel metadata for one flat raster page."""

    index: int
    raw_shape: tuple[int, int]
    canonical_shape: tuple[int, int]
    original_mode: str
    canonical_mode: str
    bit_depth: int
    color_space: ColorSpace
    alpha_semantics: AlphaSemantics
    exif_orientation: int
    estimated_decoded_bytes: int
    conversions: tuple[RasterConversion, ...] = ()


@dataclass(frozen=True, slots=True)
class RasterSourceInfo:
    """Safe description of a source; it intentionally contains no path."""

    format_name: str
    frame_count: int
    pages: tuple[RasterPageInfo, ...]
    lossy_compression: bool
    access_model: Literal["flat_pages"] = "flat_pages"
    wsi_pyramid: bool = False


@dataclass(frozen=True, slots=True)
class RasterTileCacheInfo:
    entries: int
    decoded_bytes: int
    max_entries: int
    max_decoded_bytes: int
    hits: int
    misses: int


class RasterTileCache:
    """Thread-safe, entry- and byte-bounded LRU for immutable decoded regions."""

    def __init__(
        self,
        *,
        max_entries: int = 64,
        max_decoded_bytes: int = 134_217_728,
    ) -> None:
        if max_entries <= 0 or max_decoded_bytes <= 0:
            raise ValidationError("Raster tile cache limits must be positive.")
        self._max_entries = int(max_entries)
        self._max_decoded_bytes = int(max_decoded_bytes)
        self._items: OrderedDict[tuple[object, ...], tuple[RasterImage2D, int]] = OrderedDict()
        self._decoded_bytes = 0
        self._hits = 0
        self._misses = 0
        self._lock = RLock()

    def get(self, key: tuple[object, ...]) -> RasterImage2D | None:
        with self._lock:
            cached = self._items.get(key)
            if cached is None:
                self._misses += 1
                return None
            self._items.move_to_end(key)
            self._hits += 1
            return cached[0]

    def put(self, key: tuple[object, ...], image: RasterImage2D) -> None:
        decoded_bytes = int(image.array.nbytes)
        if decoded_bytes > self._max_decoded_bytes:
            return
        with self._lock:
            previous = self._items.pop(key, None)
            if previous is not None:
                self._decoded_bytes -= previous[1]
            self._items[key] = (image, decoded_bytes)
            self._decoded_bytes += decoded_bytes
            while (
                len(self._items) > self._max_entries
                or self._decoded_bytes > self._max_decoded_bytes
            ):
                _, (_, evicted_bytes) = self._items.popitem(last=False)
                self._decoded_bytes -= evicted_bytes

    def discard_source(self, source_token: object) -> None:
        """Remove entries owned by one source without exposing its local path."""

        with self._lock:
            keys = [key for key in self._items if key and key[0] is source_token]
            for key in keys:
                _, decoded_bytes = self._items.pop(key)
                self._decoded_bytes -= decoded_bytes

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._decoded_bytes = 0

    @property
    def info(self) -> RasterTileCacheInfo:
        with self._lock:
            return RasterTileCacheInfo(
                entries=len(self._items),
                decoded_bytes=self._decoded_bytes,
                max_entries=self._max_entries,
                max_decoded_bytes=self._max_decoded_bytes,
                hits=self._hits,
                misses=self._misses,
            )


@runtime_checkable
class WsiPyramidTileSource(Protocol):
    """Plugin boundary for true WSI pyramid/level streaming.

    ``RasterTileSource`` intentionally does not implement this protocol.
    Plugins must validate level geometry and provide their own bounded tile
    scheduler rather than treating an ordinary multipage TIFF as a WSI.
    """

    capability_id: Literal["wsi_pyramid"]

    @property
    def level_shapes(self) -> tuple[tuple[int, int], ...]: ...

    def read_level_tile(
        self,
        level: int,
        box: tuple[int, int, int, int],
        *,
        cancel: CancelCheck = None,
    ) -> RasterImage2D: ...


class RasterTileSource:
    """Cancellable random page/region access suitable for background tasks.

    Every request opens its own Pillow decoder, so concurrent callers never
    share mutable decoder state.  Pillow is a flat-page backend rather than a
    streaming region decoder: tiles and thumbnails therefore require the full
    page's estimated decoded size to fit ``max_decoded_bytes`` before decoding.
    The cache is separately protected by a re-entrant lock.  Cancellation is
    cooperative at decoder boundaries; Pillow itself cannot be interrupted in
    the middle of one codec call.
    """

    def __init__(
        self,
        source: PathLike,
        *,
        limits: LoadLimits | None = None,
        max_dimension: int = 100_000,
        cache: RasterTileCache | None = None,
        cancel: CancelCheck = None,
    ) -> None:
        if max_dimension <= 0:
            raise ValidationError("max_dimension must be positive.")
        self._path = Path(source)
        self._limits = limits or LoadLimits()
        self._max_dimension = int(max_dimension)
        self._cache = cache or RasterTileCache()
        self._source_token = object()
        self._identity: tuple[int, int, int, int] | None = None
        self._info = self._inspect(cancel)

    @property
    def info(self) -> RasterSourceInfo:
        return self._info

    @property
    def cache_info(self) -> RasterTileCacheInfo:
        return self._cache.info

    def close(self) -> None:
        """Release this source's cached pixels; no decoder handle is retained."""

        self._cache.discard_source(self._source_token)

    def read_page(
        self,
        page: int = 0,
        *,
        cancel: CancelCheck = None,
    ) -> RasterImage2D:
        """Decode one full canonical page without decoding other pages."""

        page_info = self._page_info(page)
        self._validate_decoded_budget(page_info.estimated_decoded_bytes)
        key = (self._source_token, "page", page)
        return self._read_cached(key, page_info, "page", None, None, cancel)

    def read_tile(
        self,
        page: int,
        box: tuple[int, int, int, int],
        *,
        cancel: CancelCheck = None,
    ) -> RasterImage2D:
        """Decode a canonical-page region given as ``(left, top, right, bottom)``."""

        page_info = self._page_info(page)
        normalized = self._validate_box(box, page_info.canonical_shape)
        left, top, right, bottom = normalized
        output_bytes = (
            (right - left) * (bottom - top) * _decoded_bytes_per_pixel(page_info.canonical_mode)
        )
        self._validate_decoded_budget(output_bytes)
        key = (self._source_token, "tile", page, normalized)
        return self._read_cached(key, page_info, "tile", normalized, None, cancel)

    def read_thumbnail(
        self,
        page: int = 0,
        *,
        max_size: tuple[int, int] = (1024, 1024),
        cancel: CancelCheck = None,
    ) -> RasterImage2D:
        """Decode a page and return an aspect-preserving canonical thumbnail."""

        page_info = self._page_info(page)
        width, height = self._validate_size(max_size)
        source_h, source_w = page_info.canonical_shape
        scale = min(width / source_w, height / source_h, 1.0)
        output_w = max(1, int(round(source_w * scale)))
        output_h = max(1, int(round(source_h * scale)))
        output_bytes = output_w * output_h * _decoded_bytes_per_pixel(page_info.canonical_mode)
        self._validate_decoded_budget(output_bytes)
        normalized_size = (width, height)
        key = (self._source_token, "thumbnail", page, normalized_size)
        return self._read_cached(
            key,
            page_info,
            "thumbnail",
            None,
            normalized_size,
            cancel,
        )

    def _read_cached(
        self,
        key: tuple[object, ...],
        page_info: RasterPageInfo,
        access: Literal["page", "tile", "thumbnail"],
        box: tuple[int, int, int, int] | None,
        max_size: tuple[int, int] | None,
        cancel: CancelCheck,
    ) -> RasterImage2D:
        raise_if_cancelled(cancel)
        cached = self._cache.get(key)
        if cached is not None:
            raise_if_cancelled(cancel)
            return cached
        image = self._decode(page_info, access, box, max_size, cancel)
        raise_if_cancelled(cancel)
        self._cache.put(key, image)
        return image

    def _decode(
        self,
        page_info: RasterPageInfo,
        access: Literal["page", "tile", "thumbnail"],
        box: tuple[int, int, int, int] | None,
        max_size: tuple[int, int] | None,
        cancel: CancelCheck,
    ) -> RasterImage2D:
        # Pillow's ordinary image codecs materialize a full page before crop or
        # thumbnail operations.  Enforce that real peak allocation, not merely
        # the smaller returned region's byte count.
        self._validate_decoded_budget(page_info.estimated_decoded_bytes)
        Image, _, UnidentifiedImageError = self._pillow()
        try:
            with (
                self._validated_stream() as (stream, detected, _),
                Image.open(stream) as image,
            ):
                if str(image.format or "").upper() != detected:
                    raise DecodeError("Raster decoder format disagrees with the file signature.")
                if page_info.index:
                    image.seek(page_info.index)
                raise_if_cancelled(cancel)
                # Read the tag before decoding: Pillow can remove it from a
                # TIFF frame while leaving raw pixel geometry behind.
                orientation = int(image.getexif().get(274, 1) or 1)
                frame = image.copy()
                if orientation != page_info.exif_orientation:
                    raise DecodeError("Raster page metadata changed while it was being read.")
                frame, conversions = _canonicalize_image(frame)
                decoded = np.asarray(frame).copy()
                canonical = Image.fromarray(decoded)
                # Pillow >=10 applies TIFF Orientation in TiffImagePlugin's
                # load_end and removes the tag. JPEG/PNG retain raw pixels
                # here and need the explicit transform. Format semantics,
                # not a shape comparison, are essential for square pages.
                if orientation != 1 and detected != "TIFF":
                    transpose = {
                        2: Image.Transpose.FLIP_LEFT_RIGHT,
                        3: Image.Transpose.ROTATE_180,
                        4: Image.Transpose.FLIP_TOP_BOTTOM,
                        5: Image.Transpose.TRANSPOSE,
                        6: Image.Transpose.ROTATE_270,
                        7: Image.Transpose.TRANSVERSE,
                        8: Image.Transpose.ROTATE_90,
                    }[orientation]
                    canonical = canonical.transpose(transpose)
                if (canonical.height, canonical.width) != page_info.canonical_shape:
                    raise DecodeError(
                        "Raster decoder produced geometry inconsistent with EXIF orientation."
                    )
                transform = TransformRecord.from_exif_orientation(
                    orientation,
                    page_info.raw_shape,
                )
                operation_steps: list[dict[str, Any]] = []
                if orientation != 1:
                    operation_steps.append({"operation": "exif_orientation", "value": orientation})
                operation_steps.extend(conversions)
                if access == "tile":
                    assert box is not None
                    left, top, right, bottom = box
                    canonical = canonical.crop(box)
                    crop = TransformRecord.crop(
                        page_info.canonical_shape,
                        left=left,
                        top=top,
                        width=right - left,
                        height=bottom - top,
                    )
                    transform = transform.then(crop)
                    operation_steps.append(
                        {
                            "operation": "tile_crop",
                            "box": box,
                        }
                    )
                elif access == "thumbnail":
                    assert max_size is not None
                    resampling = Image.Resampling.LANCZOS
                    scale = min(
                        max_size[0] / canonical.width,
                        max_size[1] / canonical.height,
                        1.0,
                    )
                    output_size = (
                        max(1, int(round(canonical.width * scale))),
                        max(1, int(round(canonical.height * scale))),
                    )
                    # ``thumbnail`` may enter Pillow's unsupported ``reduce``
                    # fast path for I;16/I/F images.  An explicit resize keeps
                    # the high-bit-depth mode and dynamic range intact.
                    canonical = canonical.resize(
                        output_size,
                        resampling,
                        reducing_gap=None,
                    )
                    output_shape = (canonical.height, canonical.width)
                    resize = TransformRecord.resize(
                        page_info.canonical_shape,
                        output_shape,
                    )
                    transform = transform.then(resize)
                    operation_steps.append(
                        {
                            "operation": "thumbnail",
                            "max_size": max_size,
                            "output_shape": output_shape,
                            "interpolation": "lanczos",
                        }
                    )
                raise_if_cancelled(cancel)
                array = np.asarray(canonical).copy()
        except (ResourceLimitError, FormatMismatchError):
            raise
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError, EOFError) as exc:
            raise DecodeError(
                "Could not decode the requested raster page; the file may be corrupt or truncated: "
                f"{exc}"
            ) from exc

        if array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]
        self._validate_decoded_budget(int(array.nbytes))
        color_space, alpha = _color_semantics(canonical.mode)
        semantics = (
            IntensitySemantics.GRAYSCALE
            if color_space is ColorSpace.GRAYSCALE
            else IntensitySemantics.COLOR
        )
        metadata: dict[str, Any] = {
            "loader": "raster_tile_source",
            "format": self._info.format_name,
            "access": access,
            "page_index": page_info.index,
            "frame_count": self._info.frame_count,
            "source_shape": page_info.canonical_shape,
            "original_mode": page_info.original_mode,
            "canonical_mode": canonical.mode,
            "conversion_records": ({"frame": page_info.index, "steps": operation_steps},),
            "lossy_compression": self._info.lossy_compression,
            "decoded_bytes": int(array.nbytes),
            "spatial_semantics": "pixel_coordinates_only",
            "access_model": "flat_pages",
            "wsi_pyramid": False,
        }
        if box is not None:
            metadata["region"] = box
        return RasterImage2D(
            array=array,
            source_type=SourceType.RASTER,
            intensity_semantics=semantics,
            runtime_metadata=metadata,
            bit_depth=_mode_bit_depth(page_info.original_mode, array.dtype),
            color_space=color_space,
            alpha_semantics=alpha,
            transform_record=transform,
        )

    def _inspect(self, cancel: CancelCheck) -> RasterSourceInfo:
        Image, _, UnidentifiedImageError = self._pillow()
        try:
            with (
                self._validated_stream(require_identity=False) as (stream, detected, identity),
                Image.open(stream) as image,
            ):
                actual_format = str(image.format or "").upper()
                if actual_format != detected:
                    raise DecodeError(
                        f"Decoder reported {actual_format or 'unknown'} for a {detected} signature."
                    )
                frame_count = int(getattr(image, "n_frames", 1))
                if frame_count <= 0 or frame_count > self._limits.max_frames:
                    raise ResourceLimitError(
                        f"Raster reports {frame_count:,} frames, above the configured limit."
                    )
                pages: list[RasterPageInfo] = []
                for index in range(frame_count):
                    raise_if_cancelled(cancel)
                    if index:
                        image.seek(index)
                    width, height = _raw_dimensions(image, detected)
                    self._validate_dimensions(width, height)
                    original_mode = image.mode
                    orientation = int(image.getexif().get(274, 1) or 1)
                    if orientation not in range(1, 9):
                        raise DecodeError(
                            f"Raster page {index} has invalid EXIF orientation {orientation}."
                        )
                    canonical_mode, records = _canonicalization_plan(
                        original_mode,
                        has_transparency="transparency" in image.info,
                    )
                    canonical_shape = (
                        (width, height) if orientation in {5, 6, 7, 8} else (height, width)
                    )
                    color_space, alpha = _color_semantics(canonical_mode)
                    conversions = tuple(
                        RasterConversion(
                            operation=str(item["operation"]),
                            source_mode=str(item["from"]) if "from" in item else None,
                            target_mode=str(item["to"]) if "to" in item else None,
                        )
                        for item in records
                    )
                    decoded_bytes = width * height * _decoded_bytes_per_pixel(canonical_mode)
                    pages.append(
                        RasterPageInfo(
                            index=index,
                            raw_shape=(height, width),
                            canonical_shape=canonical_shape,
                            original_mode=original_mode,
                            canonical_mode=canonical_mode,
                            bit_depth=_mode_bit_depth_hint(original_mode),
                            color_space=color_space,
                            alpha_semantics=alpha,
                            exif_orientation=orientation,
                            estimated_decoded_bytes=decoded_bytes,
                            conversions=conversions,
                        )
                    )
            self._identity = identity
        except (ResourceLimitError, FormatMismatchError, DecodeError):
            raise
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError, EOFError) as exc:
            raise DecodeError(
                f"Could not inspect the raster source; it may be corrupt or truncated: {exc}"
            ) from exc
        return RasterSourceInfo(
            format_name=detected,
            frame_count=frame_count,
            pages=tuple(pages),
            lossy_compression=detected == "JPEG",
        )

    @contextmanager
    def _validated_stream(
        self,
        *,
        require_identity: bool = True,
    ) -> Iterator[tuple[BinaryIO, str, tuple[int, int, int, int]]]:
        if not self._path.is_file():
            raise DecodeError("The selected raster source is not a readable file.")
        expected = _EXTENSIONS.get(self._path.suffix.lower())
        try:
            with self._path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                identity = self._fingerprint(stat)
                if require_identity and self._identity is not None and identity != self._identity:
                    raise DecodeError("The raster source changed after it was inspected.")
                detected = _signature_format(stream.read(16))
                if detected is None:
                    raise DecodeError(
                        "The selected file has no complete PNG, JPEG, or TIFF signature."
                    )
                if expected != detected:
                    raise FormatMismatchError(
                        "File extension identifies "
                        f"{expected or 'an unsupported format'}, but the signature identifies "
                        f"{detected}. Rename or replace the file."
                    )
                stream.seek(0)
                yield stream, detected, identity
        except (DecodeError, FormatMismatchError):
            raise
        except OSError as exc:
            raise DecodeError(f"Could not read the selected raster source: {exc}") from exc

    @staticmethod
    def _fingerprint(stat: os.stat_result) -> tuple[int, int, int, int]:
        return (
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
        )

    @staticmethod
    def _pillow() -> tuple[Any, Any, type[Exception]]:
        try:
            from PIL import Image, ImageOps, UnidentifiedImageError
        except ImportError as exc:  # pragma: no cover - base dependency
            raise MissingDependencyError(
                "Raster tile access requires Pillow. Install the base project dependencies."
            ) from exc
        return Image, ImageOps, UnidentifiedImageError

    def _validate_dimensions(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise DecodeError(f"Raster reports invalid dimensions {width}x{height}.")
        if width > self._max_dimension or height > self._max_dimension:
            raise ResourceLimitError(
                f"Raster dimensions {width}x{height} exceed the configured "
                f"{self._max_dimension:,}-pixel axis limit."
            )
        pixels = width * height
        if pixels > self._limits.max_pixels:
            raise ResourceLimitError(
                f"Raster page has {pixels:,} pixels, above the "
                f"{self._limits.max_pixels:,} pixel limit."
            )

    def _validate_decoded_budget(self, decoded_bytes: int) -> None:
        if decoded_bytes > self._limits.max_decoded_bytes:
            raise ResourceLimitError(
                f"Requested decoded raster data needs approximately {decoded_bytes:,} bytes, "
                f"above the {self._limits.max_decoded_bytes:,}-byte limit."
            )

    def _page_info(self, page: int) -> RasterPageInfo:
        if isinstance(page, bool) or not isinstance(page, int):
            raise ValidationError("page must be an integer index.")
        if page < 0 or page >= self._info.frame_count:
            raise ValidationError(f"page index {page} is outside 0..{self._info.frame_count - 1}.")
        return self._info.pages[page]

    @staticmethod
    def _validate_box(
        box: tuple[int, int, int, int],
        shape: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        if len(box) != 4 or any(
            isinstance(item, bool) or not isinstance(item, int) for item in box
        ):
            raise ValidationError("Tile box must contain four integer coordinates.")
        left, top, right, bottom = box
        height, width = shape
        if left < 0 or top < 0 or right <= left or bottom <= top:
            raise ValidationError("Tile box must have positive area within the canonical page.")
        if right > width or bottom > height:
            raise ValidationError(
                f"Tile box {box!r} extends outside the canonical page {width}x{height}."
            )
        return left, top, right, bottom

    @staticmethod
    def _validate_size(size: tuple[int, int]) -> tuple[int, int]:
        if len(size) != 2 or any(
            isinstance(item, bool) or not isinstance(item, int) for item in size
        ):
            raise ValidationError("max_size must contain integer width and height.")
        width, height = size
        if width <= 0 or height <= 0:
            raise ValidationError("max_size dimensions must be positive.")
        return width, height


__all__ = [
    "RasterConversion",
    "RasterPageInfo",
    "RasterSourceInfo",
    "RasterTileCache",
    "RasterTileCacheInfo",
    "RasterTileSource",
    "WsiPyramidTileSource",
]
