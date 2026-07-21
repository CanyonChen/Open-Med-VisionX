"""Atomic, background-friendly image loading orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..domain.geometry import resample_to_ras_grid
from ..domain.images import ImageData, ImageSequence2D, ImageVolume
from ..errors import DecodeError, ValidationError
from ..io import (
    ImageLoaderRegistry,
    LoadLimits,
    RasterImageLoader,
    RasterTileSource,
    default_loader_registry,
)
from ..runtime import AtomicSessionState, BackgroundTask, TaskContext, TaskRunner


@dataclass(frozen=True, slots=True)
class LoadedStudy:
    """Complete dataset-derived state installed with one atomic pointer swap."""

    image: ImageData
    display_image: ImageData
    imported_at: datetime
    source_kind: str
    volume_projection: np.ndarray | None = None


class ImageService:
    """Load and validate a source without exposing format internals to the UI."""

    def __init__(
        self,
        *,
        registry: ImageLoaderRegistry | None = None,
        limits: LoadLimits | None = None,
        runner: TaskRunner | None = None,
        state: AtomicSessionState[LoadedStudy] | None = None,
        raster_preview_threshold_bytes: int = 64 * 1024 * 1024,
        raster_preview_max_size: tuple[int, int] = (2048, 2048),
    ) -> None:
        if raster_preview_threshold_bytes <= 0:
            raise ValidationError("raster_preview_threshold_bytes must be positive.")
        if len(raster_preview_max_size) != 2 or any(
            int(value) <= 0 for value in raster_preview_max_size
        ):
            raise ValidationError("raster_preview_max_size must contain two positive values.")
        self.registry = registry or default_loader_registry()
        self.limits = limits or LoadLimits()
        self.runner = runner or TaskRunner(max_workers=2, thread_name_prefix="openmedvisionx-io")
        self.state = state or AtomicSessionState()
        self.raster_preview_threshold_bytes = int(raster_preview_threshold_bytes)
        self.raster_preview_max_size = tuple(int(value) for value in raster_preview_max_size)
        self._active_task: BackgroundTask[LoadedStudy] | None = None

    @property
    def active_task(self) -> BackgroundTask[LoadedStudy] | None:
        return self._active_task

    def begin_load(
        self,
        source: str | Path,
        *,
        prepare_axis_aligned_volume: bool = False,
    ) -> BackgroundTask[LoadedStudy]:
        """Clear derived state and submit one generation-guarded load.

        Starting another load increments the state generation, so an older
        worker can never overwrite the later user action even if cancellation
        reaches the decoder too late.
        """

        if self._active_task is not None and not self._active_task.done:
            self._active_task.cancel()
        generation = self.state.reset().generation
        task = self.runner.submit(
            self._load_operation,
            Path(source),
            generation,
            prepare_axis_aligned_volume,
        )
        self._active_task = task
        return task

    def _load_operation(
        self,
        context: TaskContext,
        source: Path,
        generation: int,
        prepare_axis_aligned_volume: bool,
    ) -> LoadedStudy:
        context.report_progress(0.02, message="Probing image format")
        loader, probe = self.registry.probe(source)
        context.raise_if_cancelled()
        context.report_progress(0.08, message=f"Loading {probe.format_name or loader.name}")
        if isinstance(loader, RasterImageLoader):
            image = self._load_raster_for_display(context, source, loader)
        else:
            image = loader.load(
                source,
                limits=self.limits,
                cancel=lambda: context.cancelled,
            )
        context.raise_if_cancelled()
        display_image: ImageData = image
        if prepare_axis_aligned_volume and isinstance(image, ImageVolume):
            context.report_progress(0.72, message="Resampling physical volume to RAS+ display grid")
            display_image = resample_to_ras_grid(
                image,
                target_spacing=image.spacing,
                max_output_voxels=max(
                    1,
                    self.limits.max_decoded_bytes // max(1, image.array.dtype.itemsize),
                ),
            )
            context.raise_if_cancelled()
        volume_projection: np.ndarray | None = None
        if isinstance(display_image, ImageVolume):
            context.report_progress(0.88, message="Preparing maximum-intensity projection")
            volume_projection = np.max(display_image.array, axis=0)
            volume_projection.setflags(write=False)
            context.raise_if_cancelled()
        study = LoadedStudy(
            image=image,
            display_image=display_image,
            imported_at=datetime.now(timezone.utc),
            source_kind=probe.format_name or loader.name,
            volume_projection=volume_projection,
        )
        context.report_progress(0.96, message="Committing validated image state")
        self.state.replace(study, expected_generation=generation)
        context.report_progress(1.0, message="Image ready")
        return study

    def _load_raster_for_display(
        self,
        context: TaskContext,
        source: Path,
        loader: RasterImageLoader,
    ) -> ImageData:
        """Use bounded thumbnails when a flat raster would be expensive to materialize."""

        tiled = RasterTileSource(
            source,
            limits=self.limits,
            cancel=lambda: context.cancelled,
        )
        try:
            decoded_bytes = sum(page.estimated_decoded_bytes for page in tiled.info.pages)
            if decoded_bytes <= self.raster_preview_threshold_bytes:
                return loader.load(
                    source,
                    limits=self.limits,
                    cancel=lambda: context.cancelled,
                )
            context.report_progress(
                0.18,
                message="Large flat raster detected; preparing bounded thumbnails",
            )
            if tiled.info.frame_count == 1:
                return tiled.read_thumbnail(
                    max_size=self.raster_preview_max_size,
                    cancel=lambda: context.cancelled,
                )

            source_shapes = {page.canonical_shape for page in tiled.info.pages}
            source_modes = {page.canonical_mode for page in tiled.info.pages}
            if len(source_shapes) != 1 or len(source_modes) != 1:
                raise DecodeError(
                    "Large TIFF pages use incompatible geometry or color modes and cannot form "
                    "one thumbnail sequence."
                )
            preview_budget = min(
                self.raster_preview_threshold_bytes,
                max(1, self.limits.max_decoded_bytes // 4),
            )
            bytes_per_pixel = max(
                1,
                max(
                    page.estimated_decoded_bytes
                    // max(1, page.canonical_shape[0] * page.canonical_shape[1])
                    for page in tiled.info.pages
                ),
            )
            per_frame_pixels = max(
                1,
                preview_budget // max(1, tiled.info.frame_count * bytes_per_pixel),
            )
            side = max(1, min(self.raster_preview_max_size, key=int, default=1))
            side = min(side, max(1, int(np.sqrt(per_frame_pixels))))
            max_size = (side, side)
            frames = [
                tiled.read_thumbnail(
                    page=index,
                    max_size=max_size,
                    cancel=lambda: context.cancelled,
                )
                for index in range(tiled.info.frame_count)
            ]
            arrays = [frame.array for frame in frames]
            if len({array.shape for array in arrays}) != 1 or len(
                {array.dtype.str for array in arrays}
            ) != 1:
                raise DecodeError("Large TIFF thumbnails use incompatible dimensions or dtypes.")
            first = frames[0]
            return ImageSequence2D(
                array=np.stack(arrays, axis=0),
                source_type=first.source_type,
                intensity_semantics=first.intensity_semantics,
                runtime_metadata={
                    "loader": "raster_tile_source",
                    "format": tiled.info.format_name,
                    "access": "thumbnail_sequence",
                    "preview_only": True,
                    "source_shape": tiled.info.pages[0].canonical_shape,
                    "source_decoded_bytes": decoded_bytes,
                    "decoded_bytes": sum(int(array.nbytes) for array in arrays),
                    "frame_count": len(frames),
                    "lossy_compression": tiled.info.lossy_compression,
                    "spatial_semantics": "pixel_coordinates_only",
                },
                bit_depth=max(page.bit_depth for page in tiled.info.pages),
                color_space=first.color_space,
                channel_order=first.channel_order,
                alpha_semantics=first.alpha_semantics,
                frame_transforms=tuple(frame.transform_record for frame in frames),
            )
        finally:
            tiled.close()

    def cancel_active(self) -> bool:
        return bool(self._active_task is not None and self._active_task.cancel())

    def close(self) -> None:
        self.runner.shutdown(wait=False, cancel_pending=True)
