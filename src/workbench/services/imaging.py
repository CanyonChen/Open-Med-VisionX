"""Atomic, background-friendly image loading orchestration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

import numpy as np

from ..domain.geometry import resample_to_ras_grid
from ..domain.images import ImageData, ImageSequence2D, ImageVolume, SourceType
from ..domain.studies import (
    ImageSeries,
    ImageStudy,
    LayerCreator,
    LayerValidationState,
    SourceFormat,
    SourceReference,
    SpatialGeometry,
    VolumeLayer,
)
from ..errors import DecodeError, OperationCancelled, ValidationError
from ..io import (
    DicomLoader,
    DicomLoadResult,
    DicomSeriesDiscovery,
    ImageLoaderRegistry,
    LoadLimits,
    NiftiLoader,
    RasterImageLoader,
    RasterTileSource,
    default_loader_registry,
)
from ..io.base import CancelCheck
from ..runtime import AtomicSessionState, BackgroundTask, TaskContext, TaskRunner


@dataclass(frozen=True, slots=True)
class LoadedStudy:
    """Complete dataset-derived state installed with one atomic pointer swap."""

    image: ImageData
    display_image: ImageData
    imported_at: datetime
    source_kind: str
    volume_projection: np.ndarray | None = None
    domain_study: ImageStudy | None = field(default=None, repr=False)

    @property
    def reference_series(self) -> ImageSeries | None:
        """Return the sole base series used to match DICOM annotations."""

        if self.domain_study is None or len(self.domain_study.series) != 1:
            return None
        return self.domain_study.series[0]


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
        self._load_lock = RLock()
        self._load_id = 0
        self._active_load_id: int | None = None
        self._committed_load_id: int | None = None
        self._active_task: BackgroundTask[LoadedStudy] | None = None

    @property
    def active_task(self) -> BackgroundTask[LoadedStudy] | None:
        with self._load_lock:
            return self._active_task

    @property
    def committed_study(self) -> LoadedStudy | None:
        """The last fully validated study, unaffected by pending work."""

        return self.state.value

    @property
    def pending_load(self) -> BackgroundTask[LoadedStudy] | None:
        """The uncommitted active load, if one is still in progress."""

        with self._load_lock:
            task = self._active_task
            if task is None or task.done or self._active_load_id == self._committed_load_id:
                return None
            return task

    def discover_dicom_series(
        self,
        source: str | Path,
        *,
        cancel: CancelCheck = None,
    ) -> DicomSeriesDiscovery:
        """Inspect DICOM headers without decoding pixels or changing session state."""

        return DicomLoader().discover_series(source, limits=self.limits, cancel=cancel)

    def begin_load(
        self,
        source: str | Path,
        *,
        prepare_axis_aligned_volume: bool = False,
        dicom_series_selector: str | None = None,
        nifti_volume_index: int | None = None,
    ) -> BackgroundTask[LoadedStudy]:
        """Submit one generation-guarded load without clearing committed state.

        A pending load is built and validated away from the committed session.
        Starting another load invalidates the older request without changing
        ``state.value``.  Only the current request can atomically replace the
        committed study, so failure and cancellation leave the previous study
        intact.
        """

        source_path = Path(source)
        with self._load_lock:
            previous = self._active_task
            previous_id = self._active_load_id
            self._load_id += 1
            load_id = self._load_id
            self._active_load_id = load_id
            committed_generation = self.state.generation
            try:
                task = self.runner.submit(
                    self._load_operation,
                    source_path,
                    load_id,
                    committed_generation,
                    prepare_axis_aligned_volume,
                    dicom_series_selector,
                    nifti_volume_index,
                )
            except BaseException:
                self._active_load_id = previous_id
                raise
            self._active_task = task
            if (
                previous is not None
                and not previous.done
                and previous_id != self._committed_load_id
            ):
                previous.cancel()
            return task

    def _load_operation(
        self,
        context: TaskContext,
        source: Path,
        load_id: int,
        committed_generation: int,
        prepare_axis_aligned_volume: bool,
        dicom_series_selector: str | None,
        nifti_volume_index: int | None,
    ) -> LoadedStudy:
        context.report_progress(0.02, message="Probing image format")
        loader, probe = self.registry.probe(source)
        context.raise_if_cancelled()
        context.report_progress(0.08, message=f"Loading {probe.format_name or loader.name}")
        domain_study: ImageStudy | None = None
        if isinstance(loader, RasterImageLoader):
            image = self._load_raster_for_display(context, source, loader)
        elif isinstance(loader, DicomLoader):
            if nifti_volume_index is not None:
                raise ValidationError("A NIfTI volume index cannot be applied to a DICOM source.")
            dicom_result = loader.load_with_reference(
                source,
                limits=self.limits,
                cancel=lambda: context.cancelled,
                series_selector=dicom_series_selector,
            )
            image = dicom_result.volume
            domain_study = self._dicom_domain_study(dicom_result)
        elif isinstance(loader, NiftiLoader):
            if dicom_series_selector is not None:
                raise ValidationError(
                    "A DICOM series selector cannot be applied to a NIfTI source."
                )
            image = loader.load(
                source,
                limits=self.limits,
                cancel=lambda: context.cancelled,
                volume_index=nifti_volume_index,
            )
        else:
            if dicom_series_selector is not None:
                raise ValidationError(
                    "A DICOM series selector cannot be applied to a non-DICOM source."
                )
            if nifti_volume_index is not None:
                raise ValidationError(
                    "A NIfTI volume index cannot be applied to this image source."
                )
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
            if domain_study is None and display_image.source_type is SourceType.NIFTI:
                domain_study = self._volume_domain_study(
                    display_image,
                    source_format=SourceFormat.NIFTI,
                )
        study = LoadedStudy(
            image=image,
            display_image=display_image,
            imported_at=datetime.now(timezone.utc),
            source_kind=probe.format_name or loader.name,
            volume_projection=volume_projection,
            domain_study=domain_study,
        )
        context.report_progress(0.96, message="Validated image is ready to commit")
        with self._load_lock:
            context.raise_if_cancelled()
            if load_id != self._active_load_id:
                raise OperationCancelled("Image load was superseded by a newer request.")
            context.enter_commit_phase()
            # ``cancel_active`` treats a cancellation request after this point
            # as too late.  That keeps task status and committed state aligned
            # during the short interval before TaskRunner marks success.  Set
            # this before ``replace`` because session listeners run
            # synchronously and may immediately start another load.
            previous_committed_load_id = self._committed_load_id
            self._committed_load_id = load_id
            try:
                self.state.replace(study, expected_generation=committed_generation)
            except Exception:
                if self._committed_load_id == load_id:
                    self._committed_load_id = previous_committed_load_id
                raise
        return study

    @staticmethod
    def _dicom_domain_study(result: DicomLoadResult) -> ImageStudy:
        """Create one immutable base study without serializing exact DICOM UIDs."""

        volume = result.volume
        reference = result.reference
        source = SourceReference(
            source_id=reference.selector,
            source_type=volume.source_type,
            source_format=SourceFormat.DICOM,
            provenance={
                "loader": "dicom",
                "canonical_orientation": "RAS+",
            },
        )
        geometry = SpatialGeometry.from_volume(volume)
        base_layer = VolumeLayer(
            layer_id=f"{reference.selector}:base",
            series_id=reference.selector,
            name=f"{volume.modality} base image",
            source=source,
            created_by=LayerCreator.IMPORT,
            validation_state=LayerValidationState.VALIDATED,
            volume=volume,
            is_base_image=True,
            display_mapping={
                "display_inverted": bool(volume.runtime_metadata.get("display_inverted", False)),
            },
        )
        series = ImageSeries(
            series_id=reference.selector,
            modality=volume.modality,
            source=source,
            geometry=geometry,
            intensity_semantics=volume.intensity_semantics,
            layers=(base_layer,),
            study_instance_uid=reference.study_instance_uid,
            series_instance_uid=reference.series_instance_uid,
            frame_of_reference_uid=reference.frame_of_reference_uid,
        )
        return ImageStudy(
            study_id=reference.study_identifier,
            source_type=volume.source_type,
            series=(series,),
            provenance={
                "loader": "dicom",
                "reference_identity": "retained-in-memory-only",
            },
        )

    @staticmethod
    def _volume_domain_study(
        volume: ImageVolume,
        *,
        source_format: SourceFormat,
    ) -> ImageStudy:
        """Create a path-free base study for a non-DICOM medical volume."""

        if source_format is not SourceFormat.NIFTI or volume.source_type is not SourceType.NIFTI:
            raise ValidationError("Only NIfTI volumes can use the generic clinical study bridge.")
        digest = hashlib.sha256()
        digest.update(volume.array.dtype.str.encode("ascii"))
        digest.update(np.asarray(volume.shape, dtype=np.int64).tobytes())
        digest.update(np.asarray(volume.affine, dtype=np.float64).tobytes())
        # Hash one plane at a time so an oblique/non-contiguous volume does not
        # require a second full-volume allocation merely to obtain an identity.
        for plane in volume.array:
            digest.update(np.ascontiguousarray(plane).view(np.uint8))
        content_sha256 = digest.hexdigest()
        series_id = f"sha256:{content_sha256[:24]}"
        source = SourceReference(
            source_id=series_id,
            source_type=volume.source_type,
            source_format=source_format,
            content_sha256=content_sha256,
            provenance={
                "loader": "nifti",
                "canonical_orientation": "RAS+",
                "content_identity": "decoded-volume-and-affine-sha256",
            },
        )
        geometry = SpatialGeometry.from_volume(volume)
        base_layer = VolumeLayer(
            layer_id=f"{series_id}:base",
            series_id=series_id,
            name=f"{volume.modality} base image",
            source=source,
            created_by=LayerCreator.IMPORT,
            validation_state=LayerValidationState.VALIDATED,
            volume=volume,
            is_base_image=True,
        )
        series = ImageSeries(
            series_id=series_id,
            modality=volume.modality,
            source=source,
            geometry=geometry,
            intensity_semantics=volume.intensity_semantics,
            layers=(base_layer,),
        )
        return ImageStudy(
            study_id=f"volume:{content_sha256[:24]}",
            source_type=volume.source_type,
            series=(series,),
            provenance={
                "loader": "nifti",
                "reference_identity": "decoded-content-only",
            },
        )

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
            if (
                len({array.shape for array in arrays}) != 1
                or len({array.dtype.str for array in arrays}) != 1
            ):
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
        with self._load_lock:
            if self._active_task is None or self._active_load_id == self._committed_load_id:
                return False
            return self._active_task.cancel()

    def close(self) -> None:
        self.runner.shutdown(wait=False, cancel_pending=True)
