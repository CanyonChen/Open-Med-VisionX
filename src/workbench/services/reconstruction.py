"""Traditional reconstruction service with a single algorithm registry."""

from __future__ import annotations

import numpy as np

from ..algorithms import (
    BackProjection,
    DirectFourierReconstruction,
    FilteredBackProjection,
    MetricReport,
    ReconstructionAlgorithm,
    ReconstructionRequest,
    ReconstructionResult,
    ReconstructionSourceKind,
    SARTReconstruction,
    SinogramResult,
    compute_metrics,
    generate_sinogram,
)
from ..errors import ValidationError
from ..runtime import BackgroundTask, TaskContext, TaskRunner


class ReconstructionService:
    def __init__(self, *, runner: TaskRunner | None = None) -> None:
        self.runner = runner or TaskRunner(
            max_workers=1,
            thread_name_prefix="openmedvisionx-reconstruction",
        )
        self._active_task: BackgroundTask[object] | None = None

    def begin_sinogram(
        self,
        image: np.ndarray,
        *,
        projection_count: int,
        angle_range: int,
        circle: bool,
        source_kind: ReconstructionSourceKind | str = (
            ReconstructionSourceKind.IMAGE_DERIVED_SIMULATION
        ),
    ) -> BackgroundTask[SinogramResult]:
        self.cancel_active()
        task = self.runner.submit(
            self._sinogram_operation,
            np.asarray(image),
            projection_count,
            angle_range,
            circle,
            source_kind,
        )
        self._active_task = task
        return task

    @staticmethod
    def _sinogram_operation(
        context: TaskContext,
        image: np.ndarray,
        projection_count: int,
        angle_range: int,
        circle: bool,
        source_kind: ReconstructionSourceKind | str,
    ) -> SinogramResult:
        return generate_sinogram(
            image,
            projection_count=projection_count,
            angle_range=angle_range,
            circle=circle,
            source_kind=source_kind,
            cancel=lambda: context.cancelled,
            progress=lambda fraction, message: context.report_progress(
                fraction,
                message=message,
            ),
        )

    def begin_reconstruction(
        self,
        request: ReconstructionRequest,
        *,
        algorithm: str,
        interpolation: str = "linear",
        filter_name: str = "ramp",
        iterations: int = 5,
        relaxation: float = 0.15,
    ) -> BackgroundTask[ReconstructionResult]:
        selected = self._algorithm(
            algorithm,
            interpolation=interpolation,
            filter_name=filter_name,
            iterations=iterations,
            relaxation=relaxation,
        )
        self.cancel_active()
        task = self.runner.submit(self._reconstruct_operation, selected, request)
        self._active_task = task
        return task

    @staticmethod
    def compute_metrics(
        reference: np.ndarray,
        reconstruction: np.ndarray,
        *,
        rois: dict[str, tuple[int, int, int, int]] | None = None,
        intensity_range: tuple[float, float] | None = None,
        unit: str | None = None,
    ) -> MetricReport:
        """Evaluate a result through the algorithm layer's shared metric contract."""

        return compute_metrics(
            reference,
            reconstruction,
            rois=rois,
            intensity_range=intensity_range,
            unit=unit,
        )

    @staticmethod
    def _reconstruct_operation(
        context: TaskContext,
        algorithm: ReconstructionAlgorithm,
        request: ReconstructionRequest,
    ) -> ReconstructionResult:
        return algorithm.reconstruct(
            request,
            cancel=lambda: context.cancelled,
            progress=lambda fraction, message: context.report_progress(
                fraction,
                message=message,
            ),
        )

    @staticmethod
    def _algorithm(
        name: str,
        *,
        interpolation: str,
        filter_name: str,
        iterations: int,
        relaxation: float,
    ) -> ReconstructionAlgorithm:
        normalized = name.strip().lower()
        if normalized == "dfr":
            return DirectFourierReconstruction(interpolation)  # type: ignore[arg-type]
        if normalized == "bp":
            return BackProjection()
        if normalized == "fbp":
            return FilteredBackProjection(filter_name)
        if normalized == "sart":
            return SARTReconstruction(iterations, relaxation)
        raise ValidationError("algorithm must be one of dfr, bp, fbp, or sart.")

    def cancel_active(self) -> bool:
        return bool(self._active_task is not None and self._active_task.cancel())

    def close(self) -> None:
        self.runner.shutdown(wait=False, cancel_pending=True)
