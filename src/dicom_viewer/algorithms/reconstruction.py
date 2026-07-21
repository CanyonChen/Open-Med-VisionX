"""Correct, cancellable Radon/DFR/BP/FBP/SART teaching implementations."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..errors import MissingDependencyError, ValidationError
from .base import (
    CancelCheck,
    ProgressCallback,
    ReconstructionAlgorithm,
    ReconstructionRequest,
    ReconstructionResult,
    SinogramResult,
    check_cancelled,
    report_progress,
)

Interpolation = Literal["nearest", "linear", "cubic"]


def _skimage_transform() -> object:
    try:
        import skimage.transform as transform
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise MissingDependencyError("CT reconstruction requires scikit-image.") from exc
    return transform


def generate_angles(count: int, angle_range: int = 180) -> np.ndarray:
    if int(count) < 2:
        raise ValidationError("At least two projection angles are required.")
    if int(angle_range) not in {180, 360}:
        raise ValidationError("Parallel-beam teaching scans support 180 or 360 degrees.")
    return np.linspace(0.0, float(angle_range), int(count), endpoint=False)


def generate_sinogram(
    image: np.ndarray,
    *,
    theta_degrees: np.ndarray | None = None,
    projection_count: int = 180,
    angle_range: int = 180,
    circle: bool = True,
    cancel: CancelCheck = None,
    progress: ProgressCallback = None,
) -> SinogramResult:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 2 or not np.all(np.isfinite(array)):
        raise ValidationError("Radon input must be a finite 2-D image.")
    theta = (
        generate_angles(projection_count, angle_range)
        if theta_degrees is None
        else np.asarray(theta_degrees, dtype=np.float64)
    )
    check_cancelled(cancel)
    report_progress(progress, 0.05, "Preparing Radon transform")
    transform = _skimage_transform()
    sinogram = transform.radon(array, theta=theta, circle=bool(circle), preserve_range=True)
    check_cancelled(cancel)
    snapshots: dict[str, np.ndarray] = {}
    for fraction in (0.25, 0.5, 1.0):
        count = max(1, int(round(len(theta) * fraction)))
        snapshots[f"projections_{count}"] = sinogram[:, :count].copy()
    report_progress(progress, 1.0, "Radon transform complete")
    return SinogramResult(sinogram, theta, bool(circle), snapshots)


def _canonical_parallel_data(
    sinogram: np.ndarray,
    theta_degrees: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Fold 360-degree parallel-beam data into its non-redundant 180 degrees."""

    groups: dict[float, list[np.ndarray]] = {}
    angles: dict[float, float] = {}
    used_redundancy = False
    for index, raw_angle in enumerate(theta_degrees):
        angle_360 = float(raw_angle % 360.0)
        projection = np.asarray(sinogram[:, index], dtype=np.float64)
        if angle_360 >= 180.0:
            angle_360 -= 180.0
            projection = projection[::-1]
            used_redundancy = True
        key = round(angle_360, 9)
        groups.setdefault(key, []).append(projection)
        angles[key] = angle_360
    ordered_keys = sorted(groups, key=angles.__getitem__)
    folded = np.column_stack(
        [np.mean(np.stack(groups[key], axis=1), axis=1) for key in ordered_keys]
    )
    theta = np.asarray([angles[key] for key in ordered_keys], dtype=np.float64)
    if len(theta) < 2:
        raise ValidationError("Parallel-beam reconstruction requires at least two unique angles.")
    return folded, theta, used_redundancy


def _circle_mask(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    y, x = np.ogrid[:height, :width]
    radius = min(height, width) / 2.0
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
    return np.where(mask, image, 0.0)


def _partial_backprojections(
    transform: object,
    sinogram: np.ndarray,
    theta: np.ndarray,
    request: ReconstructionRequest,
    *,
    filter_name: str | None,
    cancel: CancelCheck,
    progress: ProgressCallback,
) -> tuple[np.ndarray, ...]:
    snapshots: list[np.ndarray] = []
    for step, fraction in enumerate((0.25, 0.5, 1.0), start=1):
        check_cancelled(cancel)
        count = max(2, int(round(len(theta) * fraction)))
        snapshot = transform.iradon(
            sinogram[:, :count],
            theta=theta[:count],
            filter_name=filter_name,
            output_size=request.output_size,
            circle=request.circle,
            preserve_range=True,
        )
        if request.circle:
            snapshot = _circle_mask(snapshot)
        snapshots.append(np.asarray(snapshot, dtype=np.float64))
        report_progress(progress, step / 3.0, f"Back-projected {count}/{len(theta)} angles")
    return tuple(snapshots)


class BackProjection(ReconstructionAlgorithm):
    name = "bp"

    def reconstruct(
        self,
        request: ReconstructionRequest,
        *,
        cancel: CancelCheck = None,
        progress: ProgressCallback = None,
    ) -> ReconstructionResult:
        sinogram, theta, redundant = _canonical_parallel_data(
            request.sinogram, request.theta_degrees
        )
        transform = _skimage_transform()
        snapshots = _partial_backprojections(
            transform,
            sinogram,
            theta,
            request,
            filter_name=None,
            cancel=cancel,
            progress=progress,
        )
        return ReconstructionResult(
            snapshots[-1],
            self.name,
            {"backprojection_steps": snapshots},
            {
                "circle": request.circle,
                "output_size": request.output_size,
                "unique_angles": len(theta),
                "folded_360_redundancy": redundant,
            },
        )


def _filter_response(size: int, filter_name: str) -> np.ndarray:
    frequency = np.abs(np.fft.fftfreq(size))
    response = frequency.copy()
    normalized = np.clip(frequency / 0.5, 0.0, 1.0)
    if filter_name == "shepp-logan":
        response *= np.sinc(normalized / 2.0)
    elif filter_name == "cosine":
        response *= np.cos(np.pi * normalized / 2.0)
    elif filter_name == "hamming":
        response *= 0.54 + 0.46 * np.cos(np.pi * normalized)
    elif filter_name == "hann":
        response *= (1.0 + np.cos(np.pi * normalized)) / 2.0
    return response


class FilteredBackProjection(ReconstructionAlgorithm):
    name = "fbp"
    _VALID_FILTERS = {"ramp", "shepp-logan", "cosine", "hamming", "hann"}

    def __init__(self, filter_name: str = "ramp") -> None:
        normalized = "ramp" if filter_name.lower() == "ram-lak" else filter_name.lower()
        if normalized not in self._VALID_FILTERS:
            raise ValidationError(
                f"Unsupported FBP filter {filter_name!r}; choose {sorted(self._VALID_FILTERS)}."
            )
        self.filter_name = normalized

    def reconstruct(
        self,
        request: ReconstructionRequest,
        *,
        cancel: CancelCheck = None,
        progress: ProgressCallback = None,
    ) -> ReconstructionResult:
        sinogram, theta, redundant = _canonical_parallel_data(
            request.sinogram, request.theta_degrees
        )
        transform = _skimage_transform()
        spectrum = np.fft.fft(sinogram, axis=0)
        response = _filter_response(sinogram.shape[0], self.filter_name)[:, None]
        filtered = np.real(np.fft.ifft(spectrum * response, axis=0))
        snapshots = _partial_backprojections(
            transform,
            sinogram,
            theta,
            request,
            filter_name=self.filter_name,
            cancel=cancel,
            progress=progress,
        )
        return ReconstructionResult(
            snapshots[-1],
            self.name,
            {
                "detector_frequency_magnitude": np.abs(np.fft.fftshift(spectrum, axes=0)),
                "filter_response": np.fft.fftshift(response[:, 0]),
                "teaching_filtered_sinogram": filtered,
                "backprojection_steps": snapshots,
            },
            {
                "filter": self.filter_name,
                "circle": request.circle,
                "output_size": request.output_size,
                "unique_angles": len(theta),
                "folded_360_redundancy": redundant,
            },
        )


class DirectFourierReconstruction(ReconstructionAlgorithm):
    name = "dfr"

    def __init__(self, interpolation: Interpolation = "linear") -> None:
        if interpolation not in {"nearest", "linear", "cubic"}:
            raise ValidationError("DFR interpolation must be nearest, linear, or cubic.")
        self.interpolation: Interpolation = interpolation

    def reconstruct(
        self,
        request: ReconstructionRequest,
        *,
        cancel: CancelCheck = None,
        progress: ProgressCallback = None,
    ) -> ReconstructionResult:
        try:
            from scipy.interpolate import griddata
        except ImportError as exc:  # pragma: no cover - depends on installation
            raise MissingDependencyError("DFR reconstruction requires SciPy.") from exc
        sinogram, theta_degrees, redundant = _canonical_parallel_data(
            request.sinogram, request.theta_degrees
        )
        report_progress(progress, 0.05, "Computing detector-axis Fourier transform")
        check_cancelled(cancel)
        detector_count = sinogram.shape[0]
        projection_spectrum = np.fft.fftshift(
            np.fft.fft(np.fft.ifftshift(sinogram, axes=0), axis=0),
            axes=0,
        )
        radial = np.fft.fftshift(np.fft.fftfreq(detector_count))
        theta = np.deg2rad(theta_degrees)
        kx = radial[:, None] * np.cos(theta)[None, :]
        ky = radial[:, None] * np.sin(theta)[None, :]
        points = np.column_stack((kx.ravel(), ky.ravel()))
        values = projection_spectrum.ravel()
        # Average the repeated origin and any folded 360-degree samples so
        # QHull-based linear/cubic interpolation sees a well-defined grid.
        rounded = np.round(points, decimals=12)
        unique_points, inverse = np.unique(rounded, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        real_values = np.bincount(inverse, weights=values.real) / counts
        imag_values = np.bincount(inverse, weights=values.imag) / counts
        unique_values = real_values + 1j * imag_values

        target_frequency = np.fft.fftshift(np.fft.fftfreq(request.output_size))
        target_x, target_y = np.meshgrid(target_frequency, target_frequency)
        report_progress(progress, 0.2, f"Interpolating Fourier slices with {self.interpolation}")
        check_cancelled(cancel)
        real_grid = griddata(
            unique_points,
            unique_values.real,
            (target_x, target_y),
            method=self.interpolation,
            fill_value=np.nan,
        )
        imag_grid = griddata(
            unique_points,
            unique_values.imag,
            (target_x, target_y),
            method=self.interpolation,
            fill_value=np.nan,
        )
        cartesian = real_grid + 1j * imag_grid
        radial_limit = float(np.max(np.abs(radial)))
        inside = np.hypot(target_x, target_y) <= radial_limit
        missing = inside & ~np.isfinite(cartesian)
        if np.any(missing) and self.interpolation != "nearest":
            nearest_real = griddata(
                unique_points,
                unique_values.real,
                (target_x[missing], target_y[missing]),
                method="nearest",
            )
            nearest_imag = griddata(
                unique_points,
                unique_values.imag,
                (target_x[missing], target_y[missing]),
                method="nearest",
            )
            cartesian[missing] = nearest_real + 1j * nearest_imag
        cartesian[~inside | ~np.isfinite(cartesian)] = 0.0
        check_cancelled(cancel)
        report_progress(progress, 0.85, "Applying inverse 2-D Fourier transform")
        reconstruction = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(cartesian))))
        if request.circle:
            reconstruction = _circle_mask(reconstruction)
        report_progress(progress, 1.0, "Direct Fourier reconstruction complete")
        return ReconstructionResult(
            reconstruction,
            self.name,
            {
                "projection_spectrum": projection_spectrum,
                "cartesian_spectrum": cartesian,
            },
            {
                "interpolation": self.interpolation,
                "circle": request.circle,
                "output_size": request.output_size,
                "unique_angles": len(theta_degrees),
                "folded_360_redundancy": redundant,
            },
        )


class SARTReconstruction(ReconstructionAlgorithm):
    name = "sart"

    def __init__(self, iterations: int = 5, relaxation: float = 0.15) -> None:
        if int(iterations) <= 0:
            raise ValidationError("SART iterations must be positive.")
        if not 0.0 < float(relaxation) <= 1.0:
            raise ValidationError("SART relaxation must be in (0, 1].")
        self.iterations = int(iterations)
        self.relaxation = float(relaxation)

    def reconstruct(
        self,
        request: ReconstructionRequest,
        *,
        cancel: CancelCheck = None,
        progress: ProgressCallback = None,
    ) -> ReconstructionResult:
        transform = _skimage_transform()
        sinogram, theta, redundant = _canonical_parallel_data(
            request.sinogram, request.theta_degrees
        )
        reconstruction: np.ndarray | None = None
        snapshots: list[np.ndarray] = []
        for iteration in range(self.iterations):
            check_cancelled(cancel)
            reconstruction = transform.iradon_sart(
                sinogram,
                theta=theta,
                image=reconstruction,
                relaxation=self.relaxation,
            )
            if not np.all(np.isfinite(reconstruction)):
                raise ValidationError(
                    f"SART generated non-finite values at iteration {iteration + 1}."
                )
            snapshots.append(np.asarray(reconstruction, dtype=np.float64).copy())
            report_progress(
                progress,
                (iteration + 1) / self.iterations,
                f"SART iteration {iteration + 1}/{self.iterations}",
            )
        assert reconstruction is not None
        if reconstruction.shape != (request.output_size, request.output_size):
            reconstruction = transform.resize(
                reconstruction,
                (request.output_size, request.output_size),
                order=1,
                mode="reflect",
                anti_aliasing=True,
                preserve_range=True,
            )
            snapshots = [
                transform.resize(
                    item,
                    (request.output_size, request.output_size),
                    order=1,
                    mode="reflect",
                    anti_aliasing=True,
                    preserve_range=True,
                )
                for item in snapshots
            ]
        if request.circle:
            reconstruction = _circle_mask(reconstruction)
            snapshots = [_circle_mask(item) for item in snapshots]
        return ReconstructionResult(
            reconstruction,
            self.name,
            {"iteration_images": tuple(snapshots)},
            {
                "iterations": self.iterations,
                "relaxation": self.relaxation,
                "circle": request.circle,
                "output_size": request.output_size,
                "unique_angles": len(theta),
                "folded_360_redundancy": redundant,
            },
        )
