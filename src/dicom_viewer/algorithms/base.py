"""Stable reconstruction interfaces and validated value objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from ..errors import OperationCancelled, ValidationError

CancelCheck = Callable[[], bool] | None
ProgressCallback = Callable[[float, str], None] | None


def check_cancelled(cancel: CancelCheck) -> None:
    if cancel is not None and cancel():
        raise OperationCancelled("Reconstruction was cancelled.")


def report_progress(callback: ProgressCallback, value: float, message: str) -> None:
    if callback is not None:
        callback(float(np.clip(value, 0.0, 1.0)), message)


@dataclass(frozen=True, slots=True)
class ReconstructionRequest:
    """One consistent sinogram geometry shared by every algorithm."""

    sinogram: np.ndarray  # (detectors, projections)
    theta_degrees: np.ndarray
    output_size: int
    circle: bool = True

    def __post_init__(self) -> None:
        sinogram = np.asarray(self.sinogram, dtype=np.float64)
        theta = np.asarray(self.theta_degrees, dtype=np.float64)
        if sinogram.ndim != 2 or min(sinogram.shape) < 2:
            raise ValidationError(
                f"sinogram must be a 2-D detector x angle array, got {sinogram.shape}."
            )
        if theta.ndim != 1 or len(theta) != sinogram.shape[1]:
            raise ValidationError(
                "theta_degrees length must match the sinogram projection dimension."
            )
        if not np.all(np.isfinite(sinogram)) or not np.all(np.isfinite(theta)):
            raise ValidationError("sinogram/theta contain NaN or infinity.")
        if int(self.output_size) < 2:
            raise ValidationError("output_size must be at least 2.")
        sinogram = sinogram.copy()
        theta = theta.copy()
        sinogram.setflags(write=False)
        theta.setflags(write=False)
        object.__setattr__(self, "sinogram", sinogram)
        object.__setattr__(self, "theta_degrees", theta)
        object.__setattr__(self, "output_size", int(self.output_size))


@dataclass(frozen=True, slots=True)
class SinogramResult:
    sinogram: np.ndarray
    theta_degrees: np.ndarray
    circle: bool
    intermediate: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sinogram", np.asarray(self.sinogram, dtype=np.float64))
        object.__setattr__(self, "theta_degrees", np.asarray(self.theta_degrees, dtype=np.float64))
        object.__setattr__(self, "intermediate", MappingProxyType(dict(self.intermediate)))


@dataclass(frozen=True, slots=True)
class ReconstructionResult:
    image: np.ndarray
    algorithm: str
    intermediate: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        image = np.asarray(self.image, dtype=np.float64)
        if image.ndim != 2 or not np.all(np.isfinite(image)):
            raise ValidationError("Reconstruction result must be a finite 2-D array.")
        image.setflags(write=False)
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "intermediate", MappingProxyType(dict(self.intermediate)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class ReconstructionAlgorithm(ABC):
    """Stable algorithm extension point used by services and plugins."""

    name: str

    @abstractmethod
    def reconstruct(
        self,
        request: ReconstructionRequest,
        *,
        cancel: CancelCheck = None,
        progress: ProgressCallback = None,
    ) -> ReconstructionResult:
        """Reconstruct an image using the request's shared geometry."""
