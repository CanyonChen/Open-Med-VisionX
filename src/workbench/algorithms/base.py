"""Stable reconstruction interfaces and validated value objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

import numpy as np

from ..errors import OperationCancelled, ValidationError

CancelCheck = Callable[[], bool] | None
ProgressCallback = Callable[[float, str], None] | None


class ReconstructionSourceKind(str, Enum):
    """Acquisition-domain provenance that must travel with reconstruction data."""

    RAW_KSPACE = "raw-kspace"
    SIMULATED_KSPACE = "simulated-kspace"
    SINOGRAM = "sinogram"
    IMAGE_DERIVED_SIMULATION = "image-derived-simulation"


def _reconstruction_source_kind(value: object) -> ReconstructionSourceKind:
    try:
        return ReconstructionSourceKind(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Unknown reconstruction source_kind {value!r}.") from exc


def _validated_sinogram_geometry(
    sinogram_value: object,
    theta_value: object,
) -> tuple[np.ndarray, np.ndarray]:
    if np.iscomplexobj(np.asanyarray(sinogram_value)) or np.iscomplexobj(
        np.asanyarray(theta_value)
    ):
        raise ValidationError("sinogram/theta must be real-valued arrays.")
    try:
        sinogram = np.asarray(sinogram_value, dtype=np.float64)
        theta = np.asarray(theta_value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValidationError("sinogram/theta must contain numeric values.") from exc
    if sinogram.ndim != 2 or min(sinogram.shape, default=0) < 2:
        raise ValidationError(
            f"sinogram must be a 2-D detector x angle array, got {sinogram.shape}."
        )
    if theta.ndim != 1 or len(theta) != sinogram.shape[1]:
        raise ValidationError("theta_degrees length must match the sinogram projection dimension.")
    if not np.all(np.isfinite(sinogram)) or not np.all(np.isfinite(theta)):
        raise ValidationError("sinogram/theta contain NaN or infinity.")
    immutable_sinogram = np.array(sinogram, copy=True)
    immutable_theta = np.array(theta, copy=True)
    immutable_sinogram.setflags(write=False)
    immutable_theta.setflags(write=False)
    return immutable_sinogram, immutable_theta


def _immutable_array_mapping(values: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    immutable: dict[str, np.ndarray] = {}
    for raw_name, value in values.items():
        name = str(raw_name)
        if name in immutable:
            raise ValidationError(f"Duplicate intermediate name after normalization: {name!r}.")
        array = np.array(np.asanyarray(value), copy=True)
        if array.dtype == object:
            raise ValidationError(f"Intermediate {name!r} cannot use object dtype.")
        array.setflags(write=False)
        immutable[name] = array
    return MappingProxyType(immutable)


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
    source_kind: ReconstructionSourceKind = ReconstructionSourceKind.SINOGRAM

    def __post_init__(self) -> None:
        sinogram, theta = _validated_sinogram_geometry(
            self.sinogram,
            self.theta_degrees,
        )
        if int(self.output_size) < 2:
            raise ValidationError("output_size must be at least 2.")
        source_kind = _reconstruction_source_kind(self.source_kind)
        if source_kind in {
            ReconstructionSourceKind.RAW_KSPACE,
            ReconstructionSourceKind.SIMULATED_KSPACE,
        }:
            raise ValidationError(
                "A parallel-beam sinogram request cannot use a k-space source_kind."
            )
        object.__setattr__(self, "sinogram", sinogram)
        object.__setattr__(self, "theta_degrees", theta)
        object.__setattr__(self, "output_size", int(self.output_size))
        object.__setattr__(self, "source_kind", source_kind)

    @property
    def requires_image_derived_warning(self) -> bool:
        return self.source_kind is ReconstructionSourceKind.IMAGE_DERIVED_SIMULATION

    @classmethod
    def from_sinogram_result(
        cls,
        result: SinogramResult,
        *,
        output_size: int,
    ) -> ReconstructionRequest:
        """Create a request without dropping the sinogram provenance."""

        return cls(
            result.sinogram,
            result.theta_degrees,
            output_size=output_size,
            circle=result.circle,
            source_kind=result.source_kind,
        )


@dataclass(frozen=True, slots=True)
class SinogramResult:
    sinogram: np.ndarray
    theta_degrees: np.ndarray
    circle: bool
    intermediate: Mapping[str, np.ndarray] = field(default_factory=dict)
    source_kind: ReconstructionSourceKind = ReconstructionSourceKind.SINOGRAM

    def __post_init__(self) -> None:
        sinogram, theta = _validated_sinogram_geometry(
            self.sinogram,
            self.theta_degrees,
        )
        source_kind = _reconstruction_source_kind(self.source_kind)
        if source_kind in {
            ReconstructionSourceKind.RAW_KSPACE,
            ReconstructionSourceKind.SIMULATED_KSPACE,
        }:
            raise ValidationError("SinogramResult cannot carry a k-space source_kind.")
        object.__setattr__(self, "sinogram", sinogram)
        object.__setattr__(self, "theta_degrees", theta)
        object.__setattr__(self, "intermediate", _immutable_array_mapping(self.intermediate))
        object.__setattr__(self, "source_kind", source_kind)


@dataclass(frozen=True, slots=True)
class ReconstructionResult:
    image: np.ndarray
    algorithm: str
    intermediate: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if np.iscomplexobj(np.asanyarray(self.image)):
            raise ValidationError("Reconstruction result must be real-valued.")
        image = np.array(self.image, dtype=np.float64, copy=True)
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
