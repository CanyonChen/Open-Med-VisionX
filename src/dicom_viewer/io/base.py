"""Public loader contracts and common resource-limit enforcement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, TypeAlias

from ..domain.images import ImageData
from ..errors import OperationCancelled

PathLike: TypeAlias = str | Path
CancelCheck: TypeAlias = Event | Callable[[], bool] | None


@dataclass(frozen=True, slots=True)
class LoadLimits:
    """Conservative defaults for untrusted local inputs."""

    max_pixels: int = 100_000_000
    max_frames: int = 2_048
    max_decoded_bytes: int = 1_073_741_824
    max_zip_members: int = 10_000
    max_zip_member_bytes: int = 268_435_456
    max_zip_total_bytes: int = 1_073_741_824
    max_zip_compression_ratio: float = 200.0


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Non-destructive format inspection result."""

    accepted: bool
    format_name: str | None = None
    confidence: int = 0
    details: Mapping[str, Any] = field(default_factory=dict)


def is_cancelled(cancel: CancelCheck) -> bool:
    if cancel is None:
        return False
    if isinstance(cancel, Event):
        return cancel.is_set()
    return bool(cancel())


def raise_if_cancelled(cancel: CancelCheck) -> None:
    if is_cancelled(cancel):
        raise OperationCancelled("Image loading was cancelled.")


class ImageLoader(ABC):
    """Stable extension point implemented by built-in and third-party loaders."""

    name: str

    def can_load(self, source: PathLike) -> bool:
        return self.probe(source).accepted

    @abstractmethod
    def probe(self, source: PathLike) -> ProbeResult:
        """Inspect signatures and metadata without decoding the full image."""

    @abstractmethod
    def load(
        self,
        source: PathLike,
        *,
        limits: LoadLimits | None = None,
        cancel: CancelCheck = None,
    ) -> ImageData:
        """Decode and validate an image without mutating application state."""
