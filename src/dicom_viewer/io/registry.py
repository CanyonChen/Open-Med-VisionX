"""Loader discovery without hard-coding formats in the GUI."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..domain.images import ImageData
from ..errors import UnsupportedFormatError
from .base import CancelCheck, ImageLoader, LoadLimits, PathLike, ProbeResult


class ImageLoaderRegistry:
    def __init__(self, loaders: Iterable[ImageLoader] = ()) -> None:
        self._loaders: list[ImageLoader] = []
        for loader in loaders:
            self.register(loader)

    @property
    def loaders(self) -> tuple[ImageLoader, ...]:
        return tuple(self._loaders)

    def register(self, loader: ImageLoader) -> None:
        if any(existing.name == loader.name for existing in self._loaders):
            raise ValueError(f"A loader named {loader.name!r} is already registered.")
        self._loaders.append(loader)

    def probe(self, source: PathLike) -> tuple[ImageLoader, ProbeResult]:
        path = Path(source)
        candidates: list[tuple[int, int, ImageLoader, ProbeResult]] = []
        for index, loader in enumerate(self._loaders):
            result = loader.probe(path)
            if result.accepted:
                candidates.append((result.confidence, -index, loader, result))
        if not candidates:
            supported = ", ".join(loader.name for loader in self._loaders)
            raise UnsupportedFormatError(
                f"No image loader recognizes {path.name!r}. Registered loaders: {supported}."
            )
        _, _, loader, result = max(candidates, key=lambda item: (item[0], item[1]))
        return loader, result

    def load(
        self,
        source: PathLike,
        *,
        limits: LoadLimits | None = None,
        cancel: CancelCheck = None,
    ) -> ImageData:
        loader, _ = self.probe(source)
        return loader.load(source, limits=limits, cancel=cancel)


def default_loader_registry() -> ImageLoaderRegistry:
    # Imports remain local so importing the core package never imports optional
    # pydicom/nibabel/Pillow backends.
    from .dicom import DicomLoader
    from .nifti import NiftiLoader
    from .raster import RasterImageLoader

    return ImageLoaderRegistry((DicomLoader(), NiftiLoader(), RasterImageLoader()))
