"""Safe, pluggable image loading interfaces."""

from .base import ImageLoader, LoadLimits, ProbeResult
from .dicom import DicomLoader
from .nifti import NiftiLoader
from .raster import RasterImageLoader
from .registry import ImageLoaderRegistry, default_loader_registry
from .tiles import (
    RasterConversion,
    RasterPageInfo,
    RasterSourceInfo,
    RasterTileCache,
    RasterTileCacheInfo,
    RasterTileSource,
    WsiPyramidTileSource,
)

__all__ = [
    "DicomLoader",
    "ImageLoader",
    "ImageLoaderRegistry",
    "LoadLimits",
    "NiftiLoader",
    "ProbeResult",
    "RasterConversion",
    "RasterImageLoader",
    "RasterPageInfo",
    "RasterSourceInfo",
    "RasterTileCache",
    "RasterTileCacheInfo",
    "RasterTileSource",
    "WsiPyramidTileSource",
    "default_loader_registry",
]
