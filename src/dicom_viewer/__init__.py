"""Medical image and computer-vision teaching platform.

The package deliberately keeps optional GUI, model-runtime, and cloud-provider
dependencies out of module import time.  Core domain objects can therefore be
used in tests and notebooks with only NumPy installed.
"""

from .domain.images import (
    AlphaSemantics,
    Capability,
    ColorSpace,
    ImageData,
    ImageSequence2D,
    ImageVolume,
    IntensitySemantics,
    RasterImage2D,
    SourceType,
    SpacingSource,
)
from .domain.transforms import TransformRecord

__all__ = [
    "AlphaSemantics",
    "Capability",
    "ColorSpace",
    "ImageData",
    "ImageSequence2D",
    "ImageVolume",
    "IntensitySemantics",
    "RasterImage2D",
    "SourceType",
    "SpacingSource",
    "TransformRecord",
]

__version__ = "0.1.0"
