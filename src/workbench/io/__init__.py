"""Safe, pluggable image loading interfaces."""

from .base import ImageLoader, LoadLimits, ProbeResult
from .dicom import DicomLoader, DicomLoadResult, DicomReferenceIdentity
from .dicom_annotations import (
    RT_STRUCTURE_SET_STORAGE_UID,
    SEGMENTATION_STORAGE_UID,
    DicomAnnotationImport,
    DicomAnnotationKind,
    DicomAnnotationLimits,
    annotation_layers,
    import_dicom_annotation,
)
from .dicom_series import (
    DicomSeriesDiscovery,
    SeriesSelectionRequiredError,
    SeriesSummary,
    discover_dicom_series,
)
from .label_maps import LabelMapLimits, import_label_map
from .nifti import NiftiLoader, NiftiVolumeSelectionRequiredError
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
    "DicomLoadResult",
    "DicomReferenceIdentity",
    "DicomAnnotationImport",
    "DicomAnnotationKind",
    "DicomAnnotationLimits",
    "DicomSeriesDiscovery",
    "ImageLoader",
    "ImageLoaderRegistry",
    "LoadLimits",
    "LabelMapLimits",
    "NiftiLoader",
    "NiftiVolumeSelectionRequiredError",
    "ProbeResult",
    "RasterConversion",
    "RasterImageLoader",
    "RasterPageInfo",
    "RasterSourceInfo",
    "RasterTileCache",
    "RasterTileCacheInfo",
    "RasterTileSource",
    "RT_STRUCTURE_SET_STORAGE_UID",
    "SEGMENTATION_STORAGE_UID",
    "SeriesSelectionRequiredError",
    "SeriesSummary",
    "WsiPyramidTileSource",
    "annotation_layers",
    "default_loader_registry",
    "discover_dicom_series",
    "import_dicom_annotation",
    "import_label_map",
]
