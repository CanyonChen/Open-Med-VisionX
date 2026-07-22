from __future__ import annotations

import numpy as np
import pytest

from workbench.domain import (
    CompressionKind,
    FractionalType,
    InterpolationMode,
    LabelDefinition,
    LayerReference,
    SegmentationLayer,
    SegmentationValueType,
    SourceFormat,
    SourceReference,
    SourceType,
    SpatialGeometry,
    resample_segmentation_layer,
)
from workbench.errors import ValidationError


def _geometry(shape: tuple[int, int, int], spacing: float = 1.0) -> SpatialGeometry:
    affine = np.diag((spacing, spacing, spacing, 1.0))
    return SpatialGeometry(shape_zyx=shape, affine_ras=affine)


def _layer(*, fractional: bool = False) -> SegmentationLayer:
    geometry = _geometry((2, 2, 2), spacing=2.0)
    values = np.zeros((2, 2, 2), dtype=np.uint8)
    values[1, 1, 1] = 255 if fractional else 1
    return SegmentationLayer(
        layer_id="source-seg",
        series_id="series-1",
        name="Imported segment",
        source=SourceReference(
            source_id="source-1",
            source_type=SourceType.DICOM,
            source_format=SourceFormat.DICOM_SEG,
            compression=CompressionKind.LOSSLESS,
        ),
        array=values,
        geometry=geometry,
        value_type=(
            SegmentationValueType.FRACTIONAL if fractional else SegmentationValueType.BINARY
        ),
        reference=LayerReference(
            local_series_id="series-1",
            dicom_series_uid="1.2.3",
            referenced_sop_instance_uids=("1.2.3.4",),
        ),
        labels=(LabelDefinition(1, "Region", "#FF4040"),),
        reference_geometry=_geometry((4, 4, 4)),
        maximum_fractional_value=255 if fractional else None,
        fractional_type=FractionalType.PROBABILITY if fractional else None,
    )


def test_discrete_resampling_requires_confirmation_and_nearest() -> None:
    layer = _layer()
    target = _geometry((4, 4, 4))
    with pytest.raises(ValidationError, match="explicit user confirmation"):
        resample_segmentation_layer(
            layer,
            target,
            layer_id="derived",
            interpolation=InterpolationMode.NEAREST,
            user_confirmed=False,
        )
    with pytest.raises(ValidationError, match="nearest-neighbour"):
        resample_segmentation_layer(
            layer,
            target,
            layer_id="derived",
            interpolation=InterpolationMode.LINEAR,
            user_confirmed=True,
        )


def test_nearest_resampling_preserves_labels_and_provenance() -> None:
    layer = _layer()
    derived = resample_segmentation_layer(
        layer,
        _geometry((4, 4, 4)),
        layer_id="derived",
        interpolation=InterpolationMode.NEAREST,
        user_confirmed=True,
    )
    assert derived.array.shape == (4, 4, 4)
    assert set(np.unique(derived.array)).issubset({0, 1})
    assert derived.derived_from_layer_ids == ("source-seg",)
    assert derived.transform_chain[-1].interpolation is InterpolationMode.NEAREST
    assert derived.transform_chain[-1].user_confirmed
    assert layer.array.shape == (2, 2, 2)


def test_fractional_resampling_keeps_native_integer_scale() -> None:
    layer = _layer(fractional=True)
    derived = resample_segmentation_layer(
        layer,
        _geometry((4, 4, 4)),
        layer_id="fractional-derived",
        interpolation=InterpolationMode.LINEAR,
        user_confirmed=True,
    )
    assert derived.array.dtype == np.uint8
    assert int(derived.array.max()) <= 255
    assert derived.maximum_fractional_value == 255
    assert derived.fractional_type is FractionalType.PROBABILITY
    assert derived.transform_chain[-1].parameters["fractional_native_scale_preserved"]
