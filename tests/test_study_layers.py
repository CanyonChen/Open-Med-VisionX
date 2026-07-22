from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from workbench.domain.images import ImageVolume, IntensitySemantics, SourceType
from workbench.domain.studies import (
    CompressionKind,
    ContourLayer,
    ContourPath,
    DeidentificationStatus,
    FractionalType,
    GeometryMatchStatus,
    GeometryTransformRecord,
    ImageSeries,
    ImageStudy,
    InterpolationMode,
    LabelDefinition,
    LayerPresentation,
    LayerReference,
    RasterizationSettings,
    RegionOfInterest,
    SegmentationLayer,
    SegmentationValueType,
    SourceFormat,
    SourceReference,
    SpatialGeometry,
    TransformKind,
    VolumeLayer,
)
from workbench.errors import ValidationError

STUDY_UID = "1.2.826.0.1.3680043.10.100.1"
SERIES_UID = "1.2.826.0.1.3680043.10.100.2"
FRAME_UID = "1.2.826.0.1.3680043.10.100.3"
SOP_UID = "1.2.826.0.1.3680043.10.100.4"


def _geometry(
    shape: tuple[int, int, int] = (2, 3, 4),
    *,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> SpatialGeometry:
    affine = np.eye(4, dtype=np.float64)
    affine[:3, 3] = origin
    return SpatialGeometry(shape, affine)


def _dicom_source(
    source_format: SourceFormat = SourceFormat.DICOM,
) -> SourceReference:
    return SourceReference(
        source_id=f"source-{source_format.value}",
        source_type=SourceType.DICOM,
        source_format=source_format,
        content_sha256="a" * 64,
        provenance={"PatientName": "removed", "decoder": "pydicom"},
    )


def _nifti_source() -> SourceReference:
    return SourceReference(
        source_id="source-nifti",
        source_type=SourceType.NIFTI,
        source_format=SourceFormat.NIFTI,
        content_sha256="b" * 64,
    )


def _series(geometry: SpatialGeometry | None = None) -> ImageSeries:
    geometry = geometry or _geometry()
    return ImageSeries(
        series_id="series-main",
        modality="mr",
        source=_dicom_source(),
        geometry=geometry,
        intensity_semantics=IntensitySemantics.ARBITRARY_SIGNAL,
        study_instance_uid=STUDY_UID,
        series_instance_uid=SERIES_UID,
        frame_of_reference_uid=FRAME_UID,
    )


def _reference(*, frame_uid: str = FRAME_UID) -> LayerReference:
    return LayerReference(
        local_series_id="series-main",
        dicom_series_uid=SERIES_UID,
        frame_of_reference_uid=frame_uid,
        referenced_sop_instance_uids=(SOP_UID,),
    )


def _label(value: int = 1) -> LabelDefinition:
    return LabelDefinition(value=value, name="Region", color="#ff375f")


def _binary_layer(
    *,
    geometry: SpatialGeometry | None = None,
    reference_geometry: SpatialGeometry | None = None,
    source: SourceReference | None = None,
    presentation: LayerPresentation | None = None,
) -> SegmentationLayer:
    geometry = geometry or _geometry()
    array = np.zeros(geometry.shape_zyx, dtype=np.uint8)
    array[0, 1, 1] = 1
    return SegmentationLayer(
        layer_id="seg-original",
        series_id="series-main",
        name="Binary region",
        source=source or _nifti_source(),
        presentation=presentation or LayerPresentation(),
        array=array,
        geometry=geometry,
        value_type=SegmentationValueType.BINARY,
        reference=_reference(),
        labels=(_label(),),
        reference_geometry=reference_geometry,
    )


def test_geometry_is_deeply_immutable_and_reports_specific_differences() -> None:
    affine = np.eye(4)
    geometry = SpatialGeometry((2, 3, 4), affine)
    affine[0, 3] = 99.0

    assert geometry.origin_ras == (0.0, 0.0, 0.0)
    assert not geometry.affine_ras.flags.writeable
    moved = _geometry(origin=(2.0, 0.0, 0.0))
    difference = geometry.compare_to(moved)
    assert difference.shape_matches
    assert difference.spacing_matches
    assert difference.orientation_matches
    assert not difference.affine_matches
    assert difference.mismatched_components == ("affine",)
    with pytest.raises(ValueError):
        geometry.affine_ras[0, 0] = 2.0


def test_source_reference_is_opaque_versioned_and_phi_minimized() -> None:
    source = _dicom_source()
    assert "PatientName" not in source.provenance
    assert source.provenance["decoder"] == "pydicom"
    assert source.version == "1"
    with pytest.raises(ValidationError, match="opaque identifier"):
        SourceReference(
            source_id="C:/private/case.dcm",
            source_type=SourceType.DICOM,
            source_format=SourceFormat.DICOM,
        )


def test_jpeg_and_unverified_tiff_are_rejected_as_segmentation_sources() -> None:
    jpeg = SourceReference(
        source_id="jpeg-mask",
        source_type=SourceType.RASTER,
        source_format=SourceFormat.JPEG,
    )
    with pytest.raises(ValidationError, match="Lossy"):
        _binary_layer(source=jpeg)

    tiff = SourceReference(
        source_id="tiff-mask",
        source_type=SourceType.RASTER,
        source_format=SourceFormat.TIFF,
    )
    with pytest.raises(ValidationError, match="verified as lossless"):
        _binary_layer(source=tiff)

    lossless_tiff = SourceReference(
        source_id="lossless-tiff-mask",
        source_type=SourceType.RASTER,
        source_format=SourceFormat.TIFF,
        compression=CompressionKind.LOSSLESS,
    )
    assert _binary_layer(source=lossless_tiff).source.compression is CompressionKind.LOSSLESS


def test_binary_and_fractional_dicom_seg_have_distinct_lossless_contracts() -> None:
    geometry = _geometry()
    fractional_pixels = np.zeros(geometry.shape_zyx, dtype=np.uint8)
    fractional_pixels[0, 1, 1] = 128
    layer = SegmentationLayer(
        layer_id="fractional-seg",
        series_id="series-main",
        name="Fractional region",
        source=_dicom_source(SourceFormat.DICOM_SEG),
        array=fractional_pixels,
        geometry=geometry,
        value_type=SegmentationValueType.FRACTIONAL,
        reference=_reference(),
        labels=(_label(4),),
        reference_geometry=geometry,
        maximum_fractional_value=255,
        fractional_type=FractionalType.PROBABILITY,
        threshold=0.5,
    )
    fractional_pixels[0, 1, 1] = 0

    assert int(layer.array[0, 1, 1]) == 128
    assert not layer.array.flags.writeable
    assert layer.value_type is SegmentationValueType.FRACTIONAL
    assert layer.maximum_fractional_value == 255
    assert layer.statistics[0].maximum == pytest.approx(128 / 255)
    assert layer.geometry_match_status is GeometryMatchStatus.MATCHED

    with pytest.raises(ValidationError, match="referenced series and SOP"):
        SegmentationLayer(
            layer_id="invalid-seg",
            series_id="series-main",
            name="Missing references",
            source=_dicom_source(SourceFormat.DICOM_SEG),
            array=np.zeros(geometry.shape_zyx, dtype=np.uint8),
            geometry=geometry,
            value_type=SegmentationValueType.BINARY,
            reference=LayerReference(local_series_id="series-main"),
            labels=(_label(),),
        )


def test_geometry_mismatch_blocks_overlay_and_lists_every_difference() -> None:
    reference_geometry = _geometry()
    shifted_geometry = _geometry(origin=(5.0, 0.0, 0.0))
    layer = _binary_layer(
        geometry=shifted_geometry,
        reference_geometry=reference_geometry,
    )

    assert layer.geometry_match_status is GeometryMatchStatus.REQUIRES_RESAMPLING
    report = layer.reference_report(_series(reference_geometry))
    assert report.status is GeometryMatchStatus.REQUIRES_RESAMPLING
    assert not report.overlay_allowed
    assert report.geometry_difference is not None
    assert report.geometry_difference.mismatched_components == ("affine",)


def test_resampling_requires_confirmation_and_preserves_original_layer() -> None:
    reference_geometry = _geometry()
    shifted_geometry = _geometry(origin=(5.0, 0.0, 0.0))
    original = _binary_layer(
        geometry=shifted_geometry,
        reference_geometry=reference_geometry,
    )

    with pytest.raises(ValidationError, match="explicit user confirmation"):
        original.derive_resampled(
            layer_id="seg-resampled",
            name="Resampled",
            array=np.zeros(reference_geometry.shape_zyx, dtype=np.uint8),
            target_geometry=reference_geometry,
            interpolation=InterpolationMode.NEAREST,
            user_confirmed=False,
        )

    result = original.derive_resampled(
        layer_id="seg-resampled",
        name="Resampled",
        array=np.zeros(reference_geometry.shape_zyx, dtype=np.uint8),
        target_geometry=reference_geometry,
        interpolation=InterpolationMode.NEAREST,
        user_confirmed=True,
    )
    assert original.geometry.origin_ras == (5.0, 0.0, 0.0)
    assert result.geometry.origin_ras == (0.0, 0.0, 0.0)
    assert result.derived_from_layer_ids == ("seg-original",)
    assert result.transform_chain[-1].interpolation is InterpolationMode.NEAREST
    assert result.reference_report(_series()).status is GeometryMatchStatus.RESAMPLED

    with pytest.raises(ValidationError, match="nearest-neighbour"):
        original.derive_resampled(
            layer_id="seg-invalid-linear",
            name="Invalid resampling",
            array=np.zeros(reference_geometry.shape_zyx, dtype=np.uint8),
            target_geometry=reference_geometry,
            interpolation=InterpolationMode.LINEAR,
            user_confirmed=True,
        )


def test_probability_resampling_requires_continuous_interpolation() -> None:
    source_geometry = _geometry(origin=(4.0, 0.0, 0.0))
    target_geometry = _geometry()
    transform = GeometryTransformRecord.resampling(
        source_geometry,
        target_geometry,
        interpolation=InterpolationMode.NEAREST,
        user_confirmed=True,
    )
    with pytest.raises(ValidationError, match="continuous interpolation"):
        SegmentationLayer(
            layer_id="probability",
            series_id="series-main",
            name="Probability map",
            source=_nifti_source(),
            original_geometry=source_geometry,
            transform_chain=(transform,),
            array=np.zeros(target_geometry.shape_zyx, dtype=np.float32),
            geometry=target_geometry,
            value_type=SegmentationValueType.PROBABILITY,
            reference=_reference(),
            labels=(_label(),),
            reference_geometry=target_geometry,
        )


def test_frame_of_reference_mismatch_blocks_overlay_even_when_affine_matches() -> None:
    layer = _binary_layer(geometry=_geometry(), reference_geometry=_geometry())
    mismatched = ImageSeries(
        series_id="series-main",
        modality="MR",
        source=_dicom_source(),
        geometry=_geometry(),
        intensity_semantics=IntensitySemantics.ARBITRARY_SIGNAL,
        study_instance_uid=STUDY_UID,
        series_instance_uid=SERIES_UID,
        frame_of_reference_uid="1.2.826.0.1.3680043.10.100.99",
    )
    report = layer.reference_report(mismatched)
    assert report.status is GeometryMatchStatus.REFERENCE_MISMATCH
    assert report.frame_of_reference_matches is False
    assert not report.overlay_allowed


def test_rtstruct_contours_keep_world_points_and_require_both_dicom_references() -> None:
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    contour = ContourPath(points, referenced_sop_instance_uid=SOP_UID)
    roi = RegionOfInterest(1, "Target", "#34c759", (contour,))
    layer = ContourLayer(
        layer_id="rtstruct",
        series_id="series-main",
        name="RT Structure Set",
        source=_dicom_source(SourceFormat.RTSTRUCT),
        reference=_reference(),
        rois=(roi,),
        reference_geometry=_geometry(),
    )
    points[0, 0] = 99.0

    assert float(layer.rois[0].contours[0].points_ras[0, 0]) == 0.0
    assert not layer.rois[0].contours[0].points_ras.flags.writeable
    assert layer.reference_report(_series()).overlay_allowed

    with pytest.raises(ValidationError, match="Frame of Reference"):
        ContourLayer(
            layer_id="invalid-rtstruct",
            series_id="series-main",
            name="Invalid RT Structure Set",
            source=_dicom_source(SourceFormat.RTSTRUCT),
            reference=LayerReference(
                local_series_id="series-main",
                dicom_series_uid=SERIES_UID,
            ),
            rois=(roi,),
        )


def test_contour_rasterization_is_explicit_and_versioned() -> None:
    geometry = _geometry()
    points = ContourPath(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]]))
    roi = RegionOfInterest(1, "Target", "#34c759", (points,))
    settings = RasterizationSettings(geometry, supersampling=4, user_confirmed=True)
    record = GeometryTransformRecord(
        kind=TransformKind.RASTERIZE,
        source_geometry_fingerprint=geometry.fingerprint,
        target_geometry_fingerprint=geometry.fingerprint,
        user_confirmed=True,
    )
    layer = ContourLayer(
        layer_id="rasterized-contour",
        series_id="series-main",
        name="Rasterized contour preview",
        source=_dicom_source(SourceFormat.RTSTRUCT),
        reference=_reference(),
        rois=(roi,),
        reference_geometry=geometry,
        rasterization=settings,
        transform_chain=(record,),
    )
    assert layer.current_geometry is geometry
    assert layer.rasterization is not None
    assert layer.rasterization.user_confirmed


def test_volume_series_and_study_updates_return_new_versions() -> None:
    geometry = _geometry()
    volume = ImageVolume(
        np.zeros(geometry.shape_zyx, dtype=np.float32),
        SourceType.DICOM,
        IntensitySemantics.ARBITRARY_SIGNAL,
        modality="MR",
    )
    base = VolumeLayer(
        layer_id="base-volume",
        series_id="series-main",
        name="Base volume",
        source=_dicom_source(),
        volume=volume,
        is_base_image=True,
    )
    series = _series().with_layer(base)
    study = ImageStudy(
        study_id="anonymous-study-1",
        source_type=SourceType.DICOM,
        series=(series,),
        deidentification_status=DeidentificationStatus.VERIFIED,
        provenance={"PatientID": "removed", "importer": "test"},
    )
    segmentation = _binary_layer(reference_geometry=geometry)
    updated = study.with_layer("series-main", segmentation)

    assert len(study.find_series("series-main").layers) == 1
    assert len(updated.find_series("series-main").layers) == 2
    assert updated.revision == study.revision + 1
    assert updated.find_series("series-main").revision == series.revision + 1
    assert "PatientID" not in updated.provenance
    with pytest.raises(FrozenInstanceError):
        updated.study_id = "changed"  # type: ignore[misc]


def test_presentation_is_controlled_and_locked_layers_cannot_be_replaced() -> None:
    original = _binary_layer(reference_geometry=_geometry())
    hidden = original.with_presentation(
        visible=False,
        locked=True,
        opacity=0.25,
        color="#0a84ff",
    )
    assert original.presentation.visible
    assert hidden.presentation == LayerPresentation(False, True, 0.25, "#0A84FF")
    assert hidden.revision == original.revision + 1

    series = _series().with_layer(hidden)
    with pytest.raises(ValidationError, match="locked"):
        series.replace_layer(hidden.with_presentation(opacity=0.5))
    with pytest.raises(ValidationError, match="locked"):
        hidden.derive_resampled(
            layer_id="new-layer",
            name="New layer",
            array=hidden.array,
            target_geometry=hidden.geometry,
            interpolation=InterpolationMode.NEAREST,
            user_confirmed=True,
        )


def test_series_presentation_and_label_edits_are_immutable_without_copying_voxels() -> None:
    original = _binary_layer(reference_geometry=_geometry())
    series = _series().with_layer(original)

    hidden_series = series.with_layer_presentation(original.layer_id, visible=False, locked=True)
    hidden = hidden_series.layers[-1]

    assert hidden is not original
    assert isinstance(hidden, SegmentationLayer)
    assert hidden.array is original.array
    assert not hidden.presentation.visible
    assert hidden.presentation.locked

    unlocked_series = hidden_series.with_layer_presentation(hidden.layer_id, locked=False)
    unlocked = unlocked_series.layers[-1]
    assert not unlocked.presentation.locked
    assert unlocked.array is original.array

    label_series = unlocked_series.with_segmentation_label_visibility(
        unlocked.layer_id,
        unlocked.labels[0].value,
        False,
    )
    updated = label_series.layers[-1]
    assert isinstance(updated, SegmentationLayer)
    assert updated.array is original.array
    assert not updated.labels[0].visible


def test_four_dimensional_series_requires_an_explicit_nonspatial_axis() -> None:
    geometry = _geometry()
    with pytest.raises(ValidationError, match="exactly one"):
        ImageSeries(
            series_id="series-4d",
            modality="MR",
            source=_nifti_source(),
            geometry=geometry,
            intensity_semantics=IntensitySemantics.ARBITRARY_SIGNAL,
            data_shape=(5, *geometry.shape_zyx),
        )

    series = ImageSeries(
        series_id="series-4d",
        modality="MR",
        source=_nifti_source(),
        geometry=geometry,
        intensity_semantics=IntensitySemantics.ARBITRARY_SIGNAL,
        data_shape=(5, *geometry.shape_zyx),
        time_axis=0,
    )
    assert series.shape == (5, 2, 3, 4)
    assert series.time_axis == 0


def test_base_volume_must_match_series_world_geometry_not_only_shape() -> None:
    shifted = _geometry(origin=(1.0, 0.0, 0.0))
    affine = np.array(shifted.affine_ras, copy=True)
    volume = ImageVolume(
        np.zeros(shifted.shape_zyx, dtype=np.float32),
        SourceType.DICOM,
        IntensitySemantics.ARBITRARY_SIGNAL,
        affine=affine,
        origin=shifted.origin_ras,
        modality="MR",
    )
    layer = VolumeLayer(
        layer_id="shifted-base",
        series_id="series-main",
        name="Shifted base",
        source=_dicom_source(),
        volume=volume,
        is_base_image=True,
    )
    with pytest.raises(ValidationError, match="series geometry"):
        _series(_geometry()).with_layer(layer)
